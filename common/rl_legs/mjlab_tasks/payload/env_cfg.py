"""G1 velocity task with payload, commanded crouch height, and recovery pushes.

Built on top of the deployable model-#2 layout (no ``base_lin_vel`` in the
actor, 0-20 ms actor observation latency) and adds four things, each phased in
by its own curriculum so the policy is never asked to learn all of them at once:

  A  iters     0-1500   walk, crouch-walk and squat to a commanded height
  B  iters  1500-5000   drivetrain DR opens up (friction / gains / armature)
  C  iters  4000-9000   payload 0 -> 15 kg with a forward CoM shift
  D  iters 14000-20000  pushes grow from a nudge to a shove (lowest priority,
                        so it lands last and never competes with C)

Phases overlap on purpose: by the time the payload arrives the policy already
walks on an uncertain drivetrain, and by the time the hard pushes arrive it is
already carrying load.

Two design points worth stating up front.

*Height is observed through forward kinematics*, not a height sensor -- see
``mdp.pelvis_height_fk``. The reward uses the same quantity the actor sees, so
nothing here depends on a signal the real G1 cannot produce.

*The actor gets observation history.* Payload mass and CoM are not directly
observable from a single frame; they only show up as a mismatch between what
the policy commanded and how the body actually responded. Without memory the
actor has nothing to infer them from, so history is what makes the payload
robustness learnable rather than merely averaged over.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.payload import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

# common_step_counter = iteration * num_steps_per_env (24).
_IT = 24

NOMINAL_HEIGHT = 0.78
CROUCH_FLOOR = 0.585  # 75% of nominal: the lowest height that still walks.
SQUAT_RANGE = (0.45, 0.55)  # Stationary only -- twist is zeroed down here.
HISTORY_LEN = 5  # 100 ms of proprioception at 50 Hz.

# Joints the G1 system identification covered; the rest get gentler treatment.
_LEG_RE = r"^.*_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_joint$"
_ARM_RE = r"^(?!.*(hip|knee|ankle)).*_joint$"

# Actuator group index -> sysid-derived kp/kd scale band. Groups with no
# identification data sit at a neutral +-5%. Values carry the measured
# downward bias: the real drivetrain is softer than the MJCF nominal.
_GAIN_BANDS: dict[int, tuple[float, float]] = {
  0: (0.95, 1.05),  # shoulders, elbows, wrist roll -- no data
  1: (0.805, 1.000),  # hip pitch/yaw, waist yaw
  2: (0.779, 0.983),  # hip roll, knee
  3: (0.95, 1.05),  # wrist pitch/yaw -- no data
  4: (0.95, 1.05),  # waist roll/pitch -- no data
  5: (0.717, 1.050),  # ankles
}


def _stages(*rows: tuple[int, dict[str, Any]]) -> list[Any]:
  return [{"step": it * _IT, "params": p} for it, p in rows]


def _build_schedule() -> dict[str, list[Any]]:
  """All four phases, keyed by event name."""
  s: dict[str, list[Any]] = {}

  # -- Phase B: drivetrain. Grouped rather than per-joint: 12 separate regex
  # keys per term cost ~50% of step time at 4096 envs for no extra fidelity
  # once the ranges within a group are the same.
  s["leg_friction"] = _stages(
    (0, {"ranges": {_LEG_RE: (0.0, 0.3)}}),
    (1500, {"ranges": {_LEG_RE: (0.1, 1.0)}}),
    (5000, {"ranges": {_LEG_RE: (0.2, 2.0)}}),
  )
  s["leg_armature"] = _stages(
    (
      0,
      {
        "ranges": {
          r"^.*_(hip_pitch|hip_yaw)_joint$": (0.0092, 0.0112),
          r"^.*_(hip_roll|knee)_joint$": (0.0226, 0.0276),
          r"^.*_ankle_(pitch|roll)_joint$": (0.0065, 0.0079),
        }
      },
    ),
    (
      1500,
      {
        "ranges": {
          r"^.*_(hip_pitch|hip_yaw)_joint$": (0.0071, 0.0184),
          r"^.*_(hip_roll|knee)_joint$": (0.0176, 0.0452),
          r"^.*_ankle_(pitch|roll)_joint$": (0.0050, 0.0130),
        }
      },
    ),
    (
      5000,
      {
        "ranges": {
          r"^.*_(hip_pitch|hip_yaw)_joint$": (0.0051, 0.0306),
          r"^.*_(hip_roll|knee)_joint$": (0.0126, 0.0753),
          r"^.*_ankle_(pitch|roll)_joint$": (0.0036, 0.0216),
        }
      },
    ),
  )
  s["leg_damping"] = _stages(
    (0, {"ranges": {r"^.*_hip_.*_joint$": (0.3, 3.2), _LEG_RE: (0.1, 0.4)}}),
    (1500, {"ranges": {r"^.*_hip_.*_joint$": (0.25, 4.0), _LEG_RE: (0.05, 1.0)}}),
    (5000, {"ranges": {r"^.*_hip_.*_joint$": (0.23, 4.1), _LEG_RE: (0.01, 2.0)}}),
  )
  for g, (lo, hi) in _GAIN_BANDS.items():
    mid = (lo + hi) / 2
    s[f"pd_gains_{g}"] = _stages(
      (
        0,
        {
          "kp_range": (mid * 0.975, mid * 1.025),
          "kd_range": (mid * 0.975, mid * 1.025),
        },
      ),
      (
        1500,
        {
          "kp_range": (lo + (mid - lo) * 0.5, hi - (hi - mid) * 0.5),
          "kd_range": (lo + (mid - lo) * 0.5, hi - (hi - mid) * 0.5),
        },
      ),
      (5000, {"kp_range": (lo, hi), "kd_range": (lo, hi)}),
    )

  # -- Phase C: payload. Modelled as mass added to the torso with a
  # forward-biased CoM shift, i.e. a load held in close to the chest rather
  # than out at arm's length. The arms stay free to counterbalance.
  s["payload_mass"] = _stages(
    (0, {"ranges": (0.0, 0.0)}),
    (4000, {"ranges": (0.0, 6.0)}),
    (9000, {"ranges": (0.0, 15.0)}),
  )
  s["payload_com"] = _stages(
    (0, {"ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)}}),
    (4000, {"ranges": {0: (-0.03, 0.07), 1: (-0.04, 0.04), 2: (-0.03, 0.08)}}),
    (9000, {"ranges": {0: (-0.05, 0.12), 1: (-0.08, 0.08), 2: (-0.05, 0.15)}}),
  )

  # -- Phase D: disturbance recovery. push_by_setting_velocity writes the base
  # velocity directly, which is a good model of an impulsive hit (a kick, a
  # shove, walking into something) rather than a sustained force.
  def push(lin: float, vert: float, rot: float) -> dict[str, Any]:
    return {
      "velocity_range": {
        "x": (-lin, lin),
        "y": (-lin, lin),
        "z": (-vert, vert),
        "roll": (-rot, rot),
        "pitch": (-rot, rot),
        "yaw": (-rot * 1.5, rot * 1.5),
      }
    }

  s["push_robot"] = _stages(
    (0, push(0.3, 0.2, 0.3)),
    (14000, push(0.6, 0.4, 0.5)),
    (20000, push(1.0, 0.6, 0.8)),
  )
  return s


SCHEDULE = _build_schedule()


def g1_payload_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 flat velocity + payload + crouch-height + push recovery."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Strip model #2's hand-tuned drivetrain DR; replaced below, grouped and
  # phased. effort_limits is kept -- torque headroom is real and unrelated.
  for name in ("joint_friction", "pd_gains"):
    cfg.events.pop(name, None)
    cfg.curriculum.pop(f"dr_{name}", None)

  ##
  # Commands: add a target height alongside the existing twist.
  ##
  cfg.commands["height"] = mdp.UniformHeightCommandCfg(
    resampling_time_range=(3.0, 8.0),
    nominal_height=NOMINAL_HEIGHT,
    crouch_floor=CROUCH_FLOOR,
    squat_range=SQUAT_RANGE,
    nominal_prob=0.4,
    squat_prob=0.2,
    min_speed_scale=0.35,
  )

  ##
  # Observations.
  ##
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms

  for group in (actor, critic):
    group["height_command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "height"}
    )
    group["pelvis_height"] = ObservationTermCfg(func=mdp.pelvis_height_fk)

  # History on proprioception only. The command terms are piecewise-constant,
  # so stacking them wastes input width without adding information.
  for name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "pelvis_height",
  ):
    actor[name] = replace(actor[name], history_length=HISTORY_LEN)

  ##
  # Rewards.
  ##
  cfg.rewards["track_height"] = RewardTermCfg(
    func=mdp.track_height,
    weight=1.0,
    params={"command_name": "height", "std": 0.08},
  )

  # Swap the stock posture reward for one whose tolerance follows the
  # commanded height. The stock term rewards standing up, which directly
  # opposes track_height -- the two would bid against each other on every
  # crouch. Here the leg pitch chain is freed in proportion to commanded
  # depth, while roll/yaw, waist and arms stay pinned at any depth.
  base_walking = dict(cfg.rewards["pose"].params["std_walking"])
  cfg.rewards["pose"] = RewardTermCfg(
    func=mdp.crouch_aware_posture,
    weight=cfg.rewards["pose"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "command_name": "twist",
      "height_command_name": "height",
      "std_standing": {".*": 0.05},
      "std_walking": base_walking,
      # Extra std at full crouch depth, leg pitch chain only.
      "std_crouch_extra": {
        r".*hip_pitch.*": 0.45,
        r".*knee.*": 0.60,
        r".*ankle_pitch.*": 0.30,
        r"^(?!.*(hip_pitch|knee|ankle_pitch)).*$": 0.0,
      },
      "walking_threshold": 0.5,
    },
  )

  # Upper-body stability while squatting. The torso already has two guards:
  # `upright` penalises tilt *angle* (weight 1.0) and `body_ang_vel` penalises
  # roll+pitch *rate* (yaw is excluded). Adding a third, harder penalty on
  # pitch angle would be counterproductive -- a deep squat needs forward lean
  # to keep the CoM over the feet, and a forward-held payload needs backward
  # lean to compensate. Penalising the angle fights both. Penalising the rate
  # suppresses wobble without forbidding a steady, deliberate lean, so that is
  # the knob turned up here.
  cfg.rewards["body_ang_vel"].weight = -0.10  # was -0.05

  ##
  # Events.
  ##
  legs = SceneEntityCfg("robot")
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  stage0 = {k: v[0]["params"] for k, v in SCHEDULE.items()}

  cfg.events["leg_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={"asset_cfg": legs, "operation": "abs", **stage0["leg_friction"]},
  )
  cfg.events["leg_damping"] = EventTermCfg(
    mode="reset",
    func=dr.joint_damping,
    params={"asset_cfg": legs, "operation": "abs", **stage0["leg_damping"]},
  )
  cfg.events["leg_armature"] = EventTermCfg(
    mode="reset",
    func=dr.joint_armature,
    params={"asset_cfg": legs, "operation": "abs", **stage0["leg_armature"]},
  )
  cfg.events["arm_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(_ARM_RE,)),
      "operation": "abs",
      "ranges": (0.0, 0.3),
    },
  )
  for g in _GAIN_BANDS:
    cfg.events[f"pd_gains_{g}"] = EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[g]),
        "operation": "scale",
        **stage0[f"pd_gains_{g}"],
      },
    )
  cfg.events["payload_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={"asset_cfg": torso, "operation": "add", **stage0["payload_mass"]},
  )
  cfg.events["payload_com"] = EventTermCfg(
    mode="reset",
    func=dr.body_com_offset,
    params={"asset_cfg": torso, "operation": "add", **stage0["payload_com"]},
  )
  # Play mode deliberately runs without disturbances so a rollout is
  # reproducible; the base config pops push_robot and we must not re-add it.
  if not play:
    cfg.events["push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params=dict(stage0["push_robot"]),
    )

  # Redrawing DR on every reset costs ~50% of step time at 4096 envs, because
  # early training resets constantly. Holding each env's draw for 200 steps
  # (4 s of sim) recovers most of that and is the more physical semantics
  # anyway -- a robot's drivetrain does not change between episodes.
  for name in SCHEDULE:
    if name in cfg.events and cfg.events[name].mode == "reset":
      cfg.events[name].min_step_count_between_reset = 200
  cfg.events["arm_friction"].min_step_count_between_reset = 200

  ##
  # Curriculum.
  ##
  if play:
    for name, stages in SCHEDULE.items():
      if name in cfg.events:
        cfg.events[name].params.update(stages[-1]["params"])
  else:
    for name, stages in SCHEDULE.items():
      assert name in cfg.events, f"schedule targets missing event '{name}'"
      cfg.curriculum[f"ph_{name}"] = CurriculumTermCfg(
        func=envs_mdp.event_curriculum,
        params={"event_name": name, "stages": deepcopy(stages)},
      )

  return cfg
