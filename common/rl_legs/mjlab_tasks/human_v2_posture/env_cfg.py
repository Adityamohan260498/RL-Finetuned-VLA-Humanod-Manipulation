"""LEGS-ONLY V2 + POSTURE: the legs-only paper method (15 joints, arms external) with
three extra commands the actor observes -- pelvis height, torso pitch (lean), waist
yaw -- and payloads that appear / disappear mid-episode (picking up / putting down).

Built on human_v2_legs (same gait modes, clock, Table IV rewards, symmetry, sysid DR);
see ~/g1/training_legs/README.md. Changes:
  * PostureCommand (posture.py): height 0.50-0.80 m, lean -0.1..0.6 rad, waist yaw
    +-0.8 rad while standing; walk band 0.70-0.80 m / -0.05..0.25 / +-0.3 while
    moving; rate-limited; STAND -> WALK only once back in the walk band.
  * Paper rewards re-targeted to the command: base height -> commanded height (+ an exp
    tracking term), orientation -> pelvis roll + pitch beyond the commanded lean, lin
    vel z / min torso ang vel minus the commanded rates, waist-yaw deviation -> waist-yaw
    tracking, soft upper pose -> waist pose, straight knee fades out below 0.72 m,
    base motion / stillness only once the posture target settled and only on the
    policy's joints. New: torso pitch tracking.
  * carry_toggle: every second, free envs pick up a box (p_pick) and carrying envs put it
    down (p_drop); palm mass steps in one physics step.
  * ArmCommand gets a 'pick' mode (arms forward / down).
  * Standing is 25% of commands (warehouse: lots of standing work).

Curriculum (iterations), squat depth conditioned on payload:
  P1     0- 4000  WALK only, posture nominal (commands widen to full range by 2500)
  P2  4000- 8000  + STAND / W2S; posture: height >= 0.62, lean <= 0.3, yaw +-0.5;
                  arms + reach / pick
  P3  8000-13000  full posture range; payload 0-5 kg (reset 30%, pick-ups 8%/s);
                  with a load the squat floor is lerp(0.50, 0.70, kg/10); arms + carry
  P4 13000-18000  payload 0-11 kg; squat floor at 10 kg -> 0.60
  P5 18000-22000  final: 50% of carriers at 11 kg; full squat depth at any load
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.human.mdp.carry import PALMS
from mjlab.tasks.human_v2 import mdp
from mjlab.tasks.human_v2.env_cfg import PROBS_WALK
from mjlab.tasks.human_v2.mdp import STAND, WALK
from mjlab.tasks.human_v2_legs.env_cfg import LEG_WAIST, g1_human_v2_legs_env_cfg
from mjlab.tasks.payload.mdp.observations import pelvis_height_fk
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import posture as P

IT = 24
P2, P3, P4, P5 = 4000, 8000, 13000, 18000
# Run 3 (resumed): switching stand/W2S + squats/lean + reach/pick on together at 4000 dropped
# episode length 990 -> ~720 with no recovery by 4300. P2 is now staggered (resumed from
# model_4000): 4000 stand/W2S only, 5000 + reach/pick arms, 6000 + squat/lean/yaw.
P2B, P2C = 5000, 6000
MAX_ITERATIONS = 22_000
ACTOR_HISTORY = 10  # IMU + joint terms (actions keep V2's 5)

PROBS_FULL = {  # V2's mix with standing raised to 25%
  "stand": 0.25,
  "slow": 0.07,
  "vx": 0.14,
  "vy": 0.06,
  "wz": 0.12,
  "vx_wz": 0.14,
  "vy_wz": 0.06,
  "vx_vy": 0.06,
  "all": 0.10,
}
TWIST_STAGES = [
  {
    "step": 0,
    "vx_max": 1.0,
    "mixed_vx_max": 1.0,
    "vy_range": (0.6, 0.8),
    "wz_range": (0.5, 1.0),
    "category_probs": PROBS_WALK,
  },
  {
    "step": 2500 * IT,
    "vx_max": 1.6,
    "mixed_vx_max": 1.2,
    "vy_range": (0.6, 1.0),
    "wz_range": (0.5, 1.2),
  },
  {"step": P2 * IT, "category_probs": PROBS_FULL, "resampling_time_range": (2.0, 5.0)},
]
GAIT_STAGES = [{"step": 0, "walk_only": True}, {"step": P2 * IT, "walk_only": False}]
POSTURE_STAGES = [
  {
    "step": 0,
    "p_nominal": 1.0,
    "height_range": (0.62, 0.80),
    "pitch_range": (-0.05, 0.3),
    "yaw_max": 0.5,
    "heavy_floor": 0.70,
    "walk_height": (0.70, 0.80),
    "walk_pitch": (-0.05, 0.25),
  },
  {"step": P2C * IT, "p_nominal": 0.5},
  {
    "step": P3 * IT,
    "p_nominal": 0.35,
    "height_range": (P.H_MIN, 0.80),
    "pitch_range": (-0.1, 0.6),
    "yaw_max": 0.8,
  },
  {"step": P4 * IT, "heavy_floor": 0.60},
  {"step": P5 * IT, "heavy_floor": P.H_MIN},
]
# EXTENDED (default): crouch-walking and a deeper lean, phased in after the base skills.
#   P3: walk down to 0.65 m, walking lean <= 0.35;  P4: walk down to 0.60 m (speed x0.3
#   at the bottom), walking lean <= 0.45, standing lean <= 0.8.
# If these fail, resume the same run with the SAFE task (identical observations), which
# keeps the original bands: walk >= 0.70 m, walking lean <= 0.25, standing lean <= 0.6.
EXTENDED_POSTURE = {
  P3: {"walk_height": (0.65, 0.80), "walk_pitch": (-0.05, 0.35)},
  P4: {
    "walk_height": (0.60, 0.80),
    "walk_pitch": (-0.05, 0.45),
    "pitch_range": (-0.1, 0.8),
  },
}


def posture_stages(extended: bool) -> list[dict]:
  stages = deepcopy(POSTURE_STAGES)
  if extended:
    for st in stages:
      st.update(EXTENDED_POSTURE.get(st["step"] // IT, {}))
  return stages


_NO = {"p_carry": 0.0, "mass_range": (0.0, 0.0), "p_full": 0.0}
CARRY_STAGES = [
  {"step": 0, "params": dict(_NO)},
  {
    "step": P3 * IT,
    "params": {"p_carry": 0.3, "mass_range": (0.0, 5.0), "p_full": 0.0},
  },
  {
    "step": P4 * IT,
    "params": {"p_carry": 0.5, "mass_range": (0.0, 11.0), "p_full": 0.2},
  },
  {
    "step": P5 * IT,
    "params": {"p_carry": 0.5, "mass_range": (0.0, 11.0), "p_full": 0.5},
  },
]
TOGGLE_STAGES = [
  {
    "step": 0,
    "params": {"p_pick": 0.0, "p_drop": 0.0, "mass_range": (0.0, 0.0), "p_full": 0.0},
  },
  {
    "step": P3 * IT,
    "params": {"p_pick": 0.08, "p_drop": 0.06, "mass_range": (0.5, 5.0), "p_full": 0.0},
  },
  {
    "step": P4 * IT,
    "params": {
      "p_pick": 0.08,
      "p_drop": 0.06,
      "mass_range": (0.5, 11.0),
      "p_full": 0.2,
    },
  },
  {
    "step": P5 * IT,
    "params": {
      "p_pick": 0.08,
      "p_drop": 0.06,
      "mass_range": (0.5, 11.0),
      "p_full": 0.5,
    },
  },
]
# Collision-free reach (CPU check: 0.2 % of poses touch the body vs 50 % for +-0.6 rad on
# every joint). Carry pose: the shared one hits the hips in 12-35 % of deep squat + lean poses;
# CARRY_SAFE is 0 % at 0.55-0.76 m and 0-0.4 rad lean (palms 0.28 m fwd, 0.50 m apart).
# NOT active in the running run; enable with SAFE_ARMS = True on a resume.
SAFE_ARMS = False
CARRY_SAFE = {"shoulder_pitch": -0.55, "shoulder_roll": 0.40, "elbow": 0.95}
REACH_BOX = {
  "shoulder_pitch": (-1.2, 0.5),
  "shoulder_roll": (0.25, 0.8),
  "shoulder_yaw": (-0.2, 0.4),
  "elbow": (0.2, 1.2),
  "wrist_roll": (-0.4, 0.4),
  "wrist_pitch": (-0.4, 0.4),
  "wrist_yaw": (-0.4, 0.4),
}
ARM_STAGES = [
  {"step": 0, "mode_probs": {"swing": 0.7, "default": 0.3}},
  {
    "step": P2B * IT,
    "mode_probs": {"swing": 0.4, "default": 0.2, "reach": 0.25, "pick": 0.15},
  },
  {
    "step": P3 * IT,
    "mode_probs": {
      "swing": 0.3,
      "default": 0.15,
      "reach": 0.2,
      "pick": 0.2,
      "carry": 0.15,
    },
  },
]


def _final(stages):
  out = {}
  for st in stages:
    out.update({k: v for k, v in st.items() if k != "step"})
  return out


def _set(cfg_obj, values: dict):
  for k, v in values.items():
    setattr(cfg_obj, k, deepcopy(v))


def g1_human_v2_posture_env_cfg(
  play: bool = False, extended: bool = True, safe_arms: bool = SAFE_ARMS
) -> ManagerBasedRlEnvCfg:
  cfg = g1_human_v2_legs_env_cfg(play=play)
  policy_joints = SceneEntityCfg("robot", joint_names=LEG_WAIST)

  # ---- commands ------------------------------------------------------------------
  cfg.commands["posture"] = P.PostureCommandCfg()
  cfg.commands["gait"].posture_command_name = "posture"
  twist, posture = cfg.commands["twist"], cfg.commands["posture"]
  arm = cfg.commands["arm_command"]
  if safe_arms:
    arm.reach_box = dict(REACH_BOX)
    arm.carry_override = dict(CARRY_SAFE)
    arm.default_roll_out = (
      0.30  # sim check: hand-hip grazing 40-60 % -> ~0-9 % of frames
    )
  for k in ("twist_stages", "gait_stages", "carry_payload", "arm_stages"):
    cfg.curriculum.pop(k, None)
  if play:
    _set(twist, _final(TWIST_STAGES))
    cfg.commands["gait"].walk_only = False
    _set(posture, _final(posture_stages(extended)))
    arm.mode_probs = dict(ARM_STAGES[-1]["mode_probs"])
  else:
    _set(twist, {k: v for k, v in TWIST_STAGES[0].items() if k != "step"})
    cfg.commands["gait"].walk_only = True
    stages_p = posture_stages(extended)
    _set(posture, {k: v for k, v in stages_p[0].items() if k != "step"})
    arm.mode_probs = dict(ARM_STAGES[0]["mode_probs"])
    for name, cmd, stages in (
      ("twist_stages", "twist", TWIST_STAGES),
      ("gait_stages", "gait", GAIT_STAGES),
      ("posture_stages", "posture", stages_p),
      ("arm_stages", "arm_command", ARM_STAGES),
    ):
      cfg.curriculum[name] = CurriculumTermCfg(
        func=mdp.command_cfg_stages,
        params={"command_name": cmd, "stages": deepcopy(stages)},
      )

  # ---- payload: at reset + mid-episode pick-up / put-down ---------------------------
  palms = SceneEntityCfg("robot", body_names=PALMS.body_names)
  carry_p = CARRY_STAGES[-1 if play else 0]["params"]
  cfg.events["carry_payload"].params.update(deepcopy(carry_p))
  toggle_p = TOGGLE_STAGES[-1 if play else 0]["params"]
  cfg.events["carry_toggle"] = EventTermCfg(
    mode="interval",
    func=P.carry_toggle,
    interval_range_s=(1.0, 1.0),
    is_global_time=True,
    params={**deepcopy(toggle_p), "asset_cfg": palms},
  )
  if not play:
    for name, stages in (
      ("carry_payload", CARRY_STAGES),
      ("carry_toggle", TOGGLE_STAGES),
    ):
      cfg.curriculum[name] = CurriculumTermCfg(
        func=envs_mdp.event_curriculum,
        params={"event_name": name, "stages": deepcopy(stages)},
      )

  # ---- observations -------------------------------------------------------------------
  # Actor: the posture command + the pelvis height it can compute itself (forward
  # kinematics: pelvis above the lower foot, as policy #4 did), so it knows the error.
  for g in ("actor", "critic"):
    cfg.observations[g].terms["posture"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "posture"}
    )
    cfg.observations[g].terms["pelvis_height"] = ObservationTermCfg(
      func=pelvis_height_fk,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
      },
    )

  # Actor extras (all computable on the robot):
  #   torso gravity  : pelvis IMU + waist joints by FK (the lean is commanded in torso terms)
  #   posture_goal   : where the posture target is heading + time to get there
  #   w2s_progress   : 0 -> 1 through walk-to-stand
  #   mass_estimate  : (kg / 10, known) from the arm policy / operator; (0, 0) = unknown
  #   history 5 -> 10 steps (0.2 s) on IMU and joint terms (actions stay 5)
  actor = cfg.observations["actor"].terms
  actor["torso_projected_gravity"] = ObservationTermCfg(
    func=mdp.body_projected_gravity,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  actor["posture_goal"] = ObservationTermCfg(func=P.posture_goal)
  actor["w2s_progress"] = ObservationTermCfg(func=P.w2s_progress)
  actor["mass_estimate"] = ObservationTermCfg(
    func=P.mass_estimate, params={"p_known": 0.75, "rel_err": 0.3, "step_std": 0.1}
  )
  for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    actor[name] = replace(actor[name], history_length=ACTOR_HISTORY)

  # Critic: everything the actor has that it lacked + the randomised physics (sim only).
  critic = cfg.observations["critic"].terms
  critic["posture_goal"] = ObservationTermCfg(func=P.posture_goal)
  critic["w2s_progress"] = ObservationTermCfg(func=P.w2s_progress)
  critic["mass_estimate"] = ObservationTermCfg(func=P.mass_estimate_shared)
  critic["sysid_params"] = ObservationTermCfg(func=P.sysid_params)

  # ---- rewards: paper set re-targeted to the posture command ------------------------------
  feet = "feet_ground_contact"
  knees = SceneEntityCfg("robot", joint_names=("left_knee_joint", "right_knee_joint"))
  rw = cfg.rewards
  retargeted = {
    # Run 1 (stopped at it 1657): lin-vel tracking flat at 23 % while stability terms
    # sat at 94-98 % -> tracking must dominate.
    "track_lin_vel": (P.track_lin_vel, 3.0, {"std": 0.5}),
    # Run 2 (it 300-1200): turning flat at 28-31 % while speed rose to 64 % -> turning
    # weighted like speed; resumed from run 2 model_1250.
    "track_yaw": (P.track_yaw, 3.0, {"std": 0.5}),
    "base_height": (P.height_l2, -10.0, {}),
    "track_height": (P.track_height, 2.0, {"std": 0.08}),  # as #4; gait bob
    "track_torso_pitch": (P.track_torso_pitch, 2.0, {}),
    "orientation": (P.pelvis_orientation, -1.0, {}),
    "uprightness": (P.pelvis_orientation, -1.0, {"modes": (STAND,)}),
    "lin_vel_z": (P.lin_vel_z, -2.0, {}),
    "torso_min_ang_vel": (P.torso_min_ang_vel, 2.0, {}),
    "waist_yaw": (P.track_waist_yaw, 1.2, {}),
    "straight_knee": (
      P.straight_knee_stance,
      0.1,
      {"modes": (WALK,), "sensor_name": feet, "asset_cfg": knees},
    ),
    # it 4000-4950: base_motion 0.0009 / stillness 0.0000 while standing -> kernels too tight
    # to give any gradient (robot keeps moving in STAND). Widened; resumed from model_4750.
    "base_motion": (P.base_motion, 2.5, {"modes": (STAND,), "std": 0.5}),
    "stillness": (
      P.stillness,
      2.0,
      {"modes": (STAND,), "asset_cfg": policy_joints, "std": 3.0},
    ),
    "soft_upper_pose": (P.waist_pose, 4.0, {"modes": (STAND,)}),
    "dof_vel": (envs_mdp.joint_vel_l2, -1e-3, {"asset_cfg": policy_joints}),
    "dof_acc": (envs_mdp.joint_acc_l2, -2.5e-7, {"asset_cfg": policy_joints}),
  }
  for name, (func, w, params) in retargeted.items():
    rw[name] = RewardTermCfg(func=func, weight=w, params=params)
  rw["waist_pitch"].weight = 0.5  # the lean may use some waist pitch; hips do most
  # Paper eq. (2) assumes the policy swings the arms to cancel leg yaw momentum. Here the
  # arms are external, so at weight 5 the cheapest fix was walking slowly (run 1).
  # 1.0 lost the whole-body yaw regulation (pelvis yaw wobble, turning flat) -> 2.0.
  rw["arm_leg_momentum"].weight = 2.0
  rw["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_WAIST
  )

  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(70.0)
  return cfg


def g1_human_v2_stance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """STANCE for tabletop manipulation (VLA drives the arms): the safe-arms posture policy
  fine-tuned to stand still while the arms reach / pick / carry light objects.

  Same observations as the posture task (resume-compatible). Differences: twist = stand
  only; posture goals for table work (0.68-0.80 m, lean -0.05..0.35, waist yaw +-0.5);
  arms mostly reach / pick (safe box); light payloads 0.2-3 kg picked up / put down;
  standing-still rewards up-weighted. All curricula removed (fixed final values), so the
  restored step counter cannot re-apply the walking stages.
  """
  cfg = g1_human_v2_posture_env_cfg(play=play, safe_arms=True)
  for k in (
    "twist_stages",
    "gait_stages",
    "posture_stages",
    "arm_stages",
    "carry_payload",
    "carry_toggle",
  ):
    cfg.curriculum.pop(k, None)
  tw = cfg.commands["twist"]
  tw.category_probs = {k: 0.0 for k in PROBS_FULL}
  tw.category_probs["stand"] = 1.0
  cfg.commands["gait"].walk_only = False
  post = cfg.commands["posture"]
  _set(
    post,
    {
      "p_nominal": 0.4,
      "height_range": (0.68, 0.80),
      "pitch_range": (-0.05, 0.35),
      "yaw_max": 0.5,
      "heavy_floor": P.H_MIN,
      "resampling_time_range": (3.0, 8.0),
    },
  )
  cfg.commands["arm_command"].mode_probs = {
    "reach": 0.4,
    "pick": 0.25,
    "default": 0.15,
    "carry": 0.2,
  }
  cfg.events["carry_payload"].params.update(
    {"p_carry": 0.3, "mass_range": (0.2, 3.0), "p_full": 0.0}
  )
  cfg.events["carry_toggle"].params.update(
    {"p_pick": 0.08, "p_drop": 0.08, "mass_range": (0.2, 3.0), "p_full": 0.0}
  )
  rw = cfg.rewards
  rw["stillness"].weight = 4.0
  rw["base_motion"].weight = 4.0
  rw["stand_contact"].weight = 4.0
  rw["stand_drag"].weight = -1.0
  return cfg


PALMIN_TABLE = __import__("os").path.expanduser(
  "~/g1/humanoid_challenge/common/policy/palmin_arm_table.npy"
)


def g1_human_v2_stance_palmin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """STANCE + palm-in right-arm poses (humanoid_challenge): the stance task where 70 % of the
  right arm's reach goals are palm-in grasp poses (IK table: the hand coming from the right of
  an object, palm facing it, thumb up). Those poses swing the arm out and knocked the stance
  policy over; this teaches the legs to balance them. Everything else as the stance task.
  """
  cfg = g1_human_v2_stance_env_cfg(play=play)
  ac = cfg.commands["arm_command"]
  ac.reach_table = PALMIN_TABLE
  ac.reach_table_prob = 0.7
  ac.mode_probs = {"reach": 0.6, "pick": 0.1, "default": 0.1, "carry": 0.2}
  return cfg
