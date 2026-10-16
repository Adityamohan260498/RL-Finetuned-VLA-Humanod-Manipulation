"""G1 human-like locomotion + box-carrying payload, trained FROM SCRATCH.

Base: ``unitree_g1_flat_sysid_env_cfg`` -- keeps the drivetrain DR anchored on
resources/g1_legs_identified.json (friction / damping / armature / per-group
gains, ramped to full range by iteration 6000), plus the stock foot-friction,
encoder-bias, CoM, effort-limit DR, 0-20 ms obs latency and base pushes.

What is new (see NOTES in ~/g1/isaac_warehouse/training/README.md):
  * Commands: category sampler, sign-symmetric (vx, vy, wz alone + every mix).
  * Gait clock in the actor obs; stride frequency and duty follow the command.
  * Human-gait rewards (contact pattern, swing profile, straight stance knee,
    anti-phase arm swing, foot drag, stance width) after Peng et al. Table IV.
  * Head/torso stability: torso roll/pitch rate, torso yaw-rate tracking,
    torso vertical speed, torso tilt.
  * Box carrying: total 0-10 kg split across both palms, carry flag in the obs,
    arm swing -> carry pose routing; final 5000 iterations at 11 kg.
  * 5-frame proprioceptive history (payload is only observable through it).
  * Left/right symmetry augmentation in PPO (rl_cfg.py).

Iteration budget: 15,000 (~12.5 h at the measured ~2.9-3.0 s/iter, 4096 envs).
Curriculum (iterations):
     0- 2000  vx 0.6-1.0, vy 0.6-0.8, wz 0.5-1.0; no payload; style weights low
  2000- 6000  vx up to 1.5, vy 0.6-1.0, wz 0.5-1.2; style weights to full
  4000        carrying starts: 30% of envs, 0-4 kg total
  6000-       vx up to 2.0
  7000        carrying 50% of envs, 0-11 kg (20% of them at 11)
 10000-15000  FINAL 5000: 40% no box, 30% exactly 11 kg, 30% 0-11 kg
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
from mjlab.tasks.human import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs_sysid import unitree_g1_flat_sysid_env_cfg

IT = 24  # common_step_counter per iteration (num_steps_per_env)
HISTORY_LEN = 5  # 100 ms at 50 Hz
MAX_ITERATIONS = 15_000
FINAL_PHASE_START = MAX_ITERATIONS - 5_000

# Twist command curriculum (HumanVelocityCommandCfg fields).
COMMAND_STAGES = [
  {"step": 0, "vx_max": 1.0, "vy_range": (0.6, 0.8), "wz_range": (0.5, 1.0)},
  {"step": 2000 * IT, "vx_max": 1.5, "vy_range": (0.6, 1.0), "wz_range": (0.5, 1.2)},
  {"step": 6000 * IT, "vx_max": 2.0},
]

# Carry payload curriculum (carry_payload event params).
CARRY_STAGES = [
  {"step": 0, "params": {"p_carry": 0.0, "mass_range": (0.0, 0.0), "p_full": 0.0}},
  {
    "step": 4000 * IT,
    "params": {"p_carry": 0.3, "mass_range": (0.0, 4.0), "p_full": 0.0},
  },
  {
    "step": 7000 * IT,
    "params": {"p_carry": 0.5, "mass_range": (0.0, 11.0), "p_full": 0.2},
  },
  # Final 5000: never 11 kg only. 40% no box, 30% exactly 11 kg, 30% 0-11 kg.
  {
    "step": FINAL_PHASE_START * IT,
    "params": {"p_carry": 0.6, "mass_range": (0.0, 11.0), "p_full": 0.5},
  },
]

# Style terms start at a fraction of their weight so the policy first learns to
# stand and step at all (from scratch), then the gait is shaped.
STYLE_RAMP = (0, 1500 * IT, 4000 * IT)  # 20% -> 60% -> 100%


def g1_human_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_sysid_env_cfg(play=play)

  ##
  # Commands
  ##
  old = cfg.commands["twist"]
  ranges = deepcopy(old.ranges)
  ranges.heading = None
  cfg.commands["twist"] = mdp.HumanVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),  # mid-episode switches = transitions
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=False,
    debug_vis=old.debug_vis,
    ranges=ranges,
    viz=old.viz,
    **(
      {}
      if not play
      else {"vx_max": 2.0, "vy_range": (0.6, 1.0), "wz_range": (0.5, 1.2)}
    ),
  )
  cfg.commands["gait"] = mdp.GaitClockCommandCfg(twist_command_name="twist")
  cfg.curriculum.pop("command_vel", None)
  if not play:
    cfg.curriculum["command_human"] = CurriculumTermCfg(
      func=mdp.human_command_curriculum,
      params={"command_name": "twist", "stages": deepcopy(COMMAND_STAGES)},
    )

  ##
  # Payload at the palms
  ##
  carry0 = (
    CARRY_STAGES[0]["params"]
    if not play
    else {"p_carry": 0.5, "mass_range": (0.0, 10.0)}
  )
  cfg.events["carry_payload"] = EventTermCfg(
    mode="reset",
    func=mdp.carry_payload,
    params={
      **carry0,
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link")
      ),
    },
  )
  if not play:
    cfg.curriculum["carry_payload"] = CurriculumTermCfg(
      func=envs_mdp.event_curriculum,
      params={"event_name": "carry_payload", "stages": deepcopy(CARRY_STAGES)},
    )

  ##
  # Observations
  ##
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  for group in (actor, critic):
    group["gait_phase"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "gait"}
    )
    group["carry_flag"] = ObservationTermCfg(func=mdp.carry_flag)
  critic["carry_mass"] = ObservationTermCfg(func=mdp.carry_mass)
  # Torso state is privileged for the critic. The actor does not need it
  # explicitly: torso orientation/rate = pelvis IMU composed with the three
  # waist joints it already observes (exact rigid-body FK).
  critic["torso_ang_vel"] = ObservationTermCfg(
    func=mdp.body_ang_vel_b,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  critic["torso_projected_gravity"] = ObservationTermCfg(
    func=mdp.body_projected_gravity,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  for name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  ):
    actor[name] = replace(actor[name], history_length=HISTORY_LEN)

  ##
  # Rewards
  ##
  R = cfg.rewards
  # Task: tracking (std 0.5 m/s is fine up to 2 m/s commands).
  R["track_linear_velocity"].weight = 2.0
  R["track_angular_velocity"].weight = 2.0

  # Replaced by the clock-driven terms below.
  for name in ("air_time", "foot_clearance", "foot_swing_height"):
    R.pop(name, None)

  # Posture: freer knee/hip pitch (straight stance knee + longer stride),
  # tighter waist (head stability), arms handled by arm_swing / carry_pose.
  walk = dict(R["pose"].params["std_walking"])
  walk.update(
    {
      r".*hip_pitch.*": 0.45,
      r".*knee.*": 0.5,
      r".*ankle_pitch.*": 0.3,
      r".*waist_yaw.*": 0.1,
      r".*waist_roll.*": 0.06,
      r".*waist_pitch.*": 0.08,
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.3,
      r".*elbow.*": 0.4,
    }
  )
  run = dict(R["pose"].params["std_running"])
  run.update(
    {
      r".*waist_yaw.*": 0.15,
      r".*shoulder_pitch.*": 0.6,
      r".*shoulder_roll.*": 0.35,
      r".*elbow.*": 0.5,
    }
  )
  R["pose"].params["std_walking"] = walk
  R["pose"].params["std_running"] = run
  R["pose"].params["running_threshold"] = 1.6

  feet = "feet_ground_contact"

  # Every SceneEntityCfg MUST be passed in params: the managers only resolve
  # params entries; a function-default SceneEntityCfg stays "all bodies/sites".
  def sites():
    return SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

  def torso():
    return SceneEntityCfg("robot", body_names=("torso_link",))

  knees = SceneEntityCfg("robot", joint_names=("left_knee_joint", "right_knee_joint"))
  robot = SceneEntityCfg("robot")
  new = {
    # --- gait (paper "Walking" block) ---
    "contact_pattern": (mdp.contact_pattern, 1.5, {"sensor_name": feet}),
    "swing_height": (
      mdp.swing_height_tracking,
      -20.0,
      {"apex": 0.08, "asset_cfg": sites()},
    ),
    "straight_knee": (
      mdp.straight_knee_stance,
      0.5,
      {"sensor_name": feet, "asset_cfg": knees},
    ),
    "feet_drag": (mdp.feet_drag, -0.5, {"sensor_name": feet, "asset_cfg": sites()}),
    "feet_width": (mdp.feet_lateral_distance, -2.0, {"asset_cfg": sites()}),
    "stand_still": (mdp.stand_still, -0.02, {"asset_cfg": robot}),
    # --- arms (paper "Arm swing" block) ---
    "arm_swing": (mdp.arm_swing, 1.0, {}),
    "carry_pose": (mdp.carry_pose, 2.0, {}),
    "yaw_angmom": (
      mdp.yaw_angular_momentum,
      -0.1,
      {"sensor_name": "robot/root_angmom"},
    ),
    # --- head / torso stability (user request) ---
    "torso_yaw_rate": (mdp.torso_yaw_rate_tracking, -0.3, {"asset_cfg": torso()}),
    "torso_vz": (mdp.body_lin_vel_z, -1.0, {"asset_cfg": torso()}),
    "torso_tilt": (mdp.body_flat_orientation, -2.0, {"asset_cfg": torso()}),
    # --- regulation ---
    "base_height": (
      mdp.base_height_l2,
      -10.0,
      {"target": 0.78, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "action_acc": (envs_mdp.action_acc_l2, -0.01, {}),
    "joint_acc": (
      envs_mdp.joint_acc_l2,
      -2.5e-7,
      {"asset_cfg": SceneEntityCfg("robot")},
    ),
    "joint_torque": (
      envs_mdp.joint_torques_l2,
      -1e-5,
      {"asset_cfg": SceneEntityCfg("robot")},
    ),
  }
  for name, (func, w, params) in new.items():
    R[name] = RewardTermCfg(func=func, weight=w, params=params)
  R["body_ang_vel"].weight = -0.5  # torso roll/pitch rate: was -0.05
  R["angular_momentum"].weight = -0.02
  R["action_rate_l2"].weight = -0.05
  R["foot_slip"].weight = -0.25
  R["soft_landing"].weight = -2e-4  # was -1e-5 (old touchdowns 0.66 m/s)
  R["dof_pos_limits"].weight = -5.0

  # Ramp the style terms in so learning to walk at all comes first.
  if not play:
    style = (
      "contact_pattern",
      "swing_height",
      "straight_knee",
      "feet_drag",
      "arm_swing",
      "yaw_angmom",
      "torso_yaw_rate",
      "torso_vz",
      "torso_tilt",
      "body_ang_vel",
      "soft_landing",
      "action_acc",
      "joint_torque",
    )
    for name in style:
      w = R[name].weight
      cfg.curriculum[f"style_{name}"] = CurriculumTermCfg(
        func=envs_mdp.reward_curriculum,
        params={
          "reward_name": name,
          "stages": [
            {"step": STYLE_RAMP[0], "weight": 0.2 * w},
            {"step": STYLE_RAMP[1], "weight": 0.6 * w},
            {"step": STYLE_RAMP[2], "weight": w},
          ],
        },
      )
    for name in style:  # stage 0 applies from step 0; start the term there too
      R[name].weight = 0.2 * R[name].weight

  ##
  # Episode / termination
  ##
  cfg.episode_length_s = 20.0 if not play else cfg.episode_length_s
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)

  return cfg
