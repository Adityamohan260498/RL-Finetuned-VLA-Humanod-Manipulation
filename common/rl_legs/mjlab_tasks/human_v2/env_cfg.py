"""V2: Peng, Bao & Zhou's gait-conditioned method on the 29-DoF G1 + the user's goals.

From scratch. See ~/g1/isaac_warehouse/training/PAPER_GAP_ANALYSIS.md and
~/g1/isaac_warehouse/training/V2_README.md.

  * Gait modes STAND / WALK / W2S (one-hot + sin/cos clock in the actor obs) with the
    paper's reward routing; no RUN (user: max 1.6 m/s, goal is carrying).
  * Full Table IV reward set (regulation, arm-swing, walking, standing, W2S, task).
  * Conflict fixes: residual yaw momentum (turn in place), W2S tapered speed target
    (smooth slowdown through low speed), straight-knee only while WALKing, arm-swing
    block OFF while carrying (carry flag = extra mask bit, carry pose ON).
  * User goals: vx 0.6-1.6, vy 0.6-1.0, wz 0.5-1.2, every mix; palm payload 0-11 kg;
    head/torso stability; left/right symmetry (MLP + history + mirror augmentation).
  * Curriculum (paper Sec. III-D + carrying), iterations:
      P1     0- 8000  WALK only (no stand), commands widen to the full range by 4000
      P2  8000-16000  + STAND and W2S (stand 12% of commands, switches every 2-5 s)
      P3 16000-25000  + carrying: 30% of envs 0-5 kg, from 20000 50% of envs 0-11 kg
      P4 25000-30000  final mix: 40% none / 30% exactly 11 kg / 30% 0-11 kg
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
from mjlab.tasks.human_v2 import mdp
from mjlab.tasks.human_v2.mdp import STAND, W2S, WALK
from mjlab.tasks.velocity.config.g1.env_cfgs_sysid import unitree_g1_flat_sysid_env_cfg

IT = 24
HISTORY_LEN = 5
MAX_ITERATIONS = 30_000
P2, P3, P3B, P4 = 8000, 16000, 20000, 25000

PROBS_WALK = {  # P1: no stand / slow
  "stand": 0.0,
  "slow": 0.0,
  "vx": 0.20,
  "vy": 0.08,
  "wz": 0.16,
  "vx_wz": 0.20,
  "vy_wz": 0.10,
  "vx_vy": 0.10,
  "all": 0.16,
}
PROBS_FULL = {
  "stand": 0.12,
  "slow": 0.07,
  "vx": 0.16,
  "vy": 0.07,
  "wz": 0.15,
  "vx_wz": 0.17,
  "vy_wz": 0.08,
  "vx_vy": 0.07,
  "all": 0.11,
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
    "step": 4000 * IT,
    "vx_max": 1.6,
    "mixed_vx_max": 1.2,
    "vy_range": (0.6, 1.0),
    "wz_range": (0.5, 1.2),
  },
  {"step": P2 * IT, "category_probs": PROBS_FULL, "resampling_time_range": (2.0, 5.0)},
]
GAIT_STAGES = [
  {"step": 0, "walk_only": True},
  {"step": P2 * IT, "walk_only": False},
]
CARRY_STAGES = [
  {"step": 0, "params": {"p_carry": 0.0, "mass_range": (0.0, 0.0), "p_full": 0.0}},
  {
    "step": P3 * IT,
    "params": {"p_carry": 0.3, "mass_range": (0.0, 5.0), "p_full": 0.0},
  },
  {
    "step": P3B * IT,
    "params": {"p_carry": 0.5, "mass_range": (0.0, 11.0), "p_full": 0.2},
  },
  {
    "step": P4 * IT,
    "params": {"p_carry": 0.6, "mass_range": (0.0, 11.0), "p_full": 0.5},
  },
]


def _sites():
  return SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def g1_human_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_sysid_env_cfg(play=play)

  # ---- commands -------------------------------------------------------------
  old = cfg.commands["twist"]
  ranges = deepcopy(old.ranges)
  ranges.heading = None
  twist = mdp.HumanVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=False,
    debug_vis=old.debug_vis,
    ranges=ranges,
    viz=old.viz,
    vx_max=1.6,
    mixed_vx_max=1.2,
    vy_range=(0.6, 1.0),
    wz_range=(0.5, 1.2),
    category_probs=dict(PROBS_FULL),
  )
  cfg.commands["twist"] = twist
  cfg.commands["gait"] = mdp.GaitModeCommandCfg(
    twist_command_name="twist", w2s_time=1.5
  )
  cfg.curriculum.pop("command_vel", None)
  if not play:
    s0 = TWIST_STAGES[0]
    for k, v in s0.items():
      if k != "step":
        setattr(twist, k, deepcopy(v))
    cfg.commands["gait"].walk_only = True
    cfg.curriculum["twist_stages"] = CurriculumTermCfg(
      func=mdp.command_cfg_stages,
      params={"command_name": "twist", "stages": deepcopy(TWIST_STAGES)},
    )
    cfg.curriculum["gait_stages"] = CurriculumTermCfg(
      func=mdp.command_cfg_stages,
      params={"command_name": "gait", "stages": deepcopy(GAIT_STAGES)},
    )

  # ---- payload ----------------------------------------------------------------
  palms = SceneEntityCfg(
    "robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link")
  )
  carry0 = (
    CARRY_STAGES[0]["params"]
    if not play
    else {"p_carry": 0.5, "mass_range": (0.0, 11.0), "p_full": 0.5}
  )
  cfg.events["carry_payload"] = EventTermCfg(
    mode="reset", func=mdp.carry_payload, params={**carry0, "asset_cfg": palms}
  )
  if not play:
    cfg.curriculum["carry_payload"] = CurriculumTermCfg(
      func=envs_mdp.event_curriculum,
      params={"event_name": "carry_payload", "stages": deepcopy(CARRY_STAGES)},
    )

  # ---- observations -------------------------------------------------------------
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  for group in (actor, critic):
    group["gait_mode"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "gait"}
    )
    group["carry_flag"] = ObservationTermCfg(func=mdp.carry_flag)
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  critic["carry_mass"] = ObservationTermCfg(func=mdp.carry_mass)
  critic["torso_ang_vel"] = ObservationTermCfg(
    func=mdp.body_ang_vel_b, params={"asset_cfg": torso}
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

  # ---- rewards: Table IV + fixes. (func, weight, params) --------------------------
  feet = "feet_ground_contact"
  knees = SceneEntityCfg("robot", joint_names=("left_knee_joint", "right_knee_joint"))
  r = mdp.routed
  new = {
    # task
    "track_lin_vel": (mdp.track_lin_vel, 1.5, {"std": 0.5}),
    "track_yaw": (mdp.track_yaw, 1.5, {"std": 0.5}),  # paper 1.0; user goal: turning
    # regulation (all modes)
    "lin_vel_z": (mdp.base_lin_vel_z, -2.0, {}),
    "ang_vel_xy": (mdp.base_ang_vel_xy, -0.05, {}),
    "orientation": (mdp.base_orientation, -1.0, {}),
    "base_height": (
      mdp.base_height_l2,
      -10.0,
      {"target": 0.78, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "dof_acc": (envs_mdp.joint_acc_l2, -2.5e-7, {"asset_cfg": SceneEntityCfg("robot")}),
    "dof_vel": (mdp.joint_vel_l2, -1e-3, {}),
    "action_rate": (envs_mdp.action_rate_l2, -0.01, {}),
    "alive": (mdp.alive, 0.15, {}),
    "hip_deviation": (mdp.hip_deviation, -1.0, {}),
    "torso_min_ang_vel": (mdp.torso_min_ang_vel, 2.0, {}),
    "waist_pitch": (mdp.waist_deviation, 1.0, {"joint": "pitch"}),
    "waist_roll": (mdp.waist_deviation, 1.0, {"joint": "roll"}),
    "waist_yaw": (mdp.waist_deviation, 1.2, {"joint": "yaw"}),
    "torso_yaw_smooth": (mdp.torso_yaw_smoothness, 0.8, {}),
    "shoulder_roll": (mdp.shoulder_roll_control, 3.0, {}),
    # arm swing (WALK, not carrying)
    "arm_leg_momentum": (
      r(mdp.arm_leg_momentum_balance),
      5.0,
      {"modes": (WALK,), "carry": False},
    ),
    "arm_energy": (r(mdp.arm_swing_energy), 0.3, {"modes": (WALK,), "carry": False}),
    "elbow_phase": (
      r(mdp.elbow_phase_tracking),
      2.5,
      {"modes": (WALK,), "carry": False},
    ),
    "arm_symmetry": (
      r(mdp.arm_swing_symmetry),
      2.0,
      {"modes": (WALK,), "carry": False},
    ),
    "arm_leg_amp": (mdp.arm_swing, 1.0, {}),  # already WALK/W2S + not-carrying masked
    # carrying (our addition): hold the box pose
    "carry_pose": (mdp.carry_pose, 2.0, {}),
    # walking
    "swing_height": (
      r(mdp.swing_height_tracking),
      -15.0,
      {"modes": (WALK,), "apex": 0.08, "asset_cfg": _sites()},
    ),
    "contact": (r(mdp.contact_pattern), 1.0, {"modes": (WALK,), "sensor_name": feet}),
    "straight_knee": (
      r(mdp.straight_knee_stance),
      0.1,
      {"modes": (WALK,), "sensor_name": feet, "asset_cfg": knees},
    ),
    "feet_drag": (
      r(mdp.feet_drag),
      -0.5,
      {"modes": (WALK,), "sensor_name": feet, "asset_cfg": _sites()},
    ),
    # standing
    "stand_contact": (
      r(mdp.stand_contact),
      2.5,
      {"modes": (STAND,), "sensor_name": feet},
    ),
    "base_motion": (r(mdp.base_motion), 2.5, {"modes": (STAND,)}),
    "soft_upper_pose": (mdp.soft_upper_pose, 4.0, {"modes": (STAND,)}),
    "feet_alignment": (
      r(mdp.feet_alignment),
      0.5,
      {"modes": (STAND,), "asset_cfg": _sites()},
    ),
    "uprightness": (r(mdp.base_orientation), -1.0, {"modes": (STAND,)}),
    "stand_drag": (
      r(mdp.feet_drag),
      -0.2,
      {"modes": (STAND,), "sensor_name": feet, "asset_cfg": _sites()},
    ),
    "feet_flatness": (r(mdp.feet_flatness), 2.5, {"modes": (STAND,)}),
    "stillness": (r(mdp.stillness), 2.0, {"modes": (STAND,)}),
    # walk -> stand
    "w2s_swing_height": (
      r(mdp.swing_height_tracking),
      -20.0,
      {"modes": (W2S,), "apex": 0.08, "asset_cfg": _sites()},
    ),
    "w2s_contact": (
      r(mdp.contact_pattern),
      1.0,
      {"modes": (W2S,), "sensor_name": feet},
    ),
    "w2s_slowdown": (mdp.smooth_slowdown, 0.1, {"modes": (W2S,)}),
    "w2s_drag": (
      r(mdp.feet_drag),
      -0.1,
      {"modes": (W2S,), "sensor_name": feet, "asset_cfg": _sites()},
    ),
  }
  keep = {k: cfg.rewards[k] for k in ("dof_pos_limits", "self_collisions")}
  keep["dof_pos_limits"].weight = -5.0
  keep["self_collisions"].weight = -10.0
  cfg.rewards = dict(keep)
  for name, (func, w, params) in new.items():
    cfg.rewards[name] = RewardTermCfg(func=func, weight=w, params=params)

  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
  return cfg
