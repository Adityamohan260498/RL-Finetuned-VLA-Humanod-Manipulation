"""Human-like gait rewards for the G1 (after Peng, Bao & Zhou, "Gait-Conditioned
RL with Multi-Phase Curriculum for Humanoid Locomotion", Table IV), adapted to
mjlab and to what we measured on the old policies (3.6-4.4 steps/s shuffle,
0.66 m/s touchdowns, ~45 deg/s head rotation).

Reward routing (the paper's gait mask) is done with two signals:
  * the gait clock's ``standing`` flag   -> stand vs. walk terms
  * the carry flag (env._carry_flag)     -> arm swing vs. box-holding pose
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .carry import _state as _carry_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_FEET_SITES = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
_TORSO = SceneEntityCfg("robot", body_names=("torso_link",))


def _gait(env: ManagerBasedRlEnv, name: str = "gait"):
  return env.command_manager.get_term(name)


def _contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found > 0  # [B, 2] (left, right)


def _foot_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # flat ground: z = height


# ---- walking / gait-timing terms ------------------------------------------


def contact_pattern(env: ManagerBasedRlEnv, sensor_name: str, gait_name: str = "gait"):
  """+1 per foot whose contact state matches the clock (stance vs swing); /2.

  Standing envs want both feet down. This single term is what replaces the
  shuffle: the clock dictates ~2.3 steps/s at 0.6 m/s.
  """
  g = _gait(env, gait_name)
  match = (_contact(env, sensor_name) == g.desired_stance()).float()
  return match.mean(dim=1)


def swing_height_tracking(
  env: ManagerBasedRlEnv,
  apex: float,
  gait_name: str = "gait",
  asset_cfg: SceneEntityCfg = _FEET_SITES,
):
  """Squared error to a sin-shaped swing profile (apex at mid-swing). Penalty.

  Only scored on feet that the clock says are swinging, and not while standing.
  """
  g = _gait(env, gait_name)
  prog = g.swing_progress()  # [B, 2]
  swinging = (~g.desired_stance()).float()
  target = apex * torch.sin(torch.pi * prog)
  err = torch.square(_foot_z(env, asset_cfg) - target) * swinging
  return err.sum(dim=1)


def straight_knee_stance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target: float = 0.3,
  std: float = 0.15,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", joint_names=("left_knee_joint", "right_knee_joint")
  ),
  gait_name: str = "gait",
):
  """Reward a straighter (human) stance knee. Default pose is 0.67 rad (crouched)."""
  asset: Entity = env.scene[asset_cfg.name]
  knee = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 2] (left, right)
  stance = _contact(env, sensor_name).float()
  walking = (~_gait(env, gait_name).standing).float().unsqueeze(1)
  r = torch.exp(-torch.square(knee - target) / std**2) * stance * walking
  return r.sum(dim=1) / 2


def feet_drag(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  height: float = 0.03,
  asset_cfg: SceneEntityCfg = _FEET_SITES,
):
  """Horizontal foot speed^2 while the foot is low but not in contact. Penalty."""
  asset: Entity = env.scene[asset_cfg.name]
  vxy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  low = (_foot_z(env, asset_cfg) < height) & ~_contact(env, sensor_name)
  return (torch.sum(torch.square(vxy), dim=-1) * low.float()).sum(dim=1)


def feet_lateral_distance(
  env: ManagerBasedRlEnv,
  min_dist: float = 0.17,
  max_dist: float = 0.35,
  asset_cfg: SceneEntityCfg = _FEET_SITES,
):
  """Penalty when stance width (in the heading frame) leaves [min, max]."""
  asset: Entity = env.scene[asset_cfg.name]
  p = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
  d_w = p[:, 0, :] - p[:, 1, :]
  d_b = quat_apply_inverse(asset.data.root_link_quat_w, d_w)
  lat = d_b[:, 1]  # left foot should be +y of right foot
  return torch.relu(min_dist - lat) + torch.relu(lat - max_dist)


def stand_still(
  env: ManagerBasedRlEnv,
  gait_name: str = "gait",
  asset_cfg: SceneEntityCfg = _ROBOT,
):
  """Joint speed^2 while commanded to stand. Penalty."""
  asset: Entity = env.scene[asset_cfg.name]
  standing = _gait(env, gait_name).standing.float()
  return torch.sum(torch.square(asset.data.joint_vel), dim=1) * standing


# ---- head / torso stability -----------------------------------------------


def torso_yaw_rate_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _TORSO,
):
  """(torso yaw rate - commanded yaw rate)^2. Penalty. This is the head-camera
  yaw jitter we measured (2.9-6.5 deg per step on the old policies)."""
  asset: Entity = env.scene[asset_cfg.name]
  w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  cmd = env.command_manager.get_command(command_name)
  return torch.square(w[:, 2] - cmd[:, 2])


def body_lin_vel_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _TORSO):
  """Vertical speed^2 of a body (torso => head bob). Penalty."""
  asset: Entity = env.scene[asset_cfg.name]
  v = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.square(v[:, 2])


def body_flat_orientation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _TORSO):
  """Squared xy of gravity in the body frame (tilt). Penalty."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  g = quat_apply_inverse(q, asset.data.gravity_vec_w.expand(q.shape[0], 3))
  return torch.sum(torch.square(g[:, :2]), dim=1)


def base_height_l2(
  env: ManagerBasedRlEnv, target: float, asset_cfg: SceneEntityCfg = _ROBOT
):
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target)


def yaw_angular_momentum(env: ManagerBasedRlEnv, sensor_name: str):
  """Whole-body yaw angular momentum^2 (paper eq. 2, total part). Penalty.
  Arms swinging against the legs is what drives this down."""
  angmom = env.scene[sensor_name].data  # [B, 3]
  return torch.square(angmom[:, 2])


# ---- arms: swing when free, hold a box when carrying ------------------------


class arm_swing:
  """Anti-phase arm swing: shoulder pitch offset = -gain * same-side hip pitch
  offset (a G1 pitch joint moves the limb backward for +angle, so a leg going
  forward drives its arm backward, and the opposite arm forward).

  Active only while walking and NOT carrying.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    names = list(asset.joint_names)
    self.sh = [names.index(f"{s}_shoulder_pitch_joint") for s in ("left", "right")]
    self.hip = [names.index(f"{s}_hip_pitch_joint") for s in ("left", "right")]
    self.default = asset.data.default_joint_pos

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    gain: float = 0.8,
    std: float = 0.15,
    gait_name: str = "gait",
  ):
    asset: Entity = env.scene["robot"]
    q = asset.data.joint_pos
    d = self.default
    hip_off = q[:, self.hip] - d[:, self.hip]
    target = d[:, self.sh] - gain * hip_off
    err = torch.sum(torch.square(q[:, self.sh] - target), dim=1)
    flag, _ = _carry_state(env)
    active = (~_gait(env, gait_name).standing).float() * (1.0 - flag)
    return torch.exp(-err / std**2) * active


class carry_pose:
  """Hold the box: arm joints near a carry pose while carrying. Reward."""

  POSE = {  # rad. Palms ~0.45 m apart in front of the pelvis, elbows bent.
    "left_shoulder_pitch_joint": -0.35,
    "right_shoulder_pitch_joint": -0.35,
    "left_shoulder_roll_joint": 0.28,
    "right_shoulder_roll_joint": -0.28,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.95,
    "right_elbow_joint": 0.95,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
  }

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    names = list(asset.joint_names)
    self.ids = [names.index(n) for n in self.POSE]
    self.target = torch.tensor(list(self.POSE.values()), device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, std: float = 0.12):
    asset: Entity = env.scene["robot"]
    err = torch.mean(
      torch.square(asset.data.joint_pos[:, self.ids] - self.target), dim=1
    )
    flag, _ = _carry_state(env)
    return torch.exp(-err / std**2) * flag


def track_yaw_rate(env: ManagerBasedRlEnv, std: float, command_name: str = "twist"):
  """Yaw-rate tracking ONLY. The stock track_angular_velocity also counts pelvis
  roll/pitch rate as error; with a human gait the pelvis rocks every step, which
  kept that reward near 10% of max even while turning (measured in run 1)."""
  asset: Entity = env.scene["robot"]
  cmd = env.command_manager.get_command(command_name)
  err = torch.square(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-err / std**2)
