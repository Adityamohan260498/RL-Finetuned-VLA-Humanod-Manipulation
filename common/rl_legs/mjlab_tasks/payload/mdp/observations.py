"""Observations for the payload / crouch-walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_FEET = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def pelvis_height_fk(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _FEET,
) -> torch.Tensor:
  """Height of the pelvis above the lower foot, in metres.

  This is deliberately *not* a base-height-above-terrain reading. The G1 has no
  height sensor, and ``rt/lowstate`` carries only quaternion, gyro,
  accelerometer and joint encoders. What it does give you is enough to run
  forward kinematics: with joint angles you get each foot's position relative
  to the pelvis, and with the IMU quaternion you can rotate that into the world
  frame and read off the vertical component. That is exactly what this returns,
  so a policy conditioned on it stays deployable.

  Taking the *lower* foot means the value tracks the stance leg during a step
  rather than jumping when the swing foot rises. While both feet are planted it
  is simply the crouch depth.
  """
  asset = env.scene[asset_cfg.name]
  pelvis_z = asset.data.root_link_pos_w[:, 2]
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return (pelvis_z - feet_z.min(dim=1).values).unsqueeze(-1)
