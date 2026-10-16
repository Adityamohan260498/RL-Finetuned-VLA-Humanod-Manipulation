"""Extra (critic) observations for the human-like gait task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def body_ang_vel_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return quat_apply_inverse(q, w)


def body_projected_gravity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  g = asset.data.gravity_vec_w
  return quat_apply_inverse(q, g.expand(q.shape[0], 3) if g.dim() == 1 else g)
