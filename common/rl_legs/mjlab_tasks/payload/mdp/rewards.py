"""Rewards for the payload / crouch-walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .observations import pelvis_height_fk

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_FEET = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def track_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.08,
  asset_cfg: SceneEntityCfg = _FEET,
) -> torch.Tensor:
  """Exponential kernel on the pelvis-above-foot height error.

  Uses the same FK-derived height the actor observes, so the policy is never
  rewarded for matching something it cannot measure on hardware.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target = command[:, 0]
  measured = pelvis_height_fk(env, asset_cfg)[:, 0]
  return torch.exp(-torch.square(target - measured) / std**2)


class crouch_aware_posture:
  """Posture reward whose target tolerance follows the commanded height.

  The stock ``variable_posture`` pulls every joint toward the default standing
  pose. That is a reward for *standing up*, so it fights every commanded
  crouch: the policy gets paid to undo exactly what ``track_height`` asks for,
  and the two rewards end up bidding against each other.

  Here the per-joint tolerance is widened in proportion to how deep a crouch
  was commanded. At nominal height the std is unchanged, so upright posture is
  held just as tightly as before. At the crouch floor (and below, in the squat
  regime) the leg pitch chain gets ``std_crouch_extra`` added, which makes deep
  knee and hip flexion essentially free. Joints that should stay tidy at any
  depth -- roll/yaw, waist, arms -- get no extra and remain pinned.

  The commanded *depth* itself is still rewarded, by ``track_height``. This
  term only stops the posture prior from arguing with it.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def _resolve(key: str) -> torch.Tensor:
      _, _, values = resolve_matching_names_values(
        data=cfg.params[key], list_of_strings=joint_names
      )
      return torch.tensor(values, device=env.device, dtype=torch.float32)

    self.std_standing = _resolve("std_standing")
    self.std_walking = _resolve("std_walking")
    self.std_crouch_extra = _resolve("std_crouch_extra")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_crouch_extra,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    height_command_name: str,
    walking_threshold: float = 0.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_crouch_extra  # Resolved in __init__.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (speed < walking_threshold).float().unsqueeze(1)
    std = self.std_standing * standing + self.std_walking * (1.0 - standing)

    # Duck-typed on purpose: any command exposing depth_frac() -> [0, 1] can
    # drive this, which lets the squat/stand fine-tune reuse the reward with
    # its own cyclic command rather than duplicating it.
    height_term = env.command_manager.get_term(height_command_name)
    depth_fn = getattr(height_term, "depth_frac", None)
    assert callable(depth_fn), (
      f"Command '{height_command_name}' must expose depth_frac(); got "
      f"{type(height_term).__name__}."
    )
    depth = depth_fn().unsqueeze(1)
    std = std + depth * self.std_crouch_extra

    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current - desired)
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
