"""Box-carrying payload: mass at both palms + a carry flag the actor observes.

Per episode (on reset) each env either carries (prob ``p_carry``) or not. A
carrying env gets a TOTAL mass drawn from ``mass_range`` split between the two
``*_wrist_yaw_link`` bodies (the palms) with a left share in ``split_range`` --
a box held between the hands is never perfectly centred. Mass is added at the
body CoM with no extra inertia (point mass), which is what ``dr.body_mass`` also
assumes.

State lives on the env so commands / observations / rewards can read it:
  env._carry_flag   [B]     1.0 while carrying
  env._carry_mass   [B, 2]  added kg at (left, right) palm
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _select_default_values
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PALMS = SceneEntityCfg(
  "robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link")
)


def _state(env: ManagerBasedRlEnv):
  if getattr(env, "_carry_flag", None) is None:
    env._carry_flag = torch.zeros(env.num_envs, device=env.device)
    env._carry_mass = torch.zeros(env.num_envs, 2, device=env.device)
  return env._carry_flag, env._carry_mass


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def carry_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  p_carry: float,
  mass_range: tuple[float, float],
  split_range: tuple[float, float] = (0.4, 0.6),
  p_full: float = 0.0,
  asset_cfg: SceneEntityCfg = PALMS,
) -> None:
  """p_full: fraction of carrying envs pinned at mass_range[1] (the rest uniform)."""
  flag, mass = _state(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  n = len(env_ids)
  carrying = (torch.rand(n, device=env.device) < p_carry).float()
  total = torch.empty(n, device=env.device).uniform_(*mass_range)
  full = torch.rand(n, device=env.device) < p_full
  total = torch.where(full, torch.full_like(total, mass_range[1]), total) * carrying
  left = torch.empty(n, device=env.device).uniform_(*split_range)
  flag[env_ids] = carrying
  mass[env_ids, 0] = total * left
  mass[env_ids, 1] = total * (1.0 - left)

  asset = env.scene[asset_cfg.name]
  body_idx = asset.indexing.body_ids[asset_cfg.body_ids]  # GLOBAL model ids
  default = _select_default_values(
    env, "body_mass", env_ids.to(torch.int), body_idx
  )  # [n, 2]
  # Same (env, entity) meshgrid indexing as dr._randomize_model_field: mixing
  # an env-id tensor with a 0-d entity tensor silently no-ops on TorchArray.
  env_grid, ent_grid = torch.meshgrid(
    env_ids.to(torch.int), body_idx.to(torch.int), indexing="ij"
  )
  env.sim.model.body_mass[env_grid, ent_grid] = default + mass[env_ids]


# ---- observations ----------------------------------------------------------


def carry_flag(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 1] -- known to the operator on the real robot (am I holding a box?)."""
  flag, _ = _state(env)
  return flag.unsqueeze(1)


def carry_mass(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 2] privileged (critic only): kg at the left / right palm."""
  _, mass = _state(env)
  return mass
