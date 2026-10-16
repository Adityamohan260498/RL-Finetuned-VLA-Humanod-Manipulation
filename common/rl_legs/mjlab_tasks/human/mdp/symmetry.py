"""Left/right (sagittal-plane) mirroring for rsl_rl symmetry augmentation.

Every PPO mini-batch is doubled with mirrored (obs, action) pairs, so the
policy and value function are trained on both a left turn and the identical
right turn. This is the fix for the old policies' asymmetry (3_b: +1.0 rad/s
turn -> 1.03, -1.0 -> 0.57).

Mirror about the robot's x-z plane (y -> -y):
  angular velocity  (wx, wy, wz) -> (-wx,  wy, -wz)
  gravity / lin vel (x, y, z)    -> ( x,  -y,  z)
  command           (vx, vy, wz) -> (vx, -vy, -wz)
  joints            swap left <-> right; negate roll and yaw joints
                    (incl. waist_roll, waist_yaw); pitch joints unchanged
  gait clock        left phase <-> right phase = phase + 0.5 -> (sin, cos) negated
  per-foot terms    swap left <-> right (forces also negate y)
  palm masses       swap
  posture command   (height, torso pitch, waist yaw) -> (height, pitch, -yaw)
Terms are found by name, history-aware (each frame mirrored separately).
"""

from __future__ import annotations

import torch

_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")
_ANGVEL = torch.tensor([-1.0, 1.0, -1.0])
_VEC = torch.tensor([1.0, -1.0, 1.0])
_CACHE: dict = {}


def joint_mirror(joint_names: list[str]) -> tuple[list[int], list[float]]:
  perm, sign = [], []
  for n in joint_names:
    if n.startswith("left_"):
      m = "right_" + n[5:]
    elif n.startswith("right_"):
      m = "left_" + n[6:]
    else:
      m = n
    perm.append(joint_names.index(m))
    sign.append(-1.0 if ("roll" in n or "yaw" in n) else 1.0)
  return perm, sign


def _subset_mirror(names: list[str], device):
  perm, sign = joint_mirror(names)
  return torch.tensor(perm, device=device), torch.tensor(sign, device=device)


_SUBSETS: dict = {}
# name -> builder(env, device) returning a one-frame mirror fn, for terms whose layout
# only the task knows (registered by the task module, e.g. the critic's sysid_params).
CUSTOM_MIRRORS: dict = {}


def _frame_fn(name: str, dim: int, jperm, jsign, device):
  """Function mirroring ONE frame [N, dim] of term `name`."""
  if name in _JOINT_TERMS and dim != len(jperm):
    # Partial joint vector (legs-only policy actions, arm command): mirror within the subset.
    p, s = _SUBSETS[(name, dim)]
    return lambda x: x[:, p] * s
  if name == "arm_command":
    p, s = _SUBSETS[(name, dim)]
    return lambda x: x[:, p] * s
  if name in _JOINT_TERMS:
    p = torch.tensor(jperm, device=device)
    s = torch.tensor(jsign, device=device)
    return lambda x: x[:, p] * s
  if name in ("base_ang_vel", "torso_ang_vel"):
    s = _ANGVEL.to(device)
    return lambda x: x * s
  if name in ("projected_gravity", "base_lin_vel", "torso_projected_gravity"):
    s = _VEC.to(device)
    return lambda x: x * s
  if name == "command":
    s = torch.tensor([1.0, -1.0, -1.0], device=device)
    return lambda x: x * s
  if name == "gait_phase":
    return lambda x: -x
  if (
    name == "gait_mode"
  ):  # [sin, cos, onehot(Stand, Walk, W2S)]: phase flips, mode same
    assert dim == 5, (name, dim)
    return lambda x: torch.cat([-x[:, :2], x[:, 2:]], dim=1)
  if name == "posture":  # [height, torso pitch, waist yaw]: yaw flips
    assert dim == 3, (name, dim)
    s = torch.tensor([1.0, 1.0, -1.0], device=device)
    return lambda x: x * s
  if name == "posture_goal":  # [height, pitch, yaw, time]: yaw flips
    assert dim == 4, (name, dim)
    s = torch.tensor([1.0, 1.0, -1.0, 1.0], device=device)
    return lambda x: x * s
  if name in (
    "carry_flag",
    "pelvis_height",
    "height_command",
    "w2s_progress",
    "mass_estimate",
  ):
    return lambda x: x
  if name in ("foot_height", "foot_air_time", "foot_contact", "carry_mass"):
    assert dim == 2, (name, dim)
    return lambda x: x[:, [1, 0]]
  if name == "foot_contact_forces":
    assert dim == 6, (name, dim)
    s = _VEC.to(device).repeat(2)
    return lambda x: x.view(-1, 2, 3)[:, [1, 0], :].reshape(-1, 6) * s
  raise KeyError(f"no mirror rule for observation term '{name}' (dim {dim})")


def _group_fn(env, group: str, device):
  key = (id(env), group)
  if key in _CACHE:
    return _CACHE[key]
  om = env.observation_manager
  names = om.active_terms[group]
  dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim[group]]
  terms_cfg = env.cfg.observations[group].terms
  jperm, jsign = joint_mirror(list(env.scene["robot"].joint_names))
  act_names = list(env.action_manager.get_term("joint_pos").target_names)
  _SUBSETS[("actions", len(act_names))] = _subset_mirror(act_names, device)
  if "arm_command" in env.command_manager.active_terms:
    arm_names = list(env.command_manager.get_term("arm_command").joint_names)
    _SUBSETS[("arm_command", len(arm_names))] = _subset_mirror(arm_names, device)
  pieces = []
  for name, dim in zip(names, dims):
    h = max(int(getattr(terms_cfg[name], "history_length", 0) or 0), 1)
    per = dim // h
    if name in CUSTOM_MIRRORS:
      f = CUSTOM_MIRRORS[name](env, device)
    else:
      f = _frame_fn(name, per, jperm, jsign, device)
    pieces.append((dim, h, per, f))

  def fn(x: torch.Tensor) -> torch.Tensor:
    out, i = [], 0
    for dim, h, per, f in pieces:
      seg = x[:, i : i + dim]
      if h > 1:
        seg = f(seg.reshape(-1, per)).reshape(-1, dim)
      else:
        seg = f(seg)
      out.append(seg)
      i += dim
    assert i == x.shape[1], f"group '{group}': mirrored {i} of {x.shape[1]} dims"
    return torch.cat(out, dim=1)

  _CACHE[key] = fn
  return fn


def mirror_obs_and_actions(env, obs=None, actions=None):
  """rsl_rl ``data_augmentation_func``: returns [original; mirrored]."""
  base = env.unwrapped if hasattr(env, "unwrapped") else env
  obs_aug = None
  if obs is not None:
    keys = list(obs.keys())
    mirrored = {g: _group_fn(base, g, obs[g].device)(obs[g]) for g in keys}
    if isinstance(obs, dict):
      obs_aug = {g: torch.cat([obs[g], mirrored[g]], dim=0) for g in keys}
    else:  # TensorDict (what rsl_rl passes)
      m = obs.clone()
      for g in keys:
        m[g] = mirrored[g]
      obs_aug = torch.cat([obs, m], dim=0)
  act_aug = None
  if actions is not None:
    act_names = list(base.action_manager.get_term("joint_pos").target_names)
    p, s = _subset_mirror(act_names, actions.device)
    act_aug = torch.cat([actions, actions[:, p] * s], dim=0)
  return obs_aug, act_aug
