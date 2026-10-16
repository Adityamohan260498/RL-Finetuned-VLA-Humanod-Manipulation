"""V2 reward terms: Table IV of Peng, Bao & Zhou mapped onto the G1 (29 DoF), plus
the fixes for the conflicts with the user's goals (PAPER_GAP_ANALYSIS.md, sec. 3).

Routing (the paper's gait mask) is applied INSIDE each term with ``modes`` (a tuple of
gait-mode ids it is active in) and ``carry`` (None = both, True/False = only while
carrying / only while free). Terms reused from mjlab.tasks.human.mdp take the
same routing through thin wrappers below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.human.mdp import rewards as R1
from mjlab.tasks.human.mdp.carry import _state as _carry_state
from mjlab.utils.lab_api.math import quat_apply_inverse

from .modes import STAND, W2S, WALK

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ALL = (STAND, WALK, W2S)


def _mask(env: ManagerBasedRlEnv, modes=ALL, carry=None) -> torch.Tensor:
  g = env.command_manager.get_term("gait")
  m = torch.zeros(env.num_envs, device=env.device)
  for k in modes:
    m = m + g.is_mode(k)
  if carry is not None:
    flag, _ = _carry_state(env)
    m = m * (flag if carry else (1.0 - flag))
  return m


def routed(func):
  """Wrap a reward so it accepts ``modes`` / ``carry`` routing params."""

  def f(env, modes=ALL, carry=None, **kw):
    return func(env, **kw) * _mask(env, modes, carry)

  f.__name__ = f"routed_{func.__name__}"
  return f


def _robot(env) -> Entity:
  return env.scene["robot"]


def _jid(env, names):
  jn = list(_robot(env).joint_names)
  return [jn.index(n) for n in names]


# ---- task --------------------------------------------------------------------


def track_lin_vel(env, std: float = 0.5, command_name: str = "twist"):
  """Paper 'tracking lin vel'. In W2S the target tapers linearly from the speed at
  W2S entry to 0 ('smooth slowdown' instead of an instant stop)."""
  r = _robot(env)
  cmd = env.command_manager.get_command(command_name)[:, :2]
  g = env.command_manager.get_term("gait")
  taper = (1.0 - g.w2s_progress()).unsqueeze(1) * g.w2s_v0[:, :2]
  tgt = torch.where((g.mode == W2S).unsqueeze(1), taper, cmd)
  err = torch.sum(torch.square(tgt - r.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-err / std**2)


def track_yaw(env, std: float = 0.5, command_name: str = "twist"):
  """Yaw-rate only (the stock kernel also scored pelvis roll/pitch rate)."""
  cmd = env.command_manager.get_command(command_name)
  err = torch.square(cmd[:, 2] - _robot(env).data.root_link_ang_vel_b[:, 2])
  return torch.exp(-err / std**2)


# ---- regulation (all modes) -----------------------------------------------------


def base_lin_vel_z(env):
  return torch.square(_robot(env).data.root_link_lin_vel_b[:, 2])


def base_ang_vel_xy(env):
  return torch.sum(torch.square(_robot(env).data.root_link_ang_vel_b[:, :2]), dim=1)


def base_orientation(env):
  return torch.sum(torch.square(_robot(env).data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(env):
  return torch.sum(torch.square(_robot(env).data.joint_vel), dim=1)


def hip_deviation(env):
  """Paper 'hip pos deviation': hip roll + yaw away from default."""
  r = _robot(env)
  ids = _jid(
    env, [f"{s}_hip_{j}_joint" for s in ("left", "right") for j in ("roll", "yaw")]
  )
  d = r.data.joint_pos[:, ids] - r.data.default_joint_pos[:, ids]
  return torch.sum(torch.square(d), dim=1)


def torso_min_ang_vel(env, std: float = 1.0, command_name: str = "twist"):
  """Paper 'min torso ang vel' (+): torso rate minus the commanded yaw rate."""
  r = _robot(env)
  t = r.body_names.index("torso_link")
  q = r.data.body_link_quat_w[:, t]
  w = quat_apply_inverse(q, r.data.body_link_ang_vel_w[:, t])
  w = w.clone()
  w[:, 2] = w[:, 2] - env.command_manager.get_command(command_name)[:, 2]
  return torch.exp(-torch.sum(torch.square(w), dim=1) / std**2)


def waist_deviation(env, joint: str, std: float = 0.08):
  """Paper 'waist pitch/roll/yaw deviation' (+). This is what stops the ±40° twist."""
  q = _robot(env).data.joint_pos[:, _jid(env, [f"waist_{joint}_joint"])[0]]
  return torch.exp(-torch.square(q) / std**2)


class torso_yaw_smoothness:
  """Paper 'torso yaw smoothness' (+): small torso yaw acceleration."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.prev = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env, std: float = 4.0, modes=ALL, carry=None):
    r = _robot(env)
    t = r.body_names.index("torso_link")
    wz = r.data.body_link_ang_vel_w[:, t, 2]
    acc = (wz - self.prev) / env.step_dt
    self.prev = wz.clone()
    return torch.exp(-torch.square(acc) / std**2) * _mask(env, modes, carry)

  def reset(self, env_ids):
    self.prev[env_ids] = 0.0


def shoulder_roll_control(env, std: float = 0.1):
  """Paper 'shoulder roll control' (+). Target = default when free, carry pose when carrying."""
  r = _robot(env)
  ids = _jid(env, ["left_shoulder_roll_joint", "right_shoulder_roll_joint"])
  flag, _ = _carry_state(env)
  free_t = r.data.default_joint_pos[:, ids]
  carry_t = torch.tensor([0.28, -0.28], device=env.device).expand_as(free_t)
  tgt = torch.where(flag.unsqueeze(1) > 0.5, carry_t, free_t)
  return torch.exp(
    -torch.sum(torch.square(r.data.joint_pos[:, ids] - tgt), dim=1) / std**2
  )


# ---- arm swing (WALK, not carrying) ---------------------------------------------

_ARM_BODIES = {
  side: tuple(
    f"{side}_{b}_link"
    for b in (
      "shoulder_pitch",
      "shoulder_roll",
      "shoulder_yaw",
      "elbow",
      "wrist_roll",
      "wrist_pitch",
      "wrist_yaw",
    )
  )
  for side in ("left", "right")
}


def _yaw_momentum(env):
  """(L_total_z, L_left_arm_z, L_right_arm_z) about the whole-body CoM (translational
  part m (r-c) x (v-v_c) + rotational part I_zz w_z with I_zz ~ mean principal inertia)."""
  r = _robot(env)
  gid = r.indexing.body_ids
  m = env.sim.model.body_mass[:, gid]  # [B, nb] (includes palm payload)
  pose = r.data.body_com_pose_w
  vel = r.data.body_com_vel_w
  p, v, w = pose[..., :3], vel[..., :3], vel[..., 3:]
  M = m.sum(1, keepdim=True)
  c = (m.unsqueeze(-1) * p).sum(1) / M
  vc = (m.unsqueeze(-1) * v).sum(1) / M
  rel, vrel = p - c.unsqueeze(1), v - vc.unsqueeze(1)
  lz = m * (rel[..., 0] * vrel[..., 1] - rel[..., 1] * vrel[..., 0])
  inert = (
    env.sim.model.body_inertia[:, gid].mean(-1)
    if env.sim.model.body_inertia.dim() == 3
    else env.sim.model.body_inertia[gid].mean(-1).unsqueeze(0)
  )
  lz = lz + inert * w[..., 2]
  names = list(r.body_names)
  la = [names.index(b) for b in _ARM_BODIES["left"]]
  ra = [names.index(b) for b in _ARM_BODIES["right"]]
  return lz.sum(1), lz[:, la].sum(1), lz[:, ra].sum(1)


def arm_leg_momentum_balance(
  env,
  izz: float = 1.2,
  std: float = 1.0,
  command_name: str = "twist",
  arm_weight: float = 0.4,
):
  """Paper eq. (2) as a positive kernel: exp(-[(L_z - I_zz*wz_cmd)^2 + 0.4 (L_la,z - L_ra,z)^2]).
  Residual L_z (not raw L_z) so a COMMANDED turn is not penalised (conflict #2)."""
  lt, la, ra = _yaw_momentum(env)
  wz = env.command_manager.get_command(command_name)[:, 2]
  x = torch.square(lt - izz * wz) + arm_weight * torch.square(la - ra)
  return torch.exp(-x / std**2)


def elbow_phase_tracking(env, amp: float = 0.25, std: float = 0.2):
  """Paper 'elbow phase tracking' (+): elbow flexes more as its arm swings forward
  (forward arm = opposite leg in swing): target = default + amp*(0.5 - 0.5cos)."""
  r = _robot(env)
  g = env.command_manager.get_term("gait")
  ids = _jid(env, ["left_elbow_joint", "right_elbow_joint"])
  # Left arm swings forward while the RIGHT leg swings (anti-phase).
  ph = torch.stack([(g.phase + 0.5) % 1.0, g.phase], dim=1)
  tgt = r.data.default_joint_pos[:, ids] + amp * (
    0.5 - 0.5 * torch.cos(2 * torch.pi * ph)
  )
  return torch.exp(
    -torch.sum(torch.square(r.data.joint_pos[:, ids] - tgt), dim=1) / std**2
  )


def arm_swing_symmetry(env, std: float = 0.15):
  """Paper 'arm swing symmetry' (+): left and right shoulder-pitch offsets opposite and equal."""
  r = _robot(env)
  ids = _jid(env, ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])
  d = r.data.joint_pos[:, ids] - r.data.default_joint_pos[:, ids]
  return torch.exp(-torch.square(d[:, 0] + d[:, 1]) / std**2)


def arm_swing_energy(env, std: float = 2.0, command_name: str = "twist"):
  """Paper 'human-like arm swing energy' (+): shoulder-pitch speed^2 near a
  speed-dependent target (no flailing, no rigid arms)."""
  r = _robot(env)
  ids = _jid(env, ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])
  e = torch.sum(torch.square(r.data.joint_vel[:, ids]), dim=1)
  v = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
  tgt = 1.5 * torch.square(v)  # ~1.2 rad/s rms each at 1 m/s
  return torch.exp(-torch.square(e - tgt) / std**2)


# ---- standing block (STAND) -----------------------------------------------------


def stand_contact(env, sensor_name: str):
  return (env.scene[sensor_name].data.found > 0).float().prod(dim=1)


def base_motion(env, std: float = 0.3):
  r = _robot(env)
  x = torch.sum(torch.square(r.data.root_link_lin_vel_b), 1) + torch.sum(
    torch.square(r.data.root_link_ang_vel_b), 1
  )
  return torch.exp(-x / std**2)


class soft_upper_pose:
  """Paper 'pose (soft upper)' (+): waist + arms near their pose; the arm target is
  the carry pose while carrying."""

  def __init__(self, cfg: RewardTermCfg, env):
    r = _robot(env)
    names = list(r.joint_names)
    self.ids = [
      i
      for i, n in enumerate(names)
      if any(k in n for k in ("waist", "shoulder", "elbow", "wrist"))
    ]
    self.free = r.data.default_joint_pos[:, self.ids].clone()
    self.carry = self.free.clone()
    for n, v in R1.carry_pose.POSE.items():
      if names.index(n) in self.ids:
        self.carry[:, self.ids.index(names.index(n))] = v

  def __call__(self, env, std: float = 0.2, modes=ALL, carry=None, parts=None):
    flag, _ = _carry_state(env)
    tgt = torch.where(flag.unsqueeze(1) > 0.5, self.carry, self.free)
    err = torch.mean(torch.square(_robot(env).data.joint_pos[:, self.ids] - tgt), dim=1)
    return torch.exp(-err / std**2) * _mask(env, modes, carry)


def feet_alignment(
  env, std: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
  """Paper 'feet alignment' (+): feet side by side (small fore-aft offset)."""
  r = _robot(env)
  p = r.data.site_pos_w[:, asset_cfg.site_ids, :]
  d = quat_apply_inverse(r.data.root_link_quat_w, p[:, 0] - p[:, 1])
  return torch.exp(-torch.square(d[:, 0]) / std**2)


def feet_flatness(env, std: float = 0.1):
  """Paper 'feet flatness' (+): soles parallel to the ground."""
  r = _robot(env)
  names = list(r.body_names)
  ids = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
  q = r.data.body_link_quat_w[:, ids]
  gvec = r.data.gravity_vec_w
  g = quat_apply_inverse(
    q.reshape(-1, 4),
    gvec.expand(q.shape[0] * 2, 3) if gvec.dim() == 1 else gvec.repeat_interleave(2, 0),
  ).reshape(-1, 2, 3)
  return torch.exp(-torch.sum(torch.square(g[..., :2]), dim=(1, 2)) / std**2)


def stillness(env, std: float = 1.0):
  return torch.exp(-torch.sum(torch.square(_robot(env).data.joint_vel), dim=1) / std**2)


def alive(env):
  return (~env.termination_manager.terminated).float()


class smooth_slowdown:
  """Paper W2S 'smooth slowdown' (+): small horizontal base acceleration."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.prev = torch.zeros(env.num_envs, 2, device=env.device)

  def __call__(self, env, std: float = 2.0, modes=(W2S,), carry=None):
    v = _robot(env).data.root_link_lin_vel_b[:, :2]
    a = (v - self.prev) / env.step_dt
    self.prev = v.clone()
    return torch.exp(-torch.sum(torch.square(a), dim=1) / std**2) * _mask(
      env, modes, carry
    )

  def reset(self, env_ids):
    self.prev[env_ids] = 0.0
