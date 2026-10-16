"""Posture command, mid-episode payload pick-up / put-down, and the paper rewards
re-targeted to the posture command.

PostureCommand -- what the actor sees: (pelvis height - NOMINAL_H, torso pitch, waist yaw)
  * raw goal resampled every 2-6 s: nominal with p_nominal, else uniform in the
    (curriculum) stand ranges;
  * walking band: while the twist command asks to move (or the gait is WALK / W2S)
    the goal is clamped to the walk band (crouch-walk down to walk_height[0]);
  * crouch speed scaling (as policy #4): the twist command is scaled from 1.0 at
    crouch_top (0.70 m) to min_speed_scale at crouch_floor (0.60 m) (written into the twist
    term, so the actor observes the scaled command; deploy: same formula);
  * heavy-squat floor (curriculum only): with m kg in the hands the height goal is
    >= lerp(H_MIN, heavy_floor, m / heavy_mass); heavy_floor reaches H_MIN in the
    last phase, so at deploy time this rule is a no-op;
  * the target follows the goal at a limited rate (height 0.3 m/s, pitch 0.8 rad/s,
    yaw 1.0 rad/s). The actor observes the rate-limited target.
  * ``walk_ready``: target inside the walk band -> GaitModeCommand may leave STAND.
  Deploy: compute the same clamp + rate limit on the robot (no sensing needed).

carry_toggle -- global interval event: every second each free env picks up a box
  with p_pick (mass drawn like carry_payload) and each carrying env puts it down
  with p_drop. The palm mass changes in ONE step (worst case of a real lift); the
  carry flag the actor sees flips in the same step; ArmCommand moves to the carry
  pose while carrying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _select_default_values
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.human.mdp import rewards as R1
from mjlab.tasks.human.mdp.carry import PALMS
from mjlab.tasks.human.mdp.carry import _state as _carry_state
from mjlab.tasks.human.mdp.commands import gait_params
from mjlab.tasks.human.mdp.symmetry import CUSTOM_MIRRORS
from mjlab.tasks.human_v2.mdp.modes import STAND, W2S
from mjlab.tasks.human_v2.mdp.rewards import ALL, _mask
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

NOMINAL_H = 0.78  # m, pelvis height of the V2 walking pose
H_MIN = (
  0.50  # m, deepest squat with flat feet inside the soft ankle limit (kinematic check)
)


# ---- posture command ---------------------------------------------------------------


class PostureCommand(CommandTerm):
  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n = self.num_envs
    self.nominal = torch.tensor([NOMINAL_H, 0.0, 0.0], device=self.device)
    self.goal = self.nominal.repeat(n, 1)
    self.target = self.nominal.repeat(n, 1)
    self.rate = torch.zeros(n, 3, device=self.device)  # d target / dt
    self.settle = torch.zeros(n, device=self.device)
    self.walk_ready = torch.ones(n, dtype=torch.bool, device=self.device)
    self.goal_eff = self.nominal.repeat(n, 1)  # goal after clamps (actor obs)
    self._raw_twist = torch.zeros(n, 3, device=self.device)
    self._last_twist = torch.zeros(n, 3, device=self.device)
    for k in ("height", "pitch", "yaw", "height_err", "pitch_err"):
      self.metrics[k] = torch.zeros(n, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target - self.nominal

  @property
  def settled(
    self,
  ) -> torch.Tensor:  # [B] 1.0 once the target has been still for settle_s
    return (self.settle <= 0.0).float()

  def reset(self, env_ids):
    if env_ids is None:
      env_ids = slice(None)
    self.target[env_ids] = self.nominal  # episodes start in the default pose
    self.rate[env_ids] = 0.0
    self.settle[env_ids] = 0.0
    self._raw_twist[env_ids] = 0.0
    self._last_twist[env_ids] = 0.0
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    c, k = self.cfg, len(env_ids)
    u = lambda lo, hi: torch.empty(k, device=self.device).uniform_(lo, hi)  # noqa: E731
    g = torch.stack(
      [u(*c.height_range), u(*c.pitch_range), u(-c.yaw_max, c.yaw_max)], dim=1
    )
    nominal = torch.rand(k, device=self.device) < c.p_nominal
    self.goal[env_ids] = torch.where(nominal.unsqueeze(1), self.nominal.expand(k, 3), g)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    c, dt = self.cfg, self._env.step_dt
    tw = self._env.command_manager.get_term(c.twist_command_name)
    vel = tw.vel_command_b
    changed = (vel != self._last_twist).any(dim=1)  # resampled / rewritten upstream
    self._raw_twist[changed] = vel[changed]
    twist = self._raw_twist
    gait = self._env.command_manager.get_term(c.gait_command_name)
    _, _, low = gait_params(twist)
    want_walk = ~low | (gait.mode != STAND)
    goal = self.goal.clone()
    # heavy-squat floor (curriculum)
    _, mass = _carry_state(self._env)
    frac = torch.clamp(mass.sum(1) / c.heavy_mass, 0.0, 1.0)
    goal[:, 0] = torch.maximum(goal[:, 0], H_MIN + (c.heavy_floor - H_MIN) * frac)
    # walking band
    wb = torch.tensor(
      [c.walk_height[0], c.walk_pitch[0], -c.walk_yaw_max], device=self.device
    )
    wt = torch.tensor(
      [c.walk_height[1], c.walk_pitch[1], c.walk_yaw_max], device=self.device
    )
    goal = torch.where(want_walk.unsqueeze(1), torch.clamp(goal, wb, wt), goal)
    self.goal_eff = goal
    # rate limit
    lim = torch.tensor(c.rate_limits, device=self.device) * dt
    step = torch.clamp(goal - self.target, -lim, lim)
    self.target = self.target + step
    self.rate = step / dt
    moving = (step.abs() > 1e-5).any(dim=1)
    self.settle = torch.where(
      moving, torch.full_like(self.settle, c.settle_s), self.settle - dt
    )
    self.walk_ready = ((self.target >= wb - 0.01) & (self.target <= wt + 0.01)).all(
      dim=1
    )
    # crouch-walk speed scaling
    span = max(c.crouch_top - c.crouch_floor, 1e-3)
    depth = torch.clamp((c.crouch_top - self.target[:, 0]) / span, 0.0, 1.0)
    scale = 1.0 - depth * (1.0 - c.min_speed_scale)
    vel.copy_(self._raw_twist * scale.unsqueeze(1))
    self._last_twist = vel.clone()

  @property
  def time_to_goal(self) -> torch.Tensor:  # [B] s until the target reaches the goal
    rates = torch.tensor(self.cfg.rate_limits, device=self.device)
    return ((self.goal_eff - self.target).abs() / rates).max(dim=1).values

  def _update_metrics(self) -> None:
    r = self._env.scene["robot"]
    self.metrics["height"] = self.target[:, 0]
    self.metrics["pitch"] = self.target[:, 1]
    self.metrics["yaw"] = self.target[:, 2].abs()
    self.metrics["height_err"] = (
      r.data.root_link_pos_w[:, 2] - self.target[:, 0]
    ).abs()
    self.metrics["pitch_err"] = (
      torso_pitch_roll(self._env)[0] - self.target[:, 1]
    ).abs()


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  twist_command_name: str = "twist"
  gait_command_name: str = "gait"
  resampling_time_range: tuple[float, float] = (2.0, 6.0)
  p_nominal: float = 1.0  # curriculum: 1.0 = posture always nominal
  height_range: tuple[float, float] = (H_MIN, 0.80)
  pitch_range: tuple[float, float] = (-0.1, 0.6)  # rad, torso forward lean (world)
  yaw_max: float = 0.8  # rad, waist yaw (torso vs pelvis)
  walk_height: tuple[float, float] = (0.70, 0.80)
  walk_pitch: tuple[float, float] = (-0.05, 0.25)
  walk_yaw_max: float = 0.3
  crouch_top: float = 0.70  # m, full speed at or above this height
  crouch_floor: float = 0.60  # m, speed scale reaches min_speed_scale here
  min_speed_scale: float = 0.3  # crouch-walking is slow (1.6 m/s cmd -> 0.48 m/s)
  rate_limits: tuple[float, float, float] = (0.3, 0.8, 1.0)  # m/s, rad/s, rad/s
  settle_s: float = 0.3
  heavy_floor: float = H_MIN  # curriculum: lowest height target at heavy_mass kg
  heavy_mass: float = 10.0

  def build(self, env: ManagerBasedRlEnv) -> PostureCommand:
    return PostureCommand(self, env)


def _posture(env) -> PostureCommand:
  return env.command_manager.get_term("posture")


# ---- mid-episode pick-up / put-down ---------------------------------------------------


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def carry_toggle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  p_pick: float,
  p_drop: float,
  mass_range: tuple[float, float],
  split_range: tuple[float, float] = (0.4, 0.6),
  p_full: float = 0.0,
  asset_cfg: SceneEntityCfg = PALMS,
) -> None:
  del env_ids  # global-time event: decide per env here
  flag, mass = _carry_state(env)
  n, dev = env.num_envs, env.device
  r = torch.rand(n, device=dev)
  pick = (flag < 0.5) & (r < p_pick)
  drop = (flag > 0.5) & (r < p_drop)
  ids = (pick | drop).nonzero().flatten()
  if len(ids) == 0:
    return
  k = len(ids)
  total = torch.empty(k, device=dev).uniform_(*mass_range)
  full = torch.rand(k, device=dev) < p_full
  total = torch.where(full, torch.full_like(total, mass_range[1]), total)
  total = total * pick[ids].float()
  left = torch.empty(k, device=dev).uniform_(*split_range)
  flag[ids] = pick[ids].float()
  mass[ids, 0] = total * left
  mass[ids, 1] = total * (1.0 - left)
  asset = env.scene[asset_cfg.name]
  if asset_cfg.body_ids is None or isinstance(asset_cfg.body_ids, slice):
    asset_cfg.resolve(env.scene)
  body_idx = asset.indexing.body_ids[asset_cfg.body_ids]
  default = _select_default_values(env, "body_mass", ids.to(torch.int), body_idx)
  eg, bg = torch.meshgrid(ids.to(torch.int), body_idx.to(torch.int), indexing="ij")
  env.sim.model.body_mass[eg, bg] = default + mass[ids]


# ---- extra actor observations (all computable on the robot) -----------------------------


def posture_goal(env) -> torch.Tensor:
  """[B, 4] where the posture target is heading (goal after clamps, minus nominal) and
  the time (s) until the rate-limited target gets there: lets the policy start early."""
  p = _posture(env)
  return torch.cat([p.goal_eff - p.nominal, p.time_to_goal.unsqueeze(1)], dim=1)


def w2s_progress(env) -> torch.Tensor:
  """[B, 1] 0 -> 1 through walk-to-stand, 0 outside it (from the gait state machine)."""
  g = env.command_manager.get_term("gait")
  prog = torch.clamp(g.low_time / g.cfg.w2s_time, 0.0, 1.0)
  return torch.where(g.mode == W2S, prog, torch.zeros_like(prog)).unsqueeze(1)


class mass_estimate:
  """[B, 2] = (estimated payload kg / 10, known flag). Deploy: the arm policy / operator
  supplies the estimate, or (0, 0) = unknown.

  Training: per episode and at every pick-up / put-down, each env redraws
    * known with p_known (else the obs is (0, 0) for that whole segment),
    * a multiplicative error in [1 - rel_err, 1 + rel_err] held for the segment,
  plus small per-step noise. Not carrying + known reads (0, 1).
  """

  def __init__(self, cfg, env):
    n = env.num_envs
    self.known = torch.ones(n, device=env.device)
    self.scale = torch.ones(n, device=env.device)
    self.last = torch.full((n,), -1.0, device=env.device)
    self.p_known = cfg.params.get("p_known", 0.75)
    self.rel_err = cfg.params.get("rel_err", 0.3)

  def _redraw(self, ids, dev):
    k = len(ids)
    self.known[ids] = (torch.rand(k, device=dev) < self.p_known).float()
    self.scale[ids] = 1.0 + torch.empty(k, device=dev).uniform_(
      -self.rel_err, self.rel_err
    )

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.last[env_ids] = -1.0  # forces a redraw on the next call

  def __call__(
    self, env, p_known: float = 0.75, rel_err: float = 0.3, step_std: float = 0.1
  ):
    del p_known, rel_err  # read in __init__
    _, mass = _carry_state(env)
    total = mass.sum(dim=1)
    changed = (total - self.last).abs() > 1e-4
    ids = changed.nonzero().flatten()
    if len(ids) > 0:
      self._redraw(ids, env.device)
    self.last = total.clone()
    est = torch.clamp(total * self.scale + step_std * torch.randn_like(total), min=0.0)
    est = est * (total > 0).float()
    out = torch.stack([est / 10.0 * self.known, self.known], dim=1)
    env._mass_est_obs = out  # the critic reads the same draw (mass_estimate_shared)
    return out


def mass_estimate_shared(env) -> torch.Tensor:
  """Critic copy of the actor's mass_estimate this step (actor group is computed first)."""
  out = getattr(env, "_mass_est_obs", None)
  if out is None:
    return torch.zeros(env.num_envs, 2, device=env.device)
  return out


def _policy_and_leg_joints(r):
  jn = list(r.joint_names)
  leg = [i for i, n in enumerate(jn) if any(k in n for k in ("hip", "knee", "ankle"))]
  pol = [
    i
    for i, n in enumerate(jn)
    if any(k in n for k in ("hip", "knee", "ankle", "waist"))
  ]
  return jn, leg, pol


def _batched(x: torch.Tensor, n: int) -> torch.Tensor:
  return x.unsqueeze(0).expand(n, *x.shape) if x.dim() == 1 or x.shape[0] != n else x


class sysid_params:
  """Critic only: the randomised physics of THIS env, read back from the model.
  [foot friction 1, torso CoM offset 3 (m), leg frictionloss 12 (Nm), leg damping 12,
   leg armature x10 12, kp scale of the 15 policy joints, encoder bias of the 15] = 70."""

  def __init__(self, cfg, env):
    r = env.scene["robot"]
    jn, self.leg, self.pol = _policy_and_leg_joints(r)
    self.leg_dof = r.indexing.joint_v_adr[self.leg]
    an = list(r.actuator_names)
    self.pol_ctrl = r.indexing.ctrl_ids[[an.index(jn[i]) for i in self.pol]]
    gn = list(r.geom_names)
    foot = [i for i, n in enumerate(gn) if "foot" in n and "collision" in n]
    assert foot, "no foot collision geoms"
    self.foot_geom = r.indexing.geom_ids[foot]
    self.torso = int(r.indexing.body_ids[list(r.body_names).index("torso_link")])
    self.kp0 = env.sim.get_default_field("actuator_gainprm")[self.pol_ctrl, 0]
    self.ipos0 = env.sim.get_default_field("body_ipos")[self.torso]

  def __call__(self, env):
    m, n, r = env.sim.model, env.num_envs, env.scene["robot"]
    fr = _batched(m.geom_friction, n)[:, self.foot_geom, 0].mean(1, keepdim=True)
    com = _batched(m.body_ipos, n)[:, self.torso] - self.ipos0
    fl = _batched(m.dof_frictionloss, n)[:, self.leg_dof]
    dp = _batched(m.dof_damping, n)[:, self.leg_dof]
    ar = _batched(m.dof_armature, n)[:, self.leg_dof] * 10.0
    kp = _batched(m.actuator_gainprm, n)[:, self.pol_ctrl, 0] / self.kp0
    enc = r.data.encoder_bias[:, self.pol]
    return torch.cat([fr, com, fl, dp, ar, kp, enc], dim=1)


def _sysid_mirror(env, device):
  from mjlab.tasks.human.mdp.symmetry import joint_mirror

  r = env.scene["robot"]
  jn, leg, pol = _policy_and_leg_joints(r)
  lp, _ = joint_mirror([jn[i] for i in leg])
  pp, ps = joint_mirror([jn[i] for i in pol])
  idx, sign, o = [0], [1.0], 1
  idx += [1, 2, 3]
  sign += [1.0, -1.0, 1.0]  # CoM offset: y flips
  o = 4
  for _ in range(3):  # frictionloss, damping, armature: swap sides, positive
    idx += [o + j for j in lp]
    sign += [1.0] * len(lp)
    o += len(lp)
  idx += [o + j for j in pp]  # kp scale: swap
  sign += [1.0] * len(pp)
  o += len(pp)
  idx += [o + j for j in pp]  # encoder bias: joint offsets -> joint mirror signs
  sign += list(ps)
  ti = torch.tensor(idx, device=device)
  ts = torch.tensor(sign, device=device)
  return lambda x: x[:, ti] * ts


CUSTOM_MIRRORS["sysid_params"] = _sysid_mirror


# ---- helpers ---------------------------------------------------------------------------


def _robot(env):
  return env.scene["robot"]


def torso_pitch_roll(env):
  """(forward pitch, gravity-y) of the torso in the world; pitch > 0 = leaning forward."""
  r = _robot(env)
  t = r.body_names.index("torso_link")
  q = r.data.body_link_quat_w[:, t]
  gw = r.data.gravity_vec_w
  g = quat_apply_inverse(q, gw.expand(q.shape[0], 3) if gw.dim() == 1 else gw)
  return torch.atan2(g[:, 0], -g[:, 2]), g[:, 1]


def _twist_eff(env, command_name: str = "twist"):
  """Twist the robot is actually asked to follow: zero while the gait is STAND
  (e.g. still rising out of a squat before it may step off)."""
  cmd = env.command_manager.get_command(command_name)
  stand = env.command_manager.get_term("gait").mode == STAND
  return torch.where(stand.unsqueeze(1), torch.zeros_like(cmd), cmd)


# ---- task ------------------------------------------------------------------------------


def track_lin_vel(env, std: float = 0.5, command_name: str = "twist"):
  """V2 tracking (W2S taper) on the effective twist."""
  r = _robot(env)
  cmd = _twist_eff(env, command_name)[:, :2]
  g = env.command_manager.get_term("gait")
  taper = (1.0 - g.w2s_progress()).unsqueeze(1) * g.w2s_v0[:, :2]
  tgt = torch.where((g.mode == W2S).unsqueeze(1), taper, cmd)
  err = torch.sum(torch.square(tgt - r.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-err / std**2)


def track_yaw(env, std: float = 0.5, command_name: str = "twist"):
  cmd = _twist_eff(env, command_name)
  err = torch.square(cmd[:, 2] - _robot(env).data.root_link_ang_vel_b[:, 2])
  return torch.exp(-err / std**2)


def track_height(env, std: float = 0.05):
  err = _robot(env).data.root_link_pos_w[:, 2] - _posture(env).target[:, 0]
  return torch.exp(-torch.square(err) / std**2)


def height_l2(env):
  """Paper 'base height' penalty, target = commanded height."""
  err = _robot(env).data.root_link_pos_w[:, 2] - _posture(env).target[:, 0]
  return torch.square(err)


def track_torso_pitch(env, std: float = 0.15):
  """Torso lean to the commanded pitch, torso roll to zero."""
  pitch, gy = torso_pitch_roll(env)
  err = torch.square(pitch - _posture(env).target[:, 1]) + torch.square(gy)
  return torch.exp(-err / std**2)


def track_waist_yaw(env, std: float = 0.08):
  """Replaces the paper's waist-yaw deviation: deviation from the COMMANDED yaw."""
  r = _robot(env)
  j = list(r.joint_names).index("waist_yaw_joint")
  err = r.data.joint_pos[:, j] - _posture(env).target[:, 2]
  return torch.exp(-torch.square(err) / std**2)


# ---- regulation, re-targeted -------------------------------------------------------------


def lin_vel_z(env):
  """Vertical pelvis speed minus the commanded height rate."""
  vz = _robot(env).data.root_link_lin_vel_w[:, 2]
  return torch.square(vz - _posture(env).rate[:, 0])


def pelvis_orientation(env, slack: float = 0.35, modes=ALL, carry=None):
  """Paper 'orientation' on the pelvis: roll fully; pitch only beyond +-slack of the
  commanded torso lean (a hip hinge tilts the pelvis)."""
  r = _robot(env)
  g = r.data.projected_gravity_b
  pitch = torch.atan2(g[:, 0], -g[:, 2])
  over = torch.relu(torch.abs(pitch - _posture(env).target[:, 1]) - slack)
  return (torch.square(g[:, 1]) + torch.square(over)) * _mask(env, modes, carry)


def torso_min_ang_vel(env, std: float = 1.0, command_name: str = "twist"):
  """Paper 'min torso ang vel' minus the commanded yaw rate and lean rate."""
  r = _robot(env)
  t = r.body_names.index("torso_link")
  q = r.data.body_link_quat_w[:, t]
  w = quat_apply_inverse(q, r.data.body_link_ang_vel_w[:, t]).clone()
  w[:, 1] = w[:, 1] - _posture(env).rate[:, 1]
  w[:, 2] = w[:, 2] - _twist_eff(env, command_name)[:, 2]
  return torch.exp(-torch.sum(torch.square(w), dim=1) / std**2)


def straight_knee_stance(
  env, sensor_name: str, asset_cfg: SceneEntityCfg, modes=ALL, carry=None
):
  """Straight stance knee only near walking height (fades out below 0.72 m)."""
  h = _posture(env).target[:, 0]
  fade = torch.clamp((h - 0.72) / 0.05, 0.0, 1.0)
  return (
    R1.straight_knee_stance(env, sensor_name=sensor_name, asset_cfg=asset_cfg)
    * fade
    * _mask(env, modes, carry)
  )


# ---- standing block, re-targeted -----------------------------------------------------------


def base_motion(env, std: float = 0.3, modes=(STAND,), carry=None):
  """Paper 'base motion' (+) in STAND, only once the posture target has settled."""
  r = _robot(env)
  x = torch.sum(torch.square(r.data.root_link_lin_vel_b), 1) + torch.sum(
    torch.square(r.data.root_link_ang_vel_b), 1
  )
  return torch.exp(-x / std**2) * _posture(env).settled * _mask(env, modes, carry)


def stillness(
  env, asset_cfg: SceneEntityCfg, std: float = 1.0, modes=(STAND,), carry=None
):
  """Joint stillness of the POLICY's joints (arms are external), once settled."""
  v = _robot(env).data.joint_vel[:, asset_cfg.joint_ids]
  return (
    torch.exp(-torch.sum(torch.square(v), dim=1) / std**2)
    * _posture(env).settled
    * _mask(env, modes, carry)
  )


def waist_pose(env, std: float = 0.2, modes=(STAND,), carry=None):
  """Paper 'pose (soft upper)' for the legs-only policy: waist roll -> 0, waist yaw ->
  commanded (waist pitch is left to the lean term)."""
  r = _robot(env)
  jn = list(r.joint_names)
  q = r.data.joint_pos
  err = 0.5 * (
    torch.square(q[:, jn.index("waist_roll_joint")])
    + torch.square(q[:, jn.index("waist_yaw_joint")] - _posture(env).target[:, 2])
  )
  return torch.exp(-err / std**2) * _mask(env, modes, carry)
