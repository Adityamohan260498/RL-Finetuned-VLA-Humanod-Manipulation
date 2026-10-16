"""Commands for the human-like G1 locomotion task.

Two terms:

``HumanVelocityCommand``  -- twist command drawn from explicit, sign-symmetric
  categories (vx / vy / wz alone, every pairwise mix, all three, standing,
  slow) instead of three independent uniforms. Each category's magnitudes start
  at the walking floor the user asked for (|vx| >= 0.6, |vy| >= 0.6,
  |wz| >= 0.5), so the deadband the old policies had is never rewarded.

``GaitClockCommand``  -- a phase clock the actor observes as (sin, cos). Stride
  frequency and duty factor follow the commanded speed (see ``gait_params``),
  so the policy is told *when* each foot should be down instead of discovering
  a 4 steps/s shuffle on its own. The clock is deployable: the robot computes
  the same formula from the same command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# --------------------------------------------------------------------------
# Gait timing (shared with the deploy side -- keep the two in sync)
# --------------------------------------------------------------------------

STAND_SPEED = 0.1  # |v| and |wz| below this = stand still, both feet down


def gait_params(cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """(stride_freq_hz, duty_factor, is_standing) from a [B, 3] twist command.

  Human cadence Froude-scaled to the G1's ~0.6 m leg: ~2.3 steps/s at
  0.6 m/s, ~2.6 at 1.0, 3.4 at 2.0 (the old policies stepped 3.6-4.4/s at
  0.6 m/s). Duty factor falls with speed so that at ~2 m/s (Froude 0.68,
  running for a human) a short flight phase appears instead of being forbidden.
  """
  speed = torch.norm(cmd[:, :2], dim=1)
  yaw = torch.abs(cmd[:, 2])
  eff = speed + 0.25 * yaw
  freq = torch.clamp(0.9 + 0.4 * eff, 0.9, 1.7)
  duty = torch.clamp(0.62 - 0.1 * speed, 0.42, 0.62)
  standing = (speed < STAND_SPEED) & (yaw < STAND_SPEED)
  return freq, duty, standing


class GaitClockCommand(CommandTerm):
  """Per-env gait phase in [0, 1). Left foot phase = phase, right = phase + 0.5."""

  cfg: GaitClockCommandCfg

  def __init__(self, cfg: GaitClockCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.freq = torch.full((self.num_envs,), 0.9, device=self.device)
    self.duty = torch.full((self.num_envs,), 0.6, device=self.device)
    self.standing = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    # Seconds the twist command has been ~zero (walk-to-stand hysteresis).
    self.zero_time = torch.full((self.num_envs,), 1e3, device=self.device)
    self.metrics["stride_freq"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    a = 2 * math.pi * self.phase
    sc = torch.stack([torch.sin(a), torch.cos(a)], dim=1)
    if self.cfg.zero_when_standing:  # explicit "stand" mode signal (paper's gait ID)
      sc = sc * (~self.standing).float().unsqueeze(1)
    return sc

  def foot_phase(self) -> torch.Tensor:
    """[B, 2] phase of (left, right) foot."""
    return torch.stack([self.phase, (self.phase + 0.5) % 1.0], dim=1)

  def desired_stance(self) -> torch.Tensor:
    """[B, 2] bool: foot should be on the ground. Both while standing."""
    stance = self.foot_phase() < self.duty.unsqueeze(1)
    return stance | self.standing.unsqueeze(1)

  def swing_progress(self) -> torch.Tensor:
    """[B, 2] in [0, 1] through the swing phase (0 outside swing)."""
    fp = self.foot_phase()
    d = self.duty.unsqueeze(1)
    prog = (fp - d) / torch.clamp(1.0 - d, min=1e-3)
    return torch.where(fp >= d, prog, torch.zeros_like(prog))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Only fires at reset (resampling_time_range is huge): random start phase.
    self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
    self.zero_time[env_ids] = 1e3  # episodes start at rest => already "settled"

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    twist = self._env.command_manager.get_command(self.cfg.twist_command_name)
    assert twist is not None
    self.freq, self.duty, small = gait_params(twist)
    dt = self._env.step_dt
    self.zero_time = torch.where(
      small, self.zero_time + dt, torch.zeros_like(self.zero_time)
    )
    # Walk-to-stand (paper's W2S mode): after the command drops to zero, keep the
    # clock and the walking rewards for stand_settle_s so the robot can take the
    # recovery steps it needs; only then switch to the stand rewards / signal.
    self.standing = small & (self.zero_time >= self.cfg.stand_settle_s)
    self.phase = (self.phase + self.freq * dt) % 1.0

  def _update_metrics(self) -> None:
    self.metrics["stride_freq"] = self.freq


@dataclass(kw_only=True)
class GaitClockCommandCfg(CommandTermCfg):
  twist_command_name: str = "twist"
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  stand_settle_s: float = 0.0
  """Seconds of zero command before 'standing' (run 3+: 1.0). Deploy must match."""
  zero_when_standing: bool = False
  """Observe (0, 0) instead of (sin, cos) while commanded to stand (run 2+)."""

  def build(self, env: ManagerBasedRlEnv) -> GaitClockCommand:
    return GaitClockCommand(self, env)


# --------------------------------------------------------------------------
# Category-based, sign-symmetric twist command
# --------------------------------------------------------------------------

CATEGORIES = ("stand", "slow", "vx", "vy", "wz", "vx_wz", "vy_wz", "vx_vy", "all")


def _default_probs() -> dict[str, float]:
  return {
    "stand": 0.08,
    "slow": 0.07,  # |v| 0.2-0.6: defined behaviour for precise approach
    "vx": 0.18,
    "vy": 0.09,
    "wz": 0.12,
    "vx_wz": 0.16,
    "vy_wz": 0.08,
    "vx_vy": 0.10,
    "all": 0.12,
  }


class HumanVelocityCommand(UniformVelocityCommand):
  cfg: HumanVelocityCommandCfg

  def __init__(self, cfg: HumanVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    for c in CATEGORIES:
      self.metrics[f"frac_{c}"] = torch.zeros(self.num_envs, device=self.device)

  def _signed(self, lo: float, hi: float, n: int, p_neg: float = 0.5) -> torch.Tensor:
    mag = torch.empty(n, device=self.device).uniform_(lo, max(lo, hi))
    sign = torch.where(torch.rand(n, device=self.device) < p_neg, -1.0, 1.0)
    return mag * sign

  def _vx(self, n: int, hi: float) -> torch.Tensor:
    """Forward |vx| in [vx_min, hi]; with p_backward, backward in back_range."""
    fwd = torch.empty(n, device=self.device).uniform_(
      self.cfg.vx_min, max(self.cfg.vx_min, hi)
    )
    back = -torch.empty(n, device=self.device).uniform_(*self.cfg.vx_back_range)
    is_back = torch.rand(n, device=self.device) < self.cfg.p_backward
    return torch.where(is_back, back, fwd)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    cfg = self.cfg
    probs = torch.tensor(
      [cfg.category_probs[c] for c in CATEGORIES], device=self.device
    )
    cat = torch.multinomial(probs / probs.sum(), n, replacement=True)
    self.category[env_ids] = cat
    cmd = torch.zeros(n, 3, device=self.device)
    vx_hi, mix_hi = cfg.vx_max, min(cfg.vx_max, cfg.mixed_vx_max)

    def sel(name):
      return cat == CATEGORIES.index(name)

    for name in CATEGORIES:
      m = sel(name)
      k = int(m.sum())
      if k == 0:
        continue
      if name == "slow":  # random direction, low speed, a little yaw
        ang = torch.rand(k, device=self.device) * 2 * math.pi
        sp = torch.empty(k, device=self.device).uniform_(*cfg.slow_range)
        cmd[m, 0] = sp * torch.cos(ang)
        cmd[m, 1] = sp * torch.sin(ang)
        cmd[m, 2] = self._signed(0.0, 0.3, k)
        continue
      if "vx" in name or name == "all":
        cmd[m, 0] = self._vx(k, vx_hi if name == "vx" else mix_hi)
      if "vy" in name or name == "all":
        cmd[m, 1] = self._signed(*cfg.vy_range, k)
      if "wz" in name or name == "all":
        cmd[m, 2] = self._signed(*cfg.wz_range, k)

    self.vel_command_b[env_ids] = cmd
    self.vel_command_w[env_ids] = cmd
    self.is_standing_env[env_ids] = sel("stand")
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    # Carrying a box: cap forward speed (an 11 kg box at 2 m/s is not a use case).
    carry = getattr(self._env, "_carry_flag", None)
    if carry is not None and self.cfg.carry_vx_max is not None:
      v = self.vel_command_b[:, 0]
      capped = torch.clamp(v, -self.cfg.carry_vx_max, self.cfg.carry_vx_max)
      self.vel_command_b[:, 0] = torch.where(carry > 0.5, capped, v)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    for i, c in enumerate(CATEGORIES):
      self.metrics[f"frac_{c}"] = (self.category == i).float()


@dataclass(kw_only=True)
class HumanVelocityCommandCfg(UniformVelocityCommandCfg):
  vx_min: float = 0.6
  vx_max: float = 1.0  # raised to 2.0 by the curriculum
  mixed_vx_max: float = 1.5  # vx inside mixed categories
  vx_back_range: tuple[float, float] = (0.6, 1.0)
  p_backward: float = 0.2
  vy_range: tuple[float, float] = (0.6, 1.0)
  wz_range: tuple[float, float] = (0.5, 1.2)
  slow_range: tuple[float, float] = (0.2, 0.6)
  carry_vx_max: float | None = 1.2
  category_probs: dict[str, float] = field(default_factory=_default_probs)

  def build(self, env: ManagerBasedRlEnv) -> HumanVelocityCommand:
    return HumanVelocityCommand(self, env)


def human_command_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[dict],
) -> dict[str, torch.Tensor]:
  """Step-indexed updates of HumanVelocityCommandCfg fields (vx_max, ranges...)."""
  del env_ids
  term = env.command_manager.get_term(command_name)
  cfg = term.cfg
  for st in stages:
    if env.common_step_counter >= st["step"]:
      for k, v in st.items():
        if k != "step":
          setattr(cfg, k, v)
  return {
    "vx_max": torch.tensor(cfg.vx_max),
    "vy_max": torch.tensor(cfg.vy_range[1]),
    "wz_max": torch.tensor(cfg.wz_range[1]),
  }
