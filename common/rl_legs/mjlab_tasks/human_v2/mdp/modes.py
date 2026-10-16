"""Gait modes for V2 (Peng, Bao & Zhou, Sec. III-C/D, without Run/R2W).

Modes (one-hot in the actor obs, and the reward mask):
  STAND (0)  -- feet planted, still
  WALK  (1)  -- clock-driven stepping
  W2S   (2)  -- walk-to-stand: command dropped below 0.1; keep the clock and step to
                 a halt; after W2S_TIME s switch to STAND (paper: 1.5 s of low speed
                 AND double support; the G1 has no foot sensors, so the deployable
                 rule is time only).
A command >= 0.1 from STAND or W2S goes straight to WALK.

The gait clock (sin, cos) runs in WALK and W2S and reads (0, 0) in STAND.
``walk_only`` (curriculum phase 1) forces WALK everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.human.mdp.commands import gait_params

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STAND, WALK, W2S = 0, 1, 2
NUM_MODES = 3


class GaitModeCommand(CommandTerm):
  cfg: GaitModeCommandCfg

  def __init__(self, cfg: GaitModeCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n = self.num_envs
    self.phase = torch.zeros(n, device=self.device)
    self.freq = torch.full((n,), 0.9, device=self.device)
    self.duty = torch.full((n,), 0.6, device=self.device)
    self.mode = torch.full((n,), STAND, dtype=torch.long, device=self.device)
    self.low_time = torch.full((n,), 1e3, device=self.device)
    self.w2s_v0 = torch.zeros(n, 3, device=self.device)  # base vel when W2S began
    for m in ("stand", "walk", "w2s"):
      self.metrics[f"frac_{m}"] = torch.zeros(n, device=self.device)

  # -- what the actor sees: [sin, cos, onehot(3)] ------------------------------
  @property
  def command(self) -> torch.Tensor:
    a = 2 * math.pi * self.phase
    sc = torch.stack([torch.sin(a), torch.cos(a)], dim=1)
    sc = sc * (self.mode != STAND).float().unsqueeze(1)
    onehot = torch.nn.functional.one_hot(self.mode, NUM_MODES).float()
    return torch.cat([sc, onehot], dim=1)

  # -- helpers used by rewards ----------------------------------------------------
  @property
  def standing(self) -> torch.Tensor:  # name kept for the shared human rewards
    return self.mode == STAND

  def is_mode(self, m: int) -> torch.Tensor:
    return (self.mode == m).float()

  def foot_phase(self) -> torch.Tensor:
    return torch.stack([self.phase, (self.phase + 0.5) % 1.0], dim=1)

  def desired_stance(self) -> torch.Tensor:
    stance = self.foot_phase() < self.duty.unsqueeze(1)
    return stance | self.standing.unsqueeze(1)

  def swing_progress(self) -> torch.Tensor:
    fp = self.foot_phase()
    d = self.duty.unsqueeze(1)
    prog = (fp - d) / torch.clamp(1.0 - d, min=1e-3)
    return torch.where(fp >= d, prog, torch.zeros_like(prog))

  def w2s_progress(self) -> torch.Tensor:
    """0 -> 1 over the W2S window (1 outside W2S)."""
    p = torch.clamp(self.low_time / self.cfg.w2s_time, 0.0, 1.0)
    return torch.where(self.mode == W2S, p, torch.ones_like(p))

  # -- state machine ----------------------------------------------------------------
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Fires at reset only: episodes start at rest -> STAND, random phase.
    self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
    self.low_time[env_ids] = 1e3
    self.mode[env_ids] = WALK if self.cfg.walk_only else STAND

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    twist = self._env.command_manager.get_command(self.cfg.twist_command_name)
    dt = self._env.step_dt
    self.freq, self.duty, low = gait_params(twist)
    if self.cfg.walk_only:
      self.mode[:] = WALK
      self.low_time[:] = 0.0
    else:
      was_walk = self.mode == WALK
      self.low_time = torch.where(
        low, self.low_time + dt, torch.zeros_like(self.low_time)
      )
      enter_w2s = low & was_walk
      robot = self._env.scene["robot"]
      self.w2s_v0[enter_w2s] = torch.cat(
        [
          robot.data.root_link_lin_vel_b[enter_w2s, :2],
          robot.data.root_link_ang_vel_b[enter_w2s, 2:3],
        ],
        dim=1,
      )
      go = ~low
      if self.cfg.posture_command_name is not None:
        # From STAND, step off only once the posture target is back in the walk band
        # (e.g. rising out of a squat); W2S -> WALK is not gated.
        ready = self._env.command_manager.get_term(
          self.cfg.posture_command_name
        ).walk_ready
        go = go & (ready | (self.mode != STAND))
      new = torch.where(go, torch.full_like(self.mode, WALK), self.mode)
      new = torch.where(enter_w2s, torch.full_like(new, W2S), new)
      done = (new == W2S) & (self.low_time >= self.cfg.w2s_time)
      new = torch.where(done, torch.full_like(new, STAND), new)
      self.mode = new
    self.phase = (self.phase + self.freq * dt) % 1.0

  def _update_metrics(self) -> None:
    self.metrics["frac_stand"] = self.is_mode(STAND)
    self.metrics["frac_walk"] = self.is_mode(WALK)
    self.metrics["frac_w2s"] = self.is_mode(W2S)


@dataclass(kw_only=True)
class GaitModeCommandCfg(CommandTermCfg):
  twist_command_name: str = "twist"
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  w2s_time: float = 1.5
  walk_only: bool = False  # curriculum phase 1
  posture_command_name: str | None = None  # gate STAND -> WALK on posture.walk_ready

  def build(self, env: ManagerBasedRlEnv) -> GaitModeCommand:
    return GaitModeCommand(self, env)


def command_cfg_stages(env, env_ids, command_name: str, stages: list[dict]):
  """Step-indexed setattr on any command term's cfg; logs the numeric/bool fields."""
  del env_ids
  cfg = env.command_manager.get_term(command_name).cfg
  for st in stages:
    if env.common_step_counter >= st["step"]:
      for k, v in st.items():
        if k != "step":
          setattr(cfg, k, v)
  out = {}
  for st in stages:
    for k in st:
      v = getattr(cfg, k, None) if k != "step" else None
      if isinstance(v, (int, float, bool)):
        out[k] = torch.tensor(float(v))
  return out
