"""External arm controller used while training the LEGS-ONLY policy.

The policy outputs legs + waist (15 joints). These 14 arm joints are driven here,
so the policy learns to balance whatever an arm controller (IK / grasp policy / VLA /
teleop) does later. Per episode, each env draws an arm behaviour:
  swing    paper-style anti-phase swing (shoulder pitch = -0.8 * same-side hip pitch offset)
  default  arms held at the default pose
  carry    box-carry pose + per-episode offsets (different box sizes), FORCED while carrying
  reach    smooth random targets within +-reach_range of default, re-drawn every 1-3 s
  pick     forward-down reach (grasping something low / on the floor), re-drawn every 1-3 s
  traj     replay of recorded arm trajectories (optional traj_table, e.g. the pick-and-place
           demos): ONE arm per episode (right as recorded, or the left arm mirrored), the
           other at rest, random speed and start; loops to a new trajectory at the end
Targets are low-passed (tau) so they move like a real arm controller, then written as
PD targets every step. The actor observes the current arm target (14) so it can
anticipate. Probabilities are curriculum fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.human.mdp.carry import _state as carry_state
from mjlab.tasks.human.mdp.rewards import carry_pose

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ARM_KEYS = ("shoulder", "elbow", "wrist")
MODES = ("swing", "default", "carry", "reach", "pick", "traj")
RIGHT7 = (
  "shoulder_pitch",
  "shoulder_roll",
  "shoulder_yaw",
  "elbow",
  "wrist_roll",
  "wrist_pitch",
  "wrist_yaw",
)
MIRROR7 = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)  # right -> left joint signs


class ArmCommand(CommandTerm):
  cfg: ArmCommandCfg

  def __init__(self, cfg: ArmCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene["robot"]
    names = list(self.robot.joint_names)
    self.ids = [i for i, n in enumerate(names) if any(k in n for k in ARM_KEYS)]
    self.joint_names = [names[i] for i in self.ids]
    self.default = self.robot.data.default_joint_pos[:, self.ids].clone()
    self.carry = self.default.clone()
    for n, v in carry_pose.POSE.items():
      self.carry[:, self.joint_names.index(n)] = v
    for n, v in (cfg.carry_override or {}).items():  # left-arm values, right mirrored
      for side, sg in (("left", 1.0), ("right", -1.0)):
        mirror = "roll" in n or "yaw" in n
        self.carry[:, self.joint_names.index(f"{side}_{n}_joint")] = v * (
          sg if mirror else 1.0
        )
    self.sh = [
      self.joint_names.index(f"{s}_shoulder_pitch_joint") for s in ("left", "right")
    ]
    self.hip = [names.index(f"{s}_hip_pitch_joint") for s in ("left", "right")]
    self.roll_ids = [
      self.joint_names.index(f"{s}_shoulder_roll_joint") for s in ("left", "right")
    ]
    self.roll_sign = torch.tensor([1.0, -1.0], device=self.device)
    n = self.num_envs
    self.mode = torch.zeros(n, dtype=torch.long, device=self.device)
    self.offset = torch.zeros(n, len(self.ids), device=self.device)
    self.reach_tgt = self.default.clone()
    self.pick_tgt = self.default.clone()
    self.pick_sp = [
      self.joint_names.index(f"{s}_shoulder_pitch_joint") for s in ("left", "right")
    ]
    self.pick_sr = [
      self.joint_names.index(f"{s}_shoulder_roll_joint") for s in ("left", "right")
    ]
    self.pick_el = [
      self.joint_names.index(f"{s}_elbow_joint") for s in ("left", "right")
    ]
    self.reach_timer = torch.zeros(n, device=self.device)
    self.target = self.default.clone()
    if cfg.traj_table is not None:
      import numpy as np

      z = np.load(cfg.traj_table)
      self._traj = torch.as_tensor(z["traj"], device=self.device)  # (N, T, 7) right arm
      self._traj_len = torch.as_tensor(z["lengths"], device=self.device).long()
      self._traj_fps = float(z["fps"])
      self._side_ids = [
        [self.joint_names.index(f"{s}_{j}_joint") for j in RIGHT7]
        for s in ("right", "left")
      ]
      self._mirror = torch.tensor(MIRROR7, device=self.device)
      self.traj_id = torch.zeros(n, dtype=torch.long, device=self.device)
      self.traj_t = torch.zeros(n, device=self.device)  # frames
      self.traj_speed = torch.ones(n, device=self.device)
      self.traj_side = torch.zeros(n, dtype=torch.long, device=self.device)
    for m in MODES:
      self.metrics[f"frac_{m}"] = torch.zeros(n, device=self.device)

  @property
  def command(
    self,
  ) -> torch.Tensor:  # what the actor sees: current arm target - default
    return self.target - self.default

  def _resample_command(self, env_ids: torch.Tensor) -> None:  # per episode
    p = torch.tensor(
      [self.cfg.mode_probs.get(m, 0.0) for m in MODES], device=self.device
    )
    self.mode[env_ids] = torch.multinomial(p / p.sum(), len(env_ids), replacement=True)
    off = torch.zeros(len(env_ids), len(self.ids), device=self.device)
    for k, j in enumerate(self.joint_names):
      if any(s in j for s in ("shoulder_pitch", "shoulder_roll", "elbow")):
        off[:, k] = torch.empty(len(env_ids), device=self.device).uniform_(
          -self.cfg.carry_offset, self.cfg.carry_offset
        )
    self.offset[env_ids] = off
    self.reach_timer[env_ids] = 0.0
    self.target[env_ids] = self.default[env_ids]
    if self.cfg.traj_table is not None:
      self._new_traj(env_ids, random_start=True)

  def _new_traj(self, env_ids: torch.Tensor, random_start: bool) -> None:
    k = len(env_ids)
    self.traj_id[env_ids] = torch.randint(len(self._traj), (k,), device=self.device)
    lo, hi = self.cfg.traj_speed
    self.traj_speed[env_ids] = torch.empty(k, device=self.device).uniform_(lo, hi)
    self.traj_side[env_ids] = (
      torch.rand(k, device=self.device) < self.cfg.traj_p_left
    ).long()
    start = torch.rand(k, device=self.device) if random_start else torch.zeros(k)
    self.traj_t[env_ids] = start.to(self.device) * 0.5 * self._traj_len[
      self.traj_id[env_ids]
    ].float()

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    dt = self._env.step_dt
    flag, _ = carry_state(self._env)
    mode = torch.where(
      flag > 0.5, torch.full_like(self.mode, MODES.index("carry")), self.mode
    )
    q = self.robot.data.joint_pos
    # swing
    rest = self.default.clone()  # default pose, rolled outward by default_roll_out
    if self.cfg.default_roll_out:
      rest[:, self.roll_ids] += self.cfg.default_roll_out * self.roll_sign
    swing = rest.clone()
    hip_off = q[:, self.hip] - self.robot.data.default_joint_pos[:, self.hip]
    swing[:, self.sh] = rest[:, self.sh] - 0.8 * hip_off
    # reach: new random goal when the timer runs out
    self.reach_timer -= dt
    redo = self.reach_timer <= 0
    if redo.any():
      k = int(redo.sum())
      r = self.cfg.reach_range
      if self.cfg.reach_box is None:
        self.reach_tgt[redo] = self.default[redo] + torch.empty(
          k, len(self.ids), device=self.device
        ).uniform_(-r, r)
      else:  # collision-free box per joint (left side; right mirrors roll / yaw)
        tgt = self.default[redo].clone()
        for jname, (lo, hi) in self.cfg.reach_box.items():
          for side, sg in (("left", 1.0), ("right", -1.0)):
            j = self.joint_names.index(f"{side}_{jname}_joint")
            v = torch.empty(k, device=self.device).uniform_(lo, hi)
            tgt[:, j] = v * (sg if ("roll" in jname or "yaw" in jname) else 1.0)
        self.reach_tgt[redo] = tgt
      if self.cfg.reach_table is not None:  # right arm from the pose table (optional)
        if not hasattr(self, "_table"):
          import numpy as np

          self._table = torch.as_tensor(
            np.load(self.cfg.reach_table), device=self.device
          )
          self._right_ids = [
            self.joint_names.index(f"right_{j}_joint")
            for j in (
              "shoulder_pitch",
              "shoulder_roll",
              "shoulder_yaw",
              "elbow",
              "wrist_roll",
              "wrist_pitch",
              "wrist_yaw",
            )
          ]
        use = torch.rand(k, device=self.device) < self.cfg.reach_table_prob
        if use.any():
          rows = self._table[
            torch.randint(len(self._table), (int(use.sum()),), device=self.device)
          ]
          tgt = self.reach_tgt[redo].clone()
          sub = tgt[use]
          left = torch.rand(len(rows), device=self.device) < self.cfg.reach_table_p_left
          if not hasattr(self, "_left_ids"):
            self._left_ids = [i_r - 0 for i_r in [
              self.joint_names.index(f"left_{j}_joint") for j in RIGHT7]]
            self._mir = torch.tensor(MIRROR7, device=self.device)
          r_rows = sub[:, self._right_ids].clone()
          l_rows = sub[:, self._left_ids].clone()
          r_rows = torch.where(left[:, None], r_rows, rows)
          l_rows = torch.where(left[:, None], rows * self._mir, l_rows)
          sub[:, self._right_ids] = r_rows
          sub[:, self._left_ids] = l_rows
          tgt[use] = sub
          self.reach_tgt[redo] = tgt
      self.reach_timer[redo] = torch.empty(k, device=self.device).uniform_(1.0, 3.0)
      pk = self.default[redo].clone()
      u = lambda lo, hi: torch.empty(k, 2, device=self.device).uniform_(lo, hi)  # noqa: E731
      pk[:, self.pick_sp] = u(*self.cfg.pick_shoulder_pitch)
      pk[:, self.pick_el] = u(*self.cfg.pick_elbow)
      roll = u(0.0, self.cfg.pick_roll_max)  # inward/outward, mirrored per side
      pk[:, self.pick_sr] = torch.stack([roll[:, 0], -roll[:, 1]], dim=1)
      self.pick_tgt[redo] = pk
    traj = rest.clone()
    if self.cfg.traj_table is not None:
      self.traj_t += dt * self._traj_fps * self.traj_speed
      done = self.traj_t >= (self._traj_len[self.traj_id] - 1).float()
      if done.any():
        self._new_traj(torch.nonzero(done).squeeze(-1), random_start=False)
      idx = self.traj_t.long().clamp(min=0)
      pose = self._traj[self.traj_id, idx]  # (n, 7) right-arm joints
      right, left = self._side_ids
      use_l = self.traj_side == 1
      traj[:, right] = torch.where(use_l[:, None], rest[:, right], pose)
      traj[:, left] = torch.where(use_l[:, None], pose * self._mirror, rest[:, left])
    goal = torch.stack(
      [swing, rest, self.carry + self.offset, self.reach_tgt, self.pick_tgt, traj],
      dim=1,
    )
    goal = goal[torch.arange(self.num_envs, device=self.device), mode]
    a = dt / (self.cfg.tau + dt)
    self.target = self.target + a * (goal - self.target)
    self.robot.set_joint_position_target(self.target, joint_ids=self.ids)

  def _update_metrics(self) -> None:
    for i, m in enumerate(MODES):
      self.metrics[f"frac_{m}"] = (self.mode == i).float()


def _probs() -> dict[str, float]:
  return {"swing": 0.7, "default": 0.3, "carry": 0.0, "reach": 0.0}


@dataclass(kw_only=True)
class ArmCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # per episode
  mode_probs: dict[str, float] = field(default_factory=_probs)
  carry_offset: float = 0.15  # rad, box-size / grip variation
  reach_range: float = 0.6  # rad around default
  tau: float = 0.25  # s, arm controller smoothing
  # None: reach = default +- reach_range on every arm joint (hits the body in ~50% of
  # poses, CPU check 2026-10-10). Dict: per-joint (lo, hi) for the LEFT arm, right mirrored.
  reach_box: dict[str, tuple[float, float]] | None = None
  # None: carry pose = carry_pose.POSE (hands hit the hips in 12-35 % of deep-squat + lean
  # poses). Dict: LEFT-arm joint values (right mirrored) replacing it.
  carry_override: dict[str, float] | None = None
  # rad added outward to shoulder roll in the default / swing poses (arms sag ~0.06-0.1 rad
  # under gravity, and the hands then graze the hips in ~40-60 % of walking frames).
  default_roll_out: float = 0.0
  # None (default, all existing tasks): unchanged. Path to an (N, 7) .npy of RIGHT-arm joint
  # poses (shoulder p/r/y, elbow, wrist r/p/y): with probability reach_table_prob a new reach
  # goal takes a random row for the right arm (e.g. palm-in grasp poses from IK).
  reach_table: str | None = None
  reach_table_prob: float = 0.0
  reach_table_p_left: float = 0.0  # probability a table pose goes to the LEFT arm (mirrored)
  # None (default): no 'traj' mode. Path to an .npz with traj (N, T, 7) RIGHT-arm joint
  # trajectories, lengths (N,), fps: the 'traj' mode replays them (see the module doc).
  traj_table: str | None = None
  traj_speed: tuple[float, float] = (0.8, 1.5)  # playback speed range
  traj_p_left: float = 0.5  # probability the trajectory drives the LEFT arm (mirrored)
  pick_shoulder_pitch: tuple[float, float] = (-1.3, -0.3)  # rad, arms forward/down
  pick_elbow: tuple[float, float] = (0.0, 0.9)
  pick_roll_max: float = 0.35

  def build(self, env: ManagerBasedRlEnv) -> ArmCommand:
    return ArmCommand(self, env)
