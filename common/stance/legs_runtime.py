"""Numpy port of the legs-only posture policy's command side, for deployment.

Mirrors mjlab.tasks.human_v2.mdp.modes.GaitModeCommand and
mjlab.tasks.human_v2_posture.posture.PostureCommand with the FINAL (play) settings, so the
actor sees exactly the command signals it was trained on:
  gait_mode      [sin, cos (0 while STAND), onehot(STAND, WALK, W2S)]
  w2s_progress   0 -> 1 during walk-to-stand
  posture        rate-limited (height - 0.78, torso pitch, waist yaw)
  posture_goal   clamped goal - nominal, + time for the target to get there
  command        the twist, scaled down while crouched (x1 at >= 0.70 m, x0.3 at 0.60 m)
Order per 50 Hz step = mjlab's command order: twist (carry cap) -> gait -> posture.
"""

from __future__ import annotations

import math

import numpy as np

STAND, WALK, W2S = 0, 1, 2
NOMINAL_H, H_MIN = 0.78, 0.50
W2S_TIME = 1.5
STAND_SPEED = 0.1
CARRY_VX_MAX = 1.2
WALK_H, WALK_PITCH, WALK_YAW = (0.60, 0.80), (-0.05, 0.45), 0.30
STAND_H, STAND_PITCH, STAND_YAW = (H_MIN, 0.80), (-0.1, 0.8), 0.8
RATES = np.array([0.3, 0.8, 1.0])
CROUCH_TOP, CROUCH_FLOOR, MIN_SPEED_SCALE = 0.70, 0.60, 0.3


def gait_params(cmd):
  speed = float(np.hypot(cmd[0], cmd[1]))
  yaw = abs(float(cmd[2]))
  freq = min(max(0.9 + 0.4 * (speed + 0.25 * yaw), 0.9), 1.7)
  low = speed < STAND_SPEED and yaw < STAND_SPEED
  return freq, low


class LegsCommands:
  def __init__(self, dt: float):
    self.dt = dt
    self.nominal = np.array([NOMINAL_H, 0.0, 0.0])
    self.reset()

  def reset(self):
    self.mode, self.phase, self.low_time = STAND, 0.0, 1e3
    self.goal = self.nominal.copy()
    self.goal_eff = self.nominal.copy()
    self.target = self.nominal.copy()
    self.walk_ready = True
    self.twist = np.zeros(3)  # what the actor sees (scaled)
    self.scale = 1.0

  def set_posture_goal(self, height=NOMINAL_H, pitch=0.0, yaw=0.0):
    self.goal = np.array([np.clip(height, *STAND_H), np.clip(pitch, *STAND_PITCH),
                          np.clip(yaw, -STAND_YAW, STAND_YAW)])

  def step(self, user_twist, carrying: bool):
    raw = np.array(user_twist, dtype=float)
    if carrying:
      raw[0] = np.clip(raw[0], -CARRY_VX_MAX, CARRY_VX_MAX)
    # -- gait (sees the twist as scaled at the end of the previous step) --
    tw_prev = raw * self.scale
    freq, low = gait_params(tw_prev)
    if low:
      self.low_time += self.dt
    else:
      self.low_time = 0.0
    go = (not low) and (self.walk_ready or self.mode != STAND)
    if go:
      self.mode = WALK
    elif low and self.mode == WALK:
      self.mode = W2S
    if self.mode == W2S and self.low_time >= W2S_TIME:
      self.mode = STAND
    self.phase = (self.phase + freq * self.dt) % 1.0
    # -- posture --
    _, low_raw = gait_params(raw)
    want_walk = (not low_raw) or self.mode != STAND
    g = self.goal.copy()
    if want_walk:
      g = np.clip(g, [WALK_H[0], WALK_PITCH[0], -WALK_YAW], [WALK_H[1], WALK_PITCH[1], WALK_YAW])
    self.goal_eff = g
    step = np.clip(g - self.target, -RATES * self.dt, RATES * self.dt)
    self.target = self.target + step
    lo = np.array([WALK_H[0], WALK_PITCH[0], -WALK_YAW]) - 0.01
    hi = np.array([WALK_H[1], WALK_PITCH[1], WALK_YAW]) + 0.01
    self.walk_ready = bool(np.all((self.target >= lo) & (self.target <= hi)))
    depth = np.clip((CROUCH_TOP - self.target[0]) / (CROUCH_TOP - CROUCH_FLOOR), 0.0, 1.0)
    self.scale = 1.0 - depth * (1.0 - MIN_SPEED_SCALE)
    self.twist = raw * self.scale
    return self.twist

  # ---- observation terms ----
  def gait_mode_obs(self):
    a = 2 * math.pi * self.phase
    sc = np.array([math.sin(a), math.cos(a)]) * (0.0 if self.mode == STAND else 1.0)
    oh = np.zeros(3)
    oh[self.mode] = 1.0
    return np.concatenate([sc, oh])

  def w2s_obs(self):
    return np.array([min(self.low_time / W2S_TIME, 1.0) if self.mode == W2S else 0.0])

  def posture_obs(self):
    return self.target - self.nominal

  def posture_goal_obs(self):
    ttg = float(np.max(np.abs(self.goal_eff - self.target) / RATES))
    return np.concatenate([self.goal_eff - self.nominal, [ttg]])
