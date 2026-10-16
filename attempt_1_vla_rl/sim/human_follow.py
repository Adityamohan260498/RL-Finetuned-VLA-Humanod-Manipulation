"""Retargeting: the G1's right hand + Dex3 FOLLOW my tracked hand (perception/hand_track.py) in
MuJoCo physics. Contact-only grasping, legs = stance policy.

  .venv_perc/bin/python sim/human_follow.py data/tracks/<clip>.npz [--xy 0.27,-0.17 --yaw 10]
        [--video data/follow_<clip>.mp4] [--legs policy/stance_palmin.onnx]

The mapping (human table frame and robot world frame both: x forward, y left, z up):
  position  two anchors map exactly: my pinch at MY grasp -> the robot's block (grasp centre), my
            pinch at MY release -> the robot's drop point (aimed so the BLOCK centre lands on the
            star; in-hand offset measured after the grasp). The anchor slides from one to the other
            during the carry (smoothly, on my timing). My hand's excursion from the current anchor
            is kept in direction and timing and scaled by SCALE: the G1's arm is shorter and it
            stands 19 cm back from the table edge, where I stand right at it.
  yaw       robot hand yaw(t) = block yaw (fingers on two flat faces) + MY yaw change since my
            grasp, limited to +-25 deg (palm in, fingers horizontal).
  grip      my grip signal 0..1 (aperture) blends the Dex3 from its open pre-shape to the 3-point
            squeeze.
  timing    my motion is played TIME_SCALE x slower; the arm aims LEAD s ahead on my path (its
            joint targets are low-passed with tau 0.25 s, so it lags). At my grasp moment my clock
            pauses: the hand settles on my grasp point (open, <= 1.2 s), closes (0.8 s), holds
            (0.4 s) - the Dex3's servos need that; my fingers close while my hand already lifts.
  not mine  getting from the robot's rest pose to where my tracked path starts (collision-free
            planner) and back to rest at the end. Recorded and reported separately.
Robot-side corrections (reported, not hidden): the final approach (last DESCEND s before my grasp)
comes over the grasp point and straight down (CLEAR above it), the clock pauses (arrive, close, hold) at my
grasp;
safety floor: the hand target never goes below 1 cm under the grasp height.

Recorded at 25 Hz for the VLA (frame = world = table frame; independent of the torso pose):
  img    head camera 224 x 224
  state  (5) grasp centre x, y, z (m), hand yaw (rad), grip command 0..1
  act    (5) target - current: dx, dy, dz (m), dyaw (rad); grip command 0..1
  q      (14) right arm 7 + Dex3 7 joint angles (for reference)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(COMMON / "stance"))
sys.path.insert(0, str(COMMON / "scene"))
import robot_motion as M  # noqa: E402
from layout import DROP_XY, PICK_REGION, TABLE_TOP, TASK_WAIST_YAW  # noqa: E402
from stance_runner import CONTROL_DT, DEX3_OPEN  # noqa: E402

_env = __import__("os").environ.get
if _env("SQUEEZE_IDX"):  # extra index/middle squeeze past contact (rad; default in robot_motion: 0.25)
  _x = float(_env("SQUEEZE_IDX"))
  M.DEX3_SQUEEZE = M.DEX3_GRAB + np.array([0.0, 0.0, -float(_env("SQUEEZE_TH", "0.25")), _x, _x, _x, _x])
M.GRASP_H = float(_env("GRASP_H", "0.04"))  # v2: 4 cm (was 5): grip nearer the block's centre, less creep
TIME_SCALE = 2.0
CARRY_TS = float(_env("CARRY_TS", "1.2"))  # v2: carry faster than the rest (in-hand creep grows with time)  # my clock's slow-down while carrying (robot holds the block)
# grasp pause at my grasp moment (robot time): arrive <= 1.2 s (open, until within 5 mm), close 0.8 s, hold 0.4 s
GRIP_LATE = float(__import__("os").environ.get("GRIP_LATE", "0"))  # s, my time
CLEAR = float(__import__("os").environ.get("CLEAR", "0.14"))  # v2 (10-15): 0.10 let fast approaches knock the block  # final approach: clearance above the grasp (m)
REL_PAUSE = __import__("os").environ.get("REL_PAUSE", "1") == "1"  # release pause (arrive, open, up)
APPROACH = __import__("os").environ.get("APPROACH", "all")  # "all": my approach x/y at >= CLEAR height
CARRY_CLEAR = float(__import__("os").environ.get("CARRY_CLEAR", "0.04"))  # carry: hand >= this above the
               # grasp / drop height (block clear of the table), my x/y converge over the drop point in the last
               # DESCEND s before my release, then a robot-paced straight descent to touch-down (0 = v1 behaviour)
PLACE_DOWN_S = float(_env("PLACE_DOWN_S", "2.0"))  # that descent (s, robot time)
HOVER_MAX = float(_env("HOVER_MAX", "2.0"))  # settle above the drop point first (s, at most)
DESCEND = 0.6  # final approach (s, my time): rise to CLEAR in the 0.24 s before it, x/y converge over
               # the grasp point, then straight down
SCALE = 0.6          # my path's excursions from its anchors, scaled to the G1 (arm ~0.6 x mine)
LEAD = 0.25          # s (robot time)
PRE_S, POST_S = 2.0, 1.5  # my path: from its highest point 0.5-2.5 s before my grasp to 1.5 s after my release
YAW_LIM = np.radians(float(__import__("os").environ.get("YAW_LIM_DEG", "25")))
FLOOR = M.GRASP_H - 0.01
STEP_MAX = 0.03      # rad per control step on the arm goal (1.5 rad/s)
TASK = "pick up the blue block and put it on the blue star"


def smooth(s):
  s = np.clip(s, 0.0, 1.0)
  return 3 * s * s - 2 * s ** 3


def wrap(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


def hand_q(g, released=False):
  """Grip 0..1 -> Dex3 joints: open pre-shape .. 3-point squeeze. (Opening fully at the release
  swings the thumb, thumb_1 0.49 -> 0, and it flicks the block over: tried, measured, removed.)"""
  base = M.PRESHAPE
  return base + float(np.clip(g, 0, 1)) * (M.DEX3_SQUEEZE - base)


def face_yaw(yaw_deg):
  """Block yaw folded into [-45, 45] deg: the hand closes on the pair of faces nearest straight."""
  return (yaw_deg + 45.0) % 90.0 - 45.0


class Follow(M.Motion):
  def block_on_table(self):
    """The block touches the table (MuJoCo contact; a real hand would feel it as a load drop)."""
    m, d = self.bot.model, self.bot.data
    for c in d.contact[:d.ncon]:
      if (c.geom1 == self.block_geom and c.geom2 in self.bot.table_geoms) or \
         (c.geom2 == self.block_geom and c.geom1 in self.bot.table_geoms):
        return True
    return False

  def ee(self):
    Rt, pt = self.T()
    p, R = self.bot.grip_poses()["right"]
    Rw = Rt @ R
    return Rt @ p + pt, float(np.arctan2(Rw[1, 0], Rw[0, 0]))

  def run(self, track, block_xy, block_yaw_deg, record=False, video_cams=()):
    z = np.load(track, allow_pickle=True)
    t_h, pinch, yaw_h, grip = z["t"], z["pinch"], z["yaw"], z["grip"]
    tg, tr = float(z["grasp_t"]), float(z["release_t"])
    ig, ir = int(np.searchsorted(t_h, tg)), int(np.searchsorted(t_h, tr))
    # start following where my approach is highest in the 0.5-2.5 s before my grasp (clear of the table)
    win = (t_h >= tg - PRE_S - 0.5) & (t_h <= tg - 0.5)
    t0 = float(t_h[win][np.argmax(pinch[win, 2])]) if win.any() else max(t_h[0], tg - PRE_S)
    t1 = min(t_h[-1], tr + POST_S)
    interp = lambda t, x: np.array([np.interp(t, t_h, x[:, j]) for j in range(x.shape[1])]) if x.ndim == 2 \
      else float(np.interp(t, t_h, x))  # noqa: E731

    bot = self.bot
    bot.reset()
    self.worst = {"fall": False, "table": 0, "drift": 0.0, "tilt": 0.0}
    self.hand_table_steps = 0
    self.log = []
    self.k_step = 0
    self.rec = {"img": [], "wrist": [], "state": [], "act": [], "q": [], "phase": [], "video": []} if record else None
    self.video_cams = video_cams
    self.cur_target = None
    bot.set_posture(waist_yaw=TASK_WAIST_YAW)
    bot.set_hand_joints("right", DEX3_OPEN)
    self.place_block(block_xy, block_yaw_deg)
    self.P.ignore_bodies = set()
    self.phase = "settle"
    self.hold(1.5)  # as in the eval: the episode starts 1.5 s in

    yaw_g = np.radians(face_yaw(block_yaw_deg))
    G_r = np.array([*block_xy, TABLE_TOP + M.GRASP_H])
    o_g = G_r - pinch[ig]
    # first guess of the drop aim (typical in-hand offset), refined once the block is held
    off = np.array([0.0, 0.0, 0.0])
    P_r = lambda off: np.array([DROP_XY[0] - off[0], DROP_XY[1] - off[1],  # noqa: E731
                                TABLE_TOP + 0.04 + M.RELEASE_DZ - off[2]])
    o_p = P_r(off) - pinch[ir]
    offset_fixed = False

    G_h, P_h = pinch[ig], pinch[ir]

    def target(th):
      a = 0.0 if th <= tg else (1.0 if th >= tr else smooth((th - tg) / (tr - tg)))
      # anchors map exactly (my grasp -> the robot's block, my release -> its drop point); how far
      # my hand is from the current anchor is scaled by SCALE (shorter arm, body further back)
      anchor_h = (1 - a) * G_h + a * P_h
      anchor_r = (1 - a) * G_r + a * (P_h + o_p)
      p = anchor_r + SCALE * (interp(th, pinch) - anchor_h)
      # final approach (robot-specific): my hand's HEIGHT just before the grasp is the least
      # reliable signal (monocular depth drifts as my fingers close: my pinch appears to RISE into
      # the grasp), and the Dex3 must come down onto a standing block from above. So in the last
      # DESCEND s before my grasp the hand converges over the grasp point at CLEAR above it and
      # lowers vertically; before that, x/y follow me and z = max(mine, CLEAR above the grasp).
      if th < tg and CLEAR > 0:
        u = (th - (tg - DESCEND)) / DESCEND  # <0 before the final approach, 0..1 during it
        rise = 1.0 if APPROACH == "all" else smooth((u + 0.4) / 0.4)  # CLEAR above the grasp height
        conv = smooth(u / 0.5)  # then x/y converge over the grasp point
        down = 0.0  # (the straight descent is its own robot-paced phase at my grasp moment)
        p[:2] = p[:2] + conv * (G_r[:2] - p[:2])
        p[2] = max(p[2], G_r[2] + rise * CLEAR * (1.0 - down))
      if tg < th and CARRY_CLEAR > 0:  # carry (robot-specific, mirrors the approach): block clear of the table,
        a_up = smooth((th - tg) / (0.25 * (tr - tg)))  # over the drop point before lowering
        drop = P_h + o_p
        conv = smooth((th - (tr - DESCEND)) / (0.5 * DESCEND))
        p[:2] = p[:2] + conv * (drop[:2] - p[:2])
        if th <= tr:
          p[2] = max(p[2], (1 - a) * G_r[2] + a * drop[2] + a_up * CARRY_CLEAR)
      p[2] = max(p[2], TABLE_TOP + FLOOR)
      w_mine = float(np.clip(wrap(interp(th, yaw_h) - yaw_h[ig]), -YAW_LIM, YAW_LIM))
      g = interp(th, grip)
      if th < tg and CLEAR > 0:  # final approach: fingers on the block's faces, hand open
        fa = smooth((th - (tg - DESCEND)) / DESCEND)
        w_mine *= 1.0 - fa
        g *= 1.0 - fa
      w = yaw_g + w_mine
      if th < tg and GRIP_LATE > 0:  # (off by default; tried, did not help) the Dex3 closes only in the last GRIP_LATE s before my grasp (its stiffer,
        g *= smooth((th - (tg - GRIP_LATE)) / GRIP_LATE)  # shorter fingers hit the block if they close on the way in)
      if th >= tr:  # I let go (the robot opened in its release pause)
        g = 0.0
      return p, w, g

    # 1) NOT mine: planned move from rest to the start of my path (pre-shaping the hand). If my
    # path starts where the robot cannot go (near its chest / the table), start later on it.
    self.phase = "to_start"
    path, up = None, None
    for t_try in np.arange(t0, tg - 0.3, 0.1):
      p0, w0, _ = target(t_try)
      self.R = M.hand_R(0.0, np.degrees(w0))
      for lift in (0.0, 0.06, 0.12):
        try:
          self.target = p0 + [0, 0, lift]
          path, up = self.plan_to(self.target), (p0 + [0, 0, lift] if lift else None)
          break
        except RuntimeError:
          continue
      if path is not None:
        t0 = float(t_try)
        break
    if path is None:
      raise RuntimeError("no collision-free path to any start point on my approach")
    self.run_q(path, self.P.timed(path, vmax=M.FREE_VMAX), M.PRESHAPE, DEX3_OPEN.copy())
    if up is not None:
      self.target = p0
      self.track(up, p0, 1.0 + 5 * float(up[2] - p0[2]))
    self.servo(p0, 0.5, M.PRESHAPE, M.PRESHAPE)
    # 2) MINE: follow my path
    self.phase = "follow"
    goal = bot.arm_goal.copy()
    track_err = []
    diag = {}
    b0 = self.block_state()[0].copy()
    th, n = t0, 0
    mode, mode_k, ok_n = "follow", 0, 0  # follow -> (at my grasp) arrive -> close -> hold -> follow
    grasped = released = False
    freeze, lifted = None, False
    while th <= t1:
      for key, tt in (("at_grasp", tg), ("at_release", tr)):
        if key not in diag and th >= tt:
          pb_, up_ = self.block_state()
          diag[key] = {"moved_cm": round(100 * float(np.linalg.norm(pb_[:2] - b0[:2])), 1),
                       "z_cm": round(100 * float(pb_[2] - TABLE_TOP), 1), "upright": round(up_, 2)}
      if tg + 0.35 * (tr - tg) <= th <= tg + 0.7 * (tr - tg):  # held + carried: in-hand offset
        Rt, pt = self.T()
        tip = Rt @ bot.grip_poses()["right"][0] + pt
        m_off = self.block_state()[0] - tip
        off = m_off if not offset_fixed else off + 0.1 * (m_off - off)
        o_p = P_r(off) - pinch[ir]
        offset_fixed = True
        self.in_hand_offset = off
      if mode in ("r_hover", "r_arrive", "r_open", "r_up") and REL_PAUSE:  # release pause at my release point
        p, w, g = target(tr)
        if mode == "r_hover":  # CARRY_CLEAR above the drop point, holding, until the arm (+ swaying
          g = 1.0              # body) has settled: within 8 mm for 0.3 s, at most HOVER_MAX s
          p = (P_h + o_p) + [0, 0, CARRY_CLEAR]
          ok_n = ok_n + 1 if np.linalg.norm(self.ee()[0] - p) < 0.008 else 0
          if ok_n >= 15 or mode_k * CONTROL_DT >= HOVER_MAX:
            mode, mode_k, ok_n = "r_arrive", 0, 0
        if mode == "r_arrive":  # still holding: settle within 5 mm (at most 1 s) - or stop as soon
          g = 1.0                # as the block touches the table (a tilted block lands on a corner first)
          down_k = 0
          if CARRY_CLEAR > 0:  # straight down from CARRY_CLEAR above the drop point, robot-paced
            s_d = min(1.0, (mode_k * CONTROL_DT + LEAD) / PLACE_DOWN_S)
            p = (P_h + o_p) + [0, 0, CARRY_CLEAR * (1.0 - smooth(s_d))]  # (target(tr) has the floor)
            down_k = int(PLACE_DOWN_S / CONTROL_DT)
          ok_n = ok_n + 1 if mode_k >= down_k and np.linalg.norm(self.ee()[0] - p) < 0.005 else 0
          touch = lifted and self.block_on_table()
          if touch or ok_n >= 10 or mode_k >= 50 + down_k:
            freeze = self.ee()[0] + [0, 0, 0.007] if touch else p  # lift 7 mm: open in free air (the opening thumb hits the table otherwise)
            mode, mode_k = "r_open", 0
        if mode in ("r_open", "r_up"):
          p = freeze.copy()  # do not push on: hold where it was when the block landed
        if mode == "r_open":  # open over 0.6 s
          g = 1.0 - smooth(mode_k * CONTROL_DT / 0.6)
          if mode_k * CONTROL_DT >= 0.6:
            mode, mode_k = "r_up", 0
        elif mode == "r_up":  # straight up 5 cm in 0.5 s, then my motion continues (from up there)
          g = 0.0
          lift_z = 0.05 * smooth(mode_k * CONTROL_DT / 0.5)
          p = p + [0, 0, lift_z]
          if mode_k * CONTROL_DT >= 0.5:
            mode, released = "follow", True
      elif mode == "follow":
        # aim LEAD ahead on my path, but not past my grasp before the robot has closed its hand
        th_aim = min(t1, th + LEAD / TIME_SCALE)
        if not grasped:
          th_aim = min(th_aim, tg)
        elif not released and REL_PAUSE:
          th_aim = min(th_aim, tr)
        p, w, g = target(th_aim)
        if grasped and not lifted and self.block_state()[0][2] > TABLE_TOP + 0.05:
          lifted = True
        if grasped and lifted and not released and REL_PAUSE and self.block_on_table():
          mode, mode_k, ok_n = "r_arrive", 0, 0  # landed before my release moment: release now
        if grasped and not released and REL_PAUSE:
          g = 1.0  # holding until the robot's own release (aiming ahead must not open the hand)
        if released or (not REL_PAUSE and th >= tr):
          released = True  # after my release: my path, kept at least 5 cm above where it let go
          p[2] = max(p[2], target(tr)[0][2] + 0.05)
      else:  # grasp pause at my grasp point
        p, w, _ = target(tg)
        if mode == "descend":  # straight down from CLEAR above, robot-paced (1.5 s, hand open)
          g = 0.0
          s_d = min(1.0, (mode_k * CONTROL_DT + LEAD) / 1.5)
          p = p + [0, 0, CLEAR * (1.0 - smooth(s_d))]
          if mode_k * CONTROL_DT >= 1.5:
            mode, mode_k, ok_n = "arrive", 0, 0
        elif mode == "arrive":  # hand open until within 5 mm for 0.2 s (at most 1.2 s)
          g = 0.0
          ok_n = ok_n + 1 if np.linalg.norm(self.ee()[0] - p) < 0.005 else 0
          if ok_n >= 10 or mode_k >= 60:
            mode, mode_k = "close", 0
        elif mode == "close":  # close over 0.8 s
          g = smooth(mode_k * CONTROL_DT / 0.8)
          if mode_k * CONTROL_DT >= 0.8:
            mode, mode_k = "hold", 0
        else:  # hold 0.4 s, then my motion continues
          g = 1.0
          if mode_k * CONTROL_DT >= 0.4:
            mode, grasped = "follow", True
      self.phase = mode
      self.R = M.hand_R(0.0, np.degrees(w))
      self.target = p
      self.cur_target = (p, w, g)
      q = self.ik(p, goal[7:].copy(), iters=60, fallback=False)
      goal[7:] = goal[7:] + np.clip(q - goal[7:], -STEP_MAX, STEP_MAX)
      bot.set_arm_joint_targets(goal)
      bot.set_hand_joints("right", hand_q(g, released=mode in ("r_open", "r_up") or released))
      self.step()
      n += 1
      mode_k += 1
      track_err.append(float(np.linalg.norm(self.ee()[0] - target(th)[0])))
      if mode == "follow":
        if not grasped and th >= tg:
          mode, mode_k, ok_n = ("descend" if CLEAR > 0 else "arrive"), 0, 0
        elif grasped and not released and th >= tr and REL_PAUSE:
          mode, mode_k, ok_n = ("r_hover" if CARRY_CLEAR > 0 else "r_arrive"), 0, 0
        else:
          th += CONTROL_DT / (CARRY_TS if grasped and not released else TIME_SCALE)
    # 3) NOT mine: back to rest
    self.phase = "to_rest"
    self.cur_target = None
    self.P.ignore_bodies = set()
    self.P.sync(bot.data)
    self.P.clear = self.P.finger_clear = M.FREE_CLEAR
    rest = self.P.plan(bot.data.qpos[self.P.qadr].copy(), M._rest())
    self.P.clear, self.P.finger_clear = M.LINE_CLEAR, M.LINE_HAND_CLEAR
    if rest is not None:
      self.run_q(rest, self.P.timed(rest, vmax=M.FREE_VMAX), DEX3_OPEN, bot.hand_goal["right"].copy())
    self.hold(1.0)
    pb, up = self.block_state()
    err = float(np.linalg.norm(pb[:2] - np.array(DROP_XY)) * 100)
    on_table = bool(abs(pb[2] - (TABLE_TOP + 0.04)) < 0.01)
    res = {"place_err_cm": round(err, 2), "upright": round(up, 3), "on_table": on_table,
           "fallen": self.worst["fall"], "body_table": int(self.worst["table"]),
           "hand_table_steps": int(self.hand_table_steps), "drift_cm": round(100 * self.worst["drift"], 1),
           "track_err_cm_median": round(100 * float(np.median(track_err)), 2),
           "track_err_cm_p95": round(100 * float(np.percentile(track_err, 95)), 2),
           "follow_s": round(n * CONTROL_DT, 1), "t0_before_grasp_s": round(tg - t0, 2), **diag, "total_s": round(bot.t, 1),
           "in_hand_offset_cm": None if not offset_fixed else (100 * self.in_hand_offset).round(1).tolist()}
    res["success"] = bool(err <= 3.0 and up >= 0.95 and on_table and not self.worst["fall"]
                          and self.worst["table"] == 0)
    return res

  def step(self):
    """Records the VLA sample BEFORE the physics step: what the robot saw + where it was, and the
    command it then got."""
    if self.rec is not None and self.k_step % M.REC_EVERY == 0:
      import cv2
      b = self.bot
      p, w = self.ee()
      g = 0.0
      if self.cur_target is not None:
        tp, tw, g = self.cur_target
      else:  # planned (not mine) segments: the command is where the planner is taking the hand
        tp, tw = self.target, float(np.arctan2(self.R[1, 0], self.R[0, 0]))
        g = float(np.clip((b.hand_goal["right"][3] - M.PRESHAPE[3]) /
                          (M.DEX3_SQUEEZE[3] - M.PRESHAPE[3]), 0, 1))
      self.rec["img"].append(cv2.resize(b.render("head"), (224, 224), interpolation=cv2.INTER_AREA))
      self.rec["wrist"].append(cv2.resize(b.render("wrist"), (224, 224), interpolation=cv2.INTER_AREA))
      self.rec["state"].append(np.r_[p, w, g])
      self.rec["act"].append(np.r_[tp - p, wrap(tw - w), g])
      self.rec["q"].append(np.r_[b.data.qpos[self.P.qadr], b.data.qpos[b.hand_qadr["right"]]])
      self.rec["phase"].append(self.phase)
      if self.video_cams:
        self.rec["video"].append(np.hstack([b.render(c) for c in self.video_cams]))
    self.bot.step()
    self.k_step += 1
    st = self.bot.stability()
    self.worst["fall"] |= st["fallen"]
    self.worst["table"] = max(self.worst["table"], self._body_table_contacts())
    self.worst["drift"] = max(self.worst["drift"], st["base_drift_m"])
    self.worst["tilt"] = max(self.worst["tilt"], st["torso_tilt_deg"])
    if self.hook:
      self.hook(self)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("track")
  ap.add_argument("--xy", default="")
  ap.add_argument("--yaw", type=float, default=0.0)
  ap.add_argument("--video", default="")
  ap.add_argument("--legs", default=str(COMMON / "policy/stance_palmin.onnx"))
  a = ap.parse_args()
  xy = tuple(float(v) for v in a.xy.split(",")) if a.xy else tuple(np.mean(PICK_REGION, axis=1))
  fo = Follow(policy=Path(a.legs))
  res = fo.run(a.track, xy, a.yaw, record=bool(a.video), video_cams=("front", "head") if a.video else ())
  print(json.dumps(res))
  if a.video:
    import imageio
    imageio.mimsave(a.video, fo.rec["video"], fps=25)


if __name__ == "__main__":
  main()
