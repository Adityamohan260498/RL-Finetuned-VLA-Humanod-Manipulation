"""Right-hand pick-and-place MOTION with the Dex3-1 (no object): go to the pickup, grab, carry to
the drop point on the centre line, release, back to rest. Palm facing in (toward the midline),
fingers TILT deg below horizontal, straight vertical approach from above. Free-space moves are
planned collision-free on the ground-truth scene (sim/planner.py); legs = stance policy.

  .venv_perc/bin/python sim/tools/pick_place_motion.py --check     # reach table for candidates
  .venv_perc/bin/python sim/tools/pick_place_motion.py [--pick 0.30,-0.22 --drop 0.32,0.0]
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
sys.path.insert(0, str(COMMON / "stance"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(COMMON / "sim"))
sys.path.insert(0, str(COMMON / "scene"))
from layout import DROP_XY, PICK_XY, TABLE_TOP, TASK_WAIST_YAW  # noqa: E402
from planner import ArmPlanner  # noqa: E402
from stance_runner import CONTROL_DT, DEX3_GRAB, DEX3_OPEN, StanceRobot  # noqa: E402

# grip: command the fingers PAST the 3.5 cm contact pose, so the position servos press on the
# block (torque = kp x remaining error, capped by Unitree's 1.4 N m per finger joint)
DEX3_SQUEEZE = DEX3_GRAB + np.array([0.0, 0.0, -0.25, 0.25, 0.25, 0.25, 0.25])

TILT = 0.0          # fingers below horizontal (deg): horizontal = index and middle both on the block
GRASP_H = 0.05      # grasp centre above the table (m): index ~8 cm, middle ~2.4 cm = 3-point grip
APPROACH = 0.10     # vertical approach / retreat height above the grasp point (m)
LIFT = 0.10         # lift height after the grasp (m)
DROP_HOVER = 0.06   # hover above the drop before lowering (m; higher folds the elbow in)
DROP_YAWS = (0.0, 15.0, 30.0, -15.0)
CARRY_SEC = 3.0  # straight carry duration (s)
REC_EVERY = 2    # record every 2nd control step: 25 Hz  # hand yaw options at the drop (deg)
LINE_CLEAR, LINE_HAND_CLEAR = 0.01, 0.003  # straight lines near the table: arm, hand (m)
FREE_CLEAR = 0.03   # free-space planning margin, arm + hand (m)
FREE_VMAX = 0.5     # free-space speed, fastest joint (rad/s)
RELEASE_DZ = 0.005  # release this much above the pickup height (a block would drop 5 mm)
PRESHAPE = np.array([0.0, 0.49, -0.6, 0.3, 0.3, 0.3, 0.3])  # open, thumb swung toward the fingers


def hand_R(tilt_deg=TILT, yaw_deg=0.0):
  """World orientation of the right wrist_yaw frame: fingers (x) forward and tilt below
  horizontal, palm (+y of the Dex3 right hand) facing +y = in, toward the robot's midline;
  then turned yaw about the vertical (+ = fingers toward the robot's left). The block is
  spawned with the same yaw, so the fingers always close on two flat faces."""
  t, w = np.radians(tilt_deg), np.radians(yaw_deg)
  x = np.array([np.cos(t), 0.0, -np.sin(t)])
  y = np.array([0.0, 1.0, 0.0])
  Rz = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
  return Rz @ np.column_stack([x, y, np.cross(x, y)])


def hand_R_top(axis_yaw_deg, flip=False):
  """Top-down grasp of a LYING block whose long axis points along axis_yaw (world, deg): palm down
  (+y of the Dex3 right hand = -z world), the thumb-finger pinch (hand x) horizontal ACROSS the
  3.5 cm width, index and middle spread (hand z) along the 8 cm length. flip: fingers the other way."""
  w = np.radians(axis_yaw_deg)
  x = np.array([-np.sin(w), np.cos(w), 0.0]) * (-1.0 if flip else 1.0)
  y = np.array([0.0, 0.0, -1.0])
  return np.column_stack([x, y, np.cross(x, y)])


LYING_H = 0.0175  # lying block: centre height above the table (m)
LYING_GRASP_H = 0.022  # grasp centre on a lying block (slightly above its centre: thumb clears its top)
# palm-down pre-shape: thumb curled up (open, the default pre-shape's thumb reaches 1 cm below the
# table), 11.4 cm opening, lowest point 9.2 cm below the palm (table at 10.25 cm)
PRESHAPE_TOP = np.array([0.0, 0.49, -1.30, 0.5, 0.5, 0.5, 0.5])


class Motion:
  def __init__(self, policy=COMMON / "policy/stance_palmin.onnx"):
    self.bot = StanceRobot(policy=policy)
    self.bot.ik_seed_prev = True
    # margins for the slow closed-loop straight lines near the table (approach / lowering / lift);
    # free-space moves use FREE_CLEAR
    self.P = ArmPlanner(self.bot.model, "right", clear=LINE_CLEAR, finger_clear=LINE_HAND_CLEAR)
    self.R = hand_R()
    self.hook = None
    self.rec = None
    _m = self.bot.model
    _wy = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    self.hand_bodies_all = {b for b in range(_m.nbody) if b == _wy or "right_hand_" in _m.body(b).name}
    self.video_cams = ()
    self.target = np.zeros(3)
    self.log = []
    m = self.bot.model
    self.block = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
    if self.block >= 0:
      self.block_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
      self.block_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")]
      wy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
      self.hand_bodies = {b for b in range(m.nbody) if b == wy or "right_hand_" in m.body(b).name}

  def T(self):
    d = self.bot.data
    return d.xmat[self.bot.torso].reshape(3, 3).copy(), d.xpos[self.bot.torso].copy()

  def ik(self, p, seed, iters=400, extra_seeds=(), fallback=True):
    """IK for the grasp centre at world p (current torso pose). The warm start can slide along a
    joint limit away from the target; if it misses by > 5 mm, re-solve from the extra seeds and
    the rest pose and keep the most accurate answer."""
    Rt, pt = self.T()
    ch, pl, Rl = self.bot.tips["right"], Rt.T @ (p - pt), Rt.T @ self.R

    def solve(q0, it, q_rest=None, null=0.05):
      q = ch.ik(pl, Rl, q0, iters=it, w_rot=1.0, q_rest=q0 if q_rest is None else q_rest,
                null=null)
      pp, RR, *_ = ch.fk(q)
      e_p = np.linalg.norm(pp - pl)
      e_r = np.arccos(np.clip((np.trace(RR.T @ Rl) - 1) / 2, -1, 1))
      return q, e_p + 0.1 * e_r, e_p < 0.004 and e_r < np.radians(2)  # 1 cm ~ 5.7 deg
    best = solve(seed, iters)
    if fallback and not best[2]:  # warm start missed: clean seeds (rest, given, fixed set)
      for q0 in (*extra_seeds, _rest(), *IK_SEEDS):
        cand = solve(q0, 300, q_rest=_rest(), null=0.02)  # same solver as the reach map
        if cand[1] < best[1] - 1e-5:
          best = cand
        if best[2]:
          break
    return best[0]

  def tip_err(self, p):
    Rt, pt = self.T()
    return float(np.linalg.norm(Rt @ self.bot.grip_poses()["right"][0] + pt - p))

  def line(self, a, b, n=12):
    """IK sequence along a straight Cartesian line a -> b (seeded continuously)."""
    qs = [self.ik(a, self.bot.arm_goal[7:].copy())]
    for s in np.linspace(0, 1, n + 1)[1:]:
      qs.append(self.ik(a + s * (b - a), qs[-1], fallback=False))
      if np.abs(qs[-1] - qs[-2]).max() > 0.3:
        raise RuntimeError(f"line {a.round(3)} -> {b.round(3)}: arm configuration jump")
    return qs

  # -- execution ------------------------------------------------------------------------
  def track(self, a, b, sec, hand=None, hand_from=None, check=None):
    """Straight line a -> b (world) in sec: one continuous IK sequence (no configuration
    switches) played in joint space, then servo at b until the hand is really there. If no
    continuous straight line exists from the current configuration: a short joint-space move to
    the solved pose at b, collision-checked with the tight margins."""
    try:
      qs = self.line(a, b, 16)
    except RuntimeError:
      q_b = self.ik(b, self.bot.arm_goal[7:].copy())
      self.P.sync(self.bot.data)
      q_now = self.bot.arm_goal[7:].copy()
      if not self.P.segment_free(q_now, q_b):
        raise RuntimeError(f"no safe move {a.round(3)} -> {b.round(3)}: {self.P.who(q_b)}")
      qs = [q_now, q_b]
    if check:
      self.P.sync(self.bot.data)
      if not self.P.cartesian_free(qs):
        raise RuntimeError(f"{check} blocked: " + self.P.who(qs[-1]))
    self.run_q(qs, [sec / (len(qs) - 1)] * (len(qs) - 1), hand, hand_from)
    self.servo(b, 0.4, hand, hand)

  def servo(self, p, sec, hand=None, hand_from=None, tol=0.004):
    """Hold the grasp centre at world point p (closed loop from the CURRENT torso pose, so body
    sway is compensated), optionally moving the fingers; ends early once within tol for 0.2 s."""
    n = max(1, int(sec / CONTROL_DT))
    goal = self.bot.arm_goal.copy()
    q_ref = goal[7:].copy()
    ok = 0
    for k in range(int(n * 3) + 1):  # up to 3x sec while the hand is still off target
      s = min(1.0, k / n)
      # refine the CURRENT configuration only (a switch would sweep the arm through other poses)
      q = self.ik(p, goal[7:].copy(), iters=60, fallback=False)
      goal[7:] = goal[7:] + np.clip(q - goal[7:], -0.02, 0.02)
      self.bot.set_arm_joint_targets(goal)
      if hand is not None:
        self.bot.set_hand_joints("right", hand_from + (3 * s * s - 2 * s ** 3) * (hand - hand_from))
      self.step()
      ok = ok + 1 if self.tip_err(p) < tol else 0
      if k >= n and ok >= int(0.2 / CONTROL_DT):
        break

  def run_q(self, path, durations, hand=None, hand_from=None):
    """Follow a joint path (timed), optionally blending the hand from hand_from to hand."""
    total = sum(durations)
    goal = self.bot.arm_goal.copy()
    for k in range(int(total / CONTROL_DT) + 1):
      t = k * CONTROL_DT
      goal[7:] = self.P.sample(path, durations, t)
      self.bot.set_arm_joint_targets(goal)
      if hand is not None:
        s = min(1.0, t / max(total, 1e-6))
        self.bot.set_hand_joints("right", hand_from + (3 * s * s - 2 * s ** 3) * (hand - hand_from))
      self.step()

  def hold(self, sec, hand=None, hand_from=None):
    goal = self.bot.arm_goal.copy()
    self.run_q([goal[7:], goal[7:]], [sec], hand, hand_from)

  def step(self):
    self.bot.step()
    self.k_step = getattr(self, "k_step", 0) + 1
    if getattr(self, "rec", None) is not None and self.k_step % REC_EVERY == 0:
      import cv2
      b = self.bot
      self.rec["img"].append(cv2.resize(b.render("head"), (224, 224), interpolation=cv2.INTER_AREA))
      self.rec["q"].append(np.r_[b.data.qpos[self.P.qadr], b.data.qpos[b.hand_qadr["right"]]])
      self.rec["act"].append(np.r_[b.arm_goal[7:], b.hand_goal["right"]])
      if self.video_cams:
        self.rec["video"].append(np.hstack([b.render(c) for c in self.video_cams]))
    st = self.bot.stability()
    self.worst["fall"] |= st["fallen"]
    self.worst["table"] = max(self.worst["table"], self._body_table_contacts())
    self.worst["drift"] = max(self.worst["drift"], st["base_drift_m"])
    self.worst["tilt"] = max(self.worst["tilt"], st["torso_tilt_deg"])
    if self.hook:
      self.hook(self)

  def best_config(self, low, high, hand=None):
    """All strict IK solutions at `low` (grasp / drop pose, from every seed); for each, the
    straight line up to `high` solved continuously. Returns (clearance, q_high, q_low) of the
    candidate with the largest minimum clearance along that line, or (-inf, None, None)."""
    self.P.sync(self.bot.data)
    if hand is not None:
      self.P.d.qpos[self.bot.hand_qadr["right"]] = hand
    Rt, pt = self.T()
    ch, Rl = self.bot.tips["right"], Rt.T @ self.R
    best = (-np.inf, None, None)
    best_move = np.inf
    q_now = self.bot.arm_goal[7:].copy()
    seen = []
    for q0 in (self.bot.arm_goal[7:].copy(), _rest(), *IK_SEEDS):
      q = ch.ik(Rt.T @ (low - pt), Rl, q0, iters=300, w_rot=1.0, q_rest=_rest(), null=0.02)
      pp, RR, *_ = ch.fk(q)
      if np.linalg.norm(pp - Rt.T @ (low - pt)) > 0.004 or \
         np.arccos(np.clip((np.trace(RR.T @ Rl) - 1) / 2, -1, 1)) > np.radians(2):
        continue
      if any(np.abs(q - o).max() < 0.05 for o in seen):
        continue
      seen.append(q)
      qs, ok = [q], True
      for s in np.linspace(0, 1, 9)[1:]:
        qn = self.ik(low + s * (high - low), qs[-1], fallback=False)
        if np.abs(qn - qs[-1]).max() > 0.3:
          ok = False
          break
        qs.append(qn)
      if not ok:
        continue
      clear = min(self.P.distance(qq)[0] for qq in qs)
      if clear < 0:
        continue
      # among the collision-free candidates: the one CLOSEST to the current arm pose (small,
      # natural motions; the most-clearance choice raised the elbow and threw the balance off)
      move = float(np.abs(q - q_now).max())
      if best[1] is None or move < best_move - 0.05 or (abs(move - best_move) <= 0.05 and clear > best[0]):
        best, best_move = (clear, qs[-1], q), move
    return best

  # -- block ------------------------------------------------------------------------------
  def place_block(self, xy, yaw_deg, lying=False):
    """Standing: yaw about vertical. Lying: the long axis (the box's local z) horizontal along yaw."""
    d = self.bot.data
    a = self.block_q
    d.qpos[a:a + 3] = [xy[0], xy[1], TABLE_TOP + (LYING_H if lying else 0.04) + 0.0005]
    w = np.radians(yaw_deg) / 2
    qz = np.array([np.cos(w), 0, 0, np.sin(w)])
    if lying:  # yaw * (local z -> world x: 90 deg about y)
      qy = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
      q = np.zeros(4)
      mujoco.mju_mulQuat(q, qz, qy)
      qz = q
    d.qpos[a + 3:a + 7] = qz
    d.qvel[self.bot.model.jnt_dofadr[mujoco.mj_name2id(self.bot.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                       "block_joint")]:][:6] = 0
    mujoco.mj_forward(self.bot.model, d)

  def block_in_hand(self):
    """Block position in the right wrist frame (m) - its change during the carry is slip."""
    d = self.bot.data
    wy = self.bot.palms[1]
    R, p = d.xmat[wy].reshape(3, 3), d.xpos[wy]
    return R.T @ (d.xpos[self.block] - p)

  def grip_forces(self):
    """Normal force (N) of every hand-block contact, summed per finger link."""
    m, d = self.bot.model, self.bot.data
    out, f = {}, np.zeros(6)
    for i, c in enumerate(d.contact[:d.ncon]):
      for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
        if ga == self.block_geom and m.geom_bodyid[gb] in self.hand_bodies:
          mujoco.mj_contactForce(m, d, i, f)
          n = m.body(m.geom_bodyid[gb]).name.replace("right_hand_", "").replace("_link", "")
          out[n] = out.get(n, 0.0) + abs(f[0])
    return out

  def block_state(self):
    d = self.bot.data
    R = d.xmat[self.block].reshape(3, 3)
    return d.xpos[self.block].copy(), float(R[2, 2])  # position, upright (cos of tilt)

  def _body_table_contacts(self):
    """Robot-table contacts EXCLUDING the hand (fingertips brushing the table at a place are fine)."""
    m, d = self.bot.model, self.bot.data
    n = 0
    for c in d.contact[:d.ncon]:
      for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
        if ga in self.bot.table_geoms and m.geom_bodyid[gb] in self.bot.robot_bodies:
          if m.geom_bodyid[gb] in self.hand_bodies_all:  # palm / fingers: hand contact, counted
            self.hand_table_steps = getattr(self, "hand_table_steps", 0) + 1  # separately
          else:
            n += 1
    return n

  def plan_to(self, p, anchor=None, hand=None):
    """Collision-free joint path from the current arm pose to the IK pose at p. Free-space moves
    keep FREE_CLEAR from everything for the whole arm AND hand: the real arm lags the plan and
    cuts corners (0.25 s smoothing), so a tight plan lets the fingers catch on the table."""
    self.P.sync(self.bot.data)
    q0 = self.bot.data.qpos[self.P.qadr].copy()
    if anchor is None:
      q1 = self.ik(p, self.bot.arm_goal[7:].copy())
    else:  # the anchor (grasp / drop pose) configuration with the most clearance whose straight
      q1 = self.best_config(anchor, p, hand)[1]  # line up to p is continuous; p solved from it
      if q1 is None:
        raise RuntimeError(f"no arm configuration reaches {anchor.round(3)} with a clear line up")
    tight = (self.P.clear, self.P.finger_clear)
    self.P.clear = self.P.finger_clear = FREE_CLEAR
    try:
      path = self.P.plan(q0, q1)
    finally:
      self.P.clear, self.P.finger_clear = tight
    if path is None:
      raise RuntimeError(f"no collision-free path to {p.round(3)}: {self.P.who(q1)}")
    return path

  def note(self, phase, p):
    err = self.tip_err(p) * 100
    self.log.append((phase, err))
    print(f"  {phase:28s} hand error {err:4.1f} cm | {self.P.who(self.bot.data.qpos[self.P.qadr])}",
          flush=True)

  def pick_place(self, pick_xy, drop_xy, waist_yaw=0.0, yaw_pick=0.0, yaw_drop=0.0, lying=False):
    """Standing block: palm-in grasp, hand yaw = yaw_pick (= block yaw mod 90). Lying block
    (lying=True, yaw_pick = its long-axis direction): top-down grasp across the width."""
    bot = self.bot
    bot.reset()
    self.worst = {"fall": False, "table": 0, "drift": 0.0, "tilt": 0.0}
    self.hand_table_steps = 0
    self.log = []
    bot.set_posture(waist_yaw=waist_yaw)
    bot.set_hand_joints("right", DEX3_OPEN)
    self.hold(2.0)
    x = self.R[:, 0]
    pre = PRESHAPE_TOP if lying else PRESHAPE
    pick = np.array([*pick_xy, TABLE_TOP + (LYING_GRASP_H if lying else GRASP_H)])
    self.grasp_log = {}
    if self.block >= 0:
      self.place_block(pick_xy, yaw_pick, lying)
      self.P.ignore_bodies = set()  # on the way down the block is an obstacle
    drop = np.array([*drop_xy, TABLE_TOP + GRASP_H + RELEASE_DZ])
    start = pick + [0, 0, APPROACH]  # straight above the pickup
    if lying:  # fingers to either side of the block: the one with the most clearance
      cands = [hand_R_top(yaw_pick, fl) for fl in (False, True)]
      scores = []
      for R in cands:
        self.R = R
        scores.append(self.best_config(pick, start, pre)[0])
      self.R = cands[int(np.argmax(scores))]
      if max(scores) < 0:
        raise RuntimeError("lying block: no top-down grasp reachable")
    else:
      self.R = hand_R(TILT, yaw_pick)
    # 1) planned move to above the pickup, pre-shaping the hand on the way
    self.target = start
    path = self.plan_to(start, anchor=pick, hand=pre)
    self.run_q(path, self.P.timed(path, vmax=FREE_VMAX), pre, DEX3_OPEN.copy())
    self.servo(start, 0.8)
    self.note("above pickup", start)
    # 2) straight down to the grasp point (checked)
    self.target = pick
    self.track(start, pick, 1.5, check="approach")
    self.servo(pick, 0.5)
    self.note("pickup (grasp centre)", pick)
    # 3) grab: close past contact (squeeze), hold the hand still while the grip builds up
    self.servo(pick, 0.8, DEX3_SQUEEZE, pre)
    self.servo(pick, 0.4, DEX3_SQUEEZE, DEX3_SQUEEZE)
    if self.block >= 0:
      self.P.ignore_bodies = {self.block}  # held: part of the hand for the planner
      self.grasp_log["grip_N"] = self.grip_forces()
      self.grasp_log["upright_at_grab"] = self.block_state()[1]
      self.grasp_log["in_hand_at_grab"] = self.block_in_hand()
    # 4) lift straight up
    up = pick + [0, 0, LIFT]
    self.target = up
    self.track(pick, up, 1.0)
    self.servo(up, 0.3, DEX3_SQUEEZE, DEX3_SQUEEZE)
    self.note("lifted", up)
    if self.block >= 0:
      self.grasp_log["lift_slip_cm"] = float(np.linalg.norm(
        self.block_in_hand() - self.grasp_log["in_hand_at_grab"]) * 100)
      self.grasp_log["grip_N_lifted"] = self.grip_forces()
      self.grasp_log["block_z_lifted"] = float(self.block_state()[0][2] - TABLE_TOP)
      self.grasp_log["upright_lifted"] = self.block_state()[1]
    # in-hand calibration: where the block really sits relative to the grasp centre after the
    # grasp (the closing fingers drag it ~2 cm). Aim the drop so the BLOCK centre lands on the
    # star with its bottom RELEASE_DZ above the table (ground truth: the scripted demonstrator).
    if self.block >= 0:
      Rt, pt = self.T()
      tip = Rt @ self.bot.grip_poses()["right"][0] + pt
      off = self.block_state()[0] - tip  # world offset, block centre - grasp centre
      self.grasp_log["in_hand_offset_cm"] = (off * 100).round(1)
      drop = np.array([drop_xy[0] - off[0], drop_xy[1] - off[1],
                       TABLE_TOP + (LYING_H if lying else 0.04) + RELEASE_DZ - off[2]])
    # 5) planned carry to above the drop
    above = drop + [0, 0, DROP_HOVER]
    self.target = above
    # carry: a straight line with the hand orientation held fixed (closed loop) - a free joint-
    # space path turns the hand on the way and the block rotates in the grip. The planned free
    # path (with the drop yaw options) is only the fallback when the line is blocked.
    if not lying:
      self.R = hand_R(TILT, yaw_drop)
    self.yaw_drop_used = yaw_drop
    try:
      self.track(up, above, CARRY_SEC, check="carry")
    except RuntimeError:
      path = None
      R_keep = self.R
      for yd in (yaw_drop, *[v for v in DROP_YAWS if v != yaw_drop]):
        self.R = R_keep if lying else hand_R(TILT, yd)
        try:
          path = self.plan_to(above, anchor=drop, hand=DEX3_SQUEEZE)
          break
        except RuntimeError:
          path = None
      if path is None:
        raise RuntimeError(f"no arm configuration reaches the drop {drop.round(3)} at any yaw")
      self.yaw_drop_used = yd
      self.run_q(path, self.P.timed(path, vmax=FREE_VMAX))
    self.servo(above, 0.8, DEX3_SQUEEZE, DEX3_SQUEEZE)
    self.note("above drop", above)
    if self.block >= 0:
      self.grasp_log["carry_slip_cm"] = float(np.linalg.norm(
        self.block_in_hand() - self.grasp_log["in_hand_at_grab"]) * 100)
      self.grasp_log["held_above_drop"] = bool(self.grip_forces())
      self.grasp_log["upright_above_drop"] = self.block_state()[1]
    # 6) lower, release
    self.target = drop
    self.track(above, drop, 1.0, check="lowering")
    self.servo(drop, 0.5)
    self.note("drop point (grasp centre)", drop)
    self.servo(drop, 0.6, pre, DEX3_SQUEEZE)
    # 7) retreat straight up, then rest
    back = drop + [0, 0, APPROACH]
    self.target = back
    try:
      self.track(drop, back, 1.0)
    except RuntimeError:  # no continuous straight retreat from this arm configuration: plan it
      path = self.plan_to(back)
      self.run_q(path, self.P.timed(path, vmax=FREE_VMAX))
    self.P.sync(bot.data)
    self.P.clear = self.P.finger_clear = FREE_CLEAR
    rest = self.P.plan(bot.data.qpos[self.P.qadr].copy(), _rest())
    self.P.clear, self.P.finger_clear = LINE_CLEAR, LINE_HAND_CLEAR
    if rest is not None:
      self.run_q(rest, self.P.timed(rest, vmax=FREE_VMAX), DEX3_OPEN, pre)
    self.hold(1.0)
    if self.block >= 0:
      pb, up_ = self.block_state()
      self.grasp_log["place_err_cm"] = float(np.linalg.norm(pb[:2] - np.array(drop_xy)) * 100)
      self.grasp_log["upright"] = up_
      self.grasp_log["on_table"] = bool(abs(pb[2] - (TABLE_TOP + (LYING_H if lying else 0.04))) < 0.01)
      self.grasp_log["lying_flat"] = bool(abs(up_) < 0.15)
      self.P.ignore_bodies = set()
    return self.worst


def _ik_seeds():
  from arm_ik import ArmChain
  ch = ArmChain("right")
  rng = np.random.default_rng(0)
  return [rng.uniform(ch.lo, ch.hi) for _ in range(10)]  # fixed, so demos are repeatable


IK_SEEDS = _ik_seeds()


def _rest():
  from stance_runner import ready_q
  return ready_q("right")


def check():
  mo = Motion()
  print("pickup (0.30, -0.22); drop candidates on the centre line (y = 0)")
  for x in (0.34, 0.36, 0.38):
    for w in (0.0, 0.15, 0.3):
      _try(mo, (0.30, -0.22), (x, 0.0), w)


def _try(mo, pick, drop, w=0.0):
  try:
    worst = mo.pick_place(pick, drop, w)
    errs = dict(mo.log)
    ok = (max(errs.values()) < 1.5 and not worst["fall"] and worst["table"] == 0)
    print(f"pick {pick} drop {drop} waist {w}: pick err {errs['pickup (grasp centre)']:.1f} cm, "
          f"drop err {errs['drop point (grasp centre)']:.1f} cm, table {worst['table']}, "
          f"fall {worst['fall']}, drift {worst['drift'] * 100:.1f} cm  {'OK' if ok else '--'}",
          flush=True)
  except RuntimeError as e:
    print(f"pick {pick} drop {drop} waist {w}: {e}", flush=True)


def view(pick, drop, w):
  import mujoco.viewer
  mo = Motion()
  with mujoco.viewer.launch_passive(mo.bot.model, mo.bot.data) as v:
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.2, 145, -25
    v.cam.lookat[:] = [0.3, -0.1, 0.8]

    def hook(m_):
      if not v.is_running():
        raise SystemExit
      with v.lock():
        v.user_scn.ngeom = 1
        mujoco.mjv_initGeom(v.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                            m_.target, np.eye(3).flatten(), [0.1, 0.9, 0.2, 0.7])
      v.sync()
      time.sleep(CONTROL_DT * 0.7)
    mo.hook = hook
    while v.is_running():
      print(f"pick {pick} -> drop {drop}, waist {w}", flush=True)
      try:
        worst = mo.pick_place(pick, drop, w)
        print("  done:", {k: (round(val, 3) if isinstance(val, float) else val)
                          for k, val in worst.items()}, flush=True)
      except RuntimeError as e:
        print("  stopped:", e, flush=True)
        mo.hold(2.0)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", action="store_true")
  ap.add_argument("--pick", default=",".join(map(str, PICK_XY)))
  ap.add_argument("--drop", default=",".join(map(str, DROP_XY)))
  ap.add_argument("--waist", type=float, default=TASK_WAIST_YAW)
  a = ap.parse_args()
  if a.check:
    check()
  else:
    view(tuple(map(float, a.pick.split(","))), tuple(map(float, a.drop.split(","))), a.waist)
