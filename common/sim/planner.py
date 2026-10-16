"""Collision-aware arm planner on the ground-truth MuJoCo scene.

The planner keeps its own MjData copy of the live sim (robot body, legs, waist, the other arm,
the objects), moves only the planned arm's 7 joints in it and measures the exact distance
(mj_geomDistance) between every arm/hand collision geom and every other collision geom: the
table, the objects, the robot's own torso, pelvis, legs and other arm. No approximated obstacles.

  plan = ArmPlanner(model, "right")
  plan.sync(data)                                   # snapshot of the live sim
  path = plan.plan(q_now, q_goal)                   # list of 7-vectors, collision free
  traj = plan.timed(path)                           # (t, q) for execution

Joint-space segments are checked every STEP rad; if the straight segment is blocked, RRT-Connect
in the joint limits, then shortcut smoothing. `ignore` lists bodies the hand may touch (the
object being grasped); `finger_clear` is the smaller clearance allowed for the fingers near the
table at the grasp.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
              "wrist_pitch", "wrist_yaw")
STEP = 0.04  # rad between collision checks along a segment


def _subtree(m, root):
  out, kids = {root}, True
  while kids:
    kids = False
    for b in range(m.nbody):
      if b not in out and m.body_parentid[b] in out:
        out.add(b)
        kids = True
  return out


class ArmPlanner:
  def __init__(self, model, side="right", clear=0.02, finger_clear=0.01, seed=0):
    m = self.m = model
    self.d = mujoco.MjData(m)
    self.side = side
    self.clear, self.finger_clear = clear, finger_clear
    self.rng = np.random.default_rng(seed)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}_joint") for j in ARM_JOINTS]
    self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
    self.lo, self.hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")
    arm = _subtree(m, root)
    collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    plane = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    self.arm_geoms = np.array([g for g in range(m.ngeom)
                               if collides[g] and m.geom_bodyid[g] in arm])
    self.obs_geoms = np.array([g for g in range(m.ngeom)
                               if collides[g] and not plane[g] and m.geom_bodyid[g] not in arm])
    hand_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    hand = _subtree(m, hand_root)  # palm (wrist yaw link) + fingers: the hand clearance
    self.is_finger = np.array([m.geom_bodyid[g] in hand for g in self.arm_geoms])
    # the upper arm sits against the shoulder by design: those pairs only must not penetrate
    shoulder = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_yaw_link")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    self.zero_pairs = {(g1, g2) for g1 in self.arm_geoms for g2 in self.obs_geoms
                       if m.geom_bodyid[g1] == shoulder and m.geom_bodyid[g2] == torso}
    self.ignore_bodies: set[int] = set()
    self.fromto = np.zeros(6)

  # --- scene -------------------------------------------------------------------------------
  def sync(self, data):
    """Snapshot of the live sim (everything except the planned arm stays fixed while planning)."""
    self.d.qpos[:] = data.qpos
    self.d.mocap_pos[:], self.d.mocap_quat[:] = data.mocap_pos, data.mocap_quat
    mujoco.mj_kinematics(self.m, self.d)

  def body_ids(self, names):
    return {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}

  # --- collision ---------------------------------------------------------------------------
  def distance(self, q, ignore=None):
    """(smallest clearance margin (m, < 0 = too close), arm geom, obstacle geom) for arm angles q."""
    m, d = self.m, self.d
    d.qpos[self.qadr] = q
    mujoco.mj_kinematics(m, d)
    ign = self.ignore_bodies if ignore is None else ignore
    pa, po = d.geom_xpos[self.arm_geoms], d.geom_xpos[self.obs_geoms]
    ra, ro = m.geom_rbound[self.arm_geoms], m.geom_rbound[self.obs_geoms]
    need = np.where(self.is_finger, self.finger_clear, self.clear)
    gap = (np.linalg.norm(pa[:, None] - po[None], axis=2) - ra[:, None] - ro[None]
           - need[:, None])
    worst = (np.inf, -1, -1)
    for i, j in zip(*np.nonzero(gap < 0.0)):
      g1, g2 = int(self.arm_geoms[i]), int(self.obs_geoms[j])
      if m.geom_bodyid[g2] in ign:
        continue
      want = 0.0 if (g1, g2) in self.zero_pairs else need[i]
      dist = mujoco.mj_geomDistance(m, d, g1, g2, want + 0.05, self.fromto)
      if dist - want < worst[0]:
        worst = (dist - want, g1, g2)
    return worst

  def free(self, q, ignore=None):
    return self.distance(q, ignore)[0] >= 0.0

  def who(self, q, ignore=None):
    """Readable name of the closest violating pair (for reports)."""
    dist, g1, g2 = self.distance(q, ignore)
    if g1 < 0:
      return "clear"
    b = lambda g: self.m.body(self.m.geom_bodyid[g]).name  # noqa: E731
    return f"{b(g1)} vs {b(g2)} ({dist * 100:+.1f} cm)"

  def segment_free(self, qa, qb, ignore=None):
    n = max(1, int(np.ceil(np.abs(qb - qa).max() / STEP)))
    return all(self.free(qa + (k / n) * (qb - qa), ignore) for k in range(1, n + 1))

  # --- planning ----------------------------------------------------------------------------
  def plan(self, qa, qb, ignore=None, iters=3000, step=0.2):
    """Collision-free joint path qa -> qb (list of waypoints), or None."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    if not self.free(qb, ignore):
      return None
    if self.segment_free(qa, qb, ignore):
      return [qa, qb]
    trees = [[qa], [qb]]
    parents = [[-1], [-1]]
    for it in range(iters):
      a, b = (0, 1) if it % 2 == 0 else (1, 0)
      sample = self.rng.uniform(self.lo, self.hi)
      new = self._extend(trees[a], parents[a], sample, step, ignore)
      if new is None:
        continue
      # connect the other tree greedily towards the new node
      while True:
        got = self._extend(trees[b], parents[b], trees[a][new], step, ignore)
        if got is None:
          break
        if np.abs(trees[b][got] - trees[a][new]).max() < 1e-9:
          pa_, pb_ = self._branch(trees[a], parents[a], new), self._branch(trees[b], parents[b], got)
          path = pa_[::-1] + pb_[1:] if a == 0 else pb_[::-1] + pa_[1:]
          return self.shortcut(path, ignore)
    return None

  def _extend(self, tree, parent, target, step, ignore):
    near = int(np.argmin([np.abs(q - target).max() for q in tree]))
    q0 = tree[near]
    delta = target - q0
    dmax = np.abs(delta).max()
    q1 = target if dmax <= step else q0 + delta * (step / dmax)
    if not self.segment_free(q0, q1, ignore):
      return None
    tree.append(q1)
    parent.append(near)
    return len(tree) - 1

  @staticmethod
  def _branch(tree, parent, i):
    out = []
    while i >= 0:
      out.append(tree[i])
      i = parent[i]
    return out

  def shortcut(self, path, ignore=None, tries=150):
    path = list(path)
    for _ in range(tries):
      if len(path) <= 2:
        break
      i, j = sorted(self.rng.choice(len(path), 2, replace=False))
      if j - i > 1 and self.segment_free(path[i], path[j], ignore):
        path = path[:i + 1] + path[j:]
    return path

  def cartesian_free(self, qs, ignore=None):
    """Check a precomputed Cartesian (IK) joint sequence, e.g. a straight descent."""
    return all(self.segment_free(a, b, ignore) for a, b in zip(qs[:-1], qs[1:]))

  @staticmethod
  def timed(path, vmax=0.8, tmin=0.4):
    """Waypoints -> durations (s): each segment at <= vmax rad/s on its fastest joint, smoothstep
    inside (rest at each waypoint)."""
    return [max(tmin, np.abs(b - a).max() / vmax) for a, b in zip(path[:-1], path[1:])]

  @staticmethod
  def sample(path, durations, t):
    """Joint target at time t along a timed path."""
    for a, b, T in zip(path[:-1], path[1:], durations):
      if t <= T:
        s = t / T
        return a + (3 * s * s - 2 * s ** 3) * (b - a)
      t -= T
    return path[-1]
