"""Two-arm box-holding controller for the legs-only policy (numpy, no MuJoCo needed).

Kinematics come straight from the mjlab G1 MJCF (g1.xml): the chain torso_link ->
shoulder pitch/roll/yaw -> elbow -> wrist roll/pitch/yaw, plus the palm site. Everything
is expressed in the TORSO frame, so the waist / legs do not matter.

  FK          palm-site pose (p, R) and every joint's position / axis, from the 7 angles
  Jacobian    6x7 geometric Jacobian of the palm site
  IK          damped least squares, 6-D (position + orientation), joint-limit clamped
  gravity     tau = sum_k J_com_k^T (m_k g) + J_palm^T (F_box): the torque each arm joint
              must supply to hold the arm links and its share of the box

Controller (each 50 Hz step, per arm):
  q_des   = IK(palm target in torso frame)            -> what the policy sees (arm_command)
  q_cmd   = q_des + tau_ff / kp + integral(q_des - q) -> PD position target sent to PhysX
The arm PD gains are soft (kp ~14 Nm/rad), so without the feed-forward + integral terms a
10 kg box would sag the arms by ~0.5 rad.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MJCF = Path(__file__).resolve().parents[1] / "robot/g1/g1.xml"
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
# Actuator torque limits (g1_constants.py): 5020 = 25 Nm, 4010 = 5 Nm.
EFFORT = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])
# Where the box is held: the heel of the hand (wrist_yaw_link origin, 4.6 cm past the wrist
# pitch axis). At the palm-tip site the 10 kg box would need 5.7 Nm at the 5 Nm wrist-pitch
# motor, and at the hand CoM (where training put the palm point mass) 5.3 Nm.
GRIP = np.array([0.0, 0.0, 0.0])
# Safe carry pose (training: SAFE_ARMS / CARRY_SAFE), left arm; right mirrors roll / yaw.
CARRY_SAFE = {"shoulder_pitch": -0.55, "shoulder_roll": 0.40, "elbow": 0.95}


def _quat_to_rot(q):
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _axis_angle(axis, a):
  axis = np.asarray(axis, float) / np.linalg.norm(axis)
  K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
  return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _rot_err(R_des, R):
  """Rotation-vector error (world/torso frame) taking R to R_des."""
  E = R_des @ R.T
  c = np.clip((np.trace(E) - 1) / 2, -1.0, 1.0)
  ang = math.acos(c)
  v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
  if ang < 1e-6:
    return 0.5 * v
  return ang / (2 * math.sin(ang)) * v


@dataclass
class Link:
  name: str
  pos: np.ndarray
  rot: np.ndarray
  axis: np.ndarray
  lo: float
  hi: float
  mass: float
  com: np.ndarray


class ArmChain:
  def __init__(self, side: str, xml_path: Path = MJCF):
    root = ET.parse(xml_path).getroot()
    body = next(b for b in root.iter("body") if b.get("name") == "torso_link")
    self.side, self.links = side, []
    for j in ARM_JOINTS:
      body = next(b for b in body.findall("body") if b.get("name") == f"{side}_{j}_link")
      jt = body.find("joint")
      ine = body.find("inertial")
      lo, hi = (float(v) for v in jt.get("range").split())
      self.links.append(Link(
        name=jt.get("name"),
        pos=np.array([float(v) for v in (body.get("pos") or "0 0 0").split()]),
        rot=_quat_to_rot([float(v) for v in body.get("quat").split()]) if body.get("quat")
        else np.eye(3),
        axis=np.array([float(v) for v in jt.get("axis").split()]),
        lo=lo, hi=hi,
        mass=float(ine.get("mass")),
        com=np.array([float(v) for v in ine.get("pos").split()])))
    self.palm = GRIP * np.array([1.0, 1.0 if side == "left" else -1.0, 1.0])
    self.lo = np.array([lk.lo for lk in self.links])
    self.hi = np.array([lk.hi for lk in self.links])

  def fk(self, q):
    """Torso-frame: grip point (p, R) and lists of joint positions, world axes, link CoMs."""
    p, R = np.zeros(3), np.eye(3)
    jp, ja, coms, masses = [], [], [], []
    for lk, qi in zip(self.links, q, strict=True):
      p = p + R @ lk.pos
      R = R @ lk.rot
      jp.append(p.copy())
      ja.append(R @ lk.axis)
      R = R @ _axis_angle(lk.axis, qi)
      coms.append(p + R @ lk.com)
      masses.append(lk.mass)
    return p + R @ self.palm, R, jp, ja, coms, masses

  def jacobian(self, q, point=None, upto=None):
    pp, R, jp, ja, _, _ = self.fk(q)
    point = pp if point is None else point
    n = len(q) if upto is None else upto + 1
    J = np.zeros((6, len(q)))
    for i in range(n):
      J[:3, i] = np.cross(ja[i], point - jp[i])
      J[3:, i] = ja[i]
    return J

  def ik(self, p_des, R_des, q0, iters=60, damp=0.05, w_rot=0.5, q_rest=None, null=0.02):
    q = np.array(q0, float)
    q_rest = q if q_rest is None else np.asarray(q_rest)
    for _ in range(iters):
      p, R, *_ = self.fk(q)
      e = np.concatenate([p_des - p, w_rot * _rot_err(R_des, R)])
      if np.linalg.norm(e) < 1e-5:
        break
      J = self.jacobian(q)
      J[3:] *= w_rot
      JJt = J @ J.T + damp**2 * np.eye(6)
      dq = J.T @ np.linalg.solve(JJt, e)
      N = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)
      q = np.clip(q + dq + null * N @ (q_rest - q), self.lo, self.hi)
    return q

  def gravity_torque(self, q, g_torso, palm_force=np.zeros(3)):
    """Joint torques needed to hold the arm (and a force at the palm) against gravity."""
    pp, _, jp, ja, coms, masses = self.fk(q)
    tau = np.zeros(len(q))
    for i in range(len(q)):
      t = 0.0
      for k in range(i, len(q)):
        t += np.dot(ja[i], np.cross(coms[k] - jp[i], masses[k] * g_torso))
      t += np.dot(ja[i], np.cross(pp - jp[i], palm_force))
      tau[i] = -t  # actuator torque that cancels gravity's torque
    return tau


def rest_q(side: str, roll_out: float = 0.30) -> np.ndarray:
  """Training 'default' arm pose (KNEES_BENT keyframe) + SAFE_ARMS outward roll."""
  sg = 1.0 if side == "left" else -1.0
  q = np.zeros(7)
  q[ARM_JOINTS.index("shoulder_pitch")] = 0.2
  q[ARM_JOINTS.index("shoulder_roll")] = sg * (0.2 + roll_out)
  q[ARM_JOINTS.index("elbow")] = 0.6
  return q


def carry_q(side: str) -> np.ndarray:
  sg = 1.0 if side == "left" else -1.0
  q = np.zeros(7)
  for j, v in CARRY_SAFE.items():
    q[ARM_JOINTS.index(j)] = v * (sg if ("roll" in j or "yaw" in j) else 1.0)
  return q


class BoxHoldController:
  """Holds both palms at the safe-carry grip (torso frame) with gravity feed-forward."""

  def __init__(self, kp: np.ndarray, box_kg: float, ki: float = 2.0, z_max: float = 0.6,
               compensate: bool = True, pose: str = "carry"):
    self.compensate = compensate
    self.arms = {s: ArmChain(s) for s in ("left", "right")}
    self.kp = {"left": kp[:7], "right": kp[7:]}
    self.box_kg, self.ki, self.z_max = box_kg, ki, z_max
    self.q_des, self.target, self.z, self.tau_ff = {}, {}, {}, {}
    for s, arm in self.arms.items():
      q = carry_q(s) if pose == "carry" else rest_q(s)
      p, R, *_ = arm.fk(q)
      self.target[s] = (p, R)
      self.q_des[s] = arm.ik(p, R, q)
      self.z[s] = np.zeros(7)
      self.tau_ff[s] = np.zeros(7)
    pl, pr = self.target["left"][0], self.target["right"][0]
    self.palm_gap = float(np.linalg.norm(pl - pr))

  def step(self, q_meas: np.ndarray, R_torso_w: np.ndarray, dt: float):
    """q_meas: 14 measured arm angles (left 7, right 7). Returns (q_des 14, q_cmd 14)."""
    g_t = R_torso_w.T @ np.array([0.0, 0.0, -9.81])
    out_des, out_cmd = [], []
    for k, s in enumerate(("left", "right")):
      arm, qd = self.arms[s], self.q_des[s]
      qm = q_meas[7 * k: 7 * k + 7]
      tau = arm.gravity_torque(qm, g_t, palm_force=0.5 * self.box_kg * g_t)
      self.tau_ff[s] = tau
      self.z[s] = np.clip(self.z[s] + self.ki * dt * (qd - qm), -self.z_max, self.z_max)
      qc = (np.clip(qd + tau / self.kp[s] + self.z[s], arm.lo, arm.hi) if self.compensate
            else qd)
      out_des.append(qd)
      out_cmd.append(qc)
    return np.concatenate(out_des), np.concatenate(out_cmd)

  def palm_error(self, q_meas):
    errs = []
    for k, s in enumerate(("left", "right")):
      p, _, *_ = self.arms[s].fk(q_meas[7 * k: 7 * k + 7])
      errs.append(float(np.linalg.norm(p - self.target[s][0])))
    return errs
