"""G1 tabletop stance: the legs-only RL policy keeps legs + waist (15 joints) still while
YOUR arm controller (VLA / IK / teleop) drives the 14 arm joints.

  from stance_runner import StanceRobot
  bot = StanceRobot()                         # loads scene/g1_tabletop.xml + policy/stance.onnx
  bot.reset()
  while ...:
      bot.set_arm_joint_targets(q14)          # 14 arm angles (rad), left 7 then right 7
      # or bot.set_hand_targets(p_left, p_right)  (torso frame, metres; IK inside)
      bot.step()                              # 50 Hz control step (4 x 200 Hz physics)
      img = bot.render("head")                # head D435 view (or "front", "side")

Exactly the deploy contract the policy was trained with (mjlab task
Mjlab-Velocity-HumanV2-Stance-Unitree-G1, 752-dim observation): arm targets are low-passed
(tau 0.25 s) like the training arm controller and fed to the policy as 'arm_command'; the
policy is told whether the hands carry a payload and how heavy (set_payload).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before the first GL context

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from arm_ik import ARM_JOINTS, ArmChain, rest_q  # noqa: E402
from legs_runtime import LegsCommands  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "scene/g1_tabletop.xml"
POLICY = ROOT / "policy/stance.onnx"
PHYSICS_DT, DECIMATION = 0.005, 4
CONTROL_DT = PHYSICS_DT * DECIMATION
NATURAL_FREQ = 10 * 2.0 * math.pi
ARM_NAMES = [f"{s}_{j}_joint" for s in ("left", "right") for j in ARM_JOINTS]
SIDES = ("left", "right")
# IK point on the Dex3 hand (wrist_yaw frame, right hand; y mirrored for the left): the grasp
# centre between the thumb tip and the index / middle tips in DEX3_GRAB (3.5 cm apart), measured
# from Unitree's meshes. The Dex3 palm faces +y of the right wrist frame (fingers curl that way).
GRIP_SITE = np.array([0.102, 0.085, 0.0])
DEX3_OPEN = np.zeros(7)  # thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
DEX3_GRAB = np.array([0.0, 0.49, -1.49, 1.05, 1.05, 1.05, 1.05])
GRIPPER_KG = 0.53  # Dex3 hand minus the mjlab wrist link the arm model (arm_ik) already has
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
OBJECTS = ("block", "mouse")


def _grasp_R(down=np.pi / 4, palm_in=False):
  """Pinch orientation (torso frame): fingers (hand x) point forward and `down` rad below
  horizontal, jaws (hand y) close along torso y. 45 deg is how the G1 hand hangs naturally;
  a straight top-down grasp drives the wrists to their limits."""
  x = np.array([np.cos(down), 0.0, -np.sin(down)])
  y = np.array([0.0, -1.0 if palm_in else 1.0, 0.0])  # palm normal is -y: palm_in faces +y
  return np.column_stack([x, y, np.cross(x, y)])


R_PICK = _grasp_R()


def ready_q(side):
  """Rest pose with the elbow bent more, so the fingers clear the table (also when the
  body leans while the other arm works)."""
  q = rest_q(side)
  q[ARM_JOINTS.index("elbow")] += 0.8
  return q


def _quat_to_rot(q):
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.asarray(q, float))
  return m.reshape(3, 3)


class StanceRobot:
  def __init__(self, scene: Path = SCENE, policy: Path = POLICY, arm_tau: float = 0.25,
               render_size=(480, 640), arm_gravity_ff: bool = True):
    """arm_gravity_ff: add MuJoCo's exact gravity torque (arm links + held payload) to the
    arm PD, so the soft arm gains (kp ~14 Nm/rad) do not sag ~10 cm under their own weight."""
    self.arm_gravity_ff = arm_gravity_ff
    self.model = mujoco.MjModel.from_xml_path(str(scene))
    self.data = mujoco.MjData(self.model)
    m = self.model
    so = ort.SessionOptions()  # one thread: a tiny MLP, and many sims run in parallel
    so.intra_op_num_threads = so.inter_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
    self.sess = ort.InferenceSession(str(policy), so, providers=["CPUExecutionProvider"])
    self.inp = self.sess.get_inputs()[0].name
    meta = self.sess.get_modelmeta().custom_metadata_map

    def floats(k):
      return np.array([float(v) for v in meta[k].split(",") if v])

    self.jnames = [n for n in meta["joint_names"].split(",") if n]
    self.kp, self.kd = floats("joint_stiffness"), floats("joint_damping")
    self.default = floats("default_joint_pos")
    self.scale = floats("action_scale")
    self.n_act = len(self.scale)
    self.terms = meta["observation_names"].split(",")
    self.hist = [int(float(h)) for h in meta["observation_terms_history_length"].split(",")]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.jnames]
    self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
    self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
    self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                           n.replace("_joint", "")) for n in self.jnames])
    self.effort = m.actuator_ctrlrange[self.act, 1].copy()
    m.dof_armature[self.vadr] = self.kp / NATURAL_FREQ**2  # mjlab: kp = armature * wn^2
    self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    self.palms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link")
                  for s in ("left", "right")]
    self.palm_mass0 = m.body_mass[self.palms].copy()
    self.table_geoms = {i for i in range(m.ngeom)
                        if m.body(m.geom_bodyid[i]).name == "table"}
    robot_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == robot_root}
    self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot")
                 for s in ("left", "right")]
    self.arms = {s: ArmChain(s) for s in ("left", "right")}
    self.tips = {s: ArmChain(s) for s in SIDES}  # same chain, IK point = between the fingers
    for t in self.tips.values():
      t.palm = GRIP_SITE.copy()
    for s_, t in self.tips.items():
      t.palm = GRIP_SITE * np.array([1.0, -1.0 if s_ == "left" else 1.0, 1.0])
    present = [o for o in OBJECTS if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) >= 0]
    self.obj_body = {o: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) for o in present}
    self.obj_jnt = {o: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                                        f"{o}_joint")] for o in present}
    # Dex3 hands: 7 position servos per hand (scene/build_scene.py add_dex3)
    self.hand_act = {s_: np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                     f"{s_}_hand_{j}") for j in HAND_JOINTS])
                     for s_ in SIDES}
    self.hand_qadr = {s_: np.array([m.jnt_qposadr[mujoco.mj_name2id(
      m, mujoco.mjtObj.mjOBJ_JOINT, f"{s_}_hand_{j}_joint")] for j in HAND_JOINTS])
      for s_ in SIDES}
    self.hand_lo = {s_: m.actuator_ctrlrange[self.hand_act[s_], 0].copy() for s_ in SIDES}
    self.hand_hi = {s_: m.actuator_ctrlrange[self.hand_act[s_], 1].copy() for s_ in SIDES}
    # IK seed: False = the ready pose every call (same target -> same angles), True = the last
    # goal (continuous; needed when the hand turns a lot, e.g. into the palm-in grasp)
    self.ik_seed_prev = False
    self.arm_tau = arm_tau
    self.legs = LegsCommands(CONTROL_DT)
    self._renderers = {}
    self.render_size = render_size
    self.reset()

  # ---------------------------------------------------------------- setup / commands
  def reset(self, arm_q=None):
    d, m = self.data, self.model
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.78]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    q = self.default.copy()
    q[15:] = np.concatenate([ready_q("left"), ready_q("right")]) if arm_q is None else arm_q
    d.qpos[self.qadr] = q
    self.arm_goal = q[15:].copy()
    self.arm_target = q[15:].copy()
    self.q_target = q.copy()
    self.last_action = np.zeros(self.n_act, dtype=np.float32)
    self.legs.reset()
    self.carry_kg, self.carrying, self.mass_known = 0.0, False, True
    self.hand_goal = {s: np.clip(np.zeros(7), self.hand_lo[s], self.hand_hi[s]) for s in SIDES}
    for s in SIDES:  # hands start relaxed (all joints 0 = open, fingers straight)
      d.qpos[self.hand_qadr[s]] = self.hand_goal[s]
    self.auto_payload = False  # objects are held by contact only; set_payload is manual
    self._hist = {}
    self.t = 0.0
    mujoco.mj_forward(m, d)
    self.home_xy = d.qpos[0:2].copy()
    self.home_feet = d.site_xpos[self.feet].copy()

  def set_arm_joint_targets(self, q14):
    """14 arm angles (rad): left shoulder_pitch, roll, yaw, elbow, wrist roll, pitch, yaw, then
    right. Low-passed (tau) before the PD, exactly as in training."""
    lo = np.concatenate([self.arms["left"].lo, self.arms["right"].lo])
    hi = np.concatenate([self.arms["left"].hi, self.arms["right"].hi])
    self.arm_goal = np.clip(np.asarray(q14, float), lo, hi)

  def set_hand_targets(self, p_left=None, p_right=None, R_left=None, R_right=None):
    """Hand (heel-of-hand grip point) targets in the TORSO frame; IK -> arm joint goals.
    Orientation optional (keeps the current one)."""
    goal = self.arm_goal.copy()
    for k, (s, p, R) in enumerate((("left", p_left, R_left), ("right", p_right, R_right))):
      if p is None:
        continue
      arm, q0 = self.arms[s], goal[7 * k: 7 * k + 7]
      if R is None:  # position-only IK; the null space pulls toward the natural rest pose
        goal[7 * k: 7 * k + 7] = arm.ik(np.asarray(p), np.eye(3), q0, w_rot=0.0,
                                        q_rest=rest_q(s), null=0.05)
      else:
        goal[7 * k: 7 * k + 7] = arm.ik(np.asarray(p), R, q0, w_rot=0.5, q_rest=rest_q(s))
    self.set_arm_joint_targets(goal)

  def set_grip_targets(self, p_left=None, p_right=None, R_left=None, R_right=None,
                       w_rot=0.3, iters=100):
    """Fingertip-centre targets (torso frame, metres) + optional orientation (soft, w_rot);
    use R_TOPDOWN for a top-down pinch. IK -> arm joint goals."""
    goal = self.arm_goal.copy()
    for k, (s, p, R) in enumerate((("left", p_left, R_left), ("right", p_right, R_right))):
      if p is None:
        continue
      # seeded from the ready pose by default (same target -> same arm angles: consistent demos)
      # or the last goal (ik_seed_prev); IK local minima: strong mode tries both seeds
      seeds = [goal[7 * k: 7 * k + 7] if self.ik_seed_prev else ready_q(s)]
      if iters >= 300:  # strong mode: also try the other seed, keep the more accurate one
        seeds.append(ready_q(s) if self.ik_seed_prev else goal[7 * k: 7 * k + 7])
      best, best_err = None, np.inf
      for seed in seeds:
        q = self.tips[s].ik(
          np.asarray(p), np.eye(3) if R is None else R, seed, iters=iters,
          w_rot=0.0 if R is None else w_rot, q_rest=ready_q(s), null=0.05)
        pp, RR, *_ = self.tips[s].fk(q)
        err = np.linalg.norm(pp - p) + (0.0 if R is None else
                                        0.02 * np.linalg.norm(R - RR))
        if err < best_err:
          best, best_err = q, err
      goal[7 * k: 7 * k + 7] = best
    self.set_arm_joint_targets(goal)

  def set_hand_joints(self, side, q7):
    """Dex3 joint targets (rad): thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0,
    middle_1, clipped to the joint ranges; position servos with Unitree's torque limits."""
    self.hand_goal[side] = np.clip(np.asarray(q7, float), self.hand_lo[side], self.hand_hi[side])

  def grip_poses(self):
    """Current fingertip-centre (p, R) in the torso frame, from the measured arm angles."""
    q = self.data.qpos[self.qadr[15:]]
    return {s: self.tips[s].fk(q[7 * k: 7 * k + 7])[:2] for k, s in enumerate(SIDES)}

  def object_pose(self, name, frame="torso"):
    """Object position (and rotation) in the torso frame (or 'world')."""
    d = self.data
    p, R = d.xpos[self.obj_body[name]].copy(), d.xmat[self.obj_body[name]].reshape(3, 3).copy()
    if frame == "torso":
      Rt, pt = d.xmat[self.torso].reshape(3, 3), d.xpos[self.torso]
      return Rt.T @ (p - pt), Rt.T @ R
    return p, R

  def hand_poses(self):
    """Current hand (p, R) in the torso frame, from the measured arm angles."""
    q = self.data.qpos[self.qadr[15:]]
    return {s: self.arms[s].fk(q[7 * k: 7 * k + 7])[:2] for k, s in enumerate(("left", "right"))}

  def set_posture(self, height=0.78, pitch=0.0, waist_yaw=0.0):
    """Pelvis height 0.68-0.80 m, torso lean -0.05..0.35 rad, waist yaw +-0.5 (trained range)."""
    self.legs.set_posture_goal(height, pitch, waist_yaw)

  def set_payload(self, kg: float, known: bool = True):
    """Object in hand(s): adds the mass at the hands (as trained) + tells the policy."""
    self.carry_kg, self.carrying, self.mass_known = kg, kg > 0, known
    if not self.auto_payload:  # no real object in the grippers: add the mass at the hands
      self.model.body_mass[self.palms] = self.palm_mass0 + kg / 2.0

  # ---------------------------------------------------------------- policy
  def _term(self, name):
    d = self.data
    R = _quat_to_rot(d.qpos[3:7])
    if name == "base_ang_vel":
      return d.qvel[3:6]  # free joint: body-frame angular velocity
    if name == "projected_gravity":
      return R.T @ np.array([0.0, 0.0, -1.0])
    if name == "joint_pos":
      return d.qpos[self.qadr] - self.default
    if name == "joint_vel":
      return d.qvel[self.vadr]
    if name == "actions":
      return self.last_action
    if name == "command":
      return self.legs.twist
    if name == "gait_mode":
      return self.legs.gait_mode_obs()
    if name == "carry_flag":
      return np.array([1.0 if self.carrying else 0.0])
    if name == "arm_command":
      return self.arm_target - self.default[15:]
    if name == "posture":
      return self.legs.posture_obs()
    if name == "pelvis_height":
      return np.array([d.qpos[2] - d.site_xpos[self.feet, 2].min()])
    if name == "torso_projected_gravity":
      return d.xmat[self.torso].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    if name == "posture_goal":
      return self.legs.posture_goal_obs()
    if name == "w2s_progress":
      return self.legs.w2s_obs()
    if name == "mass_estimate":
      if not self.mass_known:
        return np.zeros(2)
      return np.array([self.carry_kg / 10.0 if self.carrying else 0.0, 1.0])
    raise KeyError(name)

  def _obs(self):
    parts = []
    for name, h in zip(self.terms, self.hist, strict=True):
      x = np.array(self._term(name), dtype=np.float32)
      if h > 0:
        buf = self._hist.setdefault(name, [])
        if not buf:
          buf.extend([x] * h)
        else:
          buf.append(x)
          del buf[0]
        parts.extend(buf)
      else:
        parts.append(x)
    return np.concatenate(parts)[None]

  def step(self):
    """One 50 Hz control step."""
    self.legs.step(np.zeros(3), carrying=self.carrying)  # stance: twist always zero
    a = CONTROL_DT / (self.arm_tau + CONTROL_DT)
    self.arm_target = self.arm_target + a * (self.arm_goal - self.arm_target)
    act = self.sess.run(None, {self.inp: self._obs()})[0][0]
    self.last_action = act.astype(np.float32)
    self.q_target[:15] = self.default[:15] + self.scale * self.last_action
    self.q_target[15:] = self.arm_target
    d = self.data
    for s in SIDES:
      d.ctrl[self.hand_act[s]] = self.hand_goal[s]
    for _ in range(DECIMATION):
      q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
      tau = self.kp * (self.q_target - q) - self.kd * qd
      if self.arm_gravity_ff:
        tau[15:] += self._arm_gravity(q[15:])
      tau = np.clip(tau, -self.effort, self.effort)
      d.ctrl[self.act] = tau
      mujoco.mj_step(self.model, d)
    self.t += CONTROL_DT

  def _arm_gravity(self, q_arm):
    """Gravity torque of each arm (links + half the held payload at the hand), torso frame."""
    g_t = self.data.xmat[self.torso].reshape(3, 3).T @ np.array([0.0, 0.0, -9.81])
    out = []
    for k, s in enumerate(SIDES):
      kg = GRIPPER_KG + 0.5 * self.carry_kg
      out.append(self.tips[s].gravity_torque(q_arm[7 * k: 7 * k + 7], g_t, kg * g_t))
    return np.concatenate(out)

  # ---------------------------------------------------------------- monitoring
  def stability(self):
    """Drift / slip / sway numbers for the stance (what the VLA should not disturb)."""
    d = self.data
    return {
      "base_drift_m": float(np.linalg.norm(d.qpos[0:2] - self.home_xy)),
      "foot_slip_m": float(np.linalg.norm(d.site_xpos[self.feet, :2] - self.home_feet[:, :2],
                                          axis=1).max()),
      "pelvis_angvel": float(np.linalg.norm(d.qvel[3:6])),
      "torso_tilt_deg": float(np.degrees(np.arccos(np.clip(d.xmat[self.torso][8], -1, 1)))),
      "pelvis_z": float(d.qpos[2]),
      "fallen": bool(_quat_to_rot(d.qpos[3:7])[2, 2] < 0.6),
      "robot_table_contacts": int(sum(
        1 for c in d.contact[:d.ncon]
        if (c.geom1 in self.table_geoms and self.model.geom_bodyid[c.geom2] in self.robot_bodies)
        or (c.geom2 in self.table_geoms and self.model.geom_bodyid[c.geom1] in self.robot_bodies))),
    }

  def render(self, camera="head"):
    if camera not in self._renderers:
      h, w = self.render_size
      self._renderers[camera] = mujoco.Renderer(self.model, h, w)
    r = self._renderers[camera]
    r.update_scene(self.data, camera=camera)
    return r.render()

