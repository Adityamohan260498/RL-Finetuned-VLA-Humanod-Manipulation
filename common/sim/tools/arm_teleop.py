"""Manual right-arm control while the legs policy keeps the G1 standing (MuJoCo + slider panel).

  .venv_perc/bin/python sim/tools/arm_teleop.py [--policy policy/stance_palmin.onnx]

Slider panel (Tk window):
  * the 7 right-arm joints (shoulder pitch / roll / yaw, elbow, wrist roll / pitch / yaw), within
    their joint limits; the arm PD + gravity feed-forward track them (0.25 s smoothing),
  * the 7 right Dex3-1 hand joints: thumb 0 (rotation) / 1 / 2, index 0 / 1, middle 0 / 1,
  * the legs-policy posture commands: waist yaw, torso lean, pelvis height (trained ranges).
Live readout: pinch-point position (world), closest clearance to the table / objects / body
(ground-truth geometry, planner), torso yaw, base drift, falls, robot-table contacts.
"Save pose" appends the full configuration to data/poses/manual_poses.json.
The left arm holds its rest pose. No grasp controller, no lock: the fingers are what you set
(position servos with Unitree's torque limits; objects are held by contact only).
Viewer: Ctrl + right-drag pushes the selected body (e.g. move the block), C contacts, T transparent.
"""
import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stance"))
sys.path.insert(0, str(ROOT / "sim"))
from planner import ARM_JOINTS, ArmPlanner  # noqa: E402
from stance_runner import CONTROL_DT, HAND_JOINTS, StanceRobot, ready_q  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=str(ROOT / "policy/stance_palmin.onnx"))
ap.add_argument("--out", default=str(ROOT / "data/poses/manual_poses.json"))
a = ap.parse_args()

bot = StanceRobot(policy=Path(a.policy))
bot.reset()
m, d = bot.model, bot.data
P = ArmPlanner(m, "right", clear=0.0, finger_clear=0.0)  # report raw distances
HAND = [(j.replace("_", " "), j) for j in HAND_JOINTS]
bq = m.joint("block_joint").qposadr[0]
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
waist_adr = [m.joint(j).qposadr[0] for j in WAIST]
LOG = Path(a.out).with_name("joint_log.csv")  # measured right arm + torso joints every 0.1 s

# ---- panel --------------------------------------------------------------------------------
root = tk.Tk()
root.title("G1 right arm")
sl = {}


def slider(parent, label, lo, hi, val, row):
  tk.Label(parent, text=label, anchor="w", width=18).grid(row=row, column=0, sticky="w")
  s = tk.Scale(parent, from_=lo, to=hi, resolution=0.01, orient="horizontal", length=320)
  s.set(val)
  s.grid(row=row, column=1)
  return s


arm = tk.LabelFrame(root, text="right arm joints (rad)")
arm.grid(row=0, column=0, padx=6, pady=4, sticky="we")
q_rest = ready_q("right")
for i, j in enumerate(ARM_JOINTS):
  lo, hi = P.lo[i], P.hi[i]
  sl[j] = slider(arm, j.replace("_", " "), round(lo, 2), round(hi, 2), round(q_rest[i], 2), i)
hand = tk.LabelFrame(root, text="right Dex3-1 hand joints (rad)")
hand.grid(row=1, column=0, padx=6, pady=4, sticky="we")
for i, (lab, _) in enumerate(HAND):
  lo, hi = bot.hand_lo["right"][i], bot.hand_hi["right"][i]
  sl[lab] = slider(hand, lab, round(lo, 2), round(hi, 2), 0.0, i)
post = tk.LabelFrame(root, text="legs policy posture (trained ranges)")
post.grid(row=2, column=0, padx=6, pady=4, sticky="we")
sl["waist yaw"] = slider(post, "waist yaw", -0.5, 0.5, 0.0, 0)
sl["lean"] = slider(post, "torso lean", -0.05, 0.35, 0.0, 1)
sl["height"] = slider(post, "pelvis height", 0.68, 0.80, 0.78, 2)
info = tk.Label(root, text="", justify="left", font=("Courier", 10), anchor="w")
info.grid(row=3, column=0, padx=6, sticky="we")
bar = tk.Frame(root)
bar.grid(row=4, column=0, pady=4)
tk.Label(bar, text="name").pack(side="left")
name = tk.Entry(bar, width=18)
name.pack(side="left")
state = {"reset": False, "save": False, "msg": ""}
tk.Button(bar, text="Save pose", command=lambda: state.update(save=True)).pack(side="left")
tk.Button(bar, text="Reset", command=lambda: state.update(reset=True)).pack(side="left")
tk.Button(bar, text="Arm to rest", command=lambda: [sl[j].set(round(q_rest[i], 2))
                                                     for i, j in enumerate(ARM_JOINTS)]
          ).pack(side="left")


def readout():
  Rt, pt = d.xmat[bot.torso].reshape(3, 3), d.xpos[bot.torso]
  tip = Rt @ bot.grip_poses()["right"][0] + pt
  P.sync(d)
  dist, g1, g2 = P.distance(d.qpos[P.qadr])
  near = "nothing within 5 cm" if g1 < 0 else (
    f"{m.body(m.geom_bodyid[g1]).name} - {m.body(m.geom_bodyid[g2]).name} {dist * 100:+.1f} cm")
  st = bot.stability()
  yaw = np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0]))
  arm_txt = " ".join(f"{v:+.2f}" for v in d.qpos[P.qadr])
  waist_txt = " ".join(f"{d.qpos[q]:+.2f}" for q in waist_adr)
  return tip, (f"right arm (measured) {arm_txt}\ntorso yaw/roll/pitch {waist_txt}\n"
               f"pinch point (world) {tip.round(3)}   block {d.qpos[bq:bq + 3].round(3)}\n"
               f"closest: {near}\n"
               f"torso yaw {yaw:+.1f} deg  tilt {st['torso_tilt_deg']:.1f}  drift "
               f"{st['base_drift_m'] * 100:.1f} cm  table contacts {st['robot_table_contacts']}  "
               f"fallen {st['fallen']}\n{state['msg']}")


def save(tip):
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  poses = json.loads(out.read_text()) if out.exists() else []
  Rh = d.xmat[bot.palms[1]].reshape(3, 3)
  poses.append({
    "name": name.get() or f"pose_{len(poses)}",
    "arm_cmd": {j: float(sl[j].get()) for j in ARM_JOINTS},
    "arm_measured": dict(zip(ARM_JOINTS, map(float, d.qpos[P.qadr]))),
    "hand_cmd": {j: float(sl[lab].get()) for lab, j in HAND},
    "hand_measured": dict(zip(HAND_JOINTS, map(float, d.qpos[bot.hand_qadr["right"]]))),
    "torso_measured": {j: float(d.qpos[q]) for j, q in zip(WAIST, waist_adr)},
    "posture": {k: float(sl[k].get()) for k in ("waist yaw", "lean", "height")},
    "pinch_point_world": tip.round(4).tolist(),
    "hand_R_world": Rh.round(4).tolist(),
    "block_pos": d.qpos[bq:bq + 3].round(4).tolist(),
    "stability": bot.stability(),
  })
  out.write_text(json.dumps(poses, indent=1))
  state["msg"] = f"saved '{poses[-1]['name']}' -> {out}"


LOG.parent.mkdir(parents=True, exist_ok=True)
log = LOG.open("a")
if log.tell() == 0:
  log.write("wall_time,sim_t," + ",".join(f"right_{j}" for j in ARM_JOINTS) + "," + ",".join(WAIST) + "\n")
with mujoco.viewer.launch_passive(m, d) as v:
  v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.2, 150, -25
  v.cam.lookat[:] = [0.2, -0.1, 0.85]
  k = 0
  while v.is_running():
    t0 = time.time()
    try:
      root.update()
    except tk.TclError:
      break
    if state.pop("reset", False):
      bot.reset()
      state["reset"] = False
    goal = bot.arm_goal.copy()
    goal[7:] = [sl[j].get() for j in ARM_JOINTS]
    bot.set_arm_joint_targets(goal)
    bot.set_posture(height=sl["height"].get(), pitch=sl["lean"].get(),
                    waist_yaw=sl["waist yaw"].get())
    bot.set_hand_joints("right", [sl[lab].get() for lab, _ in HAND])
    with v.lock():
      bot.step()
    v.sync()
    if k % 5 == 0:
      log.write(f"{time.time():.2f},{bot.t:.2f}," + ",".join(f"{x:.4f}" for x in d.qpos[P.qadr]) + ","
                + ",".join(f"{d.qpos[q]:.4f}" for q in waist_adr) + "\n")
      log.flush()
    if k % 10 == 0:
      tip, txt = readout()
      info.config(text=txt)
      if state.pop("save", False):
        save(tip)
        state["save"] = False
    k += 1
    time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
