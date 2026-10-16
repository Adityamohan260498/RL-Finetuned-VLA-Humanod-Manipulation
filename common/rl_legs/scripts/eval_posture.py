"""Evaluate the legs-only posture policy in mjlab (MuJoCo Warp). Run ONLY when no training runs.

  cd ~/g1/resources/mjlab && uv run python ~/g1/training_legs/eval_posture.py \
      --onnx <run>/<run>.onnx [--tag final] [--mass-known 1|0] [--safe-arms 0|1]

The env builds the 752-dim actor observation itself; the script only drives the commands
(twist, posture goal, arm mode, payload) and runs the ONNX policy. Play config (no DR, no
pushes), terminations off (falls = pelvis tilt > 53 deg). One env per scenario x payload.

Scenarios (12 s each, metrics over the window given per group):
  walk      : fwd 0.6/1.0/1.5, side L/R 0.6, spin L/R 0.5/1.0, arc L/R 0.8+0.8, stand,
              walk->stand (0.8 m/s for 4 s then 0)                 metrics t = 6-12 s
  crouch    : fwd 1.0 commanded at height 0.65 / 0.60 (expect x0.65 / x0.3 speed)
  squat     : stand at height 0.70/0.60/0.55/0.50 x lean 0/0.4/0.8, waist yaw 0.4
  pick      : squat 0.55 m + lean 0.4, payload appears at t=4 s, rise to 0.78 at t=8 s
  arms      : stand with arm mode reach / pick / carry (self-collisions, stillness)
Payloads: 0 / 5 / 10 kg (the 'pick' group always picks up 10 kg).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.human.mdp.carry import carry_payload
from mjlab.tasks.human_v2_legs.arm_command import MODES
from mjlab.tasks.human_v2_posture import env_cfg as E
from mjlab.tasks.human_v2_posture import posture as P
from mjlab.tasks.registry import load_env_cfg

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--task", default="Mjlab-Velocity-HumanV2-Posture-Unitree-G1")
ap.add_argument("--tag", default="eval")
ap.add_argument("--mass-known", type=int, default=1, help="1: estimate given, 0: unknown")
ap.add_argument("--safe-arms", type=int, default=0, help="1: collision-free reach/carry")
ap.add_argument("--seconds", type=float, default=12.0)
args = ap.parse_args()

NOM = (P.NOMINAL_H, 0.0, 0.0)
S = []  # (group, name, twist(3), posture goal(3), arm mode, kg list)
KG = [0.0, 5.0, 10.0]
for name, tw in [("fwd 0.6", (0.6, 0, 0)), ("fwd 1.0", (1.0, 0, 0)), ("fwd 1.5", (1.5, 0, 0)),
                 ("side L 0.6", (0, 0.6, 0)), ("side R 0.6", (0, -0.6, 0)),
                 ("spin L 0.5", (0, 0, 0.5)), ("spin R 0.5", (0, 0, -0.5)),
                 ("spin L 1.0", (0, 0, 1.0)), ("spin R 1.0", (0, 0, -1.0)),
                 ("arc L", (0.8, 0, 0.8)), ("arc R", (0.8, 0, -0.8)),
                 ("stand", (0, 0, 0)), ("walk->stand", (0.8, 0, 0))]:
  S.append(("walk", name, tw, NOM, "default", KG))
for h in (0.65, 0.60):
  S.append(("crouch", f"crouch-walk 1.0 @{h:.2f}", (1.0, 0, 0), (h, 0.1, 0.0), "default", KG))
for h in (0.70, 0.60, 0.55, 0.50):
  for lean in (0.0, 0.4, 0.8):
    S.append(("squat", f"squat {h:.2f} lean {lean:.1f}", (0, 0, 0), (h, lean, 0.4), "default", KG))
S.append(("pick", "pick-up 10kg @0.55", (0, 0, 0), (0.55, 0.4, 0.0), "pick", [10.0]))
for m in ("reach", "pick", "carry"):
  S.append(("arms", f"stand arms={m}", (0, 0, 0), NOM, m, [0.0, 5.0]))
ENVS = [(s, kg) for s in S for kg in s[5]]
N = len(ENVS)

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
cfg.events.pop("carry_payload", None)
cfg.events.pop("carry_toggle", None)
cfg.events.pop("push_robot", None)
for t in cfg.commands.values():
  t.resampling_time_range = (1e9, 1e9)
cfg.observations["actor"].terms["mass_estimate"].params.update(
  {"p_known": float(args.mass_known), "rel_err": 0.0, "step_std": 0.0})
if args.safe_arms:
  cfg.commands["arm_command"].reach_box = dict(E.REACH_BOX)
  cfg.commands["arm_command"].carry_override = dict(E.CARRY_SAFE)
  cfg.commands["arm_command"].default_roll_out = 0.30
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
dev = env.device
robot = env.scene["robot"]
cm = env.command_manager
tw, post, arm, gait = (cm.get_term(n) for n in ("twist", "posture", "arm_command", "gait"))

palms = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"))
palms.resolve(env.scene)
pick_ids = torch.tensor([i for i, (s, _) in enumerate(ENVS) if s[0] == "pick"], device=dev)
for kg in sorted({kg for _, kg in ENVS}):
  ids = torch.tensor([i for i, (s, k) in enumerate(ENVS) if k == kg and s[0] != "pick"],
                     device=dev)
  if len(ids):
    carry_payload(env, ids, p_carry=1.0 if kg > 0 else 0.0, mass_range=(kg, kg),
                  split_range=(0.5, 0.5), asset_cfg=palms)
carry_payload(env, pick_ids, p_carry=0.0, mass_range=(0.0, 0.0), asset_cfg=palms)
env.sim.recompute_constants(RecomputeLevel.set_const)
arm.mode[:] = torch.tensor([MODES.index(s[4]) for s, _ in ENVS], device=dev)

TW = torch.tensor([s[2] for s, _ in ENVS], dtype=torch.float32, device=dev)
GOAL = torch.tensor([s[3] for s, _ in ENVS], dtype=torch.float32, device=dev)
W2S = torch.tensor([s[1] == "walk->stand" for s, _ in ENVS], device=dev)
PICK = torch.tensor([s[0] == "pick" for s, _ in ENVS], device=dev)

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
jn = list(robot.joint_names)
knee = [jn.index("left_knee_joint"), jn.index("right_knee_joint")]
pol = [i for i, n in enumerate(jn) if any(k in n for k in ("hip", "knee", "ankle", "waist"))]
yaw_j = jn.index("waist_yaw_joint")
torso = robot.body_names.index("torso_link")
sc_col = env.reward_manager.active_terms.index("self_collisions")
steps = int(args.seconds / env.step_dt)
rec = {k: [] for k in ("t", "vx", "vy", "wz", "h", "pitch", "yaw", "jv", "knee", "col",
                       "torso_w", "tgt_h", "cmd_vx")}
fell = torch.zeros(N, dtype=torch.bool, device=dev)
fell_t = torch.full((N,), float("nan"), device=dev)
picked = False

for k in range(steps):
  t = k * env.step_dt
  c = TW.clone()
  c[W2S & (t >= 4.0)] = 0.0
  g = GOAL.clone()
  g[PICK & (t >= 8.0)] = torch.tensor(NOM, device=dev)
  tw.vel_command_b[:] = c
  tw.vel_command_w[:] = c
  tw.is_standing_env[:] = c.abs().sum(1) == 0
  tw.is_heading_env[:] = False
  tw.is_world_env[:] = False
  post.goal[:] = g
  if not picked and t >= 4.0:  # sudden 10 kg pick-up for the 'pick' group
    carry_payload(env, pick_ids, p_carry=1.0, mass_range=(10.0, 10.0),
                  split_range=(0.5, 0.5), asset_cfg=palms)
    env.sim.recompute_constants(RecomputeLevel.set_const)
    picked = True
  o = obs["actor"].cpu().numpy().astype(np.float32)
  a = np.concatenate([sess.run(None, {inp: o[i : i + 1]})[0] for i in range(N)])
  obs, *_ = env.step(torch.from_numpy(a).to(dev))
  d = robot.data
  newly = (d.projected_gravity_b[:, 2] > -0.6) & ~fell
  fell_t[newly] = t
  fell |= newly
  pitch, _ = P.torso_pitch_roll(env)
  qa = d.qfrc_actuator
  ko = 6 if qa.shape[1] == len(jn) + 6 else 0
  rec["t"].append(t)
  for key, val in (("vx", d.root_link_lin_vel_b[:, 0]), ("vy", d.root_link_lin_vel_b[:, 1]),
                   ("wz", d.root_link_ang_vel_b[:, 2]), ("h", d.root_link_pos_w[:, 2]),
                   ("pitch", pitch), ("yaw", d.joint_pos[:, yaw_j]),
                   ("jv", d.joint_vel[:, pol].pow(2).sum(1).sqrt()),
                   ("knee", qa[:, [ko + j for j in knee]].abs().max(1).values),
                   ("col", env.reward_manager._step_reward[:, sc_col]),
                   ("torso_w", d.body_link_ang_vel_w[:, torso].norm(dim=1)),
                   ("tgt_h", post.target[:, 0]), ("cmd_vx", tw.vel_command_b[:, 0])):
    rec[key].append(val.detach().clone())

T = np.array(rec["t"])
R = {k: torch.stack(v).cpu().numpy() for k, v in rec.items() if k != "t"}


def win(group):
  if group == "pick":
    return (T >= 4.0) & (T < 8.0)  # holding the squat while the box arrives
  return T >= 6.0


out = []
for i, (s, kg) in enumerate(ENVS):
  group, name, twc, goal = s[0], s[1], s[2], s[3]
  w = win(group)
  if name == "walk->stand":
    w = T >= 8.0  # 4 s after the stop command
  m = {k: float(R[k][w, i].mean()) for k in ("vx", "vy", "wz", "h", "pitch", "yaw", "cmd_vx")}
  out.append({
    "group": group, "name": name, "kg": kg, "cmd": list(twc), "goal": list(goal),
    "vx": round(m["vx"], 3), "vy": round(m["vy"], 3), "wz": round(m["wz"], 3),
    "cmd_vx_scaled": round(m["cmd_vx"], 3),
    "height": round(m["h"], 3), "lean": round(m["pitch"], 3), "waist_yaw": round(m["yaw"], 3),
    "joint_speed_rms": round(float(R["jv"][w, i].mean()), 2),
    "torso_angvel_dps": round(math.degrees(float(np.sqrt((R["torso_w"][w, i] ** 2).mean()))), 1),
    "knee_torque_peak": round(float(R["knee"][:, i].max()), 1),
    "self_collision_frac": round(float((R["col"][:, i] < 0).mean()), 3),
    "fell": bool(fell[i]), "fell_t": None if not bool(fell[i]) else round(float(fell_t[i]), 2),
    "pick_height_dip": (round(float(R["h"][(T >= 4.0) & (T < 5.0), i].min() - goal[0]), 3)
                        if group == "pick" else None),
  })

print(f"\n{'scenario':24s} {'kg':>3s} | {'cmd':>15s} | {'got vx vy wz':>17s} | "
      f"{'h':>5s} {'lean':>5s} {'yaw':>5s} | {'jv':>5s} {'knee':>5s} {'col':>5s} fell")
for r in out:
  cv = " ".join(f"{x:+.1f}" for x in r["cmd"])
  print(f"{r['name']:24s} {r['kg']:3.0f} | {cv:>15s} | {r['vx']:+.2f} {r['vy']:+.2f} "
        f"{r['wz']:+.2f} | {r['height']:.2f} {r['lean']:+.2f} {r['waist_yaw']:+.2f} | "
        f"{r['joint_speed_rms']:5.2f} {r['knee_torque_peak']:5.0f} {r['self_collision_frac']:5.2f} "
        f"{'FELL@' + str(r['fell_t']) if r['fell'] else ''}")
nf = sum(r["fell"] for r in out)
print(f"\nfalls: {nf}/{len(out)}  (mass {'known' if args.mass_known else 'UNKNOWN'}, "
      f"arms {'safe' if args.safe_arms else 'training'})")
dst = HERE / f"logs/eval_posture_{args.tag}.json"
dst.parent.mkdir(exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print("wrote", dst)
