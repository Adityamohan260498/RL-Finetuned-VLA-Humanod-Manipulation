"""Scripted 'VLA-like' arm task to check the stance policy (no VLA yet):
left hand reaches the block, 'grasps' it (payload on), hands it to the right hand in front of
the chest, the right hand places it at the tray; legs/waist are the RL policy throughout.
Objects are not physically grasped here (the G1 hands have no fingers): the payload mass is
added at the hands while 'holding'. Reports stance stability; optional video.

  python stance/demo_stance.py \
      [--policy ../policy/stance.onnx] [--video out.mp4] [--speed 1.0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stance_runner import CONTROL_DT, POLICY, StanceRobot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", type=Path, default=POLICY)
ap.add_argument("--video", type=Path, default=None)
ap.add_argument("--speed", type=float, default=1.0, help=">1 = faster arm motions")
ap.add_argument("--kg", type=float, default=0.08, help="object mass (block 0.08 kg)")
ap.add_argument("--repeats", type=int, default=2)
args = ap.parse_args()

bot = StanceRobot(policy=args.policy)
bot.reset()
rest = {s: p.copy() for s, (p, _) in bot.hand_poses().items()}
# Hover poses (torso frame, reach-checked): just above the objects, since the fingerless G1
# hand cannot grasp here -- this exercises the stance, not grasping.
BLOCK = np.array([0.27, 0.12, 0.04])  # ~2 cm above the block top (posture 0.74 m, lean 0.15)
# Hands 18 cm apart at the hand-over: the rubber hands are ~15 cm across, closer = they push.
HAND_OVER_L, HAND_OVER_R = np.array([0.26, 0.09, 0.10]), np.array([0.26, -0.09, 0.10])
TRAY = np.array([0.26, -0.15, 0.0])
# (duration s, left target, right target, payload on?, lean)
seq = [(2.0, rest["left"], rest["right"], False, 0.0),
       (2.5, BLOCK + [0, 0, 0.08], rest["right"], False, 0.15),
       (1.5, BLOCK, rest["right"], False, 0.2),
       (1.0, BLOCK, rest["right"], True, 0.2),          # grasp
       (2.0, HAND_OVER_L, HAND_OVER_R, True, 0.1),      # bring together
       (1.0, HAND_OVER_L, HAND_OVER_R, True, 0.1),      # hand-over
       (2.0, rest["left"], TRAY + [0, 0, 0.08], True, 0.15),
       (1.5, rest["left"], TRAY, True, 0.2),
       (1.0, rest["left"], TRAY, False, 0.2),           # release
       (2.0, rest["left"], rest["right"], False, 0.0)]
frames, worst = [], {"base_drift_m": 0, "foot_slip_m": 0, "pelvis_angvel": 0,
                     "torso_tilt_deg": 0, "robot_table_contacts": 0}
track = []
fell = False
for rep in range(args.repeats):
  for dur, pl, pr, hold, lean in seq:
    n = int(dur / args.speed / CONTROL_DT)
    p0 = {s: p.copy() for s, (p, _) in bot.hand_poses().items()}
    bot.set_payload(args.kg if hold else 0.0)
    bot.set_posture(height=0.74 if lean > 0 else 0.78, pitch=lean)
    for k in range(n):
      a = min(1.0, (k + 1) / max(n * 0.7, 1))  # smooth-ish interpolation, then hold
      bot.set_hand_targets(p0["left"] + a * (pl - p0["left"]),
                           p0["right"] + a * (pr - p0["right"]))
      bot.step()
      st = bot.stability()
      fell |= st["fallen"]
      for key in worst:
        worst[key] = max(worst[key], st[key])
      if args.video is not None and int(bot.t / CONTROL_DT) % 2 == 0:
        frames.append(np.concatenate([bot.render("front"), bot.render("head")], axis=1))
    hp = bot.hand_poses()
    track.append(max(np.linalg.norm(hp["left"][0] - pl), np.linalg.norm(hp["right"][0] - pr)))
print(f"{args.repeats} task repeats, {bot.t:.1f} s, fell: {fell}")
print("worst stance:", {k: round(v, 4) for k, v in worst.items()})
print(f"hand target error at the end of each segment: median {np.median(track) * 1000:.0f} mm, "
      f"max {np.max(track) * 1000:.0f} mm")
if args.video is not None:
  import imageio

  imageio.mimsave(args.video, frames, fps=25)
  print("wrote", args.video)
