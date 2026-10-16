"""Closed-loop evaluation of the trained VLA in sim: the policy alone drives the right arm + Dex3 from
the head camera + robot state + text, at 25 Hz (legs = stance policy, waist twist as in the demos).
Each episode: a standing block at a RANDOM pose in the pickup region (position and yaw drawn fresh,
not the training poses). Success = block upright on the star (<= 3 cm), robot standing.

  .venv_vla/bin/python sim/eval_vla.py --policy outputs/<run>/checkpoints/last/pretrained_model \
      [--episodes 20] [--seconds 45] [--video data/eval_vla.mp4]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
sys.path.insert(0, str(COMMON / "sim"))
sys.path.insert(0, str(COMMON / "scene"))
import pick_place_motion as M  # noqa: E402
from layout import DROP_XY, PICK_REGION, TABLE_TOP, TASK_WAIST_YAW  # noqa: E402
from make_demos import TASK  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seconds", type=float, default=45.0)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--video", default="")
ap.add_argument("--n-action-steps", type=int, default=10)
ap.add_argument("--out", default=str(ROOT / "data/eval_vla.json"))
ap.add_argument("--legs", default=str(COMMON / "policy/stance_palmin.onnx"), help="legs ONNX")
a = ap.parse_args()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

cfg = PreTrainedConfig.from_pretrained(a.policy)
policy = get_policy_class(cfg.type).from_pretrained(a.policy)
policy.config.n_action_steps = a.n_action_steps
policy.eval().to("cuda")
pre, post = make_pre_post_processors(policy.config, pretrained_path=a.policy)

mo = M.Motion(policy=Path(a.legs))
bot = mo.bot
rng = np.random.default_rng(a.seed)
region = np.array(PICK_REGION)
results, frames = [], []
for ep in range(a.episodes):
  xy = rng.uniform(region[:, 0], region[:, 1])
  yaw = float(rng.uniform(-25, 25))
  bot.reset()
  bot.set_posture(waist_yaw=TASK_WAIST_YAW)
  bot.set_hand_joints("right", M.DEX3_OPEN)
  mo.place_block(xy, yaw)
  for _ in range(int(1.5 / M.CONTROL_DT)):  # as in the demos: the episode starts 1.5 s in
    bot.step()
  policy.reset()
  z0 = mo.block_state()[0][2]
  lifted, fallen, worst_drift = False, False, 0.0
  for k in range(int(a.seconds / (M.REC_EVERY * M.CONTROL_DT))):
    head = cv2.resize(bot.render("head"), (224, 224), interpolation=cv2.INTER_AREA)
    state = np.r_[bot.data.qpos[mo.P.qadr], bot.data.qpos[bot.hand_qadr["right"]]].astype(np.float32)
    obs = {"observation.images.camera1": torch.from_numpy(head).permute(2, 0, 1).float()[None] / 255.0,
           "observation.state": torch.from_numpy(state)[None], "task": [TASK]}
    with torch.inference_mode():
      act = post(policy.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
    goal = bot.arm_goal.copy()
    goal[7:] = act[:7]
    bot.set_arm_joint_targets(goal)
    bot.set_hand_joints("right", act[7:14])
    for _ in range(M.REC_EVERY):
      bot.step()
    pb, up = mo.block_state()
    lifted |= pb[2] - z0 > 0.03
    st = bot.stability()
    worst_drift = max(worst_drift, st["base_drift_m"])
    fallen |= st["fallen"]
    if ep == 0 and a.video:
      frames.append(np.hstack([bot.render("front"), bot.render("head")]))
    if fallen:
      break
  pb, up = mo.block_state()
  err = float(np.linalg.norm(pb[:2] - np.array(DROP_XY)) * 100)
  on_table = abs(pb[2] - (TABLE_TOP + 0.04)) < 0.01
  r = {"ep": ep, "xy": np.round(xy, 3).tolist(), "yaw": round(yaw, 1), "lifted": bool(lifted),
       "place_err_cm": round(err, 1), "upright": round(float(up), 2),
       "success": bool(err <= 3.0 and up >= 0.95 and on_table and not fallen),
       "fallen": bool(fallen), "drift_cm": round(worst_drift * 100, 1)}
  results.append(r)
  print(json.dumps(r), flush=True)
n = len(results)
summary = {"episodes": n, "success": sum(r["success"] for r in results) / n,
           "lifted": sum(r["lifted"] for r in results) / n,
           "fallen": sum(r["fallen"] for r in results) / n,
           "median_place_err_cm": float(np.median([r["place_err_cm"] for r in results]))}
print(json.dumps(summary))
Path(a.out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
if frames:
  import imageio
  imageio.mimsave(a.video, frames, fps=25)
