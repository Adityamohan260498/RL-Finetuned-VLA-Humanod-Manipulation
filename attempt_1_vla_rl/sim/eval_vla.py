"""Closed-loop evaluation of the VLA trained on the robot-follows-my-hand demos: the policy alone
drives the right hand + Dex3 from the head camera + hand state + text, at 25 Hz. Its 5-dim action
(dx, dy, dz, dyaw, grip) becomes a hand-pose target, turned into arm joints by the same closed-loop
IK as in the demos (from the CURRENT torso pose: so the legs policy can change).

  .venv_vla/bin/python sim/eval_vla.py --policy outputs/<run>/checkpoints/last/pretrained_model \
      [--episodes 30] [--seconds 45] [--legs policy/stance_palmin.onnx] [--safety] [--video f.mp4]

Each episode: a standing block at a random pose in the pickup region (seed 123: the same poses for
every policy / legs). Success = block upright on the star (<= 3 cm), on the table, robot standing,
no robot body part (other than the hand) touching the table.

--safety: a filter between the VLA and the arm (does not change the VLA): the hand target is kept
>= SAFE_Z above the table (except right over the block / star, where the grasp / place happen), and
an arm step is not applied if it would bring any robot body part other than the fingers within
SAFE_CLEAR of the table (ground-truth geometry, sim/planner.py). Counted per episode.
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
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(COMMON / "scene"))
import robot_motion as M  # noqa: E402
from human_follow import STEP_MAX, TASK, Follow, hand_q, wrap  # noqa: E402
from layout import DROP_XY, PICK_REGION, TABLE_TOP, TASK_WAIST_YAW  # noqa: E402
from stance_runner import CONTROL_DT, DEX3_OPEN  # noqa: E402

SAFE_Z = 0.035      # hand grasp centre >= 3.5 cm above the table away from the block / star
SAFE_NEAR = 0.06    # "over the block / star": within 6 cm in x/y
SAFE_CLEAR = 0.01   # arm / wrist (not fingers) >= 1 cm from the table

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--seconds", type=float, default=45.0)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--video", default="")
ap.add_argument("--video-all", action="store_true", help="video of every episode (default: episode 0 only)")
ap.add_argument("--n-action-steps", type=int, default=10)
ap.add_argument("--legs", default=str(COMMON / "policy/stance_palmin.onnx"))
ap.add_argument("--safety", action="store_true")
ap.add_argument("--out", default="")
ap.add_argument("--record", default="", help="dir: save every episode (img, state, act, stage) for the RL loop")
ap.add_argument("--early-stop", action="store_true", help="end an episode once it is decided (rollouts)")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--yaw-max", type=float, default=25.0)
ap.add_argument("--relabel", type=int, default=1)
ap.add_argument("--ready", default="0.25,-0.145,0.93,-0.18",
                help="x,y,z,yaw: scripted planned move here before the VLA starts ('' = none). The median "
                     "pose where the demos hand over from the robot's planned move to my motion")
ap.add_argument("--replay", nargs="*", default=[],
                help="diagnostic: demo .npz files; their (relabelled) actions are played open loop instead of the VLA")
a = ap.parse_args()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

if not a.replay:
  cfg = PreTrainedConfig.from_pretrained(a.policy)
  policy = get_policy_class(cfg.type).from_pretrained(a.policy)
  policy.config.n_action_steps = a.n_action_steps
  policy.eval().to("cuda")
  pre, post = make_pre_post_processors(policy.config, pretrained_path=a.policy)
# a policy trained with the wrist camera (camera2) gets it; head-only policies are unchanged
# (the saved config always lists camera1-3; what the policy was trained on is in its train rename map)
_tc = Path(a.policy) / "train_config.json"
USE_WRIST = (not a.replay) and _tc.exists() and "observation.images.wrist" in _tc.read_text()

fo = Follow(policy=Path(a.legs))
bot = fo.bot
_pm = fo.P.m
_arm_geoms = [int(g) for g in fo.P.arm_geoms  # arm links only: the wrist_yaw link carries the Dex3 palm
              if not any(k in _pm.body(_pm.geom_bodyid[g]).name for k in ("hand_", "wrist_yaw"))]
_table_geoms = [int(g) for g in bot.table_geoms]
_ft = np.zeros(6)


def arm_table_gap(q):
  """Smallest distance (m) arm links / wrist (NOT the Dex3 fingers / palm) <-> table, for arm angles q
  at the current body pose (ground-truth geometry)."""
  import mujoco
  fo.P.sync(bot.data)
  fo.P.d.qpos[fo.P.qadr] = q
  mujoco.mj_kinematics(_pm, fo.P.d)
  return min(mujoco.mj_geomDistance(_pm, fo.P.d, g1, g2, 0.05, _ft) for g1 in _arm_geoms for g2 in _table_geoms)
rng = np.random.default_rng(a.seed)
region = np.array(PICK_REGION)
poses = [(rng.uniform(region[:, 0], region[:, 1]), float(rng.uniform(-a.yaw_max, a.yaw_max))) for _ in range(a.episodes)]
replay_acts = []
for f in a.replay:  # same relabelling as sim/to_lerobot.py
  z = np.load(f, allow_pickle=True)
  info, ph, st_, act_ = json.loads(str(z["info"])), z["phase"], z["state"], z["act"].copy()
  for k in np.flatnonzero(np.isin(ph, ["settle", "to_start", "to_rest"])):
    j = min(k + a.relabel, len(st_) - 1)
    act_[k, :3] = st_[j, :3] - st_[k, :3]
    act_[k, 3] = (st_[j, 3] - st_[k, 3] + np.pi) % (2 * np.pi) - np.pi
  k0 = int(np.flatnonzero(ph == "follow")[0])
  keep = ~np.isin(ph, ["settle", "to_start", "to_rest"])
  keep[:k0] = False
  replay_acts.append((act_[keep], st_[k0]))
if a.replay:
  poses = [(json.loads(str(np.load(f, allow_pickle=True)["info"]))["xy"],
            json.loads(str(np.load(f, allow_pickle=True)["info"]))["yaw_deg"]) for f in a.replay]
  a.seconds = 1e9
results, frames = [], []
STAGES = ("start", "reached", "lifted", "over_star", "placed")


def stage_now(st, pb, up, p, g_cmd, xy, z0):
  """Progress (privileged sim state), monotone: 1 hand open at the block's grasp point (<= 2.5 cm),
  2 block lifted >= 2 cm (held), 3 lifted block within 3 cm of the star, 4 block upright on the star, hand open."""
  grasp_pt = np.r_[xy, TABLE_TOP + M.GRASP_H]
  if st < 1 and np.linalg.norm(p - grasp_pt) < 0.025:
    st = 1
  if st >= 1 and st < 2 and pb[2] - z0 > 0.02:
    st = 2
  if st >= 2 and st < 3 and pb[2] - z0 > 0.01 and np.linalg.norm(pb[:2] - np.array(DROP_XY)) < 0.03:
    st = 3
  if st >= 3 and abs(pb[2] - z0) < 0.005 and up >= 0.95 and np.linalg.norm(pb[:2] - np.array(DROP_XY)) <= 0.03 \
     and g_cmd < 0.3:
    st = 4
  return st


for ep, (xy, yaw) in enumerate(poses):
  if ep % a.workers != a.worker:
    continue
  xy = np.array(xy)
  bot.reset()
  fo.worst = {"fall": False, "table": 0, "drift": 0.0, "tilt": 0.0}
  fo.hand_table_steps, fo.k_step, fo.rec, fo.hook = 0, 0, None, None
  bot.set_posture(waist_yaw=TASK_WAIST_YAW)
  bot.set_hand_joints("right", DEX3_OPEN)
  fo.place_block(xy, yaw)
  fo.hold(1.5)  # as in the demos: the episode starts 1.5 s in
  if a.ready or a.replay:  # NOT the VLA: the same planned move the demos use to get to my motion
    rdy = replay_acts[ep][1] if a.replay else np.array([float(v) for v in a.ready.split(",")])
    fo.R = M.hand_R(0.0, np.degrees(rdy[3]))
    path = fo.plan_to(np.array(rdy[:3]))
    fo.run_q(path, fo.P.timed(path, vmax=M.FREE_VMAX), M.PRESHAPE, DEX3_OPEN.copy())
    fo.servo(np.array(rdy[:3]), 0.5, M.PRESHAPE, M.PRESHAPE)
  policy.reset() if not a.replay else None
  z0 = fo.block_state()[0][2]
  lifted, interventions, g_cmd = False, 0, 0.0
  goal = bot.arm_goal.copy()
  rec = {"img": [], "wrist": [], "state": [], "act": [], "stage": [], "up": []}
  st, k_done, k_lying = 0, None, 0
  n_k = len(replay_acts[ep][0]) if a.replay else int(a.seconds / (M.REC_EVERY * CONTROL_DT))
  for k in range(n_k):
    head = cv2.resize(bot.render("head"), (224, 224), interpolation=cv2.INTER_AREA)
    if USE_WRIST:
      wrist = cv2.resize(bot.render("wrist"), (224, 224), interpolation=cv2.INTER_AREA)
    p, w = fo.ee()
    state = np.r_[p, w, g_cmd].astype(np.float32)
    obs = {"observation.images.camera1": torch.from_numpy(head).permute(2, 0, 1).float()[None] / 255.0,
           **({"observation.images.camera2": torch.from_numpy(wrist).permute(2, 0, 1).float()[None] / 255.0}
              if USE_WRIST else {}),
           "observation.state": torch.from_numpy(state)[None], "task": [TASK]}
    if a.replay:
      act = replay_acts[ep][0][k]
    else:
      with torch.inference_mode():
        act = post(policy.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
    if a.record:
      rec["img"].append(head)
      rec["wrist"].append(wrist) if USE_WRIST else None
      rec["state"].append(state)
      rec["act"].append(act.astype(np.float32))
    tp, tw, g_cmd = p + act[:3], w + float(act[3]), float(np.clip(act[4], 0, 1))
    if a.safety:
      near = min(np.linalg.norm(tp[:2] - xy), np.linalg.norm(tp[:2] - np.array(DROP_XY)))
      if near > SAFE_NEAR and tp[2] < TABLE_TOP + SAFE_Z:
        tp[2] = TABLE_TOP + SAFE_Z
        interventions += 1
    fo.R = M.hand_R(0.0, np.degrees(tw))
    for _ in range(M.REC_EVERY):
      q = fo.ik(tp, goal[7:].copy(), iters=60, fallback=False)
      q_new = goal[7:] + np.clip(q - goal[7:], -STEP_MAX, STEP_MAX)
      if a.safety and arm_table_gap(q_new) < SAFE_CLEAR:
        q_new = goal[7:]  # hold: that step would bring the arm / wrist to the table
        interventions += 1
      goal[7:] = q_new
      bot.set_arm_joint_targets(goal)
      bot.set_hand_joints("right", hand_q(g_cmd))
      fo.step()
    pb, up = fo.block_state()
    lifted |= pb[2] - z0 > 0.03
    st = stage_now(st, pb, up, fo.ee()[0], g_cmd, xy, z0)
    rec["stage"].append(st)
    rec["up"].append(up)
    if a.early_stop:  # decided: placed + hand 4 cm away for 2 s, or the block lying / off the table for 2 s
      if st == 4 and np.linalg.norm(fo.ee()[0] - pb) > 0.06:
        k_done = k if k_done is None else k_done
        if k - k_done >= 50:
          break
      k_lying = k_lying + 1 if (up < 0.5 or pb[2] < TABLE_TOP - 0.05) else 0
      if k_lying >= 50:
        break
    if a.video and (ep == 0 or a.video_all):
      frames.append(np.hstack([bot.render(c) for c in ("front", "head") + (("wrist",) if USE_WRIST else ())]))
    if fo.worst["fall"]:
      break
  pb, up = fo.block_state()
  err = float(np.linalg.norm(pb[:2] - np.array(DROP_XY)) * 100)
  on_table = abs(pb[2] - (TABLE_TOP + 0.04)) < 0.01
  r = {"ep": ep, "xy": np.round(xy, 3).tolist(), "yaw": round(yaw, 1), "lifted": bool(lifted),
       "place_err_cm": round(err, 1), "upright": round(float(up), 2),
       "success": bool(err <= 3.0 and up >= 0.95 and on_table and not fo.worst["fall"] and fo.worst["table"] == 0),
       "fallen": bool(fo.worst["fall"]), "body_table": int(fo.worst["table"]),
       "hand_table_steps": int(fo.hand_table_steps), "drift_cm": round(fo.worst["drift"] * 100, 1),
       "safety_interventions": interventions, "max_stage": int(st), "frames": len(rec["stage"])}
  if a.record:
    Path(a.record).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.record) / f"roll_s{a.seed}_{ep:04d}.npz", img=np.array(rec["img"], np.uint8),
                        **({"wrist": np.array(rec["wrist"], np.uint8)} if USE_WRIST else {}),
                        state=np.array(rec["state"], np.float32), act=np.array(rec["act"], np.float32),
                        stage=np.array(rec["stage"], np.int8), up=np.array(rec["up"], np.float32), task=TASK, info=json.dumps({**r, "clip": "rollout"}))
  results.append(r)
  print(json.dumps(r), flush=True)
n = len(results)
summary = {"episodes": n, "legs": Path(a.legs).name, "safety": a.safety,
           "success": sum(r["success"] for r in results) / n, "lifted": sum(r["lifted"] for r in results) / n,
           "fallen": sum(r["fallen"] for r in results) / n,
           "body_table_episodes": sum(r["body_table"] > 0 for r in results),
           "median_place_err_cm": float(np.median([r["place_err_cm"] for r in results])),
           "safety_interventions": sum(r["safety_interventions"] for r in results)}
print(json.dumps(summary))
out = a.out or str(ROOT / f"data/eval/vla_{Path(a.legs).stem}{'_safe' if a.safety else ''}.json")
if a.workers > 1:
  out = out.replace(".json", f".w{a.worker}.json")
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
if frames:
  import imageio
  imageio.mimsave(a.video, frames, fps=25)
