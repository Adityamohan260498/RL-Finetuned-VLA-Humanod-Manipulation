"""Sim demos from the user's 26 clips: each clip's block pose (perception/block_pose.py: standing or
lying, yaw, position) is mapped into the robot's pickup region (scene/layout.py PICK_REGION, one
affine for the whole desk area, so the clips' relative layout is kept), then the G1 does the
pick-and-place with the Dex3 hand: planner + closed-loop IK for the right arm, legs = stance
policy, grasp = contact and friction only. Kept ONLY if: block on the star (<= 3 cm), upright,
robot standing, no robot-body / table contact.

Per episode (25 Hz) -> data/sim_demos/<clip>_<k>.npz:
  img (T,224,224,3)  head camera        q (T,14)   measured right arm (7) + Dex3 (7) joints
  act (T,14)         commanded targets  task       the instruction text

  .venv_perc/bin/python sim/make_demos.py data/tracks/*.npz [--variants 0] [--video] [--only 112209]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
sys.path.insert(0, str(COMMON / "sim"))
sys.path.insert(0, str(COMMON / "scene"))
import pick_place_motion as M  # noqa: E402
from layout import DROP_XY, HUMAN_AREA, PICK_REGION, TASK_WAIST_YAW  # noqa: E402

TASK = "pick up the blue block and put it on the blue star"
OUT = ROOT / "data/sim_demos"
SETTLE_TRIM = 1.5  # s of standing still at the start that are not recorded into the episode
PICK = np.array(PICK_REGION)  # x, y ranges


def to_pick_region(xy_h):
  """Human desk (body frame) -> robot pickup region, one per-axis affine."""
  u = (np.asarray(xy_h, float) - HUMAN_AREA[:, 0]) / (HUMAN_AREA[:, 1] - HUMAN_AREA[:, 0])
  return PICK[:, 0] + np.clip(u, 0, 1) * (PICK[:, 1] - PICK[:, 0])


def wrap90_deg(a):
  return (a + 45.0) % 90.0 - 45.0


def run(mo, clip, k, xy, yaw_deg, state, video):
  mo.rec = {"img": [], "q": [], "act": [], "video": []}
  mo.video_cams = ("front", "head") if video else ()
  mo.k_step = 0
  info = {"clip": clip, "variant": k, "xy": [round(float(v), 4) for v in xy], "yaw_deg": yaw_deg,
          "state": state}
  lying = state == "lying"
  try:  # standing: hand yaw = block yaw mod 90; lying: yaw = the long-axis direction
    worst = mo.pick_place(tuple(xy), DROP_XY, TASK_WAIST_YAW,
                          yaw_pick=yaw_deg if lying else wrap90_deg(yaw_deg), lying=lying)
  except RuntimeError as e:
    info["result"] = f"motion failed: {e}"
    return info, None
  g = mo.grasp_log
  pose_ok = g.get("lying_flat") if lying else g.get("upright", 0) >= 0.95
  ok = (g.get("place_err_cm", 99) <= 3.0 and pose_ok and g.get("on_table")
        and not worst["fall"] and worst["table"] == 0)
  info.update(result="OK" if ok else "FAIL", place_err_cm=round(g.get("place_err_cm", -1), 2),
              upright=round(g.get("upright", 0), 3), fall=worst["fall"], body_table=worst["table"],
              drift_cm=round(worst["drift"] * 100, 1),
              hand_table_s=round(mo.hand_table_steps * M.CONTROL_DT, 2),
              grip_N={a: round(b, 1) for a, b in g.get("grip_N", {}).items()})
  n0 = int(SETTLE_TRIM / (M.CONTROL_DT * M.REC_EVERY))
  ep = {key: np.array(v[n0:]) for key, v in mo.rec.items() if v}
  return info, (ep if ok or video else None)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("tracks", nargs="*")
  ap.add_argument("--variants", type=int, default=0, help="extra jittered episodes per clip")
  ap.add_argument("--only", default="", help="comma list of clip suffixes")
  ap.add_argument("--video", action="store_true")
  ap.add_argument("--out", default=str(OUT))
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--uniform", type=int, default=0,
                  help="instead of the clips: N standing blocks at uniform random poses in the region "
                       "(yaw -25..25 deg, the clips' range) - fills the gaps between the clip poses")
  ap.add_argument("--jitter-only", action="store_true",
                  help="only jittered variants (skip the clip's own pose; for extra batches)")
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(a.seed)
  mo = M.Motion()
  log = []
  if a.uniform:
    for k in range(a.uniform):
      xy = rng.uniform(PICK[:, 0], PICK[:, 1])
      yaw = float(rng.uniform(-25, 25))
      info, ep = run(mo, "uniform", k, xy, round(yaw, 1), "standing", a.video)
      print(json.dumps(info), flush=True)
      log.append(info)
      if ep is not None:
        np.savez_compressed(out / f"uniform_s{a.seed}_{k}.npz", img=ep["img"], q=ep["q"],
                            act=ep["act"], task=TASK, info=json.dumps(info))
    a.tracks = []
  for f in a.tracks:
    tr = np.load(f, allow_pickle=True)
    clip = Path(f).stem.replace("blue_cube_", "")
    if a.only and not any(clip.endswith(o) for o in a.only.split(",")):
      continue
    if "block_yaw_start" not in tr.files:
      print(f"{clip}: no block pose (run perception/block_pose.py)")
      continue
    xy0 = to_pick_region(tr["block_start"])
    yaw0 = float(np.degrees(tr["block_yaw_start"]))
    state = str(tr["block_state_start"])
    for k in range(1 if a.jitter_only else 0, 1 + a.variants):
      xy, yaw = xy0.copy(), yaw0
      if k > 0:  # jitter inside the region
        xy = np.clip(xy0 + rng.uniform(-0.015, 0.015, 2), PICK[:, 0], PICK[:, 1])
        yaw = yaw0 + rng.uniform(-5, 5)
      info, ep = run(mo, clip, k, xy, round(yaw, 1), state, a.video)
      print(json.dumps(info), flush=True)
      log.append(info)
      if ep is not None:
        name = f"{clip}_s{a.seed}_{k}"
        np.savez_compressed(out / f"{name}.npz", img=ep["img"], q=ep["q"], act=ep["act"],
                            task=TASK, info=json.dumps(info))
        if a.video and "video" in ep:
          import imageio
          imageio.mimsave(out / f"{name}.mp4", list(ep["video"]), fps=25)
  ok = sum(i.get("result") == "OK" for i in log)
  tried = sum(not i.get("result", "").startswith("skipped") for i in log)
  print(f"\n{ok} / {tried} attempted demos succeeded ({len(log) - tried} skipped: lying)")
  with open(out / f"log_{a.seed}.jsonl", "a") as fh:
    for i in log:
      fh.write(json.dumps(i) + "\n")


if __name__ == "__main__":
  main()
