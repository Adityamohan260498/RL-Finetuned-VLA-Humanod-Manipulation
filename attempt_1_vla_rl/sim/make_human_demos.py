"""Demos where the robot FOLLOWS my hand (sim/human_follow.py), kept only if they physically succeed.

  .venv_perc/bin/python sim/make_human_demos.py --variants 12 --seed 0 [--clips 111826,112044]

For each good track (data/tracks/*.npz with ok = True) and each variant: the robot's block at a
random pose in PICK_REGION (yaw +-25 deg; the clip's path is re-anchored to it). The first variant
uses the pose my block had, mapped into the robot's region (layout.human_to_g1, clipped).
Writes data/human_demos/<clip>_s<seed>_<k>.npz (img, state, act, q, phase, task, info) for the
successes, and appends every attempt to data/human_demos/attempts.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(COMMON / "scene"))
from human_follow import TASK, Follow  # noqa: E402
from layout import PICK_REGION, human_to_g1  # noqa: E402

OUT = ROOT / __import__("os").environ.get("HD_OUT", "data/human_demos")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variants", type=int, default=12)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--clips", default="", help="comma list of clip ids (default: all good tracks)")
  ap.add_argument("--legs", default=str(COMMON / "policy/stance_palmin.onnx"))
  a = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  tracks = sorted(f for f in glob.glob(str(ROOT / "data/tracks/*.npz")))
  if a.clips:
    tracks = [f for f in tracks if any(c in f for c in a.clips.split(","))]
  tracks = [f for f in tracks if bool(np.load(f, allow_pickle=True)["ok"])]
  region = np.array(PICK_REGION)
  rng = np.random.default_rng(a.seed)
  fo = Follow(policy=Path(a.legs))
  for f in tracks:
    z = np.load(f, allow_pickle=True)
    clip = Path(f).stem.replace("blue_cube_", "")
    for k in range(a.variants):
      if k == 0 and a.seed == 0:
        xy = np.clip(human_to_g1(z["block_start"]), region[:, 0] + 0.01, region[:, 1] - 0.01)
      else:
        xy = rng.uniform(region[:, 0], region[:, 1])
      yaw = float(rng.uniform(-25, 25))
      t0 = time.time()
      try:
        res = fo.run(f, tuple(xy), yaw, record=True)
        err = None
      except RuntimeError as e:
        res, err = {"success": False}, str(e)
      info = {"clip": clip, "seed": a.seed, "variant": k, "xy": np.round(xy, 4).tolist(),
              "yaw_deg": round(yaw, 1), "state_human": str(z["state"]), "error": err,
              "wall_s": round(time.time() - t0, 1), **res}
      with open(OUT / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(info) + "\n")
      print(json.dumps(info), flush=True)
      if res.get("success"):
        r = fo.rec
        np.savez_compressed(OUT / f"{clip}_s{a.seed}_{k}.npz", img=np.array(r["img"], np.uint8),
                            wrist=np.array(r["wrist"], np.uint8),
                            state=np.array(r["state"], np.float32), act=np.array(r["act"], np.float32),
                            q=np.array(r["q"], np.float32), phase=np.array(r["phase"]), task=TASK,
                            info=json.dumps(info))


if __name__ == "__main__":
  main()
