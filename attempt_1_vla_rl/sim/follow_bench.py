"""Fixed benchmark for the retargeting settings: the same 24 (clip, block pose) cases every time,
in parallel; prints the success rate + failure types. Settings come from env vars read by
sim/human_follow.py (CLEAR, ...).

  OMP_NUM_THREADS=1 .venv_perc/bin/python sim/follow_bench.py --name base [--procs 12]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLIPS = "111803 111826 111847 111909 111929 111944 112022 112044 112121 112232 112255 112337".split()


def cases():
  rng = np.random.default_rng(7)
  out = []
  for c in CLIPS:
    for _ in range(2):
      out.append((c, (float(rng.uniform(0.20, 0.35)), float(rng.uniform(-0.23, -0.11))), float(rng.uniform(-25, 25))))
  return out


def run(case):
  sys.path.insert(0, str(ROOT / "sim"))
  from human_follow import Follow
  c, xy, yaw = case
  global FO
  if "FO" not in globals():
    FO = Follow()
  try:
    r = FO.run(str(ROOT / f"data/tracks/blue_cube_20261011_{c}.npz"), xy, yaw)
    r["error"] = None
  except RuntimeError as e:
    r = {"success": False, "error": str(e)[:80]}
  r.update(clip=c, xy=xy, yaw=yaw)
  return r


def kind(d):
  if d["success"]:
    return "OK"
  if d["error"]:
    return "error: " + d["error"][:30]
  g, r = d.get("at_grasp", {}), d.get("at_release", {})
  if g.get("upright", 1) < 0.9 or g.get("moved_cm", 0) > 1.5:
    return "knocked before grasp"
  if r.get("upright", 1) < 0.9 or r.get("z_cm", 4) < 3:
    return "dropped in carry"
  if d.get("upright", 1) < 0.9:
    return "tipped at release"
  return "off target / other"


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--name", required=True)
  ap.add_argument("--procs", type=int, default=12)
  a = ap.parse_args()
  with mp.Pool(a.procs) as pool:
    res = pool.map(run, cases(), chunksize=1)
  out = ROOT / "logs" / f"bench_{a.name}.jsonl"
  out.write_text("".join(json.dumps(r) + "\n" for r in res))
  from collections import Counter
  k = Counter(kind(d) for d in res)
  env = {v: os.environ.get(v) for v in ("CLEAR", "YAW_LIM_DEG", "REL_PAUSE", "APPROACH") if os.environ.get(v)}
  print(f"{a.name} {env}: {k['OK']}/{len(res)} OK | " + ", ".join(f"{n} {x}" for x, n in k.most_common() if x != "OK"))
