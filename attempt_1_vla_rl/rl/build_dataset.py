"""LeRobot dataset for one AWR training round: each source (BC demo dir or rollout dir) is converted
ONCE into its own LeRobot dataset (cached under data/lerobot/parts/), then all are aggregated, and the
per-frame AWR sampling weights are written to <out>/awr_weights.npy (frame order = aggregated order).

  python rl/build_dataset.py --out data/lerobot/rl_it1 --bc data/human_demos_v2 --roll data/rollouts/it1 [...]
      [--bc-share 0.4] [--success-only-bc]
BC demo frames: weight 1 each, then the BC part is rescaled to --bc-share of the total sampling mass.
Rollout frames: rl/advantage.py weights (<file>.w.npy).
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--bc", nargs="*", default=[])
ap.add_argument("--roll", nargs="*", default=[])
ap.add_argument("--bc-share", type=float, default=0.4)
a = ap.parse_args()

PARTS = ROOT / "data/lerobot/parts"
PARTS.mkdir(parents=True, exist_ok=True)


def part(src, pattern):
  name = str(Path(src)).strip("/").replace("/", "__")
  root = PARTS / name
  n = len(glob.glob(f"{src}/{pattern}"))
  stamp = root / "SOURCE_COUNT"
  if not (stamp.exists() and int(stamp.read_text()) == n):
    subprocess.run([sys.executable, str(ROOT / "sim/to_lerobot.py"), "--src", src, "--root", str(root),
                    "--repo-id", f"local/{name}", "--pattern", pattern], check=True)
    stamp.write_text(str(n))
  files = sorted(glob.glob(f"{src}/{pattern}"))
  return root, f"local/{name}", files


DROP = ["settle", "to_start", "to_rest"]  # = sim/to_lerobot.py --drop default (planned segments)


def n_kept(f):
  z = np.load(f, allow_pickle=True)
  return int((~np.isin(z["phase"], DROP)).sum()) if "phase" in z.files else len(z["state"])


roots, ids, weights = [], [], []
bc_w, roll_w = [], []
for src in a.bc:
  r, i, files = part(src, "*.npz")
  roots.append(r); ids.append(i)
  w = np.concatenate([np.ones(n_kept(f), np.float32) for f in files])
  bc_w.append(len(weights)); weights.append(w)
for src in a.roll:
  r, i, files = part(src, "roll_*.npz")
  roots.append(r); ids.append(i)
  w = np.concatenate([np.load(f.replace(".npz", ".w.npy")) for f in files])
  roll_w.append(len(weights)); weights.append(w)
if bc_w and roll_w:
  m_bc = sum(weights[j].sum() for j in bc_w)
  m_roll = sum(weights[j].sum() for j in roll_w)
  s = a.bc_share / (1 - a.bc_share) * m_roll / m_bc
  for j in bc_w:
    weights[j] = weights[j] * s
from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402

out = Path(a.out)
if out.exists():
  shutil.rmtree(out)
aggregate_datasets(ids, f"local/{out.name}", roots=roots, aggr_root=out)
W = np.concatenate(weights)
np.save(out / "awr_weights.npy", W)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

meta = LeRobotDatasetMetadata(f"local/{out.name}", root=out)
assert meta.total_frames == len(W), (meta.total_frames, len(W))
print(json.dumps({"out": str(out), "frames": int(len(W)), "episodes": int(meta.total_episodes),
                  "parts": [str(r) for r in roots]}))
