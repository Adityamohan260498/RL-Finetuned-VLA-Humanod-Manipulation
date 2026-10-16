"""Re-run one recorded attempt (clip, xy, yaw) and trace the grasp: phase, block height / tilt,
block offset in the wrist frame (slip), grip forces, hand speed. Finds WHEN and HOW the block is lost.
  .venv_perc/bin/python sim/diag/trace_attempt.py <clip> <x> <y> <yaw> [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(COMMON / "scene"))
from human_follow import Follow  # noqa: E402
from layout import DROP_XY, TABLE_TOP  # noqa: E402

clip, x, y, yaw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
fo = Follow()
rows, prev = [], {"p": None}


def hook(m):
  if m.k_step % 10:
    return
  pb, up = m.block_state()
  p, _ = m.ee()
  v = 0.0 if prev["p"] is None else float(np.linalg.norm(p - prev["p"]) / (10 * 0.01))
  prev["p"] = p.copy()
  f = m.grip_forces()
  d = m.bot.data
  Rw = d.xmat[m.bot.palms[1]].reshape(3, 3)
  Rb = d.xmat[m.block].reshape(3, 3)
  rel = Rw.T @ Rb  # block orientation in the wrist frame: changes = rotation in the grip
  rows.append({"hand_zw": Rw[2].round(3).tolist(), "rel_z": rel[:, 2].round(3).tolist(),"t": round(m.bot.t, 2), "ph": m.phase, "bxy": (100 * (pb[:2] - np.array(DROP_XY))).round(1).tolist(), "hxy": (100 * (p[:2] - np.array(DROP_XY))).round(1).tolist(), "bz": round(100 * (pb[2] - TABLE_TOP), 1),
               "up": round(up, 2), "inhand": (100 * m.block_in_hand()).round(1).tolist(),
               "F": {k: round(v_, 1) for k, v_ in f.items()}, "v": round(v, 2),
               "hz": round(100 * (p[2] - TABLE_TOP), 1)})


fo.hook = hook
res = fo.run(str(ROOT / f"data/tracks/blue_cube_{clip}.npz"), (x, y), yaw)
out = {"res": res, "rows": rows}
if len(sys.argv) > 5:
  Path(sys.argv[5]).write_text(json.dumps(out))
for r in rows:
  if r["ph"] not in ("settle", "to_start"):
    print(r)
print(json.dumps(res))
