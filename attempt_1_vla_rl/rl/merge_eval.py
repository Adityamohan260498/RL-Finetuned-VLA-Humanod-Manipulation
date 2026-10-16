"""Merge sim/eval_vla.py worker outputs (<out>.w<k>.json) into <out> with one summary.
  python rl/merge_eval.py data/eval/x.json [label]"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

out = sys.argv[1]
eps = []
for f in sorted(glob.glob(out.replace(".json", ".w*.json"))):
  eps += json.load(open(f))["episodes"]
eps.sort(key=lambda r: r["ep"])
n = len(eps)
s = {"label": sys.argv[2] if len(sys.argv) > 2 else Path(out).stem, "episodes": n,
     "success": round(sum(r["success"] for r in eps) / n, 3), "lifted": round(sum(r["lifted"] for r in eps) / n, 3),
     "fallen": round(sum(r["fallen"] for r in eps) / n, 3),
     "median_place_err_cm": float(np.median([r["place_err_cm"] for r in eps])),
     "stage_reach": [round(float(np.mean([r.get("max_stage", 0) >= k for r in eps])), 3) for k in range(5)]}
Path(out).write_text(json.dumps({"summary": s, "episodes": eps}, indent=1))
print(json.dumps(s))
