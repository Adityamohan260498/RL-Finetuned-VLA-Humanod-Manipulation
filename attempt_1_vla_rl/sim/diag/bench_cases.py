"""Re-run a fixed set of (clip, xy, yaw) cases in parallel with the current env settings.
  .venv_perc/bin/python sim/diag/bench_cases.py cases.json out.jsonl [procs]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data


def one(c):
  sys.path.insert(0, str(ROOT / "sim"))
  sys.path.insert(0, str(COMMON / "scene"))
  from human_follow import Follow
  fo = Follow()
  try:
    r = fo.run(str(ROOT / f"data/tracks/blue_cube_{c['clip']}.npz"), tuple(c["xy"]), c["yaw_deg"])
  except RuntimeError as e:
    r = {"success": False, "error": str(e)}
  return {**c, **{k: r.get(k) for k in ("success", "place_err_cm", "upright", "at_grasp", "at_release", "error")}}


if __name__ == "__main__":
  cases = json.load(open(sys.argv[1]))
  with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 10) as p:
    res = p.map(one, cases, chunksize=1)
  Path(sys.argv[2]).write_text("\n".join(json.dumps(r) for r in res))
  print(sys.argv[2], "success", sum(r["success"] for r in res), "/", len(res))
