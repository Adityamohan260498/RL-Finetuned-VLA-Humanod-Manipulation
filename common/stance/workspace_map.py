"""Stability map of the right-hand workspace with the leg policy (stance_palmin.onnx).

For each point of a grid on the table: reset, plan the arm (palm-in, fingers horizontal, as in both
attempts) to 12 cm above the point, straight descent to grasp height, hold 3 s. Record how well the
grasp centre reaches the point and how much the legs had to move to keep balance.

  stable   : reached (< 1 cm) and both feet stay put (max foot slip < 2 cm)
  partial  : reached (< 2 cm) but the feet shuffle / step (slip 2-10 cm), robot standing
  no       : not reachable, a big step (> 10 cm), or a fall

  ../.venv_perc/bin/python common/stance/workspace_map.py --workers 12   (from the repo root)
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# the arm motion code (planner + closed-loop IK) used by attempt 1; attempt 2 has an identical copy
sys.path.insert(0, str(REPO / "attempt_1_vla_rl/sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--step", type=float, default=0.05)
ap.add_argument("--only", nargs="*", default=None, help="x,y points to test")
ap.add_argument("--out", default=str(REPO / "common/data/workspace_map.json"))
a = ap.parse_args()

XS = np.round(np.arange(0.20, 0.4501, a.step), 3)
YS = np.round(np.arange(-0.35, 0.1501, a.step), 3)


def run(xy):
  import robot_motion as M
  mo = M.Motion()
  bot = mo.bot
  bot.reset()
  mo.worst = {"fall": False, "table": 0, "drift": 0.0, "tilt": 0.0}
  mo.hand_table_steps = 0
  mo.place_block((0.30, 0.32), 0.0)  # block parked out of the right arm's way
  bot.set_posture(waist_yaw=M.TASK_WAIST_YAW)
  bot.set_hand_joints("right", M.DEX3_OPEN)
  mo.hold(1.5)
  p = np.array([*xy, M.TABLE_TOP + M.GRASP_H])
  res = {"x": xy[0], "y": xy[1]}
  feet0, base0 = bot.data.site_xpos[bot.feet, :2].copy(), bot.data.qpos[0:2].copy()  # after settling

  def moved():  # how far the feet / pelvis moved since the reach started
    return (np.linalg.norm(bot.data.site_xpos[bot.feet, :2] - feet0, axis=1).max(),
            np.linalg.norm(bot.data.qpos[0:2] - base0))
  try:
    above = p + [0, 0, 0.12]
    path = mo.plan_to(above)
    mo.run_q(path, [0.4] * (len(path) - 1))
    mo.servo(above, 0.6)
    mo.track(above, p, 1.5)
    slip = drift = 0.0
    for _ in range(int(3.0 / M.CONTROL_DT)):
      mo.servo(p, M.CONTROL_DT)
      sl, dr = moved()
      slip, drift = max(slip, sl), max(drift, dr)
      if bot.stability()["fallen"]:
        break
    slip = max(slip, moved()[0])  # includes any step taken during the approach
    s = bot.stability()
    res.update(err_cm=round(mo.tip_err(p) * 100, 2), slip_cm=round(slip * 100, 1),
               drift_cm=round(drift * 100, 1), fallen=s["fallen"])
  except Exception as e:  # no IK / no collision-free path
    res.update(err_cm=None, error=str(e)[:80])
  e, sl = res.get("err_cm"), res.get("slip_cm", 99)
  if e is None or res.get("fallen"):
    res["cls"] = "no"
  elif e < 1 and sl < 2:
    res["cls"] = "stable"
  elif e < 2 and sl < 10:
    res["cls"] = "partial"
  else:
    res["cls"] = "no"
  print(res, flush=True)
  return res


if __name__ == "__main__":
  pts = [(float(x), float(y)) for x in XS for y in YS]
  if a.only:
    [run(tuple(map(float, q.split(",")))) for q in a.only]
    sys.exit()
  with Pool(a.workers) as pool:
    out = pool.map(run, pts, chunksize=1)
  Path(a.out).write_text(json.dumps(out, indent=1))
  sym = {"stable": "#", "partial": "+", "no": "."}
  print("rows = x (forward from the pelvis, m), cols = y (left +), # stable  + partial  . no")
  print("       " + " ".join(f"{y:+.2f}" for y in YS))
  for x in XS[::-1]:
    print(f"{x:.2f}   " + " ".join(f"  {sym[r['cls']]}  " for r in out if r["x"] == x))
