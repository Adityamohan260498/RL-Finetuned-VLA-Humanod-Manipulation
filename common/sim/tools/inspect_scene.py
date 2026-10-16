"""Interactive 3D inspection: the G1 stands on its legs policy at the table, arms in the rest
pose, both Dex3 hands relaxed. Nothing is grasped, no lock.

  .venv_perc/bin/python sim/tools/inspect_scene.py [--policy policy/stance_palmin.onnx]

Viewer keys: Space pause, mouse drag rotate / right-drag pan / scroll zoom, double-click a body to
select, Ctrl+right-drag to push it. C contact points, F contact forces, T transparent, 0-5 geom groups.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stance"))
from stance_runner import CONTROL_DT, StanceRobot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=str(ROOT / "policy/stance_palmin.onnx"))
a = ap.parse_args()

bot = StanceRobot(policy=Path(a.policy))
bot.reset()
paused = [False]


def key(k):
  if k == 32:  # space
    paused[0] = not paused[0]


with mujoco.viewer.launch_passive(bot.model, bot.data, key_callback=key) as v:
  v.cam.lookat[:] = bot.data.body("block").xpos
  v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.6, 150, -30
  while v.is_running():
    t0 = time.time()
    if not paused[0]:
      with v.lock():
        bot.step()
    v.sync()
    time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
