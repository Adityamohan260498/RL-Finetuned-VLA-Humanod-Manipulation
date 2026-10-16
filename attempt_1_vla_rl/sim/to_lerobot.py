"""Robot-follows-my-hand demos (sim/make_human_demos.py, data/human_demos/*.npz) -> a LeRobot dataset.

  .venv_vla/bin/python sim/to_lerobot.py [--src data/human_demos] [--root data/lerobot/g1_dex3_human]

Per frame (25 Hz):
  observation.images.head  224x224 RGB, the G1's head camera (the only camera)
  observation.state        5: grasp centre x, y, z (m, table/world frame), hand yaw (rad), grip 0..1
  action                   5: dx, dy, dz (m), dyaw (rad) = target - current; grip command 0..1
  task                     "pick up the blue block and put it on the blue star"
Hand-pose actions (not joint angles): the legs / torso pose can change without retraining the VLA.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "data/human_demos"))
ap.add_argument("--root", default=str(ROOT / "data/lerobot/g1_dex3_human"))
ap.add_argument("--repo-id", default="local/g1_dex3_human")
ap.add_argument("--pattern", default="*.npz")
ap.add_argument("--drop", default="settle,to_start,to_rest",
                help="phases left out: the robot's own planned moves (rest <-> start of my motion)")
a = ap.parse_args()

DROP = [x for x in a.drop.split(",") if x]
files = sorted(glob.glob(f"{a.src}/{a.pattern}"))
if not files:
  raise SystemExit("no demos")
WRIST = "wrist" in np.load(files[0], allow_pickle=True).files  # wrist camera (from 2026-10-15 demos v4)
if Path(a.root).exists():
  shutil.rmtree(a.root)
features = {
  "observation.images.head": {"dtype": "image", "shape": (224, 224, 3),
                              "names": ["height", "width", "channels"]},
  **({"observation.images.wrist": {"dtype": "image", "shape": (224, 224, 3),
                                   "names": ["height", "width", "channels"]}} if WRIST else {}),
  "observation.state": {"dtype": "float32", "shape": (5,), "names": None},
  "action": {"dtype": "float32", "shape": (5,), "names": None},
}
ds = LeRobotDataset.create(a.repo_id, fps=25, features=features, root=a.root,
                           robot_type="unitree_g1_dex3", use_videos=False,
                           image_writer_processes=4, image_writer_threads=4)
n_frames, clips = 0, {}
for f in files:
  z = np.load(f, allow_pickle=True)
  z = {k: z[k] for k in z.files}
  info = json.loads(str(z["info"]))
  clips[info["clip"]] = clips.get(info["clip"], 0) + 1
  task = str(z["task"])
  act, keep = z["act"], np.ones(len(z["state"]), bool)
  if "phase" in z:  # demos (rollouts have none: their actions are the VLA's own, all kept)
    # The planned segments' recorded "command" is the planner's final goal (22-55 cm, up to 2 rad away)
    # and the palm-in rest orientation is not a hand_R(0, yaw) pose, so a 4-D hand-pose command cannot
    # express them: replayed through the eval controller 0/6 demos succeed. The eval does the same
    # scripted planned move to a ready pose instead (sim/eval_vla.py --ready); the VLA starts there.
    keep = ~np.isin(z["phase"], DROP)
  for k in np.flatnonzero(keep):
    ds.add_frame({"observation.images.head": z["img"][k],
                  **({"observation.images.wrist": z["wrist"][k]} if WRIST else {}),
                  "observation.state": z["state"][k].astype(np.float32),
                  "action": act[k].astype(np.float32), "task": task})
  ds.save_episode()
  n_frames += int(keep.sum())
ds.finalize() if hasattr(ds, "finalize") else None
print(f"{len(files)} episodes from {len(clips)} clips {clips}, {n_frames} frames -> {a.root}")
