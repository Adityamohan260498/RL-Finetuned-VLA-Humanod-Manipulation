"""Successful sim demos (sim/make_demos.py, data/sim_demos/*.npz) -> a LeRobot dataset.

  .venv_vla/bin/python sim/to_lerobot.py [--src data/sim_demos] [--root data/lerobot/g1_dex3_block_star]

Per frame (25 Hz):
  observation.images.head  224x224 RGB, the G1's head camera (the only camera)
  observation.state        14: right-arm joint angles (7) + right Dex3 joint angles (7), rad
  action                   14: right-arm joint targets (7) + Dex3 joint targets (7), rad
  task                     "pick up the blue block and put it on the blue star"
The legs / waist are the stance policy; the left arm holds its rest pose.
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
ap.add_argument("--src", default=str(ROOT / "data/sim_demos"))
ap.add_argument("--root", default=str(ROOT / "data/lerobot/g1_dex3_block_star"))
ap.add_argument("--repo-id", default="local/g1_dex3_block_star")
ap.add_argument("--holdout", default="", help="comma list of clip ids kept OUT of training")
a = ap.parse_args()

hold = [h for h in a.holdout.split(",") if h]
files = sorted(f for f in glob.glob(f"{a.src}/*.npz") if not any(h in f for h in hold))
if not files:
  raise SystemExit("no demos")
if Path(a.root).exists():
  shutil.rmtree(a.root)
features = {
  "observation.images.head": {"dtype": "image", "shape": (224, 224, 3),
                              "names": ["height", "width", "channels"]},
  "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
  "action": {"dtype": "float32", "shape": (14,), "names": None},
}
ds = LeRobotDataset.create(a.repo_id, fps=25, features=features, root=a.root,
                           robot_type="unitree_g1_dex3", use_videos=False,
                           image_writer_processes=4, image_writer_threads=4)
n_frames, states = 0, {"standing": 0, "lying": 0}
for f in files:
  z = np.load(f, allow_pickle=True)
  z = {k: z[k] for k in z.files}  # load once (indexing an NpzFile re-decompresses)
  info = json.loads(str(z["info"]))
  states[info["state"]] = states.get(info["state"], 0) + 1
  task = str(z["task"])
  for k in range(len(z["q"])):
    ds.add_frame({"observation.images.head": z["img"][k],
                  "observation.state": z["q"][k].astype(np.float32),
                  "action": z["act"][k].astype(np.float32), "task": task})
  ds.save_episode()
  n_frames += len(z["q"])
ds.finalize() if hasattr(ds, "finalize") else None
print(f"{len(files)} episodes ({states}), {n_frames} frames -> {a.root}")
