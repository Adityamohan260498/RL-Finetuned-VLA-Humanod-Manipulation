"""lerobot-train with AWR prioritized sampling (paper sec. 2.3): frame i is drawn with probability
proportional to <dataset_root>/awr_weights.npy[i] (with replacement); the loss stays plain flow matching.
Same CLI as lerobot-train."""
import os
import sys

import numpy as np
import torch

_DL = torch.utils.data.DataLoader


class AWRDataLoader(_DL):
  def __init__(self, dataset, *args, sampler=None, shuffle=None, **kw):
    if kw.get("batch_sampler") is not None:  # accelerate re-wrapping an existing loader: keep its sampler
      super().__init__(dataset, *args, **kw)
      return
    root = getattr(dataset, "root", None)
    wf = os.path.join(str(root), "awr_weights.npy") if root is not None else ""
    if sampler is None and wf and os.path.exists(wf):
      w = np.load(wf).astype(np.float64)
      assert len(w) == len(dataset), (len(w), len(dataset))
      sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True)
      shuffle = False
      print(f"[awr] weighted sampler over {len(w)} frames, eff. sample size {w.sum()**2 / (w**2).sum():.0f}", flush=True)
    super().__init__(dataset, *args, sampler=sampler, shuffle=shuffle, **kw)


torch.utils.data.DataLoader = AWRDataLoader
from lerobot.scripts import lerobot_train  # noqa: E402

lerobot_train.torch.utils.data.DataLoader = AWRDataLoader
sys.exit(lerobot_train.main())
