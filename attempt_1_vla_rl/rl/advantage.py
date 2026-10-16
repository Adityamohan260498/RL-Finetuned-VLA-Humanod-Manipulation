"""Per-frame advantages for VLA rollouts (paper 2606.27163 sec. 6, simplified: the sim gives the
progress stages directly, so no learned value head).

Stages (sim/eval_vla.py stage_now, privileged state, monotone): 0 start, 1 hand at the grasp point,
2 block lifted, 3 lifted block over the star, 4 placed upright on the star (hand open); outcome y = success.
V(k) = P(success | episode reached stage k), estimated over the given rollouts (a GRPO-like empirical
baseline). A frame in stage k gets   A = V(k + 1) - V(k)  if the episode later reached k + 1, else  y - V(k)
(segment advantage: progress before a failure is positive, the segment where it got stuck negative).
A is divided by its std, clipped to [-2, 2]; AWR sampling weight w = exp(A).
Event credit (when the rollout recorded the block's uprightness `up`): if the block is knocked over
(up < 0.5) before it was lifted, the TIP_S seconds before that moment get the minimum A (-2): inside the
"reached" segment those are the frames that caused the failure, the rest of the segment keeps the
segment value.

  python rl/advantage.py <rollout_dir> [<rollout_dir> ...]   -> writes <file>.w.npy next to each rollout
"""
import json
import sys
from pathlib import Path

import numpy as np

files = sorted(f for d in sys.argv[1:] for f in Path(d).glob("roll_*.npz"))
eps = []
for f in files:
  z = np.load(f, allow_pickle=True)
  info = json.loads(str(z["info"]))
  up = z["up"] if "up" in z.files else None
  eps.append((f, z["stage"].astype(int), float(info["success"]), up))
n_st = 5
V = np.zeros(n_st)
for k in range(n_st):
  ys = [y for _, st, y, _ in eps if st.max() >= k]
  V[k] = np.mean(ys) if ys else 0.0
TIP_S, FPS = 1.0, 25
A_all, tips = [], []
for f, st, y, up in eps:
  A = np.zeros(len(st))
  for t, k in enumerate(st):
    A[t] = (V[k + 1] - V[k]) if (k + 1 < n_st and st.max() > k) else (y - V[k])
  tip = None
  if up is not None:
    down = np.flatnonzero((up < 0.5) & (st < 2))
    tip = int(down[0]) if len(down) else None
  A_all.append(A)
  tips.append(tip)
sd = float(np.std(np.concatenate(A_all))) + 1e-6 if A_all else 1.0
for (f, st, y, up), A, tip in zip(eps, A_all, tips):
  An = np.clip(A / sd, -2, 2)
  if tip is not None:
    An[max(0, tip - int(TIP_S * FPS)):tip + 1] = -2.0
  np.save(str(f).replace(".npz", ".w.npy"), np.exp(An).astype(np.float32))
succ = np.mean([e[2] for e in eps]) if eps else 0
reach = [float(np.mean([e[1].max() >= k for e in eps])) for k in range(n_st)]
print(json.dumps({"episodes": len(eps), "success": round(float(succ), 3), "V": V.round(3).tolist(),
                  "reach_rate": np.round(reach, 3).tolist(), "A_std": round(sd, 3),
                  "tipped_before_lift": int(sum(t is not None for t in tips))}))
