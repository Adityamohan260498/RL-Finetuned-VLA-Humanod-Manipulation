"""Phone camera intrinsics from the ChArUco calibration videos.

  .venv_perc/bin/python perception/calibrate.py data/raw/calibration/*.mp4

Board (~/g1/challenge/markers/make_markers.py, printed at 100 %): 7 x 9 squares of 2.5 cm, 1.8 cm
markers, DICT_4X4_50 ids 10-40. The clips are 4K at ~60 fps, the same phone + resolution as the
task clips.

Writes data/calib/:
  calib.npz     K (3x3), dist (5), size (w, h), rms
  report.txt    views used, RMS on the views it was fitted on AND on held-out views, FOV
  check.jpg     a board view with the detected corners (green) and the corners re-projected through
                the calibration (red); they should coincide
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SQUARE, MARKER = 0.025, 0.018  # m
FLAGS = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST


def detect(videos, board, every, min_corners):
  det = cv2.aruco.CharucoDetector(board)
  views, size = [], None
  for video in videos:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    got = 0
    for i in range(0, n, every):
      cap.set(cv2.CAP_PROP_POS_FRAMES, i)
      ok, img = cap.read()
      if not ok:
        break
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      size = gray.shape[::-1]
      cc, ci, _, _ = det.detectBoard(gray)
      if ci is not None and len(ci) >= min_corners:
        obj, imgp = board.matchImagePoints(cc, ci)
        views.append({"obj": obj, "img": imgp, "video": Path(video).name, "frame": i})
        got += 1
    print(f"{Path(video).name}: {got} board views (every {every}th of {n} frames)", flush=True)
  return views, size


def diverse(views, size, thr=0.02):
  """Greedy pick of views whose board centre, apparent size or tilt differ from all kept ones."""
  w, h = size
  K0 = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])  # rough, only for the tilt
  desc = []
  for v in views:
    p = v["img"].reshape(-1, 2)
    _, rv, _ = cv2.solvePnP(v["obj"], v["img"], K0, None)
    n = cv2.Rodrigues(rv)[0][:, 2]
    desc.append(np.r_[p.mean(0) / w, p.std(0) / w, 0.5 * n])
  desc = np.array(desc)
  kept = []
  for i in range(len(views)):
    if all(np.linalg.norm(desc[i] - desc[j]) > thr for j in kept):
      kept.append(i)
  return [views[i] for i in kept]


def view_errors(views, K, dist):
  errs = []
  for v in views:
    ok, rv, tv = cv2.solvePnP(v["obj"], v["img"], K, dist)
    p = cv2.projectPoints(v["obj"], rv, tv, K, dist)[0].reshape(-1, 2)
    errs.append(float(np.sqrt(np.mean(np.sum((p - v["img"].reshape(-1, 2)) ** 2, axis=1)))))
  return np.array(errs)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("videos", nargs="+")
  ap.add_argument("--out", default=str(COMMON / "data/calib"))
  ap.add_argument("--every", type=int, default=10, help="use every Nth frame")
  ap.add_argument("--max-views", type=int, default=60)
  ap.add_argument("--min-corners", type=int, default=12)
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)

  board = cv2.aruco.CharucoBoard((7, 9), SQUARE, MARKER, D, np.arange(10, 10 + 31))
  views, size = detect(a.videos, board, a.every, a.min_corners)
  if len(views) < 16:
    raise SystemExit(f"only {len(views)} usable board views")
  # keep only views that differ: board position / size in the image and its tilt. A clip where the
  # board lies still gives hundreds of copies of one view, which would outweigh everything else.
  div = diverse(views, size)
  rng = np.random.default_rng(0)
  idx = rng.permutation(len(div))
  fit = [div[k] for k in idx[: min(a.max_views, (2 * len(div)) // 3)]]
  held = [div[k] for k in idx[len(fit):]]
  fit_cal = lambda vs: cv2.calibrateCamera([v["obj"] for v in vs], [v["img"] for v in vs], size,  # noqa: E731
                                           None, None, flags=FLAGS)
  rms, K, dist, _, _ = fit_cal(fit)
  # drop blurry / mis-detected views (per-view error > 3x median) and refit once
  e = view_errors(fit, K, dist)
  keep = [v for v, x in zip(fit, e) if x <= 3 * np.median(e)]
  dropped = len(fit) - len(keep)
  if dropped:
    rms, K, dist, _, _ = fit_cal(keep)
  eh = view_errors(held, K, dist)
  # how well the views pin the focal length down: refit on random halves of the diverse views
  boot = np.array([fit_cal([div[k] for k in rng.permutation(len(div))[: len(div) // 2]])[1][[0, 1, 0, 1],
                                                                                         [0, 1, 2, 2]]
                   for _ in range(10)])
  hfov = np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0])))
  vfov = np.degrees(2 * np.arctan(size[1] / (2 * K[1, 1])))
  np.savez(out / "calib.npz", K=K, dist=dist, size=np.array(size), rms=rms,
           fx_std=boot[:, 0].std(), cy_std=boot[:, 3].std())
  per_clip = {c: sum(v["video"] == c for v in div) for c in sorted({v["video"] for v in views})}
  lines = [
    f"videos: {', '.join(Path(v).name for v in a.videos)}",
    f"image size: {size[0]} x {size[1]}",
    f"board views: {len(views)} detected -> {len(div)} distinct ({per_clip}) -> {len(keep)} fitted "
    f"({dropped} dropped as outliers), {len(held)} held out",
    f"lens model: k1, k2 radial (k3 and tangential fixed at 0: the full model overfits these views)",
    f"reprojection RMS: fit views {rms:.2f} px | held-out views median {np.median(eh):.2f} px, "
    f"p90 {np.percentile(eh, 90):.2f} px, max {eh.max():.2f} px",
    f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f} px",
    f"uncertainty (refits on random halves): fx +- {boot[:, 0].std():.0f} px "
    f"({100 * boot[:, 0].std() / K[0, 0]:.1f} %), cx +- {boot[:, 2].std():.0f}, cy +- {boot[:, 3].std():.0f} px",
    f"distortion k1 k2 p1 p2 k3: {np.round(dist.ravel(), 4).tolist()}",
    f"field of view: {hfov:.1f} deg horizontal, {vfov:.1f} deg vertical",
  ]
  (out / "report.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))

  # check image: the held-out view with the median error
  v = held[int(np.argsort(eh)[len(eh) // 2])]
  cap = cv2.VideoCapture(str(next(p for p in a.videos if Path(p).name == v["video"])))
  cap.set(cv2.CAP_PROP_POS_FRAMES, v["frame"])
  _, img = cap.read()
  _, rv, tv = cv2.solvePnP(v["obj"], v["img"], K, dist)
  proj = cv2.projectPoints(v["obj"], rv, tv, K, dist)[0].reshape(-1, 2)
  for p, q in zip(v["img"].reshape(-1, 2), proj):
    cv2.circle(img, tuple(np.round(p).astype(int)), 14, (0, 255, 0), 3)
    cv2.circle(img, tuple(np.round(q).astype(int)), 5, (0, 0, 255), -1)
  cv2.drawFrameAxes(img, K, dist, rv, tv, 0.05, 6)
  cv2.putText(img, f"held-out view {v['video']} frame {v['frame']}: green detected, red "
              f"re-projected, {np.median(eh):.2f} px median", (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
              1.8, (0, 0, 255), 4)
  cv2.imwrite(str(out / "check.jpg"), cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA))
  print("wrote", out)


if __name__ == "__main__":
  main()
