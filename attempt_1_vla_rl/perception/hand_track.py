"""Phone video -> the three hand signals the robot follows, in the TABLE frame (automatic):
pinch path (3D), hand yaw, grip; plus grasp / release times and the block's start / end.

  .venv_perc/bin/python perception/hand_track.py data/raw/blue_cube/*.mp4 [--overlay]

Table frame (from ArUco ids 0, 1, 3 on the table, camera intrinsics data/calib/calib.npz):
x forward into the table from where I stand, y to my left, z up from the table top; origin at the
middle of the table's near (right-in-image) edge.

Writes data/tracks/<clip>.npz:
  t (T,)              s
  pinch (T,3)         midpoint of thumb tip + index tip, m (table frame)
  yaw (T,)            hand yaw, rad: direction wrist -> middle knuckle on the table plane
  aperture (T,)       thumb tip - index tip distance, m (MediaPipe's metric hand model)
  grip (T,)           0 open .. 1 closed on the block (from the aperture, see grip_signal)
  valid (T,)          hand found in this frame (other frames interpolated)
  grasp_t, release_t  when the block starts / stops moving
  block_start, block_end (2,), star (2,)   on the table plane, m
  state               "standing" / "lying" (data/block_state.json: earlier silhouette model fit)
How the 3D works without depth: MediaPipe's metric hand model + solvePnP give the hand's position
up to a depth bias (my hand vs MediaPipe's average hand); one scale per clip along the camera rays
puts the pinch at the block's half height at the grasp (the fingers are on the block then).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
MODEL = Path(__file__).resolve().parent / "models/hand_landmarker.task"
OUT = ROOT / "data/tracks"
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_M = 0.06
FALLBACK_SCALE = 1.19  # median depth scale over the clips where the hand is seen at the grasp
HALF_H = {"standing": 0.04, "lying": 0.0175}  # block centre height (3.5 x 3.5 x 8 cm)
THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9
MP_W = 1280
BLOB_W = 960
BLUE_LO, BLUE_HI = (100, 110, 40), (130, 255, 255)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17)]


def load_calib(w, h):
  c = np.load(COMMON / "data/calib/calib.npz")
  K, dist = c["K"].copy(), c["dist"]
  sw, sh = (int(v) for v in c["size"])
  if (sw, sh) != (w, h):
    K[:2] *= w / sw
  return K, dist


class Table:
  """Table plane + frame from ArUco ids 0, 1, 3 (camera frame <-> table frame)."""

  def __init__(self, frames, K, dist):
    det = cv2.aruco.ArucoDetector(ARUCO, cv2.aruco.DetectorParameters())
    s = MARKER_M / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    pos = {}
    for img in frames:
      corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
      if ids is None:
        continue
      for c, i in zip(corners, ids.ravel()):
        ok, _, tv = cv2.solvePnP(obj, c.reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
          pos.setdefault(int(i), []).append(tv.ravel())
    missing = [i for i in (0, 1, 3) if i not in pos]
    if missing:
      raise ValueError(f"ArUco ids {missing} not visible")
    p = {i: np.median(v, axis=0) for i, v in pos.items()}
    n = np.cross(p[1] - p[0], p[3] - p[0])
    n /= np.linalg.norm(n)
    if n @ (-p[0]) < 0:
      n = -n
    x = p[1] - p[3]
    x -= n * (x @ n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    self.origin = 0.5 * (p[3] + (p[0] + p[3] - p[1]))
    self.R = np.column_stack([x, y, n])
    self.n, self.p0, self.K, self.dist = n, p[0], K, dist
    self.marker_gap = float(np.linalg.norm(p[0] - p[1]))

  def to_table(self, pc):
    return (np.asarray(pc) - self.origin) @ self.R

  def ray_hit(self, px, height=0.0):
    """Pixel -> camera-frame point on the plane `height` above the table."""
    u = cv2.undistortPoints(np.array([[px]], np.float64), self.K, self.dist).ravel()
    d = np.array([u[0], u[1], 1.0])
    t = (height + self.n @ self.p0) / (self.n @ d)
    return t * d


def blue_mask(img, scale):
  small = cv2.resize(img, (BLOB_W, int(img.shape[0] / scale)))
  m = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), BLUE_LO, BLUE_HI)
  return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fill(x, valid, win):
  x = np.array(x, float)
  idx = np.arange(len(x))
  cols = x.reshape(len(x), -1)
  for k in range(cols.shape[1]):
    cols[:, k] = np.interp(idx, idx[valid], cols[valid, k])
    if win > 1:
      cols[:, k] = np.convolve(np.pad(cols[:, k], win // 2, mode="edge"), np.ones(win) / win, "valid")
  return cols.reshape(x.shape)


def grip_signal(ap, g_k, r_k, fps):
  """0 = open, 1 = closed on the block. Scaled between the widest opening in the second before the
  grasp and the aperture while holding; forced to 1 between grasp and release (the block moves
  with the hand then, so it is held, whatever MediaPipe's fingertips say)."""
  a0 = max(0, g_k - int(1.0 * fps))
  a_open = float(np.max(ap[a0:g_k + 1]))
  a_hold = float(np.median(ap[g_k:r_k + 1]))
  g = np.clip((a_open - ap) / max(a_open - a_hold, 1e-3), 0, 1)
  g[g_k:r_k + 1] = 1.0
  return g


def process(path, overlay):
  cap = cv2.VideoCapture(str(path))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  K, dist = load_calib(w, h)
  first = []
  for _ in range(min(20, n)):
    ok, img = cap.read()
    if ok:
      first.append(img)
  table = Table(first[::4], K, dist)
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  bs = w / BLOB_W
  # star = the compact blue blob that is not the block (the block is the larger one at the start)
  m0 = blue_mask(first[0], bs)
  nb, lab, st, cen = cv2.connectedComponentsWithStats(m0)
  order = sorted(range(1, nb), key=lambda i: -st[i, 4])
  if len(order) < 2:
    raise ValueError("block / star not found")
  star_i = order[1]
  star_px = cen[star_i] * bs
  star = table.to_table(table.ray_hit(star_px, 0.0))[:2]
  star_mask = cv2.dilate((lab == star_i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
  # standing vs lying: blue area of the block at the start, in cm^2 on the table plane (the camera
  # sees a standing block's top + one side ~ 3.5 x 6; a lying one ~ 8 x 4.5)
  blk0 = lab == order[0]
  ys, xs = np.nonzero(blk0)
  pts = np.array([table.to_table(table.ray_hit((x * bs, y * bs), 0.0))[:2]
                  for x, y in zip(xs[::7], ys[::7])])
  hull = cv2.convexHull(pts.astype(np.float32) * 1000)
  area_cm2 = cv2.contourArea(hull) / 100.0

  opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
    min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4)
  pinch_c = np.zeros((n, 3))
  ap = np.zeros(n)
  valid = np.zeros(n, bool)
  px2d = np.zeros((n, 21, 2))
  block_px = np.full((n, 2), np.nan)
  smalls = []
  prev = None
  mh = int(h * MP_W / w)
  with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
    for k in range(n):
      ok, img = cap.read()
      if not ok:
        n = k
        break
      small = cv2.resize(img, (MP_W, mh), interpolation=cv2.INTER_AREA)
      if overlay and k % 2 == 0:
        smalls.append(small)
      m = blue_mask(img, bs)
      m[star_mask] = 0
      nb, _, st, cen = cv2.connectedComponentsWithStats(m)
      if nb > 1:
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i, 4] > 2000:
          block_px[k] = cen[i] * bs
      res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
                                int(1000 * k / fps))
      if not res.hand_landmarks:
        prev = None
        continue
      px = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
      wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
      if prev is not None:
        ok, rv, tv = cv2.solvePnP(wl, px, K, dist, prev[0].copy(), prev[1].copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
      else:
        ok, rv, tv = cv2.solvePnP(wl, px, K, dist, flags=cv2.SOLVEPNP_SQPNP)
      if not ok or tv[2, 0] <= 0.05:
        prev = None
        continue
      prev = (rv, tv)
      R, _ = cv2.Rodrigues(rv)
      pinch_c[k] = R @ (0.5 * (wl[THUMB_TIP] + wl[INDEX_TIP])) + tv.ravel()
      ap[k] = np.linalg.norm(wl[THUMB_TIP] - wl[INDEX_TIP])
      px2d[k] = px
      valid[k] = True
  pinch_c, ap, valid, px2d, block_px = (a[:n] for a in (pinch_c, ap, valid, px2d, block_px))
  t = np.arange(n) / fps
  if valid.sum() < 10:
    raise ValueError("hand not found")

  # block on the table + its motion -> grasp / release
  bvalid = np.isfinite(block_px[:, 0])
  state = json.loads((COMMON / "data/block_state.json").read_text())["state"][Path(path).stem]
  cls = {"standing_range": "-", "lying_range": "-"}
  hh = HALF_H[state]
  block = np.zeros((n, 2))
  for k in np.nonzero(bvalid)[0]:
    block[k] = table.to_table(table.ray_hit(block_px[k], hh))[:2]
  block = fill(block, bvalid, 9)
  q = max(3, int(0.4 * fps))
  start, end = np.median(block[:q], axis=0), np.median(block[-q:], axis=0)
  moved = np.linalg.norm(block - start, axis=1)
  left = np.linalg.norm(block - end, axis=1)
  thr = 0.015
  if not np.any(moved > thr):
    raise ValueError("block never moves")
  g_k = int(np.argmax(moved > thr))
  after = np.nonzero(left[g_k:] < thr)[0]
  if not len(after):
    raise ValueError("block never settles")
  r_k = g_k + int(after[0])
  ap_s = fill(ap, valid, 7)
  a = max(0, g_k - int(0.6 * fps))
  g_k = a + int(np.argmin(ap_s[a:g_k + 1]))  # the fingers close a moment before the block moves
  b = min(n - 1, r_k + int(0.6 * fps))
  opened = np.nonzero(ap_s[r_k:b + 1] > ap_s[r_k] + 0.01)[0]
  r_k = r_k + (int(opened[0]) if len(opened) else 0)

  # depth scale: at the grasp the pinch is at the block's half height
  win = int(0.5 * fps)
  ks = sorted([k for k in range(max(0, g_k - win), min(n, g_k + win + 1)) if valid[k]],
              key=lambda k: abs(k - g_k))[:7]
  pinch_px = 0.5 * (px2d[:, THUMB_TIP] + px2d[:, INDEX_TIP])
  if ks:
    scale = float(np.median([np.linalg.norm(table.ray_hit(pinch_px[k], hh)) /
                             max(np.linalg.norm(pinch_c[k]), 1e-6) for k in ks]))
    scale_src = "grasp"
  else:  # hand hidden at the grasp: the median of the other clips (same hand, same camera)
    scale, scale_src = FALLBACK_SCALE, "median of other clips"
  pinch = table.to_table(scale * fill(pinch_c, valid, 7))
  pinch[:, 2] = np.maximum(pinch[:, 2], 0.0)
  # hand yaw: wrist and middle knuckle pixels back-projected to the plane at the pinch height
  yaw = np.zeros(n)
  for k in np.nonzero(valid)[0]:
    zk = float(np.clip(pinch[k, 2], 0.0, 0.3))
    pw = table.to_table(table.ray_hit(px2d[k, WRIST], zk))
    pm = table.to_table(table.ray_hit(px2d[k, MIDDLE_MCP], zk))
    yaw[k] = np.arctan2(pm[1] - pw[1], pm[0] - pw[0])
  yaw[valid] = np.unwrap(yaw[valid])
  yaw = fill(yaw, valid, 7)
  grip = grip_signal(ap_s, g_k, r_k, fps)

  # quality: at the grasp the pinch must be ON the block (position from the block blob)
  grasp_gap = float(np.linalg.norm(pinch[g_k, :2] - start))
  ok = bool(grasp_gap < 0.035 and abs(pinch[g_k, 2] - hh) < 0.03 and valid[max(0, g_k - 15):g_k + 16].mean() > 0.5)
  OUT.mkdir(parents=True, exist_ok=True)
  name = Path(path).parent.name + "_" + Path(path).stem
  np.savez(OUT / f"{name}.npz", t=t, pinch=pinch, yaw=yaw, aperture=ap_s, grip=grip, valid=valid,
           grasp_t=t[g_k], release_t=t[r_k], block_start=start, block_end=end, star=star,
           state=state, block_area_cm2=area_cm2, depth_scale=scale,
           scale_src=scale_src, grasp_gap=grasp_gap, ok=ok, fps=fps, px2d=px2d,
           R_table=table.R, origin=table.origin, marker_gap=table.marker_gap)
  cm = lambda v: np.round(100 * np.asarray(v), 1)  # noqa: E731
  print(f"{name}: {'OK ' if ok else 'BAD'} gap {100 * grasp_gap:4.1f}cm {n / fps:4.1f}s hand {100 * valid.mean():3.0f}% "
        f"{state:8s} "
        f"block {cm(start)} -> {cm(end)} star {cm(star)} miss {100 * np.linalg.norm(end - star):.1f} cm | "
        f"grasp {t[g_k]:.2f}s release {t[r_k]:.2f}s | pinch@grasp {cm(pinch[g_k])} lift "
        f"{100 * (pinch[g_k:r_k + 1, 2].max() - pinch[g_k, 2]):.1f} cm | yaw@grasp "
        f"{np.degrees(yaw[g_k]):+.0f} | ap {100 * ap_s[g_k]:.1f}/{100 * ap_s.max():.1f} cm | depth x{scale:.2f} ({scale_src})",
        flush=True)
  if overlay:
    write_overlay(OUT / f"{name}_overlay.mp4", smalls, px2d * MP_W / w, valid, grip, t, fps, pinch,
                  np.degrees(yaw))


def write_overlay(path, frames, px, valid, grip, t, fps, pinch, yaw_deg):
  h, w = frames[0].shape[:2]
  vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (w, h))
  for i, im in enumerate(frames):
    k = 2 * i
    im = im.copy()
    if valid[k]:
      p = px[k].astype(int)
      for a, b in EDGES:
        cv2.line(im, tuple(p[a]), tuple(p[b]), (0, 255, 0), 2)
      c = ((p[THUMB_TIP] + p[INDEX_TIP]) / 2).astype(int)
      cv2.circle(im, tuple(c), 9, (0, 0, int(255 * grip[k])) if grip[k] > 0.5 else (255, 200, 0), -1)
    txt = (f"t {t[k]:5.2f}s grip {grip[k]:.2f} yaw {yaw_deg[k]:+4.0f} pinch "
           + " ".join(f"{100 * v:5.1f}" for v in pinch[k]) + " cm")
    cv2.putText(im, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(im, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    vw.write(im)
  vw.release()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("videos", nargs="+")
  ap.add_argument("--overlay", action="store_true")
  a = ap.parse_args()
  for v in a.videos:
    try:
      process(v, a.overlay)
    except ValueError as e:
      print(f"{v}: SKIP ({e})", flush=True)


if __name__ == "__main__":
  main()
