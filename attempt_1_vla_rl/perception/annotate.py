"""Key-frame hand annotation: you place the 12 hand points that map onto the G1 + Dex3, at the key
frames you choose. MediaPipe's guess is shown next to the real frame and pre-places the points.

  .venv_perc/bin/python perception/annotate.py [data/raw/blue_cube/<clip>.mp4 ...]

Window: top-left the real frame (click / drag here), top-right the same frame with MediaPipe's 12
points, bottom-left a 3x magnifier around the selected point, bottom-right status + help.

  a / d  or  <- / ->   previous / next frame          j / l   -10 / +10 frames
  space                play / pause                   k       make this a key frame / remove it
  1..7                 label the key frame: 1 start 2 pregrasp 3 grasp 4 lift 5 place 6 release 7 end
  n / p  (or Tab)      next / previous point          click   put the selected point there (then next)
  drag                 move a point                   right-click  selected point not visible (hidden)
  m                    reset this key frame's points to MediaPipe
  s  save              q  save + next clip            Esc  save + quit

Saves data/annotations/<clip>.json after every change (pixel coordinates of the ORIGINAL 4K video):
  {"clip", "video", "fps", "size", "points": [12 names], "keyframes": [{"frame", "t", "label",
   "points": {name: [x, y] | null}}]}
MediaPipe's 21 landmarks for every frame are cached in data/annotations/cache/<clip>_mp.npz.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT.parent / "common"  # shared robot / scene / stance / planner / raw data
MODEL = Path(__file__).resolve().parent / "models/hand_landmarker.task"
OUT = COMMON / "data/annotations"
# the 12 points used for the embodiment mapping (MediaPipe landmark ids)
POINTS = {"wrist": 0, "thumb_base": 1, "thumb_knuckle": 2, "thumb_joint": 3, "thumb_tip": 4,
          "index_knuckle": 5, "index_joint": 6, "index_tip": 8,
          "middle_knuckle": 9, "middle_joint": 10, "middle_tip": 12, "pinky_knuckle": 17}
NAMES = list(POINTS)
BONES = [("wrist", "thumb_base"), ("thumb_base", "thumb_knuckle"), ("thumb_knuckle", "thumb_joint"),
         ("thumb_joint", "thumb_tip"), ("wrist", "index_knuckle"), ("index_knuckle", "index_joint"),
         ("index_joint", "index_tip"), ("wrist", "middle_knuckle"),
         ("middle_knuckle", "middle_joint"), ("middle_joint", "middle_tip"),
         ("wrist", "pinky_knuckle"), ("index_knuckle", "middle_knuckle"),
         ("middle_knuckle", "pinky_knuckle")]
COLOR = {"wrist": (255, 255, 255), "thumb": (0, 140, 255), "index": (0, 220, 0),
         "middle": (255, 80, 80), "pinky": (200, 0, 200)}
LABELS = {ord("1"): "start", ord("2"): "pregrasp", ord("3"): "grasp", ord("4"): "lift",
          ord("5"): "place", ord("6"): "release", ord("7"): "end"}
PW, PH = 960, 540  # each top panel
PROXY_W = 1280  # frames kept in memory / given to MediaPipe at this width
KEY_LEFT, KEY_RIGHT = {81, 65361, 2424832}, {83, 65363, 2555904}


def col(name):
  return COLOR[name.split("_")[0]]


def mediapipe_all(frames, fps, size):
  """21 landmarks per frame in ORIGINAL-video pixels (NaN where no hand)."""
  opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
    min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4)
  lm = np.full((len(frames), 21, 2), np.nan)
  with mp.tasks.vision.HandLandmarker.create_from_options(opts) as det:
    for k, jpg in enumerate(frames):
      img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
      res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                                 int(1000 * k / fps))
      if res.hand_landmarks:
        lm[k] = [[p.x * size[0], p.y * size[1]] for p in res.hand_landmarks[0]]
  return lm


def load_clip(video):
  cap = cv2.VideoCapture(str(video))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  ph = int(size[1] * PROXY_W / size[0])
  frames, t0 = [], time.time()
  while True:
    ok, img = cap.read()
    if not ok:
      break
    small = cv2.resize(img, (PROXY_W, ph), interpolation=cv2.INTER_AREA)
    frames.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1])
  name = video.parent.name + "_" + video.stem
  cache = OUT / "cache" / f"{name}_mp.npz"
  if cache.exists():
    lm = np.load(cache)["lm"]
  else:
    print(f"  MediaPipe on {len(frames)} frames ...", flush=True)
    lm = mediapipe_all(frames, fps, size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, lm=lm, fps=fps, size=np.array(size))
  print(f"  loaded {len(frames)} frames in {time.time() - t0:.0f} s, hand found in "
        f"{100 * np.isfinite(lm[:, 0, 0]).mean():.0f} %", flush=True)
  return name, fps, size, frames, lm


class Annotator:
  def __init__(self, video):
    self.video = Path(video)
    self.name, self.fps, self.size, self.frames, self.mp = load_clip(self.video)
    self.s = PW / self.size[0]  # original px -> panel px
    self.path = OUT / f"{self.name}.json"
    self.keys = {}
    if self.path.exists():
      d = json.loads(self.path.read_text())
      self.keys = {kf["frame"]: {"label": kf["label"], "points": kf["points"]} for kf in d["keyframes"]}
    self.k, self.sel, self.drag, self.play = 0, 0, False, False
    self.mouse = (PW // 2, PH // 2)
    self.msg = f"loaded {len(self.keys)} key frames" if self.keys else "press k to make a key frame"

  # --- data ---------------------------------------------------------------------------------
  def mp_points(self, k):
    if not np.isfinite(self.mp[k, 0, 0]):
      return None
    return {n: self.mp[k, i].round(1).tolist() for n, i in POINTS.items()}

  def toggle_key(self):
    if self.k in self.keys:
      del self.keys[self.k]
      self.msg = f"frame {self.k}: key frame removed"
    else:
      pts = self.mp_points(self.k)
      src = "MediaPipe"
      if pts is None and self.keys:  # no hand found here: start from the nearest key frame
        near = min(self.keys, key=lambda f: abs(f - self.k))
        pts, src = json.loads(json.dumps(self.keys[near]["points"])), f"key frame {near}"
      if pts is None:
        pts, src = {n: None for n in NAMES}, "empty (no hand found)"
      self.keys[self.k] = {"label": None, "points": pts}
      self.sel = 0
      self.msg = f"frame {self.k}: key frame added, points from {src}"
    self.save()

  def save(self):
    OUT.mkdir(parents=True, exist_ok=True)
    kfs = [{"frame": f, "t": round(f / self.fps, 3), "label": v["label"], "points": v["points"]}
           for f, v in sorted(self.keys.items())]
    self.path.write_text(json.dumps({"clip": self.name, "video": str(self.video.relative_to(ROOT)),
                                     "fps": self.fps, "size": list(self.size), "points": NAMES,
                                     "keyframes": kfs}, indent=1))

  # --- drawing ------------------------------------------------------------------------------
  def draw_hand(self, img, pts, scale, sel=None, thick=2, r=5):
    for a, b in BONES:
      if pts.get(a) is not None and pts.get(b) is not None:
        cv2.line(img, tuple((np.array(pts[a]) * scale).astype(int)),
                 tuple((np.array(pts[b]) * scale).astype(int)), col(b), thick, cv2.LINE_AA)
    for i, n in enumerate(NAMES):
      if pts.get(n) is None:
        continue
      p = tuple((np.array(pts[n]) * scale).astype(int))
      cv2.circle(img, p, r, col(n), -1, cv2.LINE_AA)
      if i == sel:
        cv2.circle(img, p, r + 6, (0, 255, 255), 2, cv2.LINE_AA)

  def render(self):
    img = cv2.imdecode(self.frames[self.k], cv2.IMREAD_COLOR)
    left = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
    right = left.copy()
    mpp = self.mp_points(self.k)
    if mpp:
      self.draw_hand(right, mpp, self.s)
    else:
      cv2.putText(right, "MediaPipe: no hand", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    is_key = self.k in self.keys
    if is_key:
      self.draw_hand(left, self.keys[self.k]["points"], self.s, self.sel)
      cv2.rectangle(left, (0, 0), (PW - 1, PH - 1), (0, 255, 255), 4)
    cv2.putText(left, "YOUR POINTS (click here)" if is_key else "real frame",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(right, "MediaPipe", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    top = np.hstack([left, right])
    # magnifier: 3x around the selected point (key frame) or the mouse
    full = img  # proxy resolution
    ps = PROXY_W / self.size[0]
    if is_key and self.keys[self.k]["points"].get(NAMES[self.sel]) is not None:
      c = np.array(self.keys[self.k]["points"][NAMES[self.sel]]) * ps
    else:
      c = np.array(self.mouse) / self.s * ps
    half = 66
    x0 = int(np.clip(c[0] - half, 0, full.shape[1] - 2 * half))
    y0 = int(np.clip(c[1] - half, 0, full.shape[0] - 2 * half))
    crop = full[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    if is_key:
      sh = {n: (None if v is None else [v[0] * ps - x0, v[1] * ps - y0])
            for n, v in self.keys[self.k]["points"].items()}
      self.draw_hand(crop, sh, 1.0, self.sel, thick=1, r=2)
    mag = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_NEAREST)
    mag = cv2.copyMakeBorder(mag, 0, 0, 280, 280, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(mag, "3x", (290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    info = np.full((400, PW, 3), 30, np.uint8)
    lab = self.keys[self.k]["label"] if is_key else None
    lines = [
      f"{self.name}   frame {self.k}/{len(self.frames) - 1}   t {self.k / self.fps:5.2f} s"
      + ("   KEY FRAME" + (f" [{lab}]" if lab else " [no label: press 1-7]") if is_key else ""),
      f"selected point: {NAMES[self.sel]}" + ("  (hidden)" if is_key and
                                             self.keys[self.k]["points"].get(NAMES[self.sel]) is None else ""),
      "key frames: " + " ".join(f"{f}{'(' + v['label'] + ')' if v['label'] else ''}"
                                for f, v in sorted(self.keys.items())),
      self.msg, "",
      "a/d or arrows: +-1 frame   j/l: +-10   space: play   k: key frame on/off",
      "1 start 2 pregrasp 3 grasp 4 lift 5 place 6 release 7 end",
      "click: place selected point (then next)   drag: move   right-click: hidden",
      "n/p or Tab: next/prev point   m: reset to MediaPipe   s: save   q: next clip   Esc: quit",
    ]
    for i, t in enumerate(lines):
      cv2.putText(info, t[:110], (12, 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                  (0, 255, 255) if i < 2 else (230, 230, 230), 1, cv2.LINE_AA)
    # timeline: key frames (yellow) + where MediaPipe found the hand (green)
    tl = np.full((24, 2 * PW, 3), 50, np.uint8)
    n = len(self.frames)
    found = np.isfinite(self.mp[:, 0, 0])
    for f in range(n):
      if found[f]:
        x = int(f / n * 2 * PW)
        tl[16:, x] = (0, 160, 0)
    for f in self.keys:
      x = int(f / n * 2 * PW)
      cv2.rectangle(tl, (x - 2, 0), (x + 2, 15), (0, 255, 255), -1)
    x = int(self.k / n * 2 * PW)
    cv2.line(tl, (x, 0), (x, 23), (0, 0, 255), 2)
    return np.vstack([top, tl, np.hstack([mag, info])])

  # --- input --------------------------------------------------------------------------------
  def on_mouse(self, ev, x, y, flags, _):
    if x < PW and y < PH:
      self.mouse = (x, y)
    if self.k not in self.keys or not (x < PW and y < PH):
      if ev == cv2.EVENT_LBUTTONUP:
        self.drag = False
      return
    pts = self.keys[self.k]["points"]
    orig = [round(x / self.s, 1), round(y / self.s, 1)]
    if ev == cv2.EVENT_LBUTTONDOWN:
      near = [(np.hypot(*(np.array(p) * self.s - (x, y))), i) for i, (n, p) in
              enumerate(pts.items()) if p is not None]
      near = [v for v in near if v[0] < 10]
      if near:  # grab an existing point
        self.sel = NAMES.index(list(pts)[min(near)[1]])
        self.drag = True
      else:  # place the selected point, go to the next one
        pts[NAMES[self.sel]] = orig
        self.msg = f"{NAMES[self.sel]} placed"
        self.sel = (self.sel + 1) % len(NAMES)
        self.save()
    elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
      pts[NAMES[self.sel]] = orig
    elif ev == cv2.EVENT_LBUTTONUP and self.drag:
      self.drag = False
      self.msg = f"{NAMES[self.sel]} moved"
      self.save()
    elif ev == cv2.EVENT_RBUTTONDOWN:
      pts[NAMES[self.sel]] = None
      self.msg = f"{NAMES[self.sel]} marked hidden"
      self.save()

  def key(self, c):
    n = len(self.frames)
    if c in (ord("d"), *KEY_RIGHT):
      self.k = min(n - 1, self.k + 1)
    elif c in (ord("a"), *KEY_LEFT):
      self.k = max(0, self.k - 1)
    elif c == ord("l"):
      self.k = min(n - 1, self.k + 10)
    elif c == ord("j"):
      self.k = max(0, self.k - 10)
    elif c == ord(" "):
      self.play = not self.play
    elif c == ord("k"):
      self.toggle_key()
    elif c in LABELS and self.k in self.keys:
      self.keys[self.k]["label"] = LABELS[c]
      self.msg = f"frame {self.k} labelled {LABELS[c]}"
      self.save()
    elif c in (ord("n"), 9):
      self.sel = (self.sel + 1) % len(NAMES)
    elif c == ord("p"):
      self.sel = (self.sel - 1) % len(NAMES)
    elif c == ord("m") and self.k in self.keys and self.mp_points(self.k):
      self.keys[self.k]["points"] = self.mp_points(self.k)
      self.msg = "points reset to MediaPipe"
      self.save()
    elif c == ord("s"):
      self.save()
      self.msg = f"saved {self.path.relative_to(ROOT)}"
    elif c in (ord("q"), 27):
      self.save()
      return "quit" if c == 27 else "next"
    return None

  def run(self, win):
    cv2.setMouseCallback(win, self.on_mouse)
    while True:
      cv2.imshow(win, self.render())
      c = cv2.waitKeyEx(int(1000 / self.fps) if self.play else 30)
      if self.play:
        self.k = (self.k + 1) % len(self.frames)
      if c == -1:
        continue
      r = self.key(c if c in (*KEY_LEFT, *KEY_RIGHT) else c & 0xFF)
      if r:
        return r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("videos", nargs="*")
  a = ap.parse_args()
  videos = a.videos or sorted(str(p) for p in (COMMON / "data/raw/blue_cube").glob("*.mp4"))
  win = "hand key frames"
  cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
  for v in videos:
    print(f"{v}", flush=True)
    if Annotator(v).run(win) == "quit":
      break
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
