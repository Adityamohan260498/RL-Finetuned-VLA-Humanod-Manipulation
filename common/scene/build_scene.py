"""Build the tabletop scene: Unitree G1 (29 DoF, mjlab MJCF) standing at a table with objects.

  python scene/build_scene.py   (any env with mujoco >= 3.3)

Writes scene/g1_tabletop.xml (paths relative to scene/, meshes in robot/meshes). The G1 gets one
torque motor per joint (ctrl = torque, clamped to the real actuator limits); the PD loop and
the legs policy live in stance/stance_runner.py. Cameras: 'head' (D435 pose on torso_link,
as on the real robot), 'front' and 'side' (fixed, for videos / VLA third-person view).
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parent / "robot"
G1 = ROBOT / "g1/g1.xml"  # mjlab's G1 MJCF (copied in, see robot/README.md)
MESHES = ROBOT / "meshes"  # G1 + Dex3 meshes
OUT = HERE / "g1_tabletop.xml"

# Actuator torque limits (mjlab g1_constants.py).
EFFORT = {"hip_pitch": 88, "hip_yaw": 88, "hip_roll": 139, "knee": 139, "ankle_pitch": 50,
          "ankle_roll": 50, "waist_yaw": 88, "waist_roll": 50, "waist_pitch": 50,
          "shoulder_pitch": 25, "shoulder_roll": 25, "shoulder_yaw": 25, "elbow": 25,
          "wrist_roll": 25, "wrist_pitch": 5, "wrist_yaw": 5}
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from layout import (BLOCK_SIZE, PICK_XY, SPAWN_OBJECTS, STAR_XY, TABLE_FRONT,  # noqa: E402
                    TABLE_SIZE, TABLE_TOP, human_to_g1)

# The user's desk as filmed (white top, ArUco markers in the corners, blue block, blue star
# drop spot, black organizer + bottle at the far end, a mouse), laid out through layout.py.
TABLE_X = TABLE_FRONT + TABLE_SIZE[0] / 2
BLOCK_XY = tuple(PICK_XY)  # default spot (episodes move it)
# D435 on torso_link (Unitree URDF d435_joint), pitched 47.6 deg down.
HEAD_CAM_POS, HEAD_CAM_PITCH = (0.0576235, 0.01753, 0.42987), 0.8307767
# The G1 has no neck: the D435 is fixed. For the task, the mount is re-aimed (user request:
# "tilt the head") so it sees the right-hand work area, the block and the star: HEAD_CAM_EXTRA
# pitch down and HEAD_CAM_YAW toward the right hand, on top of the stock mount.
HEAD_CAM_EXTRA, HEAD_CAM_YAW = 0.15, -0.75  # -0.75: frames the pickup region AND the star


def build():
  spec = mujoco.MjSpec.from_file(str(G1))
  spec.meshdir = str(MESHES)
  spec.option.timestep = 0.005
  # Same solver settings as the training task (mjlab velocity cfg) ...
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  # grasping: elliptic friction cone + impratio 10 (MuJoCo's accurate friction; the pyramidal
  # default let a held block creep out of a 76 N grip). The legs policy trained with pyramidal.
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10.0
  spec.option.iterations, spec.option.ls_iterations = 10, 20
  # ... and the same robot contact model (g1_constants.FULL_COLLISION): feet condim 3,
  # priority 1, friction 0.6; every other robot collision geom condim 1 (frictionless).
  for g in spec.geoms:
    if g.name.endswith("_collision"):
      if g.name.startswith(("left_foot", "right_foot")):
        g.condim, g.priority, g.friction = 3, 1, [0.6, 0.005, 0.0001]
      else:
        g.condim = 1
  w = spec.worldbody
  spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080  # video renders
  w.add_light(pos=[0, 0, 3.0], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
  tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.25, 0.27, 0.30], rgb2=[0.32, 0.34, 0.37], width=512, height=512)
  mat = spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.05)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  del tex
  w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.05], material="grid",
             friction=[1.0, 0.005, 0.0001])

  # table: white top + one pedestal set back (front legs were hit by the hands)
  tb = w.add_body(name="table", pos=[TABLE_X, 0, 0])
  sx, sy, sz = (v / 2 for v in TABLE_SIZE)
  tb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, sz],
              pos=[0, 0, TABLE_TOP - sz], rgba=[0.93, 0.93, 0.91, 1], friction=[1.0, 0.01, 0.001])
  tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.04, (TABLE_TOP - 0.03) / 2],
              pos=[0.10, 0, (TABLE_TOP - 0.03) / 2], rgba=[0.3, 0.3, 0.3, 1])
  tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.25, 0.01],
              pos=[0.12, 0, 0.01], rgba=[0.3, 0.3, 0.3, 1])

  # decals (visual only): ArUco markers in the table corners as in the videos (id 1 far-right,
  # id 0 far-left, id 3 near-right, id 2 near-left, robot's view) and the blue star drop spot
  near, far = TABLE_FRONT + 0.05, TABLE_FRONT + TABLE_SIZE[0] - 0.06
  side = TABLE_SIZE[1] / 2 - 0.05
  for i, (x, y) in {1: (far, -side), 0: (far, side), 3: (near, -side), 2: (near, side)}.items():
    spec.add_texture(name=f"aruco_{i}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(HERE / f"textures/aruco_{i}.png"))
    mt = spec.add_material(name=f"aruco_{i}")
    mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"aruco_{i}"
    # 6 cm black square + white border; rotated so the marker's top faces the far edge
    w.add_geom(name=f"aruco_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.04, 0.0005],
               pos=[x, y, TABLE_TOP + 0.0005], euler=[0, 0, 90], material=f"aruco_{i}",
               contype=0, conaffinity=0, group=1)
  star = w.add_body(name="star", pos=[STAR_XY[0], STAR_XY[1], TABLE_TOP + 0.0005])
  for k, yaw in enumerate((20, 65)):  # two crossed strips, like the sticker
    star.add_geom(name=f"star_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.022, 0.012, 0.0005],
                  euler=[0, 0, yaw], rgba=[0.10, 0.18, 0.62, 1], contype=0, conaffinity=0,
                  group=1)
  # desk clutter from the videos (static, out of reach): black organizer + a bottle
  w.add_geom(name="organizer", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.14, 0.06, 0.045],
             pos=[far - 0.20, -side + 0.01, TABLE_TOP + 0.045], rgba=[0.08, 0.08, 0.09, 1])
  # manipulable objects (free bodies): the blue block (standing) and the mouse (left side,
  # for the left hand later)
  objs = [("block", "box", [v / 2 for v in BLOCK_SIZE], list(BLOCK_XY),
           [0.08, 0.16, 0.78, 1], 0.08),
          ("mouse", "ellipsoid", [0.055, 0.032, 0.019], list(human_to_g1([0.25, 0.10]) * [1, -1]),
           [0.22, 0.23, 0.25, 1], 0.09)]
  types = {"box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
  for name, kind, size, xy, rgba, kg in (o for o in objs if o[0] in SPAWN_OBJECTS):
    b = w.add_body(name=name, pos=[xy[0], xy[1], TABLE_TOP + size[2] + 0.002])
    b.add_freejoint(name=f"{name}_joint")
    b.add_geom(name=f"{name}_geom", type=types[kind], size=size, rgba=rgba, mass=kg,
               friction=[1.2, 0.01, 0.001], condim=4)

  add_dex3(spec)

  # cameras
  torso = spec.body("torso_link")
  c, s = np.cos(HEAD_CAM_PITCH / 2), np.sin(HEAD_CAM_PITCH / 2)
  # MuJoCo cameras look down -z; rotate so it looks along +x of the torso, pitched down.
  q_look = np.array([0.5, 0.5, -0.5, -0.5])  # x-forward, z-up camera
  q_pitch = np.array([c, 0, s, 0])
  qw = _qmul(q_pitch, q_look)
  ce, se = np.cos(HEAD_CAM_EXTRA / 2), np.sin(HEAD_CAM_EXTRA / 2)
  cy, sy = np.cos(HEAD_CAM_YAW / 2), np.sin(HEAD_CAM_YAW / 2)
  qw = _qmul(np.array([cy, 0, 0, sy]), _qmul(np.array([ce, 0, se, 0]), qw))  # yaw(torso z) * pitch
  torso.add_camera(name="head", pos=list(HEAD_CAM_POS), quat=list(qw), fovy=58)
  # front: from the robot's front-right, always aimed at the right-hand work area
  w.add_body(name="work_area", pos=[0.28, -0.20, TABLE_TOP + 0.08])
  w.add_camera(name="front", pos=[1.05, -1.05, 1.30], fovy=40,
               mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="work_area")
  # like the phone in the videos: looking down at the table from above
  # (image right = the robot, image top = its right side: same as the person in the videos)
  w.add_camera(name="phone", pos=[0.50, 0.12, 2.05], xyaxes=[-1, 0, 0, 0, -0.97, 0.24], fovy=50)
  w.add_camera(name="side", pos=[0.45, -1.8, 1.1], xyaxes=[1, 0, 0, 0, 0.25, 0.97], fovy=50)

  # one torque motor per robot joint
  for j in spec.joints:
    if (j.name.endswith("_joint") and j.type == mujoco.mjtJoint.mjJNT_HINGE
        and "_hand_" not in j.name):  # Dex3 joints have their own position servos
      key = next(k for k in EFFORT if k in j.name)
      e = EFFORT[key]
      spec.add_actuator(name=j.name.replace("_joint", ""), target=j.name,
                        trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-e, e], ctrllimited=True)
  return spec


# Unitree Dex3-1 hands (3 fingers, 7 joints per hand: thumb 0/1/2, index 0/1, middle 0/1), from
# MuJoCo Menagerie unitree_g1/g1_with_hands.xml (Unitree's own meshes, masses, joint ranges),
# replacing the G1's fingerless rubber hands. Contact and actuation set here (not Menagerie's
# generic kp=500 position servos): DEX3_* below; torque limits = Unitree's URDF joint efforts
# (Menagerie actuatorfrcrange).
DEX3_XML = ROBOT / "dex3/g1_with_hands.xml"
DEX3_KP, DEX3_KV = 10.0, 0.2       # N m / rad, N m s / rad (position servo per joint)
DEX3_FRICTION = [1.0, 0.01, 0.0001]  # rubber / silicone pad on plastic
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def _copy_body(spec, parent, sb):
  """Copy a Menagerie body subtree (inertia, joints, mesh geoms) under `parent`."""
  nb = parent.add_body(name=sb.name, pos=sb.pos, quat=sb.quat, mass=sb.mass, ipos=sb.ipos,
                       iquat=sb.iquat, inertia=sb.inertia, fullinertia=sb.fullinertia,
                       explicitinertial=True)
  for j in sb.joints:
    nb.add_joint(name=j.name, type=j.type, axis=j.axis, range=j.range, limited=j.limited,
                 armature=j.armature, damping=j.damping, frictionloss=j.frictionloss,
                 actfrclimited=j.actfrclimited, actfrcrange=j.actfrcrange)
  have = {me.name for me in spec.meshes}
  for g in sb.geoms:
    if g.type != mujoco.mjtGeom.mjGEOM_MESH:  # primitive collision geom (e.g. thumb_1 capsule)
      nb.add_geom(type=g.type, size=g.size, pos=g.pos, quat=g.quat, fromto=g.fromto, group=3,
                  density=0, rgba=[0.2, 0.6, 0.2, 0.3])
      continue
    if g.meshname not in have:
      spec.add_mesh(name=g.meshname, file=f"{g.meshname}.STL")
      have.add(g.meshname)
    vis = g.contype == 0 and g.conaffinity == 0
    nb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, pos=g.pos, quat=g.quat,
                contype=0 if vis else 1, conaffinity=0 if vis else 1, group=2 if vis else 3,
                density=0, rgba=[0.2, 0.2, 0.2, 1] if vis else [0.2, 0.6, 0.2, 0.3])
  for c in sb.bodies:
    _copy_body(spec, nb, c)


def add_dex3(spec):
  src = mujoco.MjSpec.from_file(str(DEX3_XML))
  for side in ("left", "right"):
    wrist = spec.body(f"{side}_wrist_yaw_link")
    for g in list(wrist.geoms):  # the rubber hand (visual mesh + collision capsule)
      if g.meshname.endswith("rubber_hand") or g.name == f"{side}_hand_collision":
        spec.delete(g)
    sw = src.body(f"{side}_wrist_yaw_link")  # Menagerie wrist link includes the Dex3 palm
    wrist.mass, wrist.ipos, wrist.iquat = sw.mass, sw.ipos, sw.iquat
    wrist.inertia, wrist.fullinertia, wrist.explicitinertial = sw.inertia, sw.fullinertia, True
    palm = f"{side}_hand_palm_link"
    spec.add_mesh(name=palm, file=f"{palm}.STL")
    pg = next(g for g in sw.geoms if g.meshname == palm)
    wrist.add_geom(name=f"{palm}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=palm,
                   pos=pg.pos, quat=pg.quat, contype=0, conaffinity=0, group=2, density=0,
                   rgba=[0.2, 0.2, 0.2, 1])
    wrist.add_geom(name=f"{palm}_collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=palm,
                   pos=pg.pos, quat=pg.quat, group=3, density=0, condim=4,
                   friction=DEX3_FRICTION)
    for child in sw.bodies:
      _copy_body(spec, wrist, child)
  for b in spec.bodies:  # finger links: frictional pads (the G1 body geoms stay condim 1)
    if "_hand_" in b.name:
      for g in b.geoms:
        if g.contype or g.conaffinity:
          g.condim, g.friction = 4, DEX3_FRICTION
  for side in ("left", "right"):
    for j in HAND_JOINTS:
      jn = f"{side}_hand_{j}_joint"
      spec.add_actuator(name=f"{side}_hand_{j}", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                        gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[DEX3_KP] + [0] * 9,
                        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                        biasprm=[0, -DEX3_KP, -DEX3_KV] + [0] * 7,
                        ctrlrange=spec.joint(jn).range, ctrllimited=True,
                        forcerange=spec.joint(jn).actfrcrange, forcelimited=True)


def _qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


if __name__ == "__main__":
  spec = build()
  m = spec.compile()
  # relative paths in the written file: the folder can be moved / cloned anywhere
  xml = spec.to_xml().replace(f'meshdir="{MESHES}', 'meshdir="../robot/meshes')
  OUT.write_text(xml.replace(f'file="{HERE}/', 'file="'))
  print(f"wrote {OUT}: nbody {m.nbody} njnt {m.njnt} nu {m.nu} ncam {m.ncam}")
