"""Shared layout: the user's desk (measured from the phone videos) -> the G1 tabletop.

Human body frame (perception/extract_hand.py): x forward into the table from where the person
stands, y to their left, origin at the middle of the table's near edge. Measured over the 26
blue-block clips: block start x 3..61 cm, y -14..+14 cm; drop star at (46, 5) cm.

The G1's right hand only reaches x 0.12-0.36 m, |y| 0.10-0.34 m on its own side (45-deg pinch;
crossing the midline hits the torso), and the table edge must stay >= 0.19 m ahead of the pelvis
(the hips touch it when the body leans into a reach; 0.17 still grazed). So the human work area is mapped with ONE
per-axis affine into the G1 area below; the same map is used for the block start positions, the
drop star and the scene decals, so the layout's relative arrangement is preserved.
"""

import numpy as np

HUMAN_AREA = np.array([[0.00, 0.62], [-0.15, 0.15]])   # x, y ranges (m), body frame
G1_AREA = np.array([[0.23, 0.35], [-0.34, -0.04]])     # x, y ranges (m), robot world frame
# (palm-in grasp from the right, measured reach map; your star maps to the left side, x 0.32,
#  y -0.14, as on your desk)
HUMAN_STAR = np.array([0.46, 0.05])
TABLE_FRONT, TABLE_TOP = 0.19, 0.75  # near edge ahead of the pelvis, top height (m; user: 0.75)
TABLE_SIZE = (0.75, 0.80, 0.03)      # x (depth), y (width), thickness (m): robot at the LONG edge,
                                     # on the table's centre line (y = 0)
BLOCK_SIZE = (0.035, 0.035, 0.08)    # the user's blue block, standing


def human_to_g1(p):
  p = np.asarray(p, float)
  u = (p - HUMAN_AREA[:, 0]) / (HUMAN_AREA[:, 1] - HUMAN_AREA[:, 0])
  return G1_AREA[:, 0] + u * (G1_AREA[:, 1] - G1_AREA[:, 0])


# pick-and-place task (right hand, palm in): pick on the right, drop on the centre line. Both
# from the planner reach check (sim/tools/pick_place_motion.py).
PICK_XY = (0.275, -0.17)                      # centre of the pickup region
# x, y ranges where the full motion passed (planner + IK, legs policy). Beyond it (further right /
# forward) the legs policy steps back ~13 cm and turns ~25 deg after the reach, and the drop is
# then out of reach: the limit is the legs' balance, not the arm.
PICK_REGION = ((0.20, 0.35), (-0.23, -0.11))
DROP_XY = (0.38, 0.00)  # star; the hand reaches ~2 cm further to put the BLOCK centre here
TASK_WAIST_YAW = 0.15  # torso twist (rad) to the left for the whole task: the elbow clears the torso
STAR_XY = np.array(DROP_XY)  # the blue star decal marks the drop point
SPAWN_OBJECTS = ("block",)  # free objects to spawn: any of "block", "mouse"
