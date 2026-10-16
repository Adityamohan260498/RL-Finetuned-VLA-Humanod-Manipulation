# humanoid_challenge: G1 tabletop stance for VLA manipulation

A Unitree G1 standing at a table in MuJoCo. A reinforcement-learning policy holds the **legs and
waist (15 joints)** still and balanced. **The 14 arm joints are free for your controller:** a VLA,
teleop, or IK. That's the base for the next step, where a VLA trained on your own phone data
moves one or both arms.

```
humanoid_challenge/
  scene/build_scene.py      builds scene/g1_tabletop.xml (G1 + table + your objects + cameras)
  scene/g1_tabletop.xml     the scene (absolute mesh paths, loadable from anywhere)
  stance/stance_runner.py   StanceRobot: legs policy + arm interface + cameras  <- use this
  stance/demo_stance.py     scripted pick / hand-over / place arm motions + stance metrics + video
  stance/legs_runtime.py    gait / posture command logic (numpy port of the training code)
  stance/arm_ik.py          G1 arm FK / Jacobian / IK / gravity torques (numpy, checked vs MuJoCo)
  policy/stance.onnx        the stance policy (legs + waist)
  logs/                     training log, eval results
```

## Run

All commands use the mjlab environment (`uv`), which has mujoco and onnxruntime:

```bash
cd ~/g1/resources/mjlab
uv run python ~/g1/humanoid_challenge/scene/build_scene.py          # rebuild the scene
uv run python ~/g1/humanoid_challenge/stance/demo_stance.py --video ~/g1/humanoid_challenge/demo.mp4
```

## Interface for the VLA

```python
import sys; sys.path.insert(0, "/home/aditya/g1/humanoid_challenge/stance")
from stance_runner import StanceRobot
bot = StanceRobot()
bot.reset()
while True:
    rgb = bot.render("head")               # D435 head camera ("front" / "side" also exist)
    q14 = my_vla(rgb, instruction)         # 14 arm angles: left (shoulder p/r/y, elbow,
    bot.set_arm_joint_targets(q14)         #   wrist r/p/y) then right; or:
    # bot.set_hand_targets(p_left, p_right)   hand positions in the torso frame (IK inside)
    bot.set_payload(0.08)                  # when an object is in hand (kg); 0 when released
    bot.step()                             # 50 Hz (4 x 200 Hz physics)
    print(bot.stability())                 # drift / foot slip / sway / table contacts
```

- **Control rate:** 50 Hz. A VLA running slower (5–30 Hz action chunks) just holds or interpolates targets between chunks.
- **Arm smoothing:** arm targets are low-passed with tau 0.25 s, the same as during training. The legs policy sees those targets every step and uses them to anticipate the arms.
- **Gravity compensation:** the arm PD gets an exact gravity feed-forward (links plus payload). Without it the soft arm gains (14 Nm/rad) sag the hands about 10 cm.
- **Posture:** `bot.set_posture(height, pitch, waist_yaw)` lowers or leans the body for low reaches. Trained range: 0.68–0.80 m height, −0.05 to 0.35 rad lean, ±0.5 rad waist yaw.
- **Reachable hand workspace** (heel of hand, torso frame, measured): x ≤ 0.30 m at the table top, x ≤ 0.25 m down to 6 cm below it. Objects sit at x 0.27–0.38 m.

## Scene

- **Table:** top 0.70 m, 0.50 × 0.50 m, front edge 0.20 m from the pelvis, single pedestal.
  - The G1's hands hang at about 0.79 m even at rest, pointing forward and down to x 0.25 m at y ±0.32 m.
  - A taller or wider table puts the hands on it and pushes the robot away. That was measured; it's why the table has this size.
- **Objects** (free bodies):
  - `block`: 3.5 × 3.5 × 8 cm, 80 g, standing;
  - `power_bank`: 7 × 14 × 1.5 cm, 200 g;
  - `headphones`: a 9 cm cylinder stand-in, 250 g;
  - `tray`: drop zone.
- **Contacts:** robot contact model and solver settings are the same as training (mjlab `FULL_COLLISION`): feet condim 3 with friction 0.6, other robot geoms frictionless, timestep 5 ms, implicit-fast integrator, pyramidal cone.
- **No fingers:** the G1 hands here have no fingers. For grasping, either add a gripper or fingers, or do what the demo does and treat "grasped" as payload mass at the hands plus the carry flag. A weld-on-contact grasp is the usual quick option for VLA data.

## The stance policy

- **Starting point:** fine-tuned from the legs-only posture policy (safe-arms fine-tune, `g1_human_v2_posture_safearms/model_24498`).
- **Task:** `Mjlab-Velocity-HumanV2-Stance-Unitree-G1`, 700 iterations, 4096 envs, about 45 min.
- **Inputs and outputs:** 752 → 15 (legs + waist).
- **Changes for this task:**
  - stand-only commands;
  - table-work postures;
  - arms 40% random safe-box reach / 25% pick / 20% carry / 15% rest;
  - light objects, 0.2–3 kg, picked up and put down mid-episode;
  - standing-still rewards up-weighted (stillness, base motion, stand contact 2–2.5 → 4, foot drag −0.2 → −1).
- **Unchanged:** the paper's reward set otherwise, domain randomisation, pushes.
- **Results:** see "Eval" below.

## Eval (MuJoCo, `stance/demo_stance.py`, 3 repeats of pick → hand-over → place, no falls)

| arm speed | policy | base drift | foot slip | pelvis ang. vel | hand err (median) |
|---|---|---|---|---|---|
| ×1 (49.5 s) | previous (safe-arms posture) | 2.90 m | 2.94 m | 3.2 rad/s | 8 mm |
| ×1 (49.5 s) | **stance** | **0.58 m** | **0.50 m** | **1.6 rad/s** | **4 mm** |
| ×2 (24.7 s) | previous | 1.11 m | 1.12 m | 2.8 rad/s | 16 mm |
| ×2 (24.7 s) | **stance** | **0.18 m** | **0.15 m** | **1.4 rad/s** | 70 mm |

Standing with no arm motion: 30 s, 4.5 cm drift, 2.5 cm foot slip. Training (700 iterations): no falls,
stillness reward 0.0002 → 2.16 of 4, both feet down 91 % of the time. Video: `demo_stance.mp4`
(front + head views).

**What I learned building the scene** (each one made the robot fall before it was fixed):
1. **Contact model:** the robot contact model must match training (`FULL_COLLISION`).
2. **Table size:** the G1's resting hands hang at table height, so the table is narrow and low.
3. **Hand-overs:** two hands closer than about 15 cm push into each other. Hand-overs need about 18 cm between the heel-of-hand points.
4. **IK:** position-only IK with a rest-pose null space. With an orientation term it found odd wrist poses.

**Still not perfect:** over long sessions the feet creep (0.5 m in 50 s of slow arm work). The policy
cannot observe its position, so a slow position-hold loop (or a re-centring step between tasks) is the
next fix if the VLA episodes are long.
