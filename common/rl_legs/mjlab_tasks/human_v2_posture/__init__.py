"""LEGS-ONLY V2 + posture commands (height, lean, waist yaw) + mid-episode payloads.

Posture         : default, with crouch-walking (down to 0.60 m) and a deeper lean.
PostureSafeArms : Posture with collision-free arm reach + carry pose (option C fine-tune).
PostureSafe     : same observations / experiment, original bands (walk >= 0.70 m, lean
                  <= 0.25 walking / 0.6 standing). Resume a Posture run with it if the
                  extended bands fail.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import (
  g1_human_v2_posture_env_cfg,
  g1_human_v2_stance_env_cfg,
  g1_human_v2_stance_palmin_env_cfg,
)
from .rl_cfg import g1_human_v2_posture_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-Posture-Unitree-G1",
  env_cfg=g1_human_v2_posture_env_cfg(),
  play_env_cfg=g1_human_v2_posture_env_cfg(play=True),
  rl_cfg=g1_human_v2_posture_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-PostureSafe-Unitree-G1",
  env_cfg=g1_human_v2_posture_env_cfg(extended=False),
  play_env_cfg=g1_human_v2_posture_env_cfg(play=True, extended=False),
  rl_cfg=g1_human_v2_posture_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-PostureSafeArms-Unitree-G1",
  env_cfg=g1_human_v2_posture_env_cfg(safe_arms=True),
  play_env_cfg=g1_human_v2_posture_env_cfg(play=True, safe_arms=True),
  rl_cfg=g1_human_v2_posture_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-Stance-Unitree-G1",
  env_cfg=g1_human_v2_stance_env_cfg(),
  play_env_cfg=g1_human_v2_stance_env_cfg(play=True),
  rl_cfg=g1_human_v2_posture_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-StancePalmIn-Unitree-G1",
  env_cfg=g1_human_v2_stance_palmin_env_cfg(),
  play_env_cfg=g1_human_v2_stance_palmin_env_cfg(play=True),
  rl_cfg=g1_human_v2_posture_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
