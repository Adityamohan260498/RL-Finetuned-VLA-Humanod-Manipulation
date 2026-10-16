"""LEGS-ONLY V2 (15 joints: legs + waist). Arms are driven externally."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import g1_human_v2_legs_env_cfg
from .rl_cfg import g1_human_v2_legs_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-Legs-Unitree-G1",
  env_cfg=g1_human_v2_legs_env_cfg(),
  play_env_cfg=g1_human_v2_legs_env_cfg(play=True),
  rl_cfg=g1_human_v2_legs_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
