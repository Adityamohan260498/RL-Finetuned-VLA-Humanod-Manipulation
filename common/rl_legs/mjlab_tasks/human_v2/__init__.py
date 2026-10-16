"""V2: paper-faithful gait-conditioned G1 locomotion (29 DoF) + box carrying."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import g1_human_v2_env_cfg
from .rl_cfg import g1_human_v2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-HumanV2-Unitree-G1",
  env_cfg=g1_human_v2_env_cfg(),
  play_env_cfg=g1_human_v2_env_cfg(play=True),
  rl_cfg=g1_human_v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
