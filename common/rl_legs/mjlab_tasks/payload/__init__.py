"""G1 locomotion with payload, commanded crouch height, and push recovery."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import g1_payload_env_cfg
from .rl_cfg import g1_payload_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Payload-Flat-Unitree-G1",
  env_cfg=g1_payload_env_cfg(),
  play_env_cfg=g1_payload_env_cfg(play=True),
  rl_cfg=g1_payload_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
