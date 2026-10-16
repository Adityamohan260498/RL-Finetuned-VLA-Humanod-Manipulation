"""G1 human-like locomotion with box-carrying payload (trained from scratch)."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import g1_human_env_cfg
from .rl_cfg import g1_human_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Human-Unitree-G1",
  env_cfg=g1_human_env_cfg(),
  play_env_cfg=g1_human_env_cfg(play=True),
  rl_cfg=g1_human_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from .turn_finetune_cfg import FINETUNE_ITERATIONS, g1_human_turn_env_cfg  # noqa: E402


def _turn_rl_cfg():
  cfg = g1_human_ppo_runner_cfg()
  cfg.experiment_name = "g1_human_turn"
  cfg.max_iterations = FINETUNE_ITERATIONS
  return cfg


register_mjlab_task(
  task_id="Mjlab-Velocity-Human-Turn-Unitree-G1",
  env_cfg=g1_human_turn_env_cfg(),
  play_env_cfg=g1_human_turn_env_cfg(play=True),
  rl_cfg=_turn_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from .turn_finetune_cfg import g1_human_stand_env_cfg  # noqa: E402


def _stand_rl_cfg():
  cfg = g1_human_ppo_runner_cfg()
  cfg.experiment_name = "g1_human_stand"
  cfg.max_iterations = FINETUNE_ITERATIONS
  return cfg


register_mjlab_task(
  task_id="Mjlab-Velocity-Human-Stand-Unitree-G1",
  env_cfg=g1_human_stand_env_cfg(),
  play_env_cfg=g1_human_stand_env_cfg(play=True),
  rl_cfg=_stand_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
