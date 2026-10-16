from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.human_v2.rl_cfg import g1_human_v2_ppo_runner_cfg

from .env_cfg import MAX_ITERATIONS


def g1_human_v2_posture_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = g1_human_v2_ppo_runner_cfg()
  cfg.experiment_name = "g1_human_v2_posture"
  cfg.max_iterations = MAX_ITERATIONS
  return cfg
