from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.human_v2.rl_cfg import g1_human_v2_ppo_runner_cfg


def g1_human_v2_legs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = g1_human_v2_ppo_runner_cfg()
  cfg.experiment_name = "g1_human_v2_legs"
  return cfg
