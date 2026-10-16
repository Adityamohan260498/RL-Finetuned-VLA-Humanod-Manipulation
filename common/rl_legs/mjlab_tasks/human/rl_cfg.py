"""PPO config for the human-like gait task: stock G1 hyperparameters plus
left/right symmetry data augmentation (rsl_rl Symmetry extension)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from .env_cfg import MAX_ITERATIONS


def g1_human_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_human"
  cfg.max_iterations = MAX_ITERATIONS
  cfg.save_interval = 250
  cfg.algorithm.symmetry_cfg = {
    "use_data_augmentation": True,
    "use_mirror_loss": False,
    "mirror_loss_coeff": 0.0,
    "data_augmentation_func": "mjlab.tasks.human.mdp.symmetry:mirror_obs_and_actions",
  }
  return cfg
