"""RL configuration for the G1 payload / crouch task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def g1_payload_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same PPO hyperparameters as the other G1 policies, own experiment name.

  The observation is much wider here (history on six proprioceptive terms), so
  the first layer grows, but the rest of the network and every PPO setting is
  unchanged. Keeping them identical is deliberate: it means any difference in
  hardware behaviour is attributable to the task setup, not to retuning.
  """
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_payload"
  return cfg
