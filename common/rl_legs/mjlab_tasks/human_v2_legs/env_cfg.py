"""LEGS-ONLY V2: same paper-faithful method as human_v2, but the policy commands only
legs + waist (15 joints). The 14 arm joints are driven by ArmCommand (swing / default /
carry poses / random reaching), so any external arm controller can be plugged in later
(see ~/g1/training_legs/README.md). Measured motivation: the 29-joint run-3 policy with
its arms overridden fell in 13-18% of runs (92% with random arm motion).

Differences vs human_v2: action = 15 joints; actor also sees the arm target (14);
arm-swing / shoulder / elbow / carry-pose rewards removed (arms aren't the policy's);
yaw-momentum term keeps only the total residual (legs+waist must cancel whatever the
arms do); standing pose term covers the waist only. Arm-behaviour curriculum:
  P1      0- 8000  swing 70% / default 30%
  P2   8000-16000  swing 45 / default 20 / reach 35
  P3  16000-      swing 35 / default 15 / reach 30 / carry 20 (+ carrying forces carry)
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.human_v2 import mdp
from mjlab.tasks.human_v2.env_cfg import P2, P3, g1_human_v2_env_cfg
from mjlab.tasks.human_v2.mdp import STAND

from .arm_command import ArmCommandCfg

IT = 24
LEG_WAIST = (r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*", r"waist_.*")
ARM_STAGES = [
  {"step": 0, "mode_probs": {"swing": 0.7, "default": 0.3, "carry": 0.0, "reach": 0.0}},
  {
    "step": P2 * IT,
    "mode_probs": {"swing": 0.45, "default": 0.2, "carry": 0.0, "reach": 0.35},
  },
  {
    "step": P3 * IT,
    "mode_probs": {"swing": 0.35, "default": 0.15, "carry": 0.2, "reach": 0.3},
  },
]


def g1_human_v2_legs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_human_v2_env_cfg(play=play)
  act = cfg.actions["joint_pos"]
  act.actuator_names = LEG_WAIST
  arm_keys = ("shoulder", "elbow", "wrist")
  act.scale = {k: v for k, v in act.scale.items() if not any(a in k for a in arm_keys)}

  cfg.commands["arm_command"] = ArmCommandCfg()
  if play:
    cfg.commands["arm_command"].mode_probs = dict(ARM_STAGES[-1]["mode_probs"])
  else:
    cfg.curriculum["arm_stages"] = CurriculumTermCfg(
      func=mdp.command_cfg_stages,
      params={"command_name": "arm_command", "stages": deepcopy(ARM_STAGES)},
    )
  for g in ("actor", "critic"):
    cfg.observations[g].terms["arm_command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "arm_command"}
    )

  for name in (
    "arm_energy",
    "elbow_phase",
    "arm_symmetry",
    "arm_leg_amp",
    "carry_pose",
    "shoulder_roll",
  ):
    cfg.rewards.pop(name)
  cfg.rewards["arm_leg_momentum"].params["arm_weight"] = 0.0
  cfg.rewards["soft_upper_pose"] = RewardTermCfg(
    func=mdp.soft_upper_pose,
    weight=4.0,
    params={"modes": (STAND,), "parts": ("waist",)},
  )
  return cfg
