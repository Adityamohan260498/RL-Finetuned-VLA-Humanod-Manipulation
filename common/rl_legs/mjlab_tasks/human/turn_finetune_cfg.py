"""Turning + standing fine-tune of the human-gait policy (run 2, warm start from run 1).

Run 1 (g1_human/2026-10-08_11-56-13_human_gait_carry) learned the gait and the
box carry, but turn tracking stayed at ~10% of max for 8k iterations. Two causes
in the reward, both fixed here; nothing else changes (same obs => weights load):
  1. track_angular_velocity scored pelvis roll/pitch rate as error -> yaw-only kernel.
  2. yaw_angmom (-0.1) and angular_momentum penalise the yaw momentum a commanded
     turn needs -> eased 10x / 4x. foot_slip eased (pivoting slides a foot).
Run-1 MuJoCo eval also showed STANDING is broken: 20/20 unloaded and 3/20 at 11 kg
fall within 0.7-1.9 s of a stand command (also after walking). Causes and fixes:
  3. pose std_standing 0.05 rad on every joint -> reward ~0, no gradient -> 0.15.
  4. no explicit stand mode -> gait clock observes (0, 0) while standing.
  5. falling cost nothing explicit -> is_terminated -20.
  6. stand was 8% of commands -> 12% (resampling every 3-8 s gives walk<->stand).
Every curriculum is pinned to its FINAL stage so the fine-tune trains in run 1's
end conditions (vx up to 2.0, 40/30/30 payload mix, full style weights, full DR)
regardless of whether the step counter is restored on resume.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.human import mdp
from mjlab.tasks.human.env_cfg import g1_human_env_cfg

FINETUNE_ITERATIONS = 3000

TURN_PROBS = {
  "stand": 0.12,
  "slow": 0.07,
  "vx": 0.10,
  "vy": 0.06,
  "wz": 0.18,
  "vx_wz": 0.20,
  "vy_wz": 0.10,
  "vx_vy": 0.05,
  "all": 0.12,
}


def g1_human_turn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_human_env_cfg(play=play)
  R = cfg.rewards

  R["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_yaw_rate, weight=3.0, params={"std": 0.4, "command_name": "twist"}
  )
  cfg.commands["twist"].category_probs = dict(TURN_PROBS)
  cfg.commands["gait"].zero_when_standing = True
  R["pose"].params["std_standing"] = {".*": 0.15}
  R["fall_penalty"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=-20.0)

  # Pin every curriculum at its final stage (and the matching cfg values).
  for name, term in list(cfg.curriculum.items()):
    stages = term.params.get("stages")
    if not stages:
      continue
    last = dict(stages[-1])
    last["step"] = 0
    term.params["stages"] = [last]
    if name.startswith("style_"):
      R[name[len("style_") :]].weight = last["weight"]
    elif name == "command_human":
      for k, v in last.items():
        if k != "step":
          setattr(cfg.commands["twist"], k, v)
    elif "event_name" in term.params:
      cfg.events[term.params["event_name"]].params.update(last["params"])

  # Ease the terms that fight a commanded turn (after pinning, so they stick).
  for name, scale in (
    ("yaw_angmom", 0.1),
    ("angular_momentum", 0.25),
    ("foot_slip", 0.4),
  ):
    R[name].weight *= scale
    cfg.curriculum.pop(f"style_{name}", None)
  return cfg


# ---------------------------------------------------------------------------
# Run 3: walk-to-stand. Run-2 eval: standing from rest OK (11 kg 20/20, 0 kg 14/20)
# but stopping after walking fell 20/20 (0 kg) and 16/20 (11 kg), also with a 1.5 s
# command ramp -- entering stand mode while moving forbids the recovery steps.
# Also lost fwd 2.0 m/s from standstill (pure-vx share was cut to 10%).
# ---------------------------------------------------------------------------
STAND_PROBS = {
  "stand": 0.12,
  "slow": 0.07,
  "vx": 0.14,
  "vy": 0.06,
  "wz": 0.16,
  "vx_wz": 0.18,
  "vy_wz": 0.09,
  "vx_vy": 0.06,
  "all": 0.12,
}


def g1_human_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_human_turn_env_cfg(play=play)
  cfg.commands["gait"].stand_settle_s = 1.0
  cfg.commands["twist"].category_probs = dict(STAND_PROBS)
  if not play:
    cfg.commands["twist"].resampling_time_range = (2.0, 5.0)  # more stops/starts
  return cfg
