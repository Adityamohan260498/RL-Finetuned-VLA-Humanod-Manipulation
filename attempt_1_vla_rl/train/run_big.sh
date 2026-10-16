#!/usr/bin/env bash
# Head-only fallback improvement while the wrist demos generate: AWR round with ALL head-cam demos
# (v2 + v3 = 1151) at 60 % of the sampling mass + it2/it3 rollouts, 15k steps from RL it2, then 60-pose eval.
cd "$(dirname "$0")/.."
../.venv_vla/bin/python rl/advantage.py data/rollouts/it2 data/rollouts/it3 > logs/big_adv.log 2>&1
../.venv_vla/bin/python rl/build_dataset.py --out data/lerobot/rl_big --bc data/human_demos_v2 data/human_demos_v3 \
  --roll data/rollouts/it2 data/rollouts/it3 --bc-share 0.6 > logs/big_ds.log 2>&1
NEW=$(train/train_awr.sh data/lerobot/rl_big outputs/smolvla_rl_it2/checkpoints/last/pretrained_model 15000 rl_big)
rl/run_eval.sh "$NEW" data/eval/eval_big.json 60 123 4 --n-action-steps 5 > logs/eval_big.log 2>&1
echo done > logs/big.done
