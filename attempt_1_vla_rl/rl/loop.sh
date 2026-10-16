#!/usr/bin/env bash
# RL flywheel driver: iterate from the best policy so far; keep a new one only if the fixed 30-pose
# eval improves (paper sec. 2.5 checkpoint rollback).
#   rl/loop.sh <first_it> <n_its> <best_policy> <best_success> <rollout_eps> <train_steps>
cd "$(dirname "$0")/.."
IT=$1; N=$2; BEST=$3; BEST_S=$4; NR=$5; STEPS=$6
for ((i=IT; i<IT+N; i++)); do
  rl/iteration.sh $i "$BEST" $NR $STEPS data/human_demos_v2 || { echo "it$i failed" >> logs/rl_loop.log; break; }
  S=$(python3 -c "import json;print(json.load(open('data/eval/eval_it$i.json'))['summary']['success'])")
  NEW=outputs/smolvla_rl_it$i/checkpoints/last/pretrained_model
  if python3 -c "import sys;sys.exit(0 if $S > $BEST_S else 1)"; then
    echo "[$(date +%H:%M)] it$i $S > $BEST_S: KEEP $NEW" >> logs/rl_loop.log; BEST=$NEW; BEST_S=$S
  else
    echo "[$(date +%H:%M)] it$i $S <= $BEST_S: ROLL BACK to $BEST" >> logs/rl_loop.log
  fi
  echo "$BEST $BEST_S" > logs/rl_best.txt
done
