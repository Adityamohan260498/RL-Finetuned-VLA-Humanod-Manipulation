#!/usr/bin/env bash
# RL flywheel driver: iterate from the best policy so far; keep a new one only if the fixed eval
# (EVAL_N poses, default 60) improves (paper sec. 2.5 checkpoint rollback).
#   BC_DIRS="data/human_demos_v2 ..." NAS=5 rl/loop_v.sh <first_it> <n_its> <best_policy> <best_success> <rollout_eps> <train_steps>
# touch logs/rl_stop to stop cleanly after the current iteration.
cd "$(dirname "$0")/.."
IT=$1; N=$2; BEST=$3; BEST_S=$4; NR=$5; STEPS=$6; BC=${BC_DIRS:-data/human_demos_v2}
for ((i=IT; i<IT+N; i++)); do
  [ -f logs/rl_stop ] && { echo "[$(date +%H:%M)] stop file -> stop before it$i" >> logs/rl_loop.log; rm -f logs/rl_stop; break; }
  rl/iteration.sh $i "$BEST" $NR $STEPS $BC || { echo "it$i failed" >> logs/rl_loop.log; break; }
  S=$(python3 -c "import json;print(json.load(open('data/eval/eval_it$i.json'))['summary']['success'])")
  NEW=outputs/smolvla_rl_it$i/checkpoints/last/pretrained_model
  if python3 -c "import sys;sys.exit(0 if $S > $BEST_S else 1)"; then
    echo "[$(date +%H:%M)] it$i $S > $BEST_S: KEEP $NEW" >> logs/rl_loop.log; BEST=$NEW; BEST_S=$S
  else
    echo "[$(date +%H:%M)] it$i $S <= $BEST_S: ROLL BACK to $BEST" >> logs/rl_loop.log
  fi
  echo "$BEST $BEST_S" > logs/rl_best.txt
done
