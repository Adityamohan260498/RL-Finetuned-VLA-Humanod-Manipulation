#!/usr/bin/env bash
# Parallel closed-loop eval / rollouts: K workers on the GPU, each its own MuJoCo + SmolVLA.
#   rl/run_eval.sh <policy_dir> <out.json> <episodes> <seed> <workers> [extra eval_vla args...]
cd "$(dirname "$0")/.."
POL=$1; OUT=$2; N=$3; SEED=$4; K=$5; shift 5
rm -f ${OUT%.json}.w*.json
for ((w=0; w<K; w++)); do
  ../.venv_vla/bin/python sim/eval_vla.py --policy "$POL" --episodes $N --seed $SEED --workers $K --worker $w \
     --out "$OUT" "$@" > ${OUT%.json}.w$w.log 2>&1 &
done
wait
../.venv_vla/bin/python rl/merge_eval.py "$OUT"
