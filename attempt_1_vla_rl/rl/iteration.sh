#!/usr/bin/env bash
# One RL iteration (paper-style flywheel, sequential on one GPU):
#   rollouts of the current policy -> stage advantages -> BC + recent rollouts dataset with AWR weights
#   -> continue training -> fixed 30-pose eval.
#   rl/iteration.sh <it> <policy_dir> <rollout_episodes> <train_steps> <bc_dir> [<bc_dir> ...]
set -euo pipefail
cd "$(dirname "$0")/.."
IT=$1; POL=$2; NR=$3; STEPS=$4; shift 4; BC="$@"
K=${WORKERS:-5}
EN=${EVAL_N:-60}   # 60 fixed poses (the first 30 = the original 30): +-6 % instead of +-9 %
NA=${NAS:+--n-action-steps $NAS}
log() { echo "[$(date +%H:%M)] it$IT $*" | tee -a logs/rl_loop.log; }
log "rollouts: $NR episodes of $POL"
rl/run_eval.sh "$POL" data/eval/roll_it$IT.json $NR $((1000 + IT)) $K --record data/rollouts/it$IT --early-stop $NA \
  | tee -a logs/rl_loop.log
ROLL="data/rollouts/it$IT"
[ -d data/rollouts/it$((IT - 1)) ] && ROLL="data/rollouts/it$((IT - 1)) $ROLL"
log "advantage over $ROLL"
../.venv_vla/bin/python rl/advantage.py $ROLL | tee -a logs/rl_loop.log
log "dataset"
../.venv_vla/bin/python rl/build_dataset.py --out data/lerobot/rl_it$IT --bc $BC --roll $ROLL --bc-share ${BC_SHARE:-0.4} \
  2>&1 | grep '^{' | tee -a logs/rl_loop.log
log "train $STEPS steps"
NEW=$(train/train_awr.sh data/lerobot/rl_it$IT "$POL" $STEPS rl_it$IT)
log "eval $NEW"
rl/run_eval.sh "$NEW" data/eval/eval_it$IT.json $EN 123 $K $NA | tee -a logs/rl_loop.log
log "done -> $NEW"
