#!/usr/bin/env bash
# LEGS-ONLY V2 + POSTURE (height / lean / waist yaw) + mid-episode pick-up & put-down.
# 22,000 iterations, ~21.5 h at ~3.5 s/it (measured, 4096 envs). Resumable.
#   ./train_posture.sh
#   ./train_posture.sh --agent.resume True --agent.load-run <run_dir_name>   # continue
set -euo pipefail
MJLAB=/home/aditya/g1/resources/mjlab
mkdir -p /home/aditya/g1/training_legs/logs
LOG=/home/aditya/g1/training_legs/logs/train_posture_$(date +%Y%m%d_%H%M%S).log
cd "$MJLAB"
setsid nohup uv run train Mjlab-Velocity-HumanV2-Posture-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.run-name human_v2_posture \
  "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $!   log: $LOG"
echo "watch: tail -f $LOG | grep -E 'Learning iteration|Mean reward|Mean episode length'"
echo "stop:  pkill -f 'run-name human_v2_posture'"
echo "ckpts: $MJLAB/logs/rsl_rl/g1_human_v2_posture/<date>_human_v2_posture/"
