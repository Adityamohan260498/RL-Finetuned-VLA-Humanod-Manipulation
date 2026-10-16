#!/usr/bin/env bash
# BC v1: wait for the v2 conversion, SmolVLA 20k steps from base, then the fixed 30-pose eval.
cd "$(dirname "$0")/.."
while kill -0 32545 2>/dev/null; do sleep 20; done
train/train_smolvla.sh 20000 16 data/lerobot/g1_dex3_human_v2 > logs/bc_v1_train.out 2>&1
CK=$(grep -o 'checkpoint: .*' logs/bc_v1_train.out | tail -1 | cut -d' ' -f2)
echo "$CK" > logs/bc_v1.ckpt
rl/run_eval.sh "$CK" data/eval/bc_v1.json 30 123 4 > logs/eval_bc_v1.log 2>&1
echo done > logs/bc_v1.done
