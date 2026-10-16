#!/usr/bin/env bash
# Wrist model: wait for the v4 conversion, SmolVLA head + wrist, 12k steps from RL it2, then the 60-pose eval.
cd "$(dirname "$0")/.."
while kill -0 70302 2>/dev/null; do sleep 20; done
while pgrep -f "video_it2" > /dev/null; do sleep 10; done
train/train_smolvla_wrist.sh 12000 16 data/lerobot/g1_dex3_wrist_v4 outputs/smolvla_rl_it2/checkpoints/last/pretrained_model > logs/wrist_train.out 2>&1
CK=$(grep -o 'checkpoint: .*' logs/wrist_train.out | tail -1 | cut -d' ' -f2)
echo "$CK" > logs/wrist.ckpt
rl/run_eval.sh "$CK" data/eval/wrist_v1.json 60 123 4 --n-action-steps 5 > logs/eval_wrist.log 2>&1
echo done > logs/wrist.done
