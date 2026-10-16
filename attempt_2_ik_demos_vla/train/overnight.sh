#!/usr/bin/env bash
# Overnight pipeline: wait for the demo generation -> LeRobot dataset -> SmolVLA training -> eval.
set -uo pipefail
cd "$(dirname "$0")/.."
L=logs/overnight.log
echo "$(date) waiting for demo batch 2" >> $L
while [ ! -f logs/gen_batch2.done ]; do sleep 60; done
echo "$(date) demos: $(ls data/sim_demos/*.npz | wc -l)" >> $L
../.venv_vla/bin/python sim/to_lerobot.py > logs/to_lerobot.log 2>&1
tail -1 logs/to_lerobot.log >> $L
echo "$(date) training" >> $L
train/train_smolvla.sh 20000 16 > logs/train_launch.log 2>&1
CK=$(ls -d outputs/smolvla_dex3_*/checkpoints/last/pretrained_model | tail -1)
echo "$(date) trained: $CK" >> $L
../.venv_vla/bin/python sim/eval_vla.py --policy "$CK" --episodes 20 --video data/eval_vla.mp4 > logs/eval_vla.log 2>&1
tail -1 logs/eval_vla.log >> $L
echo "$(date) ALL DONE" >> $L
