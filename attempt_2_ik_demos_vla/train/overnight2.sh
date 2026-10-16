#!/usr/bin/env bash
# Round 2: + uniform-pose demos -> rebuild dataset -> continue training from round 1 -> eval.
set -uo pipefail
cd "$(dirname "$0")/.."
L=logs/overnight.log
INIT=outputs/smolvla_dex3_1012_0420/checkpoints/last/pretrained_model
echo "$(date) round 2: waiting for uniform demos" >> $L
while [ ! -f logs/gen_uniform.done ]; do sleep 60; done
echo "$(date) demos: $(ls data/sim_demos/*.npz | wc -l) ($(ls data/sim_demos/uniform_*.npz | wc -l) uniform)" >> $L
../.venv_vla/bin/python sim/to_lerobot.py > logs/to_lerobot2.log 2>&1
tail -1 logs/to_lerobot2.log >> $L
echo "$(date) round 2 training (from $INIT)" >> $L
train/train_smolvla.sh 20000 16 data/lerobot/g1_dex3_block_star "$INIT" > logs/train_launch2.log 2>&1
CK=$(ls -d outputs/smolvla_dex3_*/checkpoints/last/pretrained_model | tail -1)
echo "$(date) round 2 trained: $CK" >> $L
../.venv_vla/bin/python sim/eval_vla.py --policy "$CK" --episodes 30 --video data/eval_vla_r2.mp4 --out data/eval_vla_r2.json > logs/eval_vla2.log 2>&1
grep '"episodes"' logs/eval_vla2.log | tail -1 >> $L
echo "$(date) ROUND 2 DONE" >> $L
