#!/usr/bin/env bash
# One AWR round: continue a SmolVLA policy on an rl/build_dataset.py dataset with weighted frame sampling.
#   train/train_awr.sh <dataset_root> <init pretrained_model dir> <steps> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$1; INIT=$2; STEPS=$3; TAG=$4
OUT=outputs/smolvla_$TAG
rm -rf "$OUT"
../.venv_vla/bin/python rl/awr_train.py \
  --policy.path="$INIT" --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=local/$(basename "$ROOT") --dataset.root="$ROOT" \
  --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=16 --steps="$STEPS" --save_freq="$STEPS" --log_freq=100 \
  --num_workers=4 --wandb.enable=false \
  --output_dir="$OUT" --job_name=$TAG > "logs/train_$TAG.log" 2>&1
echo "$OUT/checkpoints/last/pretrained_model"
