#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base, 450M: SmolVLM2 vision-language backbone frozen, the
# action expert trained) on the robot-follows-my-hand demos: head + WRIST camera, 5-dim hand-pose state / action.
#   train/train_smolvla.sh [steps] [batch] [dataset_root] [init: pretrained_model dir to continue from]
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-20000}
BATCH=${2:-16}
ROOT=${3:-data/lerobot/g1_dex3_human}
INIT=${4:-lerobot/smolvla_base}
OUT=outputs/smolvla_wrist_$(date +%m%d_%H%M)
mkdir -p logs
../.venv_vla/bin/lerobot-train \
  --policy.path="$INIT" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=local/g1_dex3_human \
  --dataset.root="$ROOT" \
  --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq=2000 --log_freq=100 \
  --num_workers=4 --wandb.enable=false \
  --output_dir="$OUT" --job_name=g1_dex3_human 2>&1 | tee "logs/train_$(basename "$OUT").log"
echo "checkpoint: $OUT/checkpoints/last/pretrained_model"
