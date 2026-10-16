#!/usr/bin/env bash
# After the wrist BC eval: one trimmed RL iteration (60 rollouts, 3k AWR steps, 60-pose eval; 2-camera training is 1.55 step/s) on the wrist
# model, keep it only if better, then a demo video of the better wrist model. Deadline 2026-10-16 01:00.
cd "$(dirname "$0")/.."
until [ -f logs/wrist.done ]; do sleep 30; done
BC=$(cat logs/wrist.ckpt)
BS=$(python3 -c "import json;print(json.load(open('data/eval/wrist_v1.json'))['summary']['success'])")
echo 584 > data/lerobot/g1_dex3_wrist_v4/SOURCE_COUNT
echo "[$(date +%H:%M)] wrist RL: from $BC ($BS)" >> logs/rl_loop.log
NAS=5 WORKERS=5 rl/iteration.sh 11 "$BC" 60 3000 data/human_demos_v4 > logs/wrist_rl.out 2>&1
S=$(python3 -c "import json;print(json.load(open('data/eval/eval_it11.json'))['summary']['success'])" 2>/dev/null || echo 0)
BEST=$BC
if python3 -c "import sys;sys.exit(0 if $S > $BS else 1)"; then BEST=outputs/smolvla_rl_it11/checkpoints/last/pretrained_model; fi
echo "[$(date +%H:%M)] wrist RL it11 $S vs BC $BS -> final wrist model $BEST" >> logs/rl_loop.log
echo "$BEST" > logs/wrist_final.txt
../.venv_vla/bin/python sim/eval_vla.py --policy "$BEST" --episodes 4 --seed 123 --n-action-steps 5 \
  --video media/wrist_final_demo.mp4 --video-all --out data/eval/video_wrist_final.json > logs/video_wrist.log 2>&1
echo done > logs/wrist_rl.done
