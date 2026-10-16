#!/usr/bin/env bash
# v2 human-motion demos (carry clearance + settle hover + place descent, grasp 4 cm, carry 1.2x): 18 clips x 50
cd "$(dirname "$0")/.."
export HD_OUT=data/human_demos_v2
mkdir -p $HD_OUT logs
CLIPS=($(../.venv_perc/bin/python -c "
import glob, numpy as np
print(' '.join(f.split('blue_cube_')[1][:-4] for f in sorted(glob.glob('data/tracks/blue_cube_*.npz')) if bool(np.load(f, allow_pickle=True)['ok'])))"))
N=${1:-12}; V=${2:-50}
for ((w=0; w<N; w++)); do
  L=""; for ((i=w; i<${#CLIPS[@]}; i+=N)); do L="$L,${CLIPS[$i]}"; done
  ../.venv_perc/bin/python sim/make_human_demos.py --variants $V --seed $((6000+w)) --clips ${L#,} > logs/gen_v2_$w.log 2>&1 &
done
wait
echo done > logs/gen_v2.done
