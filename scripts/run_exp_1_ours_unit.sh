#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset_root> <scene> <white_bg> <method_name> <exp_name>"
    exit 1
fi

root=$1
scene=$2
white_bg=$3
method_name=$4
exp_name=$5

dataset_name=$(basename $root)
args=()
[[ "$white_bg" == "1" ]] && args+=(--white_background)

# for i in {5..300..40}
# do
i=45
    python train.py \
        -s "$root"/"$scene"/colmap_"$i"/ \
        --test_iterations 1 \
        -m out_xp1/"$dataset_name"/"$scene"/"$method_name"/"$exp_name"_frame_"$i" \
        --eval \
        --exp_name "$exp_name"-frame-"$i" \
        --wandb_group "$scene"-xp-1 \
        --no_motion_mask \
        "${args[@]}"
# done
