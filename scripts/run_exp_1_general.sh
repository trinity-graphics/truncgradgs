#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <dataset_root> <scene> <white_bg> <method_name> <exp_name> <img_path>"
    exit 1
fi

root=$1
scene=$2
white_bg=$3
method_name=$4
exp_name=$5
img_path=$6

dataset_name=$(basename $root)
args=()
[[ "$white_bg" == "1" ]] && args+=(--white_background)


if [ $IS_BASELINE -eq 1 ]; then
    script=train_baseline.py
else
    script=train.py
fi

python "$script" \
        -s "$root"/"$scene"/ \
        -i "$img_path" \
        --test_iterations 1 \
        -m out_xp1/"$dataset_name"/"$scene"/"$method_name"/"$exp_name" \
        --eval \
        --exp_name "$exp_name" \
        --wandb_group "$dataset_name"-"$scene"-xp-1 \
        --no_motion_mask \
        "${args[@]}"
