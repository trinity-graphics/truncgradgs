#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_root>"
    exit 1
fi

root=$1

iter=30000

set -e
LOG_FILE="$(dirname "$0")/mipnerf.txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== MipNeRF360 experiment started at $(date) ==="
scene_list=(bonsai counter kitchen)
for scene in "${scene_list[@]}"; do
    img_path="images_2"

    #echo "Processing $scene -- full method"
    #method_name=full-colmap
    #if [ -f ./out_ablation/mipnerf360/"$scene"/"$method_name"/point_cloud/iteration_30000/point_cloud.ply ]; then
    #    echo "Skipping training for $scene, method $method_name, because the model already exists"
    #else
    #    MODEL_INIT=COLMAP python train.py \
    #        -s "$root/mipnerf360"/"$scene"/ \
    #        -i "$img_path" \
    #        --test_iterations 1 \
    #        -m out_ablation/mipnerf360/"$scene"/"$method_name" \
    #        --eval \
    #        --exp_name "$scene"-"$method_name" \
    #        --wandb_group "$scene"-ablation \
    #        --no_motion_mask \
    #        --iterations "$iter" || tee -a "$LOG_FILE"
    #fi

    echo "Processing $scene without: truncated gradient"
    method_name=no-trunc-grad
    if [ -f ./out_ablation/mipnerf360/"$scene"/"$method_name"/point_cloud/iteration_30000/point_cloud.ply ]; then
        echo "Skipping training for $scene, method $method_name, because the model already exists"
    else
        MODEL_INIT=COLMAP python train.py \
            -s "$root/mipnerf360"/"$scene"/ \
            -i "$img_path" \
            --test_iterations 1 \
            -m out_ablation/mipnerf360/"$scene"/"$method_name" \
            --eval \
            --exp_name "$scene"-"$method_name" \
            --wandb_group "$scene"-ablation \
            --iterations "$iter" \
            --no_motion_mask \
            --disable_truncated_gradient || tee -a "$LOG_FILE"
    fi

    echo "Processing $scene without: delayed pruning"
    method_name=no-delayed-pruning
    if [ -f ./out_ablation/mipnerf360/"$scene"/"$method_name"/point_cloud/iteration_30000/point_cloud.ply ]; then
        echo "Skipping training for $scene, method $method_name, because the model already exists"
    else
        MODEL_INIT=COLMAP python train.py \
            -s "$root/mipnerf360"/"$scene"/ \
            -i "$img_path" \
            --test_iterations 1 \
            -m out_ablation/mipnerf360/"$scene"/"$method_name" \
            --eval \
            --exp_name "$scene"-"$method_name" \
            --wandb_group "$scene"-ablation \
            --no_motion_mask \
            --iterations "$iter" \
            --pruning_from_iter 0 || tee -a "$LOG_FILE"
    fi

    # echo "Processing $scene without: ADC schedule"
    # method_name=no-adc-schedule
    # if [ -f ./out_ablation/mipnerf360/"$scene"/"$method_name"/point_cloud/iteration_30000/point_cloud.ply ]; then
    #     echo "Skipping training for $scene, method $method_name, because the model already exists"
    # else
    #     MODEL_INIT=COLMAP INTERLEAVED_ADC=0 python train.py \
    #         -s "$root/mipnerf360"/"$scene"/ \
    #         -i "$img_path" \
    #         --test_iterations 1 \
    #         -m out_ablation/mipnerf360/"$scene"/"$method_name" \
    #         --eval \
    #         --exp_name "$scene"-"$method_name" \
    #         --wandb_group "$scene"-ablation \
    #         --iterations "$iter" \
    #         --no_motion_mask  || tee -a "$LOG_FILE"
    # fi


    echo "Processing $scene without: pull strength schedule"
    method_name=no-pull-strength-schedule
    if [ -f ./out_ablation/mipnerf360/"$scene"/"$method_name"/point_cloud/iteration_30000/point_cloud.ply ]; then
        echo "Skipping training for $scene, method $method_name, because the model already exists"
    else
        MODEL_INIT=COLMAP python train.py \
            -s "$root/mipnerf360"/"$scene"/ \
            -i "$img_path" \
            --test_iterations 1 \
            -m out_ablation/mipnerf360/"$scene"/"$method_name" \
            --eval \
            --exp_name "$scene"-"$method_name" \
            --wandb_group "$scene"-ablation \
            --no_motion_mask \
            --iterations "$iter" \
            --disable_pull_strength_schedule || tee -a "$LOG_FILE"
    fi
done
