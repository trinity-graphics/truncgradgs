#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_root>"
    exit 1
fi

root=$1

set -e
LOG_FILE="$(dirname "$0")/mipnerf.txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== MipNeRF360 experiment started at $(date) ==="
scene_list=(bicycle bonsai counter garden kitchen room stump)
for scene in "${scene_list[@]}"; do
    echo "Processing $scene"
    if ! python convert.py --source_path "$root"/mipnerf360/$scene; then
        echo "Conversion failed for $scene"
        ./../scripts/notify.sh "Conversion failed for MipNeRF360 $scene" &2> /dev/null || exit
    else
        echo "Conversion successful for $scene"
        ./../scripts/notify.sh "Conversion successful for MipNeRF360 $scene" &2> /dev/null || exit
    fi
    for is_baseline in 1 0; do
        for model_init in colmap random random_mcmc ; do
            echo "Training MipNeRF360 $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
            if [ $is_baseline -eq 1 ]; then
                method_name=baseline
            else
                method_name=ours-latest
            fi
            if [ -f ./out_xp1/mipnerf360/$scene/$method_name/$model_init-$method_name/point_cloud/iteration_30000/point_cloud.ply ]; then
                echo "Skipping training for MipNeRF360 $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init because the model already exists"
                continue
            fi
            img_path="images_2"
            case "$scene" in bicycle|stump|garden)
                echo "Using 1/4 resolution"
                img_path="images_4"
                ;;
            esac
            if ! IS_BASELINE=$is_baseline MODEL_INIT=$model_init ./scripts/run_exp_1_general.sh "$root/mipnerf360" "$scene" 0 $method_name $model_init-$method_name "$img_path"; then
                echo "Training failed for MipNeRF360 $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
                ./../scripts/notify.sh "Training failed for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init" &2> /dev/null || exit
            else
                echo "Training successful for MipNeRF360 $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
                ./../scripts/notify.sh "Training successful for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init" &2> /dev/null || exit
            fi
        done
    done
done
