#!/bin/bash
set -e
#Barn  
LOG_FILE="$(dirname "$0")/tankntemple.txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== TanknTemple experiment started at $(date) ==="
scene_list=(Barn Church Courthouse Caterpillar Ignatius Meetingroom Truck)
for scene in "${scene_list[@]}"; do
    echo "Processing $scene"
    # if ! python convert.py --source_path /workspace/gaussian-splatting-truncated-grad/data/tankntemple/Caterpi$scene; then
    #     echo "Conversion failed for $scene"
    #     ./../scripts/notify.sh "Conversion failed for TanknTemple $scene"
    # else
    #     echo "Conversion successful for $scene"
    #     ./../scripts/notify.sh "Conversion successful for TanknTemple $scene"
    # fi
    for is_baseline in 1 0; do
        for model_init in colmap random random_mcmc; do
            echo "Training TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
            if [ $is_baseline -eq 1 ]; then
                method_name=baseline
            else
                method_name=ours-latest
            fi
            if [ -f ./out_xp1/tankntemple/$scene/$method_name/$model_init-$method_name/point_cloud/iteration_30000/point_cloud.ply ]; then
                echo "Skipping training for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init because the model already exists"
                continue
            fi
            if ! IS_BASELINE=$is_baseline MODEL_INIT=$model_init ./scripts/run_exp_1_general.sh /workspace/gaussian-splatting-truncated-grad/data/tankntemple $scene 0 $method_name $model_init-$method_name "images"; then
                echo "Training failed for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
                ./../scripts/notify.sh "Training failed for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
            else
                echo "Training successful for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
                ./../scripts/notify.sh "Training successful for TanknTemple $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
            fi
        done
    done

done
