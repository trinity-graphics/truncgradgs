#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_root>"
    exit 1
fi

root=$1
dataset_name=$(basename $root)

scene_list=(1_4_underwater 4_5_tree alley 1_1_neon cup bouncy_balls)
for scene in "${scene_list[@]}"; do
    echo "Processing $scene"
    for is_baseline in 1 0; do
        for model_init in random_ours random random_mcmc colmap; do
    # for i in {5..300..40}
    # do
            if [ "$scene" == "bouncy_balls" ]; then
                i=165
            else
                i=45
            fi
            args=()
            case "$scene" in
              4_5_tree|alley|bouncy_balls)
                echo "white background"
                args+=(--white_background)
                ;;
            esac
            if [ $is_baseline -eq 1 ]; then
                method_name=baseline
                script=train_baseline.py
            else
                method_name=ours-latest
                script=train.py
            fi
            exp_name="$model_init-$method_name"

            if [ -f ./out_xp1/"$dataset_name/$scene/$method_name/$exp_name"_frame_"$i"/point_cloud/iteration_30000/point_cloud.ply ]; then
                echo "Skipping training for $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init because the model already exists"
                continue
            fi
            IS_BASELINE=$is_baseline MODEL_INIT=$model_init python "$script" \
                -s "$root"/"$scene"/colmap_"$i"/ \
                --test_iterations 1 \
                -m out_xp1/"$dataset_name"/"$scene"/"$method_name"/"$exp_name"_frame_"$i" \
                --eval \
                --exp_name "$exp_name"-frame-"$i" \
                --wandb_group "$scene"-xp-1 \
                --no_motion_mask \
                "${args[@]}"
    # done
        done
    done
done
