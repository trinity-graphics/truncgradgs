#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root> <output_dir>"
    exit 1
fi

set -e

root=$1
output_dir=$2

scene_list=(1_4_underwater 4_5_tree alley 1_1_neon cup bouncy_balls)
for scene in "${scene_list[@]}"; do
    echo "Evaluating $scene"
    for model_init in random colmap; do
        for is_baseline in 1 0; do
            echo "Evaluating Blender $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
            if [ $is_baseline -eq 1 ]; then
                method_name=baseline
            else
                method_name=ours-latest
            fi
            exp_name="$model_init-$method_name"
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
            if [ -f "$output_dir"/"$scene/$method_name/$exp_name"_frame_"$i"/results.json ]; then
                echo "Skipping evaluation for Blender $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init because results already exist"
                continue
            fi
            if [ ! -f "$output_dir"/"$scene/$method_name/$exp_name"_frame_"$i"/point_cloud/iteration_30000/point_cloud.ply ]; then
                echo "No model found for Blender $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
		exit 1
            fi
            MODEL_INIT=colmap python render.py \
                -s "$root"/"$scene"/colmap_"$i" \
                -m "$output_dir"/"$scene"/"$method_name"/"$exp_name"_frame_"$i" \
                --eval \
                --no_motion_mask \
                --skip_train \
                "${args[@]}"

            metrics_output=$(python metrics.py \
                -m "$output_dir"/"$scene"/"$method_name"/"$exp_name"_frame_"$i")

            psnr=$(echo "$metrics_output" | grep -oP '(?m)^PSNR\s*:\s*\K[\d.]+' || echo "0")
            lpips_vgg=$(echo "$metrics_output" | grep -oP '(?m)^LPIPS \(VGG\)\s*:\s*\K[\d.]+' || echo "0")
            lpips_alex=$(echo "$metrics_output" | grep -oP '(?m)^LPIPS \(ALEX\)\s*:\s*\K[\d.]+' || echo "0")
            ssim=$(echo "$metrics_output" | grep -oP '(?m)^SSIM\s*:\s*\K[\d.]+' || echo "0")
            msssim=$(echo "$metrics_output" | grep -oP '(?m)^MSSSIM\s*:\s*\K[\d.]+' || echo "0")

            echo "PSNR: $psnr, LPIPS (VGG): $lpips_vgg, LPIPS (ALEX): $lpips_alex, SSIM: $ssim, MSSSIM: $msssim"
	done
    done
done
