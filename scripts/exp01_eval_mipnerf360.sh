#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root> <output_dir>"
    exit 1
fi

set -e

root=$1
output_dir=$2
scene_list=(bicycle bonsai counter garden kitchen room stump)
for scene in "${scene_list[@]}"; do
    echo "Evaluating $scene"
    for model_init in random colmap; do
        for is_baseline in 1 0; do
            echo "Evaluating MipNeRF360 $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
            if [ $is_baseline -eq 1 ]; then
                method_name=baseline
            else
                method_name=ours-latest-2
            fi
            if [ ! -f ./out_xp1/mipnerf360/"$scene"/$method_name/$model_init-$method_name/point_cloud/iteration_30000/point_cloud.ply ]; then
                echo "No model found for MipNeRF360 $scene with IS_BASELINE=$is_baseline and MODEL_INIT=$model_init"
		exit 1
            fi
	    # python scripts/log_wandb_metrics_to_json.py export-metrics "$model_init-$method_name" "mipnerf360-$scene-xp-1" --output "$output_dir"/"$model_init-$method_name-$scene.json"
            img_path="images_2"
            case "$scene" in bicycle|stump|garden)
                echo "Using 1/4 resolution"
                img_path="images_4"
                ;;
            esac
            MODEL_INIT=colmap python render.py \
                -s "$root"/"$scene"/ \
                -i "$img_path" \
                -m out_xp1/mipnerf360/"$scene"/"$method_name"/"$model_init-$method_name" \
                --eval \
                --no_motion_mask \
                --skip_train

            metrics_output=$(python metrics.py \
                -m out_xp1/mipnerf360/"$scene"/"$method_name"/"$model_init-$method_name")
            psnr=$(echo "$metrics_output" | grep -oP '(?m)^PSNR\s*:\s*\K[\d.]+' || echo "0")
            lpips_vgg=$(echo "$metrics_output" | grep -oP '(?m)^LPIPS \(VGG\)\s*:\s*\K[\d.]+' || echo "0")
            lpips_alex=$(echo "$metrics_output" | grep -oP '(?m)^LPIPS \(ALEX\)\s*:\s*\K[\d.]+' || echo "0")
            ssim=$(echo "$metrics_output" | grep -oP '(?m)^SSIM\s*:\s*\K[\d.]+' || echo "0")
            msssim=$(echo "$metrics_output" | grep -oP '(?m)^MSSSIM\s*:\s*\K[\d.]+' || echo "0")

            echo "PSNR: $psnr, LPIPS (VGG): $lpips_vgg, LPIPS (ALEX): $lpips_alex, SSIM: $ssim, MSSSIM: $msssim"
	done
    done
done

