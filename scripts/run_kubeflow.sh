#!/bin/bash

set -e  # Exit on any error
set -u  # Exit on undefined variables

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset_root> <white_bg_bool> <method_name> <exp_name> <scene_name>."
    exit 1
fi

if [ -z "${NOTION_ACCESS_TOKEN:-}" ]; then
    echo "Error: NOTION_ACCESS_TOKEN not set"
    exit 1
fi

if [ -z "${GITHUB_READ:-}" ]; then
    echo "Error: GITHUB_READ not set"
    exit 1
fi

dataset_root=$1
white_bg=$2
method_name=$3
exp_name=$4
scene_name=$5
model_path="/home/models/3dgs/$scene_name"
datasrc_id=29adf5ca-4835-80aa-8362-000b406a336e
method="3dgs_$method_name"

post_to_notion() {
    local scene_name=$1
    local psnr=$2
    local lpips=$3
    local ssim=$4
    
    curl --location 'https://api.notion.com/v1/pages' \
    --header 'Content-Type: application/json' \
    --header 'Notion-Version: 2022-06-28' \
    --header "Authorization: Bearer $NOTION_ACCESS_TOKEN" \
    --data "{
      \"parent\": {
        \"data_source_id\": \"$datasrc_id\"
      },
      \"properties\": {
        \"Method\": {
          \"title\": [{\"text\":{\"content\": \"$method\"}}]
        },
        \"Scene\": {
          \"rich_text\": [{\"text\":{\"content\": \"$scene_name\"}}]
        }
        \"PSNR\": {
          \"number\": $psnr
        },
        \"LPIPS\": {
          \"number\": $lpips
        },
        \"SSIM\": {
          \"number\": $ssim
        }
      }
    }"
}

post_error_to_notion() {
    local method="$method"
    local scene_name=$1
    local error_msg=$2
    local line_number=$3
    
    echo "[!] Posting error to Notion: $error_msg"
    
    curl --location 'https://api.notion.com/v1/pages' \
    --header 'Content-Type: application/json' \
    --header 'Notion-Version: 2022-06-28' \
    --header "Authorization: Bearer $NOTION_ACCESS_TOKEN" \
    --data "{
      \"parent\": {
        \"data_source_id\": \"$datasrc_id\"
      },
      \"properties\": {
        \"Method\": {
          \"title\": [{\"text\":{\"content\": \"$method\"}}]
        },
        \"Status\": {
          \"status\": {\"name\": \"Failed\"}
        },
        \"Error\": {
          \"rich_text\": [{\"text\":{\"content\": \"line $line_number\"}}]
        }
      }
    }"
}

handle_error() {
    local exit_code=$?
    local line_number=$1
    
    echo "[!] Error occurred on line $line_number with exit code $exit_code"
    local last_command="${BASH_COMMAND}"
    post_error_to_notion "$scene_name" "$last_command failed (exit code: $exit_code)" "$line_number"
}


trap 'handle_error $LINENO' ERR

if [ -d $method ]; then
    rm -rf $method 2>/dev/null || true
fi
git clone --depth 1 --recursive https://github.com/DubiousCactus/gaussian-splatting-truncated-grad $method

args=()
[[ "$white_bg" == "1" ]] && args+=(--white_background)
for i in {5..300..40};
do
    python train.py -s "$dataset_root/$scene_name/colmap_$i" \
    --test_iterations 1 \
    -m "$model_path/frame_$i" \
    --eval \
    --exp_name "$exp_name" \
    --wandb_group "$scene_name-xp-1"\
    "${args[@]}"
done

python render.py \
  -s "$dataset_root/$scene_name" \
  --conf "$config_path" \
  --model_path "$model_path" \
  --resolution 1 \
  --skip_train

metrics_output=$(python metrics_ours.py -m "$model_path" )

psnr=$(echo "$metrics_output" | grep -oP 'PSNR\s*:\s*\K[\d.]+' || echo "0")
lpips=$(echo "$metrics_output" | grep -oP 'LPIPS\s*:\s*\K[\d.]+' || echo "0")
ssim=$(echo "$metrics_output" | grep -oP 'SSIM\s*:\s*\K[\d.]+' || echo "0")

# post_to_notion "$scene_name" "$psnr" "$lpips" "$ssim"

chmod 777 -R "$model_path"

