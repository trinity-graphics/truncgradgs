#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <gt_dir> <render_dir> <n_frames_per_video> <out_dir>"
    exit 1
fi

GT_DIR="$1"
RENDER_DIR="$2"
FRAMES_PER_VIDEO=$3

OUT_GT="$4/y4m_encoded_gt"
OUT_RENDER="$4/y4m_encoded_render"

JOBS=${JOBS:-$(nproc)}

mkdir -p "$OUT_GT" "$OUT_RENDER"

detect_pattern () {
    dir="$1"

    if ls "$dir"/frame_*.* >/dev/null 2>&1; then
        echo "frame_%05d"
    else
        echo "%05d"
    fi
}

GT_PATTERN=$(detect_pattern "$GT_DIR")
RENDER_PATTERN=$(detect_pattern "$RENDER_DIR")

EXT=$(ls "$GT_DIR" | head -n1 | awk -F. '{print $NF}')

TOTAL=$(ls "$GT_DIR"/*.$EXT | wc -l)
VIDEOS=$(( TOTAL / FRAMES_PER_VIDEO ))

echo "Frames total: $TOTAL"
echo "Videos: $VIDEOS"
echo "Parallel jobs: $JOBS"

encode_pair () {

    idx="$1"
    start=$(( idx * FRAMES_PER_VIDEO ))

    gt_out=$(printf "%s/video_%04d.y4m" "$OUT_GT" "$idx")
    render_out=$(printf "%s/video_%04d.y4m" "$OUT_RENDER" "$idx")

    ffmpeg -loglevel error -y \
        -start_number "$start" \
        -i "$GT_DIR/$GT_PATTERN.$EXT" \
        -frames:v $FRAMES_PER_VIDEO \
        -pix_fmt yuv420p \
        -f yuv4mpegpipe \
        "$gt_out"

    ffmpeg -loglevel error -y \
        -start_number "$start" \
        -i "$RENDER_DIR/$RENDER_PATTERN.$EXT" \
        -frames:v $FRAMES_PER_VIDEO \
        -pix_fmt yuv420p \
        -f yuv4mpegpipe \
        "$render_out"

    echo "Finished video $idx"
}

export -f encode_pair
export GT_DIR RENDER_DIR OUT_GT OUT_RENDER
export FRAMES_PER_VIDEO GT_PATTERN RENDER_PATTERN EXT

seq 0 $((VIDEOS-1)) | xargs -I{} -P "$JOBS" bash -c 'encode_pair "$@"' _ {}
