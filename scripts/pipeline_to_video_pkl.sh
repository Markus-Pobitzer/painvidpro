#!/bin/bash

# Examlpe how several pipelines can be stored as a video_pkl dataset
BASE_DIR="$1"

PIPE_PATH_LIST=(
    "${BASE_DIR}/acrylic/acrylic"
    "${BASE_DIR}/loomis_portrait/loomis-portrai-rmbg"
    "${BASE_DIR}/oil/oil"
    "${BASE_DIR}/pencil_drawing/pencil_drawing"
)
OUT_DIR="${BASE_DIR}/video_as_pkl_combined"
SEED=42

# Make sure to remove stale dataset
echo "Cleaning $OUT_DIR"
rm -rf "$OUT_DIR"

# Iterate over each pipeline path
for PIPE_PATH in "${PIPE_PATH_LIST[@]}"; do
    echo "Exporting $PIPE_PATH"
    python src/painvidpro/cli/export_pipeline_to_vidoe_pkl.py \
        "$PIPE_PATH" \
        "$OUT_DIR" \
        --seed="$SEED"
done
