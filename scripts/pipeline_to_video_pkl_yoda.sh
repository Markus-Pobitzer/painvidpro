#!/bin/bash

# Examlpe how several pipelines can be stored as a video_pkl dataset

PIPE_PATH_LIST=(
    "/data/mpobitzer/datataset/acrylic/acrylic"
    "/data/mpobitzer/datataset/loomis_portrait/loomis-portrai-rmbg"
    "/data/mpobitzer/datataset/oil/oil"
    "/data/mpobitzer/datataset/pencil_drawing/pencil_drawing"
)
OUT_DIR="/data/mpobitzer/datataset/video_as_pkl_combined"
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
