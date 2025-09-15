#!/bin/bash

# Generates reference frame varaitions
BASE_DIR="$1"

PIPE_PATH_LIST=(
    "$BASE_DIR""acrylic/acrylic"
    "$BASE_DIR""loomis_portrait/loomis-portrai-rmbg"
    "$BASE_DIR""oil/oil"
    "$BASE_DIR""pencil_drawing/pencil_drawing"
)


# Iterate over each pipeline path
for PIPE_PATH in "${PIPE_PATH_LIST[@]}"; do
    echo "Working on $PIPE_PATH"
    sbatch scripts/process_video/ref_frame_variations_yoda_slurm.sh "$PIPE_PATH" "ProcessorRefFrameVariations"
done
