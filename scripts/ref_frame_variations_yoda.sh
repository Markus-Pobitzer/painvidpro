#!/bin/bash

# Generates reference frame varaitions
BASE_DIR="$1"

PIPE_PATH_LIST=(
    "${BASE_DIR}/acrylic/acrylic"
    "${BASE_DIR}/loomis_portrait/loomis-portrai-rmbg"
    "${BASE_DIR}/oil/oil"
    "${BASE_DIR}/pencil_drawing/pencil_drawing"
)

PROCESSOR_LIST=(
    "ProcessorQwenEditRefFrameVariations"
    "ProcessorRefFrameVariations"
    "ProcessorQwenEditRefFrameVariations"
    "ProcessorQwenEditRefFrameVariations"
)


# Iterate over each pipeline path with index
for i in "${!PIPE_PATH_LIST[@]}"; do
    PIPE_PATH="${PIPE_PATH_LIST[$i]}"
    PROCESSOR="${PROCESSOR_LIST[$i]}"
    echo "Working on $PIPE_PATH with processor $PROCESSOR"
    sbatch scripts/process_video/ref_frame_variations_yoda_slurm.sh "$PIPE_PATH" "$PROCESSOR"
done
