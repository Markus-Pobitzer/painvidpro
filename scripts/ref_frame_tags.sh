#!/bin/bash

# Activate venv if not already
source .venv/bin/activate

# Generates reference frame varaitions
BASE_DIR="$1"

PIPE_PATH_LIST=(
    "${BASE_DIR}/acrylic/acrylic"
    "${BASE_DIR}/loomis_portrait/loomis-portrai-rmbg"
    "${BASE_DIR}/oil/oil"
    "${BASE_DIR}/pencil_drawing/pencil_drawing"
)

# Activate venv if not already
source .venv/bin/activate

# Iterate over each pipeline path
for PIPE_PATH in "${PIPE_PATH_LIST[@]}"; do
    echo "Working on $PIPE_PATH"
    # sbatch scripts/process_video/ref_frame_variations_yoda_slurm.sh "$PIPE_PATH" "ProcessorRefFrameVariations"

    # Run the processing script
    python -m painvidpro.cli.create_ref_frames_for_pipeline \
        --base_dir "$PIPE_PATH" \
        --processor_name "ProcessorRefFrameTagging"
done
