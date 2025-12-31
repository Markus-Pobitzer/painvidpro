#!/bin/bash

# Compiles the complete dataset
BASE_DIR="$1"

PIPE_PATH_LIST=(
    "${BASE_DIR}/acrylic"
    "${BASE_DIR}/loomis_portrait"
    "${BASE_DIR}/oil"
    "${BASE_DIR}/pencil_drawing"
)

VIDEO_SOURCE_LIST=(
    "video_sources/compiled/acrylic.jsonl"
    "video_sources/compiled/loomis_portrait.jsonl"
    "video_sources/compiled/oil.jsonl"
    "video_sources/compiled/pencil.jsonl"
)

source .venv/bin/activate

# Iterate over each pipeline path with index
for i in "${!PIPE_PATH_LIST[@]}"; do
    PIPE_PATH="${PIPE_PATH_LIST[$i]}"
    VIDEO_SOURCE="${VIDEO_SOURCE_LIST[$i]}"
    if [ -f "${PIPE_PATH}/pipeline.json" ]; then
        echo "Pipeline already found under $PIPE_PATH skipping creation and loading of video sources"
    else
        echo "Working on $PIPE_PATH with video source $VIDEO_SOURCE"
        python -m painvidpro.cli.create_pipeline \
            --base-dir "$PIPE_PATH" \
            --files "$VIDEO_SOURCE"
    fi
done

# Actual processing
for i in "${!PIPE_PATH_LIST[@]}"; do
    PIPE_PATH="${PIPE_PATH_LIST[$i]}"
    find "$PIPE_PATH" -maxdepth 1 -type d ! -path "$PIPE_PATH" | while read -r SOURCE_DIR; do
        SOURCE_NAME=$(basename "$SOURCE_DIR")
        find "$SOURCE_DIR" -maxdepth 1 -type d ! -path "$SOURCE_DIR" | while read -r VIDEO_DIR; do
            VIDEO_ID=$(basename "$VIDEO_DIR")
            echo "Processing: Source=$SOURCE_NAME, Video ID=$VIDEO_ID"
            # Change following script to make it compatible to your cluster
            sbatch scripts/process_video/yoda_slurm.sh "$PIPE_PATH" "$SOURCE_NAME" "$VIDEO_ID"

        done
    done
done

echo "All jobs submitted!"
