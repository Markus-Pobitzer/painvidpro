#!/bin/bash

BASE_DIR="$1"

if [ $# -ne 1 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: $0 <base_directory>"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

find "$BASE_DIR" -maxdepth 1 -type d ! -path "$BASE_DIR" | while read -r SOURCE_DIR; do
    SOURCE_NAME=$(basename "$SOURCE_DIR")
    find "$SOURCE_DIR" -maxdepth 1 -type d ! -path "$SOURCE_DIR" | while read -r VIDEO_DIR; do
        VIDEO_ID=$(basename "$VIDEO_DIR")
        echo "Processing: Source=$SOURCE_NAME, Video ID=$VIDEO_ID"
        sbatch scripts/process_video/yoda_slurm.sh "$BASE_DIR" "$SOURCE_NAME" "$VIDEO_ID"

    done
done

echo "Processing complete for all videos in $BASE_DIR"
