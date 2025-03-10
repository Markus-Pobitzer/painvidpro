#!/bin/bash
#SBATCH -A staff
#SBATCH -J process_video
#SBATCH -t 06:00:00              # Max runtime 6 hours
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -N 1                     # Use 1 node
#SBATCH --cpus-per-task=4        # 4 CPU cores per task
#SBATCH --mem=8G                 # 8GB memory

# Validate input parameters
if [ $# -ne 3 ]; then
    echo "Usage: $0 <base_directory> <source> <video_id>"
    exit 1
fi

BASE_DIR="$1"
SOURCE="$2"
VIDEO_ID="$3"

# Sanitize BASE_DIR and SOURCE for use in filenames
SANITIZED_BASE_DIR=$(echo "$BASE_DIR" | tr '/' '_' | tr -cd '[:alnum:]._-')
SANITIZED_SOURCE=$(echo "$SOURCE" | tr '/' '_' | tr -cd '[:alnum:]._-')

# Set output and error log files with sanitized names
#SBATCH -o slurm_log/slurm-%j_${SANITIZED_BASE_DIR}_${SANITIZED_SOURCE}.out
#SBATCH -e slurm_log/slurm-%j_${SANITIZED_BASE_DIR}_${SANITIZED_SOURCE}.err


# Activate poetry if not already
eval $(poetry env activate)

echo "Processing video with args:" "$BASE_DIR" "$SOURCE" "$VIDEO_ID"

# Run the processing script
python -m painvidpro.pipeline.process_video_by_id \
    --dir "$BASE_DIR" \
    --source "$SOURCE" \
    --video_id "$VIDEO_ID"

# Exit with status from python command
exit $?
