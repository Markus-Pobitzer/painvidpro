#!/bin/bash
#SBATCH -A staff
#SBATCH -J process_video          # Job name
#SBATCH -t 06:00:00              # Max runtime 6 hours
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -N 1                     # Use 1 node
#SBATCH --cpus-per-task=4        # 4 CPU cores per task
#SBATCH --mem=8G                 # 8GB memory
#SBATCH -o slurm-%j.out          # Output log (job ID in name)
#SBATCH -e slurm-%j.err          # Error log (job ID in name)

# Validate input parameters
if [ $# -ne 3 ]; then
    echo "Usage: $0 <base_directory> <source> <video_id>"
    exit 1
fi

BASE_DIR="$1"
SOURCE="$2"
VIDEO_ID="$3"

# Activate poetry if not already
eval $(poetry env activate)

echo "Processing video with args:" "$BASE_DIR" "$SOURCE" "$VIDEO_ID"

# Run the processing script
python -m painvidpro.pipeline.process_video_by_id.py \
    --dir "$BASE_DIR" \
    --source "$SOURCE" \
    --video_id "$VIDEO_ID"

# Exit with status from python command
exit $?
