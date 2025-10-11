#!/bin/bash
#SBATCH -A staff
#SBATCH -J process_video
#SBATCH -t 24:00:00              # Max runtime 24 hours
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -N 1                     # Use 1 node
#SBATCH --cpus-per-task=6        # 6 CPU cores per task
#SBATCH --mem=24G                # 24GB memory
#SBATCH -o slurm_log/slurm-%j.out
#SBATCH -e slurm_log/slurm-%j.err

# Validate input parameters
if [ $# -ne 3 ]; then
    echo "Usage: $0 <base_directory> <source> <video_id>"
    exit 1
fi

BASE_DIR="$1"
SOURCE="$2"
VIDEO_ID="$3"

# Activate venv if not already
source .venv/bin/activate

echo "Processing video with args:" "$BASE_DIR" "$SOURCE" "$VIDEO_ID"

# Run the processing script
python -m painvidpro.pipeline.process_video_by_id \
    --dir "$BASE_DIR" \
    --source "$SOURCE" \
    --video_id "$VIDEO_ID"

# Exit with status from python command
exit $?
