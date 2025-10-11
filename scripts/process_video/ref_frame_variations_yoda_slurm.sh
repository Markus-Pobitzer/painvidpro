#!/bin/bash
#SBATCH -A staff
#SBATCH -J ref_frame
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1                     # Use 1 node
#SBATCH --cpus-per-task=4        # 4 CPU cores per task
#SBATCH --mem=8G                 # 8GB memory
#SBATCH -o slurm_log/slurm-%j.out
#SBATCH -e slurm_log/slurm-%j.err

# Validate input parameters
if [ $# -ne 2 ]; then
    echo "Usage: $0 <pipeline_base_dir> <ref_frame_procressor>"
    exit 1
fi

BASE_DIR="$1"
PROC_NAME="$2"

# Activate venv if not already
source .venv/bin/activate

echo "Processing pipeline" "$PIPE_PATH" "with processor" "$PROC_NAME"

# Run the processing script
python -m painvidpro.cli.create_ref_frames_for_pipeline \
    --base_dir "$BASE_DIR" \
    --processor_name "$PROC_NAME" \
    --remove_previous_ref_frames \
    --enable_sequential_cpu_offload

# Exit with status from python command
exit $?
