#!/bin/bash
#SBATCH --job-name=test-sam
#SBATCH --partition=gpu-qi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_sam_%j.out
#SBATCH --error=logs/test_sam_%j.err

set -euo pipefail
module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants
cd "$HOME/plants"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
mkdir -p logs

echo "=== TEST Run 4: SAM vit_b (2 epochs) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify SAM checkpoint
SAM_CKPT="output/checkpoints/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_CKPT" ]; then
    echo "ERROR: SAM checkpoint not found at $SAM_CKPT"
    exit 1
fi

stdbuf -oL python train/run_grid_training.py --only 4 --epochs 2 \
    2>&1 | tee "logs/test_sam_${SLURM_JOB_ID}.log"

echo "Test complete: $(date)"
