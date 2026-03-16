#!/bin/bash
# ============================================================================
# SLURM job script: Run 4 — SAM vit_b (1 GPU)
#
# Usage:
#   sbatch slurm/run_sam.sh
#   tail -f logs/sam_%j.out
# ============================================================================

#SBATCH --job-name=plants-sam
#SBATCH --partition=gpu-qi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sam_%j.out
#SBATCH --error=logs/sam_%j.err

set -euo pipefail

module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

PROJECT_DIR="$HOME/plants"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Run:    4 — SAM vit_b (1 GPU)"
echo "============================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
"

# Verify SAM checkpoint
SAM_CKPT="output/checkpoints/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_CKPT" ]; then
    echo "ERROR: SAM checkpoint not found at $SAM_CKPT"
    exit 1
fi

mkdir -p logs

stdbuf -oL python train/run_grid_training.py \
    --only 4 \
    2>&1 | tee "logs/sam_${SLURM_JOB_ID}.log"

echo ""
echo "SAM complete: $(date)"
