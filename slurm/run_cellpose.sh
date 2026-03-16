#!/bin/bash
# ============================================================================
# SLURM job script: Run 5 — Cellpose v3 per-class (1 GPU)
#
# Usage:
#   sbatch slurm/run_cellpose.sh
#   tail -f logs/cellpose_%j.out
# ============================================================================

#SBATCH --job-name=plants-cellpose
#SBATCH --partition=gpu-qi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cellpose_%j.out
#SBATCH --error=logs/cellpose_%j.err

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
echo "Run:    5 — Cellpose v3 per-class (1 GPU)"
echo "============================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
"

mkdir -p logs

stdbuf -oL python train/run_grid_training.py \
    --only 5 \
    2>&1 | tee "logs/cellpose_${SLURM_JOB_ID}.log"

echo ""
echo "Cellpose complete: $(date)"
