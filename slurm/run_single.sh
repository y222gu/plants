#!/bin/bash
# ============================================================================
# SLURM job script: Single run (debug / individual model)
#
# Usage:
#   sbatch slurm/run_single.sh --only 1              # Run only YOLO
#   sbatch slurm/run_single.sh --only 4 --epochs 5   # Quick SAM test
#   sbatch slurm/run_single.sh --only 2 3             # Both U-Net++ runs
# ============================================================================

#SBATCH --job-name=plants-single
#SBATCH --partition=gpu-qi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err

set -euo pipefail

module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

PROJECT_DIR="$HOME/plants"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# ── GPU info ─────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Args:   $@"
echo "============================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
"

echo ""
mkdir -p logs

# ── Run ──────────────────────────────────────────────────────────────────────
stdbuf -oL python train/run_grid_training.py \
    "$@" \
    2>&1 | tee "logs/single_${SLURM_JOB_ID}.log"

echo ""
echo "Job complete: $(date)"
