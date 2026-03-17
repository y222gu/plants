#!/bin/bash
# ============================================================================
# SLURM job script: Run 4 — SAM vit_b (single GPU, large batch)
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
echo "Run:    4 — SAM vit_b (single GPU, batch 32)"
echo "============================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

# Verify SAM checkpoint
SAM_CKPT="output/checkpoints/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_CKPT" ]; then
    echo "ERROR: SAM checkpoint not found at $SAM_CKPT"
    exit 1
fi

mkdir -p logs

# Single GPU — no DDP overhead for small mask decoder
stdbuf -oL python train/train_sam.py \
    --strategy A \
    --sam-type vit_b \
    --num-classes 5 \
    --img-size 1024 \
    --batch-size 32 \
    --epochs 300 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --patience 15 \
    --save-every 5 \
    --num-workers 8 \
    2>&1 | tee "logs/sam_${SLURM_JOB_ID}.log"

echo ""
echo "SAM training complete: $(date)"

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Running SAM evaluation..."
echo "============================================"

SAM_BEST=$(ls -t output/runs/sam/sam_vit_b_A_c5/*/best.pth 2>/dev/null | head -1)
if [ -z "$SAM_BEST" ]; then
    echo "ERROR: No SAM best.pth found in dated subfolders"
else
    stdbuf -oL python evaluate.py \
        --model sam \
        --sam-type vit_b \
        --strategy A \
        --num-classes 5 \
        --checkpoint "$SAM_BEST" \
        2>&1 | tee "logs/sam_eval_${SLURM_JOB_ID}.log"
fi

echo ""
echo "SAM evaluation complete: $(date)"
