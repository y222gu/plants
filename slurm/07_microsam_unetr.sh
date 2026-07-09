#!/bin/bash
# MicroSAM (SAM ViT-B vit_b_lm) + UNETR with LoRA, Strategy A — SAM-foundation comparison
# (Fig 2f, "MicroSAM+UNETR"). LoRA rank 4, alpha 1.0 on attention projections; UNETR decoder
# taps blocks 2/5/8/11. Batch size 8 per the manuscript (full fine-tune at 1024x1024 would
# exceed memory). Adapters + decoder + head trainable; rest of encoder frozen.
# Output: output/runs/sam_semantic/sam_vit_b_lm_lora_r4_semantic7c_A/
#SBATCH --job-name=microsam_unetr
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yifeigu/plants/logs/microsam_unetr_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/microsam_unetr_%j.err

set -e

cd ~/plants
module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

sleep $((${SLURM_ARRAY_TASK_ID:-0} * 30))

for attempt in 1 2 3 4 5; do
    if python -c "import torch; torch.cuda.init(); torch.randn(32,32,device='cuda')@torch.randn(32,32,device='cuda')"; then
        break
    fi
    [ "$attempt" -eq 5 ] && { scontrol requeue "$SLURM_JOB_ID"; sleep 5; exit 1; }
    sleep 60
done

echo "=== MicroSAM (vit_b_lm) + UNETR (LoRA r4), Strategy A ==="
echo "Start: $(date)"

python train/train_sam_semantic.py \
    --model-type vit_b_lm \
    --adapter lora \
    --lora-rank 4 \
    --lora-alpha 1.0 \
    --strategy A \
    --equal-weights \
    --batch-size 8

echo "Done: $(date)"
