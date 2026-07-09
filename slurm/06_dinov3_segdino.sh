#!/bin/bash
# DINOv3 + SegDINO MLP head, Strategy A — decoder-isolation comparison (Fig 2f, "SegDINO").
# Lightweight per-patch MLP head paired with the same DINOv3-S/16 encoder used by RADIX.
# Output: output/runs/timm/segdino_mlp_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A/
#SBATCH --job-name=dinov3_segdino
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yifeigu/plants/logs/dinov3_segdino_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/dinov3_segdino_%j.err

set -e

cd ~/plants
module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

export HF_HUB_OFFLINE=1

sleep $((${SLURM_ARRAY_TASK_ID:-0} * 30))

for attempt in 1 2 3 4 5; do
    if python -c "import torch; torch.cuda.init(); torch.randn(32,32,device='cuda')@torch.randn(32,32,device='cuda')"; then
        break
    fi
    [ "$attempt" -eq 5 ] && { scontrol requeue "$SLURM_JOB_ID"; sleep 5; exit 1; }
    sleep 60
done

echo "=== DINOv3-S/16 + SegDINO MLP head, Strategy A ==="
echo "Start: $(date)"

python train/train_timm_semantic.py \
    --encoder hf:facebook/dinov3-vits16-pretrain-lvd1689m \
    --decoder segdino_mlp \
    --strategy A \
    --equal-weights \
    --batch-size 8

echo "Done: $(date)"
