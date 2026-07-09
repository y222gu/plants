#!/bin/bash
# DINOv2 + multi-scale linear head, Strategy A — decoder-isolation comparison (Fig 2f, "DINOv2+MS-Linear").
# Replaces the DPT decoder with the Meta MS-Linear recipe (4 transformer blocks → 1x1 conv head).
# Output: output/runs/timm/ms_linear_vit_small_patch14_dinov2_equalw_drop_shuf_dfcel_semantic7c_A/
#SBATCH --job-name=dinov2_mslin
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yifeigu/plants/logs/dinov2_mslin_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/dinov2_mslin_%j.err

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

echo "=== DINOv2-S/14 + MS-Linear, Strategy A ==="
echo "Start: $(date)"

python train/train_timm_semantic.py \
    --encoder vit_small_patch14_dinov2.lvd142m \
    --decoder ms_linear \
    --strategy A \
    --equal-weights \
    --batch-size 8

echo "Done: $(date)"
