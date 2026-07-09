#!/bin/bash
# RADIX ablation: drop both channel-level augmentations (Supp Table 8 row).
# ChannelDropout and ChannelShuffle both off (p=0). Geometric/intensity augs unchanged.
# Output: output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_noaug_dfcel_semantic7c_A/
#SBATCH --job-name=radix_noaug
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yifeigu/plants/logs/radix_noaug_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/radix_noaug_%j.err

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

echo "=== Ablation: RADIX without channel-level augmentation (Strategy A) ==="
echo "Start: $(date)"

python train/train_timm_semantic.py \
    --encoder hf:facebook/dinov3-vits16-pretrain-lvd1689m \
    --decoder dpt_meta \
    --strategy A \
    --equal-weights \
    --channel-dropout 0.0 \
    --channel-shuffle 0.0 \
    --batch-size 8

echo "Done: $(date)"
