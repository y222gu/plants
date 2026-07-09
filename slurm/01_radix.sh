#!/bin/bash
# RADIX: DINOv3-S/16 + DPT-meta, Strategy A (unified, all species/microscopes).
# Headline model in Figs 1d, 2a-e, and the "RADIX" point in Fig 2f.
# Output: output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A/
#SBATCH --job-name=radix
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yifeigu/plants/logs/radix_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/radix_%j.err

set -e

cd ~/plants
module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

# DINOv3 weights are gated on HF — pre-downloaded; run offline on the compute node.
export HF_HUB_OFFLINE=1

# Stagger CUDA init to avoid races on shared GPU nodes (see CLAUDE.md).
sleep $((${SLURM_ARRAY_TASK_ID:-0} * 30))

# CUDA smoke test with retry + self-requeue if we land on a stuck GPU.
for attempt in 1 2 3 4 5; do
    if python -c "import torch; torch.cuda.init(); torch.randn(32,32,device='cuda')@torch.randn(32,32,device='cuda')"; then
        break
    fi
    [ "$attempt" -eq 5 ] && { scontrol requeue "$SLURM_JOB_ID"; sleep 5; exit 1; }
    sleep 60
done

echo "=== RADIX (DINOv3-S/16 + DPT-meta, Strategy A) ==="
echo "Start: $(date)"

python train/train_timm_semantic.py \
    --encoder hf:facebook/dinov3-vits16-pretrain-lvd1689m \
    --decoder dpt_meta \
    --strategy A \
    --equal-weights \
    --batch-size 8

echo "Done: $(date)"
