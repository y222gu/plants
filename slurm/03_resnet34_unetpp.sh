#!/bin/bash
# ResNet34 + UNet++, Strategy A — convolutional baseline (Fig 2f, "ResNet34+UNet++").
# Output: output/runs/unet/unetplusplus_resnet34_imagenet_equalw_drop_shuf_dfcel_semantic7c_A/
#SBATCH --job-name=resnet34_unetpp
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yifeigu/plants/logs/resnet34_unetpp_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/resnet34_unetpp_%j.err

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

echo "=== ResNet34 + UNet++, Strategy A ==="
echo "Start: $(date)"

python train/train_unet_semantic.py \
    --arch unetplusplus \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --strategy A \
    --equal-weights \
    --batch-size 8

echo "Done: $(date)"
