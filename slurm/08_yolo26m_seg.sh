#!/bin/bash
# YOLO26m-seg, Strategy A — instance-segmentation baseline (Fig 2f, "YOLO26m-seg").
# Fine-tuned from a COCO-pretrained checkpoint to detect the six annotated raw classes
# (outer epidermis contour, aerenchyma, outer/inner endodermis, outer/inner exodermis).
# At inference the polygons are rasterized and combined into 7-class semantic masks for
# comparison with the other models. overlap_mask=False is required (see CLAUDE.md).
# Output: output/runs/yolo/yolo26m-seg_semantic7c_A/
#SBATCH --job-name=yolo26m_seg
#SBATCH --partition=gpu-qi
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/yifeigu/plants/logs/yolo26m_seg_%j.out
#SBATCH --error=/home/yifeigu/plants/logs/yolo26m_seg_%j.err

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

echo "=== YOLO26m-seg, Strategy A ==="
echo "Start: $(date)"

python train/train_yolo.py \
    --model yolo26m-seg \
    --strategy A \
    --num-classes 6 \
    --batch-size 16

echo "Done: $(date)"
