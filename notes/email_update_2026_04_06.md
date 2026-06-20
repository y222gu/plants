Subject: Root segmentation update — encoder ablation, error analysis, new architectures

Hi all,

Quick update on experiments from the past two days:

**Encoder pretraining**: Tested ImageNet vs MicroNet (materials microscopy) vs random init on U-Net++ semantic. ImageNet→MicroNet gave no improvement (0.869 vs 0.871). Random init was 6.5% worse (0.806), confirming pretraining matters but ImageNet is sufficient.

**Aerenchyma weight tuning**: Millet aerenchyma occupies only 0.47% of image pixels. Increasing CE weight from 2→10 improved aerenchyma accuracy (0.73→0.87) but degraded cortex (0.89→0.86) since they share the same spatial region. Net mIoU slightly lower. This tradeoff appears inherent to the weight-based approach.

**Error analysis**: Identified 5 failure modes across models. The hardest case is Millet aerenchyma (many tiny holes vs large lacunae in Rice/Sorghum). YOLO has a unique failure where it misses Whole Root detection on ~3% of samples. Importantly, YOLO and U-Net++ errors are weakly correlated (r=0.62), suggesting ensemble potential.

**Hybrid cell→structure approach (didn't work)**: Tested using class-agnostic cell segmentation (Cellpose-SAM, PlantSeg) to get individual cell masks, then majority-voting structure labels onto cells. Even with perfect GT labels, this underperformed standalone models by 2-15% mIoU. Thin ring structures (endodermis, exodermis) can't be captured by majority-voting over cell boundaries.

**New models in progress**:
- Fine-tuning PlantSeg UNet2D (5.4M params, pretrained on Arabidopsis plant tissue) — currently training, pixel_acc 0.96 at epoch 34
- Fine-tuning Cellpose-SAM ViT-L per-class at 512 resolution — queued next

YOLO and U-Net++ semantic remain the best so far (Bio-7 mIoU 0.86-0.87). Will report new architecture results once training completes.

Best,
Yifei
