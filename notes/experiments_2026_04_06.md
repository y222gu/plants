# Experiments Summary — April 5-6, 2026

## 1. U-Net++ Encoder Pretraining Ablation
**Question**: Does microscopy-specific pretraining beat ImageNet for fluorescence images?

| Encoder weights | Bio-7 mIoU |
|----------------|-----------|
| ImageNet (baseline) | 0.871 |
| ImageNet→MicroNet (materials microscopy) | 0.869 |
| Random init (no pretraining) | 0.806 |

**Conclusion**: ImageNet pretraining is sufficient. MicroNet (materials microscopy) offers no benefit — domain gap too large. Pretraining itself matters (+6.5% over random init).

## 2. Aerenchyma CE Weight Ablation
**Question**: Does increasing aerenchyma class weight improve detection of tiny Millet aerenchyma?

| aer weight | Aer accuracy | Cortex accuracy | Bio-7 mIoU |
|-----------|-------------|----------------|-----------|
| 2 (baseline) | 0.73 | 0.89 | 0.871 |
| 10 | 0.87 | 0.86 | 0.866 |
| 20 | 0.91 | 0.83 | 0.864 |

**Conclusion**: Higher weight improves aerenchyma but degrades cortex (they share spatial region). Net effect is slightly negative on overall mIoU. The tradeoff is inherent — can't improve one without the other via weight tuning alone.

## 3. Error Analysis Across Models
**Question**: What types of errors do the models make?

- **Hardest class**: Aerenchyma (IoU 0.60), especially Millet (IoU 0.23) — tiny scattered holes vs large lacunae in other cereals
- **Hardest species**: Millet aerenchyma, Rice Wox10-50 genotype (irregular morphology)
- **YOLO-specific failure**: Completely misses Whole Root on 6/196 samples (detection threshold issue)
- **Thin ring classes** (Epidermis, Exodermis): Inherently IoU-sensitive to small boundary shifts
- **YOLO vs UNet++ errors are weakly correlated** (r=0.62), suggesting ensemble potential

## 4. Cellpose-SAM Zero-Shot Cell Segmentation
**Question**: Can Cellpose-SAM segment individual cells in our root images?

- Detects 228-480 cells per image zero-shot across all 4 species
- Best preprocessing: **CLAHE + inverted DAPI** (bright cells, dark walls) → +36% more cells detected
- Clean instance masks directly — no post-processing needed
- Better than PlantSeg zero-shot on our data

## 5. PlantSeg/PanSeg Zero-Shot Cell Segmentation
**Question**: Can PlantSeg's boundary UNet work on our images?

- Best preprocessing: **MaxProj + CLAHE + Unsharp mask** → up to 1330 cells detected
- Cortex region still has merging issues (model trained on Arabidopsis, not cereal roots)
- Requires dt_watershed post-processing step
- Weaker than Cellpose-SAM but detects different cells in some regions

## 6. Hybrid Approach: Cell Segmentation + Structure Labels
**Question**: Can we use generic cell masks + majority-vote from trained models (or GT) to produce better structural segmentation?

| Method | mIoU (4 inner classes, excl WR/Epi/Aer) |
|--------|----------------------------------------|
| YOLO standalone | 0.926-0.948 |
| UNet++ standalone | 0.809-0.960 |
| PanSeg cells + GT labels | 0.773-0.937 |
| Cellpose-SAM cells + GT labels | 0.680-0.884 |

**Conclusion**: Hybrid approach **underperforms standalone models** even with perfect GT labels. The bottleneck is that generic cell boundaries don't align with anatomical structure boundaries (endodermis is a single-cell-thick ring — cells straddle the boundary). End-to-end trained models remain superior.

## 7. Fine-Tuning PanSeg UNet2D (In Progress)
**Question**: Does a plant-tissue-pretrained UNet outperform ImageNet-pretrained U-Net++?

- PanSeg UNet2D (5.4M params, pretrained on Arabidopsis stem cells) fine-tuned for our 6 annotation classes
- Same loss (BCE+Dice), augmentations, and data as U-Net++ multilabel
- Training on Lambda GH200, batch 16, currently at epoch 34, pixel_acc 0.96
- Results pending

## 8. Fine-Tuning Cellpose-SAM (Queued)
**Question**: Can Cellpose-SAM's ViT-L be fine-tuned per-class at 512 resolution?

- One model per annotation class (6 total), fine-tuned from cpsam pretrained weights
- 512×512 input (position embeddings interpolated from 32×32 → 64×64)
- Our albumentations augmentation pipeline (not Cellpose's built-in crops)
- Queued to start after PanSeg training completes

---

## Key Takeaways

1. **YOLO and U-Net++ semantic remain the best models** for this task (Bio-7 mIoU 0.86-0.87)
2. **The hybrid cell→structure approach doesn't work** — thin ring structures can't be captured by majority-voting over cell masks
3. **Aerenchyma in Millet is the hardest problem** — 0.47% of image pixels, weight tuning has diminishing returns
4. **ImageNet pretraining is surprisingly hard to beat** with domain-specific alternatives
5. **Two new architectures being tested**: PanSeg UNet2D (plant-pretrained) and Cellpose-SAM (ViT-L, per-class)
