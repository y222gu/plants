# Plant Root Cross-Section Instance Segmentation Project

## Project Overview

Train instance segmentation models for cell-type topology in cereal plant root cross-section fluorescence microscopy images. The goal is robust segmentation across species (millet, rice, sorghum), genotypes, and microscopes (C10, Olympus, Zeiss), with potential transfer to other plants (e.g., tomato).

**GPU**: Single NVIDIA RTX 4090 (24GB VRAM)
**Dataset**: 1311 annotated samples, YOLO polygon format

---

## Directory Structure

```
C:\Users\Yifei\Documents\plants\
├── data\
│   ├── image\
│   │   └── {Species}\{Microscope}\{Exp}\{Sample}\
│   │       ├── {Sample}_DAPI.tif      # Blue channel
│   │       ├── {Sample}_FITC.tif      # Green channel
│   │       └── {Sample}_TRITC.tif     # Red channel
│   └── annotation\
│       └── {Species}_{Microscope}_{Exp}_{Sample}.txt   # YOLO polygons
├── preview\                           # Annotation preview PNGs
├── preview_annotations.py
├── CLAUDE.md
```

**Species**: Millet, Rice, Sorghum
**Microscopes**: C10, Olympus, Zeiss (Rice only)
**Channels**: DAPI (blue), FITC (green), TRITC (red) — 3 grayscale TIF images per sample

---

## Label Definitions

Annotation files use YOLO polygon format: `class_id x1 y1 x2 y2 ... xn yn` (normalized 0-1 coordinates).

### Annotated Classes (in annotation files)

| Class ID | Annotated Name | Polygon Meaning |
|----------|---------------|-----------------|
| 0 | Whole Root | Outer boundary of root (defined by epidermis edge) |
| 1 | Aerenchyma | Air-filled irregular holes in the cortex |
| 2 | Outer Endodermis | Outer ring of the endodermis layer |
| 3 | Inner Endodermis | Inner ring — actually encloses the **vascular** region |

### Actual Semantic Regions (for model training)

| Target Class | Region | How to Derive |
|-------------|--------|---------------|
| 0 | Whole Root | Use class 0 polygon directly |
| 1 | Aerenchyma | Use class 1 polygons directly |
| 2 | Endodermis | Subtract class 3 polygon from class 2 polygon (ring between outer and inner) |
| 3 | Vascular | Use class 3 polygon directly (area enclosed by inner endodermis) |

### Biology Context

- **Epidermis**: outermost cell layer, defines whole root boundary
- **Cortex**: region between epidermis and outer endodermis; contains aerenchyma
- **Endodermis**: thin layer of small rectangular cells between cortex and vascular cylinder; higher signal in 2 of 3 channels; derived by subtraction (outer - inner ring)
- **Exodermis**: exists but NOT annotated; sits between cortex and epidermis
- **Vascular cylinder**: innermost region enclosed by inner endodermis
- **Aerenchyma**: irregular bubble-like air spaces in the cortex only; never in endodermis, exodermis, or epidermis

### Annotation Rules

- Polygons can overlap but boundaries must NOT intersect
- All aerenchyma polygons are contained within the whole root polygon
- Typical per-image counts: 1 whole root, many aerenchyma, 1 outer endodermis, 1 inner endodermis

---

## Dataset Splitting

### Critical Rule
**Samples from the same experiment (`Exp{N}`) must stay together** — never split across train/test. Experiments share similar conditions, genotypes, and imaging parameters.

### Balance Requirements
- Balance across species (Millet, Rice, Sorghum)
- Balance across microscopes (C10, Olympus, Zeiss)
- Split unit = entire experiment folder

### Dataset Composition (approximate)

| Species | Microscope | Samples |
|---------|-----------|---------|
| Millet | Olympus | ~195 |
| Rice | C10 | ~50 |
| Rice | Olympus | ~518 |
| Rice | Zeiss | ~35 |
| Sorghum | C10 | ~47 |
| Sorghum | Olympus | ~466 |

### Splitting Strategies

**Strategy 1 — Standard cross-validation**: Train on all species/microscopes, test on held-out experiments from all groups. Use ~80/20 split by experiment.

**Strategy 2 — Generalizability test**: Train on Sorghum + Rice from C10 + Olympus. Test on:
- Millet/Olympus (unseen species, seen microscope)
- Rice/Zeiss (seen species, unseen microscope)

**Strategy 3 — Species-specific + ensemble**: Train separate models per species, ensemble at inference. Fallback if strategies 1-2 underperform.

---

## Data Loading & Preprocessing

### Dataset & DataLoader Requirements

- Build a configurable dataset class that accepts filters: `species=["Rice", "Sorghum"]`, `microscope=["Olympus", "C10"]`, `experiments=[...]`
- Load 3-channel TIF images and compose into a 3-channel tensor (R=TRITC, G=FITC, B=DAPI)
- Parse YOLO polygon annotations into instance masks
- Derive endodermis mask by subtracting inner endodermis from outer endodermis polygon
- Handle the label remapping: annotated class 2 → endodermis (ring), annotated class 3 → vascular

### Preprocessing

- Normalize per-channel (compute dataset mean/std, or use percentile-based normalization suitable for fluorescence microscopy)
- Handle 16-bit TIF → float32 conversion
- Resize or pad to consistent dimensions (preserve aspect ratio)

### Augmentation (fluorescence microscopy appropriate)

- Random horizontal/vertical flips
- Random 90-degree rotations
- Elastic deformation (mild)
- Random brightness/contrast per channel
- **Channel dropout**: randomly zero out 1 of 3 channels (teaches model to not rely on single channel)
- **Channel swapping/shuffling**: randomly permute channel order (improves microscope generalization)
- Random crop + resize
- Gaussian noise
- Do NOT use color jitter designed for natural images (hue shifts are meaningless for fluorescence)

---

## Models to Train & Compare

### Good practice
Have separate training script for different models. 

### 1. Fine-tune pretrained generic models

- **YOLOv8/v11-seg** (Ultralytics): Natural fit since annotations are YOLO format. Fast training, good for 4090. Start here as baseline.
- **Mask R-CNN** (torchvision / Detectron2): Classic instance segmentation. Use ResNet-50/101-FPN backbone pretrained on COCO.
- **SAM 2 (Segment Anything Model 2)**: Fine-tune with LoRA or adapter layers. Strong zero-shot capabilities may help generalization.

### 2. Fine-tune microscopy-pretrained models

- **Cellpose 2.0/3.0**: Pretrained on large fluorescence microscopy dataset. Supports custom model training. Well-suited for cell-like structures. May need adaptation for non-cell classes (whole root, vascular).
- **StarDist**: Good for convex/star-convex objects. May work well for aerenchyma but less suited for ring-shaped endodermis.

### 3. Train from scratch

- **U-Net / U-Net++** with instance segmentation head: Standard for biomedical segmentation.
- **nnU-Net**: Self-configuring architecture for biomedical images. Handles preprocessing automatically.

### 4. Explore architectures

- **MAE (Masked Autoencoder) pretraining** → fine-tune: Pretrain encoder on all unlabeled fluorescence images, then fine-tune for segmentation. Could yield a reusable fluorescence microscopy encoder.
- **Swin-Transformer + Mask2Former**: SOTA for instance/panoptic segmentation.
- **Vision Foundation Models**: Fine-tune BiomedCLIP or similar for feature extraction.

### Recommended Priority Order
1. YOLOv8/v11-seg (fast iteration, YOLO format native)
2. Mask R-CNN with Detectron2 (strong baseline)
3. Cellpose 3.0 (microscopy-specific)
4. SAM 2 fine-tuning (generalization potential)
5. MAE pretrain + U-Net (if encoder reuse is the goal)

---

## Training Best Practices

- **Early stopping**: Monitor validation loss/mAP, patience ~10-20 epochs
- **Learning rate**: Use cosine annealing or reduce-on-plateau; use lower LR for pretrained backbone (1e-5 to 1e-4), higher for new heads (1e-3 to 1e-2)
- **Checkpointing**: Save best model (by val mAP) + latest checkpoint only; do NOT save every epoch
- **Mixed precision (AMP)**: Use fp16 on 4090 to save VRAM and speed up training
- **Gradient accumulation**: If batch size is limited by VRAM
- **Batch size**: Maximize within 24GB VRAM; typically 4-8 for high-res microscopy
- **Warmup**: Use linear warmup for first few epochs when fine-tuning

---

## Evaluation Metrics

Report all metrics overall AND broken down by species AND by microscope.

### Instance Segmentation Metrics
- **mAP@0.5** and **mAP@0.5:0.95** (COCO-style) — per class and overall
- **Precision / Recall** per class at IoU=0.5

### Pixel-Level Metrics (per class)
- **IoU (Jaccard index)**
- **Dice coefficient (F1)**
- **Pixel accuracy**
- **Per-class accuracy**

### Reporting Breakdown
```
Overall | Per-Species (Millet, Rice, Sorghum) | Per-Microscope (C10, Olympus, Zeiss)
```

Save metrics to CSV/JSON after each evaluation.

---

## Downstream Analysis Tasks

After segmentation, compute per-sample:

1. **Aerenchyma area ratio**: total aerenchyma polygon area / whole root polygon area
2. **Endodermis channel intensity**: average intensity per channel (DAPI, FITC, TRITC) in the endodermis ring (area between outer and inner endodermis boundaries)
3. **Vascular channel intensity**: average intensity per channel in the vascular area (inner endodermis polygon)
4. **Aerenchyma count**: number of aerenchyma instances per sample

These analyses should work on both ground truth annotations and model predictions.

---

## Color Convention for Visualization

| Class | Name | Color |
|-------|------|-------|
| 0 | Whole Root | Blue (0, 0, 255) |
| 1 | Aerenchyma | Yellow (255, 255, 0) |
| 2 | Outer Endodermis / Endodermis | Green (0, 255, 0) |
| 3 | Inner Endodermis / Vascular | Red (255, 0, 0) |
