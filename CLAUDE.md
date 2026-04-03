# Plant Root Cross-Section Instance Segmentation Project

## Project Overview

Train instance segmentation models for cell-type topology in cereal plant root cross-section fluorescence microscopy images. Robust segmentation across species (millet, rice, sorghum, tomato), genotypes, and microscopes (C10, Olympus, Zeiss), with generalization to unseen species and platforms.

**GPU**: 1 NVIDIA A100 80GB per model | **Dataset**: ~1,717 samples, YOLO polygon format | **Target**: Nature Plants

---

## Directory Structure

```
plants/
├── data/
│   ├── image/{Species}/{Microscope}/{Exp}/{Sample}/{Sample}_{DAPI|FITC|TRITC}.tif
│   └── annotation/{Species}_{Microscope}_{Exp}_{Sample}.txt  # YOLO polygons
├── train/                    # train_yolo.py, train_unet_binary.py, train_unet_semantic.py, train_sam.py, train_cellpose.py, run_grid_training.py
├── src/                      # Shared library (config, dataset, preprocessing, annotation_utils, splits, augmentation, evaluation, metrics, postprocessing, downstream, visualization, formats/, models/)
├── predict.py                # Inference
├── evaluate.py               # Unified evaluation (PNG plots only)
├── slurm/                    # SLURM job scripts
├── analyze_downstream.py     # Biological analysis
├── polygon_editor.py         # Interactive annotation GUI
├── preview_annotations.py    # Annotation preview PNGs
├── output/                   # Training runs, exports
└── data_to_edit/             # Samples needing exodermis annotation (same structure as data/)
```

**Species**: Millet, Rice, Sorghum (monocots/cereals), Tomato (dicot)
**Microscopes**: Olympus IX83 (widefield), Cytation C10 (plate reader), Zeiss LSM 970 (confocal)
**Channels**: DAPI (blue/cell walls), FITC (green/lignin), TRITC (red/suberin) -- 3 grayscale TIFs per sample

---

## Label Definitions

YOLO polygon format: `class_id x1 y1 x2 y2 ... xn yn` (normalized 0-1 coordinates).

### Raw Annotation Classes (in files)

| ID | Name | Meaning | Present in |
|----|------|---------|-----------|
| 0 | Whole Root | Outer root boundary (epidermis edge) | All |
| 1 | Aerenchyma | Air-filled holes in cortex | Cereals only (zero in tomato) |
| 2 | Outer Endodermis | Outer ring of endodermis | All |
| 3 | Inner Endodermis | Inner ring -- encloses vascular region | All |
| 4 | Outer Exodermis | Outer ring of exodermis | All |
| 5 | Inner Exodermis | Inner ring of exodermis | All |

All samples annotated for all 6 classes. Biologically absent classes have zero polygons -- this is real biology, not missing data. Models should learn zero area for absent classes (no special handling needed).

### Target Classes (downstream only — derived via `--convert-classes`)

| Target | Region | Derivation |
|--------|--------|-----------|
| 0 | Whole Root | Class 0 directly |
| 1 | Aerenchyma | Class 1 directly |
| 2 | Endodermis | Class 2 minus class 3 (ring subtraction) |
| 3 | Vascular | Class 3 directly |
| 4 | Exodermis | Class 4 minus class 5 (ring subtraction) |

### Biology Context

- **Epidermis**: outermost layer, defines root boundary
- **Cortex**: between epidermis and endodermis; contains aerenchyma in cereals
- **Endodermis**: thin layer with Casparian strip; high FITC/TRITC signal; derived by ring subtraction
- **Exodermis**: barrier layer; present in tomato only
- **Vascular cylinder**: innermost region inside inner endodermis
- **Aerenchyma**: irregular air spaces in cortex; cereals only

### Annotation Rules

- Polygons can overlap but boundaries must NOT intersect
- All aerenchyma contained within whole root polygon
- Cereal: 1 root, many aerenchyma, 1 outer/inner endo, 0 exodermis
- Tomato: 1 root, 0 aerenchyma, 1 outer/inner endo, 1 outer/inner exodermis

**QC**: 415/1,671 flagged but all minor boundary touching (max 2.3% area). No gross errors. No preprocessing needed.

---

## Dataset Splitting

### Critical Rule
**Samples from same experiment (`Exp{N}`) must stay together** -- never split across train/test.

### Dataset Composition

| Species | Microscope | Samples | Notes |
|---------|-----------|---------|-------|
| Millet | Olympus | 110 | Monocot |
| Rice | C10 | 50 | Monocot |
| Rice | Olympus | 503 | Monocot |
| Rice | Zeiss | 35 | **Held out from all training** |
| Sorghum | C10 | 44 | Monocot |
| Sorghum | Olympus | 430 | Monocot |
| Tomato | C10 | 65 | Dicot (M82 WT) |
| Tomato | Olympus | 480 | Dicot (multiple genotypes) |

### Strategy A -- Splits (unified model)

| Species | Microscope | Train | Val | Test | Total |
|---------|------------|------:|----:|-----:|------:|
| Millet | Olympus | 67 | 29 | 14 | 110 |
| Rice | C10 | 38 | 6 | 6 | 50 |
| Rice | Olympus | 402 | 49 | 51 | 502 |
| Rice | Zeiss | 0 | 0 | 35 | 35 |
| Sorghum | C10 | 25 | 11 | 8 | 44 |
| Sorghum | Olympus | 320 | 38 | 27 | 385 |
| Tomato | C10 | 44 | 11 | 10 | 65 |
| Tomato | Olympus | 393 | 60 | 27 | 480 |
| **Total** | | **1289** | **204** | **178** | **1671** |

- Zeiss (35 samples) fully held out -- zero-shot evaluation only
- All models train on 6 raw annotation classes

### Strategy B -- Generalization
- **B-mono**: Train on monocots only, test on monocot test set
- **B-dico**: Train on tomato only, test on dicot test set
- Compare vs Strategy A to assess unified model benefit

### Deployment (zero-shot)
- Apply best Strategy A model to 35 Zeiss images (unseen microscope)

---

## Models

### Training Run Plan

| Runs | Model | Strategy | Purpose |
|------|-------|----------|---------|
| 1-5 | YOLO26m / U-Net++ multilabel / U-Net++ semantic / SAM / Cellpose | A | Benchmark |
| 6-13 | All 4 models | B-mono | Monocot generalization |
| 14-21 | All 4 models | B-dico | Dicot generalization |
| 22+ | Best model | A (ablation) | Augmentation ablation |

### YOLO Training Runs

| Run | Local path (.../yolo26m-seg/) | Strategy | overlap_mask | Total epochs | Best epoch (P15) | Best epoch (P30) | mAP50-95(M) |
|-----|------|----------|:---:|------:|------:|------:|------:|
| 1 | 2026-04-02_001 | A | True | 137 | 53 | 132 | 0.790 |
| 2 | 2026-04-02_002 | A | False | 99 | 51 | 68 | 0.921 |
| 3 | 2026-04-03_002 | B-mono | False | 116 | 59 | 59 | 0.919 |
| 4 | 2026-04-03_003 | B-dico | False | 72 | 57 | 57 | 0.989 |

Patience=15 vs 30 finds the same best checkpoint for B-mono and B-dico. For Strategy A (overlap=False), the difference is 0.0008 mAP — negligible. patience=15 is sufficient.

### Model Summary

| Model | Architecture | Params | Frozen | Input | Epochs | Batch | Loss |
|-------|-------------|--------|--------|-------|--------|-------|------|
| YOLO26m-seg | YOLO26 (COCO) | 23.6M | None | 1024 | 200 | 16 | Ultralytics internal |
| U-Net++ multilabel | resnet34 (ImageNet) | ~24.4M | None (diff LR) | 1024 | 200 | 16 | BCE+Dice (sigmoid, 6ch) |
| U-Net++ semantic | resnet34 (ImageNet) | ~24.4M | None (diff LR) | 1024 | 200 | 16 | Dice+Focal+wCE+Lovasz (softmax, 7cls) |
| SAM (ViT-B) | ViT-B (SA-1B) | 93.7M | Encoder (89.7M) | 1024 | 200 | 8 | BCE+Dice (per instance) |
| Cellpose (cyto3) | ViT-L (cyto3) | ~307M | None | 512→256 crop | 100 | 8 | Flow+distance |

**Shared defaults**: patience=15, seed=42, fp16 (except Cellpose fp32), AdamW+CosineAnnealing (U-Net++/SAM), 1 GPU

**pos_weight** (U-Net++ multilabel & SAM BCE): root=1, aer=2, o.endo=5, i.endo=1, o.exo=5, i.exo=1

**U-Net++ differential LR**: encoder 1e-5, decoder 1e-4

### Key Model Details

- **YOLO**: Pre-exports uint8 PNGs to `output/yolo_dataset/`; NMS-free; Ultralytics manages augmentation; evaluates on 6 raw classes
  - **Must use `overlap_mask=False`** for overlapping annotations (filled polygons like ours). See finding below.
  - `overlap_mask=True` (Ultralytics default) causes the model to learn ring-like masks instead of filled polygons — `polygons2masks_overlap()` paints GT masks onto a single canvas sorted by area, so larger structures (Whole Root, Outer Endo) lose overlapping pixels to smaller ones during training loss computation. The model then predicts rings because that's what it was supervised on.
  - `overlap_mask=False` gives each instance its own separate GT mask channel, so the model learns correct filled polygons.
  - Contour-fill (`_fill_mask_contours`) is applied at inference as a safety net — no-op on already-filled masks, recovers filled polygons from ring-like masks if needed.
- **U-Net++ multilabel**: 6 sigmoid channels (can overlap); evaluates on 6 raw classes; `raw_to_target` available via `--convert-classes` for downstream
- **U-Net++ semantic**: 7 mutually exclusive classes (bg + 6 regions) via softmax; rings derived by paint order
- **SAM**: Frozen encoder; trains mask decoder only (4.4%); prompts = 3 random foreground points + bbox with 5% jitter; requires prompts at inference (eval uses oracle GT boxes)
- **Cellpose**: Per-class models (`--class-id N` or `--all-classes`); images preloaded as uint8; trains on 256x256 crops; no confidence scores (set to 1.0)

---

## Data Pipeline

All models: 3 TIFs -> `load_sample_normalized()` (percentile 1st-99.5th -> [0,1] float32) -> augmentation -> model

| Model | Pre-augmentation | Final format | Notes |
|-------|-----------------|-------------|-------|
| U-Net++ | Resize to 1024, on-the-fly | float32 [0,1] | albumentations pipeline |
| SAM | Original size, on-the-fly | float32 [0,1] | albumentations + prompt generation after aug |
| YOLO | Pre-export uint8 PNG 1024 | float32 [0,1] (after /255) | Ultralytics built-in aug |
| Cellpose | Preload uint8 512 | uint8 256x256 crops | Cellpose built-in aug |

Normalization before augmentation for all models. For downstream intensity analysis, use `load_sample_raw()`.

### Augmentation (fluorescence-appropriate, no hue/saturation jitter)

**Shared albumentations** (`src/augmentation.py`, used by U-Net++ and SAM): RandomRotate90, HorizontalFlip, VerticalFlip, Affine (translate/scale/rotate/shear), ElasticTransform, RandomBrightnessContrast, GaussianBlur, GaussNoise, RandomGamma, **ChannelDropout** (p=0.2), **ChannelShuffle** (p=0.2), Resize

**YOLO**: Ultralytics built-in matched to shared pipeline; `bgr=0.2` (channel swap); no elastic/blur/noise/gamma/channel dropout

**Cellpose**: Built-in rotation (0-360), scale (0.7-1.3), flip, 256x256 crop; no elastic/brightness/channel ops

---

## Training Configuration

### Checkpoints & Outputs

Every run creates `output/runs/{model}/{config}/YYYY-MM-DD_NNN/` via `make_run_subfolder()`.

- **Best/last/periodic checkpoints**: model-specific formats (see training scripts)
- **Training history**: `results.csv` (YOLO) or `logs/metrics.csv` (U-Net++) or `training_history.json` (SAM/Cellpose)
- **Plots**: `loss_curve.png`, `hparams.yaml`
- Val metrics during training use raw output (no post-processing)

### Evaluation Pipeline

Each model evaluates on its own training class space by default (via `get_model_classes(model)`):
- **YOLO / U-Net++ multilabel**: 6 raw annotation classes (`ANNOTATED_CLASSES`, `CLASS_COLORS_RGB`)
- **U-Net++ semantic / SAM / Cellpose**: TBD (to be defined per model later)

```
# Default: no post-processing, no class conversion — raw predictions evaluated directly
python evaluate.py --model yolo --checkpoint <path>

# Opt-in post-processing (fill_holes, cleanup_whole_root, clip_aerenchyma)
python evaluate.py --model yolo --checkpoint <path> --postprocess
python evaluate.py --model yolo --checkpoint <path> --postprocess fill_holes cleanup_whole_root

# Opt-in class conversion for downstream tasks (raw 6 → target 5 via ring subtraction)
python evaluate.py --model yolo --checkpoint <path> --convert-classes
```

**Three independent stages** (all OFF by default):
1. **Post-processing** (`--postprocess`): fill_holes, cleanup_whole_root, clip_aerenchyma
2. **Class conversion** (`--convert-classes`): raw_to_target ring subtraction (separate from post-processing)
3. Both can be combined: `--postprocess --convert-classes`

**Output**: `metrics.json`, `per_sample.csv`, comparison plots (PNG), `vis/` overlay PNGs

### Metrics
- mAP@0.5 and mAP@0.5:0.95 (COCO-style), IoU, Dice, Precision/Recall -- per class, per species, per microscope

---

## Key Finding: `overlap_mask` in YOLO Segment

**Problem**: Ultralytics `overlap_mask=True` (default) stores GT masks as a single painted canvas via `polygons2masks_overlap()`, sorted by area (largest first). Overlapping pixels are assigned to the last-painted (smallest) instance. For nested annotations like ours (Whole Root ⊃ Outer Exo ⊃ Inner Exo ⊃ Outer Endo ⊃ Inner Endo), the training loss supervises ring-like GT for outer structures, so the model learns to predict rings instead of filled polygons.

**Impact on 7 derived biological classes (test set, 178 samples)**:

| Derived Class | overlap=True IoU | overlap=False IoU | overlap=True Dice | overlap=False Dice |
|--------------|------:|------:|------:|------:|
| Whole Root | 0.623 | **0.981** | 0.768 | **0.990** |
| Epidermis | 0.516 | **0.681** | 0.681 | **0.810** |
| Exodermis | 0.598 | **0.822** | 0.748 | **0.902** |
| Cortex | 0.811 | 0.811 | 0.896 | 0.896 |
| Aerenchyma | 0.674 | **0.695** | 0.805 | **0.820** |
| Endodermis | 0.892 | 0.888 | 0.943 | 0.941 |
| Vascular | 0.976 | **0.980** | 0.988 | **0.990** |
| **Mean** | 0.727 | **0.837** | 0.833 | **0.907** |

**Takeaway**: Always use `overlap_mask=False` when training YOLO segment on overlapping/nested annotations. The largest gains are on outer containing structures (Whole Root, Epidermis, Exodermis). Inner structures (Vascular, Endodermis) are unaffected since they have no inner overlaps.

---

## Hyperparameter Strategy

No systematic grid search. YOLO default hyperparameters (AdamW, cosine LR) are well-validated. Key choices justified by domain knowledge:
- **No hue/saturation augmentation** — fluorescence channels have fixed meaning
- **`overlap_mask=False`** — required for nested annotations (empirically validated above)
- **Channel swap (`bgr=0.2`)** — encourages channel-invariant features

Ablation experiments (Fig 6) focus on domain-relevant factors (channel dropout/shuffle, overlap_mask) rather than standard hyperparameters. Compute budget prioritized for generalization experiments (Strategy B, Zeiss zero-shot).

### Augmentation Ablation Plan (Fig 6)

Run on the **best model only** (determined after Strategy A benchmark of all 4 models). If reviewer requests, add one more architecture as supplementary. The ablation tests a data hypothesis ("do channel augmentations help for fluorescence?"), not an architecture hypothesis — results should generalize across models.

**4 conditions on Strategy A split:**

| Condition | ChannelDropout (p=0.2) | ChannelShuffle (p=0.2) / bgr=0.2 |
|-----------|:---:|:---:|
| Full augmentation (baseline) | ON | ON |
| No channel aug | OFF | OFF |
| Dropout only | ON | OFF |
| Shuffle only | OFF | ON |

Report per-class IoU/Dice on 7 derived biological classes. Expect channel augmentation to help generalization across microscopes (different channel intensity profiles) and species.

---

## Polygon Editor (`polygon_editor.py`)

Interactive GUI for visualizing and correcting YOLO polygon annotations.

### Modes
- **Create GT**: draw from scratch | **Correct GT**: edit GT with predictions as reference | **Correct Pred**: edit predictions, save to annotation/

### Short Class Names
Root, Aer, O.Endo, I.Endo, O.Exo, I.Exo

### Key Shortcuts
N=Draw, Enter/Space=Confirm/Edit, Esc=Cancel, S=Save, R=Split ring, Del=Delete, Ctrl+Z/Shift+Z=Undo/Redo, Left/Right=Navigate

### Brush Mode
- **Drawing new** (no selection): default=paint, Shift=erase
- **Editing existing** (selected): default=erase, Shift=paint
- Ctrl+Scroll=brush size, Scroll=zoom

### Display Adjustments (display only)
Brightness (-100 to +100) and Gamma (0.1 to 3.0) sliders. Raw image in `_raw_image`, adjustments on-the-fly.

### PyQt5 Layout Rules (IMPORTANT)
- Use `addWidget(widget, stretch=1)` NOT `addLayout(layout, stretch=1)` for claiming space
- Do NOT use `setSizePolicy()`/`setMaximumHeight()` on QGroupBoxes -- fix by adding stretch to content widget
- Do NOT wrap toolbar rows in container QWidget with maxHeight
- Row 1 is flat (no QGroupBox); Row 2 uses QGroupBoxes with tight margins `(4, 2, 4, 2)`

---

## Paper Outline

**Title**: Automated segmentation of root anatomical barriers enables scalable quantification across species and imaging platforms

| Figure | Content |
|--------|---------|
| Fig 1 | Dataset diversity, annotation protocol |
| Fig 2 | Benchmark: unified model across species/microscopes (Strategy A) |
| Fig 3 | Generalization: unified vs specialist models (Strategy B) |
| Fig 4 | Explainability: embeddings, GradCAM, channel importance |
| Fig 5 | Deployment: zero-shot on Zeiss |
| Fig 6 | Augmentation ablation: channel dropout/shuffle |
| Fig 7 | Downstream: automated vs expert measurements |

### Explainability
- UMAP of embeddings (U-Net++ 2048-dim, SAM 256-dim) colored by species/microscope/aerenchyma ratio
- GradCAM on U-Net++ encoder layer 4 per class
- Channel occlusion: zero out TRITC/FITC/DAPI, measure IoU drop
- Morphometric correlation with embeddings

### Downstream Analysis
Per-sample: aerenchyma ratio & count, endodermis/vascular channel intensities. Compare vs expert (R-squared, Bland-Altman).

---

## Color Convention

### Raw Annotation Classes (`CLASS_COLORS_RGB` — YOLO, U-Net++ multilabel)

| Class | Name | Color |
|-------|------|-------|
| 0 | Whole Root | Blue (0,0,255) |
| 1 | Aerenchyma | Yellow (255,255,0) |
| 2 | Outer Endodermis | Green (0,255,0) |
| 3 | Inner Endodermis | Red (255,0,0) |
| 4 | Outer Exodermis | Orange (255,128,0) |
| 5 | Inner Exodermis | Purple (128,0,255) |

### Target Classes (`TARGET_CLASS_COLORS_RGB` — downstream / other models)

| Class | Name | Color |
|-------|------|-------|
| 0 | Whole Root | Blue (0,0,255) |
| 1 | Aerenchyma | Yellow (255,255,0) |
| 2 | Endodermis | Green (0,255,0) |
| 3 | Vascular | Red (255,0,0) |
| 4 | Exodermis | Cyan (0,255,255) |

---

## HPC Commands

```bash
# Sync to HPC
rsync -avz --progress --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='output/' --exclude='logs/' --exclude='.DS_Store' --exclude='annotation_copy/' --exclude='*.pt' ~/Documents/Siobhan_Lab/plants/ hpc2:~/plants/

# Environment setup
module load conda3/4.13.0 && conda activate plants

# First-time setup
conda create -n plants python=3.11 -y && conda activate plants && pip install -r ~/plants/requirements.txt

# Submit jobs
sbatch slurm/run_grid.sh

# Interactive session
srun --job-name=grid_train --partition=gpu-qi --gres=gpu:a100_80:1 --cpus-per-task=8 --mem=64G --time=72:00:00 --pty bash

# Monitor
squeue -u $USER
tail -f logs/grid_<JOB_ID>.out
srun --jobid=<JOB_ID> --overlap nvidia-smi

# Transfer results back
rsync -avz hpc2:~/plants/output/ ~/Documents/Siobhan_Lab/plants/output/
rsync -avz hpc2:~/plants/logs/ ~/Documents/Siobhan_Lab/plants/logs/
```

**HPC**: Cluster `hpc2`, partition `gpu-qi`, GPU `a100_80`. Cannot SSH to compute nodes -- use `srun --overlap`.

---

## Model Training Notes

### Cellpose
- SLURM: 5 parallel jobs (one per class) + dependent eval job via `sbatch --parsable`
- Numpy truthiness: check `is not None and len(...) > 0` (Cellpose returns numpy arrays)
- `average_precision()` returns (samples x thresholds); use `.mean(axis=-1)` for per-sample AP
- No confidence scores -- set to 1.0 during eval
