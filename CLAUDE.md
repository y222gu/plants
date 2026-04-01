# Plant Root Cross-Section Instance Segmentation Project

## Project Overview

Train instance segmentation models for cell-type topology in cereal plant root cross-section fluorescence microscopy images. The goal is robust segmentation across species (millet, rice, sorghum, tomato), genotypes, and microscopes (C10, Olympus, Zeiss), with demonstrated generalization to unseen species and platforms.

**GPU**: 1 NVIDIA A100 80GB per model (1 GPU per SLURM job)
**Dataset**: ~1,172 cereal samples + 545 tomato samples (~1,717 total), YOLO polygon format
**Target journal**: Nature Plants (or Nature Machine Intelligence / Nature Methods)

---

## Directory Structure

```
plants/
├── data/
│   ├── image/
│   │   └── {Species}/{Microscope}/{Exp}/{Sample}/
│   │       ├── {Sample}_DAPI.tif      # Blue channel (Calcofluor White / cell walls)
│   │       ├── {Sample}_FITC.tif      # Green channel (Fluorol Yellow / lignin)
│   │       └── {Sample}_TRITC.tif     # Red channel (Basic Fuchsin / suberin)
│   └── annotation/
│       └── {Species}_{Microscope}_{Exp}_{Sample}.txt   # YOLO polygons
├── train/                             # Training scripts (one per model)
│   ├── train_yolo.py
│   ├── train_unet_binary.py           # 6-channel multilabel U-Net (1 model, sigmoid)
│   ├── train_unet_semantic.py         # 7-class semantic U-Net (1 model, softmax)
│   ├── train_sam.py
│   ├── train_cellpose.py
│   └── run_grid_training.py
├── src/                               # Shared library
│   ├── config.py                      # Paths, class defs, defaults
│   ├── dataset.py                     # SampleRegistry (auto-discovery)
│   ├── preprocessing.py               # Image loading, normalization
│   ├── annotation_utils.py            # YOLO parsing, mask generation
│   ├── splits.py                      # Train/val/test splitting
│   ├── augmentation.py                # Albumentations pipelines
│   ├── evaluation.py                  # PredictionResult converters
│   ├── metrics.py                     # mAP, IoU, Dice computation
│   ├── postprocessing.py              # Mask cleaning pipeline
│   ├── downstream.py                  # Aerenchyma ratios, intensities
│   ├── visualization.py               # Overlay + publication styling
│   ├── formats/                       # YOLO, COCO, Mask NPZ exporters
│   └── models/                        # Model-specific datasets/utils
├── predict.py                         # Inference + save predictions
├── evaluate.py                        # Model evaluation + metrics (PNG plots only)
├── slurm/                             # SLURM job scripts
│   ├── run_benchmark.sh               # Submit all 5 benchmark runs
│   ├── run_1_yolo.sh                  # Run 1: YOLO
│   ├── run_2_unet_4c.sh              # Run 2: U-Net++ 4-class
│   ├── run_3_unet_5c.sh              # Run 3: U-Net++ 5-class
│   ├── run_sam.sh                     # Run 4: SAM (train + eval)
│   ├── run_cellpose.sh               # Run 5: Cellpose (5 parallel + eval)
│   ├── eval_all.sh                    # Evaluate all models (no postprocess)
│   ├── eval_sam.sh                    # Evaluate SAM only
│   └── eval_cellpose.sh              # Evaluate Cellpose only
├── analyze_downstream.py              # Biological analysis
├── polygon_editor.py                  # Interactive annotation GUI
├── preview_annotations.py             # Annotation preview PNGs
├── output/                            # Training runs, exports
└── data_to_edit/                      # Samples needing exodermis annotation
    ├── image/                         # Same structure as data/image/
    └── annotation/                    # Same structure as data/annotation/
```

**`data_to_edit/`**: Working directory for annotation in progress. Same directory structure as `data/`.

**Species**: Millet, Rice, Sorghum (monocots/cereals), Tomato (dicot)
**Microscopes**: Olympus IX83 (widefield), Cytation C10 (plate reader), Zeiss LSM 970 (confocal)
**Channels**: DAPI (blue), FITC (green), TRITC (red) — 3 grayscale TIF images per sample

---

## Label Definitions

Annotation files use YOLO polygon format: `class_id x1 y1 x2 y2 ... xn yn` (normalized 0-1 coordinates).

### Annotated Classes (in annotation files)

**Cereal classes (Millet, Rice, Sorghum):**

| Class ID | Annotated Name | Polygon Meaning |
|----------|---------------|-----------------|
| 0 | Whole Root | Outer boundary of root (defined by epidermis edge) |
| 1 | Aerenchyma | Air-filled irregular holes in the cortex |
| 2 | Outer Endodermis | Outer ring of the endodermis layer |
| 3 | Inner Endodermis | Inner ring — encloses the **vascular** region |

**Tomato classes (uses 0, 2, 3, 4, 5):**

| Class ID | Annotated Name | Polygon Meaning |
|----------|---------------|-----------------|
| 0 | Whole Root | Outer boundary of root (defined by epidermis edge) |
| 2 | Outer Endodermis | Outer ring of the endodermis layer |
| 3 | Inner Endodermis | Inner ring — encloses the **vascular** region |
| 4 | Outer Exodermis | Outer ring of the exodermis layer |
| 5 | Inner Exodermis | Inner ring of the exodermis layer |

**All samples are fully annotated for all 6 classes (0-5).** Some classes are biologically absent in certain species — this is real biology, not missing data:
- **Aerenchyma (class 1)**: Air spaces that form in monocot cortex. Biologically absent in tomato (dicot) — tomato annotation files correctly have zero class 1 polygons.
- **Exodermis (classes 4-5)**: Barrier layer present in tomato. Biologically absent in cereals — cereal annotation files correctly have zero class 4/5 polygons.

Models should learn that these classes have zero area in the respective species — no special masking or missing-data handling is needed.

### Actual Semantic Regions (for model training)

| Target Class | Region | How to Derive | Biologically present in |
|-------------|--------|---------------|------------------------|
| 0 | Whole Root | Use class 0 polygon directly | All species |
| 1 | Aerenchyma | Use class 1 polygons directly | Cereals (zero area in tomato) |
| 2 | Endodermis | Subtract class 3 polygon from class 2 polygon (ring) | All species |
| 3 | Vascular | Use class 3 polygon directly (area inside inner endodermis) | All species |
| 4 | Exodermis | Subtract class 5 polygon from class 4 polygon (ring) | Tomato (zero area in cereals) |

### Biology Context

- **Epidermis**: outermost cell layer, defines whole root boundary
- **Cortex**: region between epidermis and outer endodermis; contains aerenchyma in cereals
- **Endodermis**: thin layer with Casparian strip; higher signal in FITC (lignin) and TRITC (suberin) channels; derived by subtraction (outer - inner ring)
- **Exodermis**: barrier layer between cortex and epidermis; present in tomato, biologically absent in cereals
- **Vascular cylinder**: innermost region enclosed by inner endodermis
- **Aerenchyma**: irregular air spaces in the cortex; present in cereals (monocots), biologically absent in tomato (dicot)

### Annotation Rules

- Polygons can overlap but boundaries must NOT intersect
- All aerenchyma polygons are contained within the whole root polygon
- All samples are annotated for all 6 classes — biologically absent classes simply have zero polygons
- Cereal annotations: typically 1 whole root, many aerenchyma, 1 outer endodermis, 1 inner endodermis, 0 exodermis
- Tomato annotations: 1 whole root, 0 aerenchyma, 1 outer/inner endodermis, 1 outer/inner exodermis

### Annotation QC (`qc_annotations.py`)

QC checked all 1,671 annotation files for spatial consistency:
1. All polygons (classes 1-5) within whole root (class 0)
2. All aerenchyma (class 1) within inner exodermis (class 5)
3. All aerenchyma (class 1) outside outer endodermis (class 2)

**Results**: 415/1,671 samples flagged, but all issues are minor boundary touching/overlap (max 2.3% area). No gross annotation errors. 886/1,036 issues are boundary intersections (shared edges), not containment violations. This is expected — biological structures share cell wall boundaries (e.g., aerenchyma bounded by cortex walls, exodermis adjacent to epidermis). No preprocessing needed — all models handle this naturally:
- U-Net semantic: paint order resolves shared boundaries
- U-Net multilabel: independent sigmoid channels tolerate shared pixels
- YOLO/SAM/Cellpose: per-instance masks, minor boundary overlap is negligible

---

## Dataset Splitting Strategies (for paper)

### Critical Rule
**Samples from the same experiment (`Exp{N}`) must stay together** — never split across train/test.

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
| Tomato | Olympus | 480 | Dicot (lignin mutants, Solanum spp., suberin mutants, WT) |

### Strategy A — Benchmark (unified model)

- **Train**: All 4 species (rice, sorghum, millet, tomato), Olympus + C10 only
- **Val**: 10% of training experiments
- **Test**: ~20% held-out experiments from all species/microscope groups
- **Zeiss**: Excluded entirely — reserved for deployment use case
- **Purpose**: Proves the model works; model comparison happens here
- **Classes**: All models train on the 6 raw annotation classes (0-5). U-Net++ semantic trains on 7 (bg+6 derived via paint order). Ring subtraction (raw → 5 target classes) happens in post-processing for all models.

### Strategy B — Monocot vs Dicot (generalization)

| Run | Train on | Test on | Question |
|-----|----------|---------|----------|
| B-mono | Monocots only (Rice + Sorghum + Millet) | Monocot test set | How does a monocot-only model compare to the unified model on monocots? |
| B-dico | Dicot only (Tomato) | Dicot test set | How does a dicot-only model compare to the unified model on dicots? |

- Compare B-mono performance on monocots vs Strategy A (unified) performance on monocots
- Compare B-dico performance on dicots vs Strategy A (unified) performance on dicots
- Shows whether a unified multi-species model helps or hurts per-group performance
- All use Olympus + C10 only
- Run with all 4 models

### Deployment Use Cases (zero-shot, no training)

- **Zeiss**: Apply best Strategy A model to all 35 Zeiss images (unseen microscope)

---

## Models to Train

### Training Run Plan

| Run | Model | Strategy | Classes | Purpose |
|-----|-------|----------|---------|---------|
| 1 | YOLO26m-seg | A | 6 raw | Benchmark |
| 2 | U-Net++ multilabel | A | 6 raw (sigmoid) | Benchmark |
| 3 | U-Net++ semantic | A | 7 (bg+6, softmax) | Benchmark |
| 4 | SAM (ViT-B) | A | 6 raw | Benchmark |
| 5 | Cellpose (per-class) | A | 6 raw | Benchmark |
| 6-13 | All 4 models | B-mono | — | Monocot-only generalization |
| 14-21 | All 4 models | B-dico | — | Dicot-only generalization |
| 22+ | Best model | A (augmentation ablation) | — | Ablation study |

### Benchmark Hyperparameters (Strategy A)

| Param | YOLO | U-Net++ binary | U-Net++ semantic | SAM | Cellpose |
|-------|------|----------------|------------------|-----|---------|
| Architecture | yolo26m-seg | unetplusplus | unetplusplus | vit_b | cyto3 |
| Encoder | YOLO26 backbone (COCO) | resnet34 (ImageNet) | resnet34 (ImageNet) | ViT-B (SA-1B) | cyto pretrained |
| Alt. encoder (TODO) | — | efficientnet-b4 (ImageNet) | efficientnet-b4 (ImageNet) | — | — |
| Classes | 6 raw annotation | 6 raw (multilabel sigmoid) | 7 (bg + 6, softmax) | 6 raw annotation | 6 raw (per-class models) |
| Input image size | 1024 | 1024 | 1024 | 1024 | 512 (trains on 256 patches) |
| Epochs | 200 | 200 | 200 | 200 | 100 |
| Patience | 15 | 15 | 15 | 15 | — (fixed epochs) |
| Batch size | 16 | 16 | 16 | 8 | 8 |
| LR (head/decoder) | auto (Ultralytics) | 1e-4 | 1e-4 | 1e-4 | 0.01 |
| LR (backbone) | auto (Ultralytics) | 1e-5 | 1e-5 | frozen | — |
| Weight decay | 5e-4 (Ultralytics) | 1e-4 | 1e-4 | 1e-4 | 0.01 |
| Optimizer | MuSGD (Ultralytics) | AdamW | AdamW | AdamW | Cellpose internal |
| Scheduler | auto (Ultralytics) | CosineAnnealing | CosineAnnealing | CosineAnnealing | — |
| Precision | fp16 (AMP) | fp16 (16-mixed) | fp16 (16-mixed) | fp16 (GradScaler) | fp32 |
| GPU | 1 | 1 | 1 | 1 | 1 |

### Model Parameters & Frozen Layers

| Model | Architecture | Total Params | Frozen | Trainable | Trainable % |
|-------|-------------|-------------|--------|-----------|-------------|
| **YOLO26m-seg** | YOLO26 backbone (COCO pretrained) | 23.6M | None | 23.6M | 100% |
| **U-Net++ multilabel** | resnet34 encoder (ImageNet pretrained) | ~24.4M | None (differential LR: encoder 1e-5, decoder 1e-4) | ~24.4M | 100% |
| **U-Net++ semantic** | resnet34 encoder (ImageNet pretrained) | ~24.4M | None (differential LR: encoder 1e-5, decoder 1e-4) | ~24.4M | 100% |
| **SAM (ViT-B)** | ViT-B image encoder (SA-1B pretrained) | 93.7M | Image encoder (89.7M) + prompt encoder (6K) | 4.1M (mask decoder only) | 4.4% |
| **Cellpose (cyto3)** | ViT-L via SAM (cyto3 pretrained) | ~307M | None | ~307M | 100% |

Note: Cellpose cyto3 uses a SAM ViT-L-based transformer architecture (~307M params). The older cyto2 uses a residual U-Net (~13M params).

### Loss Functions

| Model | Loss Formula | Class Weights |
|-------|-------------|---------------|
| **YOLO26** | Ultralytics internal: CIoU (box) + `0.5×BCE + 0.5×Dice` (mask) + BCE (cls) + semantic seg loss | N/A (framework-managed) |
| **U-Net++ multilabel** | `1.0 × BCE + 1.0 × Dice` (sigmoid, 6 channels) | pos_weight=[1, 2, 5, 1, 5, 1] per channel on BCE |
| **U-Net++ semantic** | `Dice + Focal + weighted CE + Lovász` (softmax, 7 classes) | CE weights=[0.5, 1, 2, 5, 1, 5, 1] |
| **SAM** | `1.0 × BCE + 1.0 × Dice` (sigmoid, per instance) | pos_weight=[1, 2, 5, 1, 5, 1] per instance class on BCE |
| **Cellpose** | Cellpose internal: flow field + distance loss | N/A (framework-managed) |

**pos_weight mapping** (shared by U-Net++ multilabel and SAM):

| Class ID | Raw Class | pos_weight | Rationale |
|----------|-----------|------------|-----------|
| 0 | Whole Root | 1.0 | Large filled region |
| 1 | Aerenchyma | 2.0 | Many small instances |
| 2 | Outer Endodermis | 5.0 | Thin boundary polygon |
| 3 | Inner Endodermis | 1.0 | Large filled region |
| 4 | Outer Exodermis | 5.0 | Thin boundary polygon |
| 5 | Inner Exodermis | 1.0 | Large filled region |

**U-Net++ multilabel**: Single model with 6 sigmoid output channels (one per raw annotation class 0-5). Channels can overlap. Post-processing derives 5 target classes via `raw_to_target` ring subtraction.

**U-Net++ semantic**: Single model predicts 7 mutually exclusive classes (bg + 6 anatomical regions) via softmax.

**All models train on 6 raw annotation classes** — no ring subtraction during training. Ring subtraction (raw → 5 target classes) is done in post-processing via `raw_to_target` step.

### Model Architecture Details

#### YOLO26m-seg
- **Pretrained on**: COCO
- **Frozen**: Nothing (full fine-tuning; Ultralytics handles warmup internally)
- **Trainable**: Entire model (backbone + neck + detection head + segmentation head)
- **NMS-free**: End-to-end design, no post-processing NMS required
- **Classes**: 6 raw annotation classes (0-5)
- **Data export**: Pre-exports composited uint8 PNG images + filtered YOLO .txt labels to `output/yolo_dataset/` via `export_yolo_dataset()`; skips re-export if counts match (use `--force-export` to override)
- **Augmentation**: Ultralytics built-in: `hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=45.0, translate=0.1, scale=0.3, shear=10.0, flipud=0.5, fliplr=0.5, bgr=0.2, mosaic=0.0, mixup=0.0` — parameters matched to shared albumentations pipeline where possible; no channel dropout
- **Embedding extraction**: Forward hook on SPPF layer → global average pool → image-level vector
- **GPU**: 1 GPU per job (single device)

#### U-Net++ multilabel (`train_unet_binary.py`)
- **Encoder**: resnet34 (ImageNet-pretrained) via `segmentation_models_pytorch` (configurable via `--encoder`)
- **Differential LR**: Encoder at LR=1e-5, decoder at LR=1e-4
- **Output**: (B, 6, H, W) — 6 sigmoid channels, one per raw annotation class. Channels can overlap (outer endo polygon contains inner endo area).
- **Loss**: BCE (pos_weight: root=1, aer=2, o.endo=5, i.endo=1, o.exo=5, i.exo=1) + Dice (1:1)
- **Post-processing**: Derives 5 target classes via `raw_to_target` ring subtraction

#### U-Net++ semantic (`train_unet_semantic.py`)
- **Encoder**: resnet34 (ImageNet-pretrained) via `segmentation_models_pytorch` (configurable via `--encoder`)
- **Differential LR**: Encoder at LR=1e-5, decoder at LR=1e-4
- **Output**: (B, 7, H, W) — 7 mutually exclusive classes (bg + 6 anatomical regions) via softmax
- **Loss**: Dice + Focal + weighted CE (bg=0.5, epidermis=1, aer=2, endo=5, vasc=1, exo=5, cortex=1) + Lovász
- **Embedding extraction**: `model.encoder(x)[-1]` → global average pool → (2048,) vector
- **GradCAM**: On encoder layer 4 for per-class activation maps

#### SAM (ViT-B) — `segment_anything` (SAM 1)
- **Pretrained on**: SA-1B
- **Frozen**: Image encoder (completely) + prompt encoder (by default; `--unfreeze-prompt-encoder` to unfreeze)
- **Trainable**: Mask decoder only
- **On-the-fly encoding**: Frozen image encoder runs every batch (not pre-computed), enabling full augmentation via `AugmentedSAMDataset`
- **Prompts**: 3 random foreground points + bounding box with 5% jitter from GT (re-randomized each `__getitem__`)
- **Loss**: `bce_weight` × BCE + `dice_weight` × Dice (custom `DiceLoss` with sigmoid + smooth=1)
- **GPU**: 1 GPU per job (DDP support available but not used)
- **Embeddings**: `image_encoder(x)` → global average pool → (256,) vector
- **Limitation**: Requires prompts at inference (not fully automatic); eval uses oracle GT bounding boxes

#### Cellpose (v3)
- **Base model**: cyto3 (default, ViT-L-based transformer, ~307M params); microscopy-pretrained. Use `--version 2` for cyto2 (residual U-Net, ~13M params).
- **Image size**: 512 (not 1024 — 4x faster per sample)
- **Training**: Per-class models (separate model for each raw annotation class via `--class-id N` or `--all-classes`)
- **Frozen**: Nothing — all parameters trainable (Cellpose manages internally)
- **Data pipeline**: Images preloaded once via `preload_cellpose_data()`, per-class labels built from cache via `build_class_labels()` (no disk I/O per class). Images converted to uint8 before training.
- **Training API**: `cellpose.train.train_seg()` with `rescale=True`, `scale_range=0.6`, `min_train_masks=1`; internally crops to 256×256 patches via `random_rotate_and_resize(bsize=256)`
- **Embedding extraction**: Style network → (256,) global vector
- **Limitation**: Black-box flow-based post-processing; harder to interpret; no confidence scores (set to 1.0 during eval)

---

## Data Loading, Preprocessing & Augmentation

### Per-Model Training Data Pipeline

All models start from the same raw data: 3 separate grayscale TIF files per sample (TRITC, FITC, DAPI). The full pipeline from raw TIF to model input differs per model:

#### U-Net++ (binary & semantic)

Pipeline runs on-the-fly in `UNetBinaryDataset.__getitem__()` / `UNetSemanticDataset.__getitem__()`:

1. **Load**: 3 TIFs → `load_sample_normalized()` → (H, W, 3) float32 [0,1] (percentile normalization: 1st–99.5th per channel)
2. **Resize**: `cv2.resize()` to 1024×1024 (no aspect ratio preservation)
3. **Augmentation**: `get_train_transform(1024)` applied via `transform(image=img, mask=mask)` — albumentations handles spatial transforms on both image and mask, photometric transforms on image only
4. **Format**: `torch.from_numpy().permute(2,0,1).float()` → (3, 1024, 1024) float32 tensor
5. **Model trains on**: 1024×1024 float32 [0,1]

Mask derivation: semantic mode uses `polygons_to_raw_semantic_mask()` (7-class painted mask); binary mode uses `polygons_to_raw_binary_masks()` (per-class binary mask).

#### SAM

Pipeline runs on-the-fly in `AugmentedSAMDataset.__getitem__()`:

1. **Load**: 3 TIFs → `load_sample_normalized()` → (H, W, 3) float32 [0,1] (cached via LRU cache)
2. **No pre-resize** — image stays at original size (e.g. 1276×1276)
3. **Augmentation**: `get_train_transform(1024)` applied via `apply_transform_with_masks()` — augments image + all N instance masks together with identical spatial transforms; `A.Resize(1024)` at end of pipeline resizes to 1024×1024
4. **Prompt generation**: After augmentation, point prompts (random foreground pixels) and box prompts (bounding box + 5% jitter) are generated from the augmented mask
5. **Format**: `torch.from_numpy().permute(2,0,1).float()` → (3, 1024, 1024) float32 tensor
6. **Model trains on**: 1024×1024 float32 [0,1] — fed directly to frozen image encoder, no additional SAM-specific normalization applied

#### YOLO

Pipeline split into two phases — pre-export + Ultralytics training:

1. **Pre-export** (runs once via `export_yolo_dataset()`):
   - Load: 3 TIFs → `load_sample_normalized()` → float32 [0,1]
   - Resize: `cv2.resize()` to 1024×1024
   - Format: `to_uint8()` (×255, clip) → save as uint8 RGB PNG on disk
   - Labels: YOLO polygon .txt files copied (exodermis classes filtered if num_classes≤4)
2. **Training** (Ultralytics `model.train()` loads pre-exported PNGs):
   - Ultralytics loads PNG, applies built-in augmentation (rotation, scale, flip, etc.)
   - Ultralytics internally divides by 255 → [0,1] float32
   - Letterbox resize to `imgsz=1024` (no-op since images are already 1024×1024 square)
3. **Model trains on**: 1024×1024 float32 [0,1]

#### Cellpose

Pipeline split into two phases — pre-load + Cellpose training:

1. **Pre-load** (runs once via `preload_cellpose_data()` + `build_class_labels()`):
   - Load: 3 TIFs → `load_sample_normalized()` → float32 [0,1]
   - Resize: `cv2.resize()` to 512×512
   - Format: `(np.clip(img, 0, 1) * 255).astype(np.uint8)` → uint8 arrays in memory
   - Labels: per-class integer instance masks built from cached annotations
2. **Training** (`train.train_seg()` internal, each batch):
   - Calls `random_rotate_and_resize(xy=(bsize, bsize))` with default `bsize=256`
   - Applies rotation (0–360°), scaling (0.7–1.3× with `scale_range=0.6`), horizontal flip (50%)
   - **Crops 256×256 patches** from the rotated/scaled 512×512 image
3. **Model trains on**: 256×256 uint8 patches (significantly smaller than other models)

### Normalization Summary

| Step | U-Net++ | SAM | YOLO | Cellpose |
|------|---------|-----|------|----------|
| Percentile norm (1st–99.5th → [0,1]) | Yes (`load_sample_normalized`) | Yes | Yes (at export) | Yes (at preload) |
| Additional normalization | None | None (no ImageNet mean/std) | Ultralytics /255 | None |
| Format into model | float32 [0,1] | float32 [0,1] | float32 [0,1] (after /255) | uint8 [0,255] |

Normalization is applied **before** augmentation for all models. Photometric augmentations intentionally shift the normalized values — re-normalizing after would undo them. For downstream intensity analysis, use `load_sample_raw()` (no normalization) to preserve true fluorescence values.

### Augmentation (fluorescence microscopy appropriate)

All models use similar spatial augmentations for fair comparison. No hue/saturation jitter (meaningless for fluorescence).

#### Shared albumentations pipeline (`src/augmentation.py: get_train_transform()`) — used by U-Net++ and SAM:

- `RandomRotate90` (p=0.5), `HorizontalFlip` (p=0.5), `VerticalFlip` (p=0.5)
- `Affine` (p=0.7): translate ±10%, scale 0.7-1.3, rotate ±45°, shear ±10°
- `ElasticTransform` (p=0.3): alpha=120, sigma=12
- `RandomBrightnessContrast` (p=0.6): ±0.3 each
- `GaussianBlur` (p=0.2): kernel 3-7
- `GaussNoise` (p=0.4): std 0.01-0.08
- `RandomGamma` (p=0.3): gamma 70-150
- **`ChannelDropout`** (p=0.2): zero out 1 of 3 channels — key for cross-microscope robustness
- **`ChannelShuffle`** (p=0.2): randomly permute channel order — key for cross-microscope robustness
- `A.Resize(img_size, img_size)` at the end

For SAM, augmentation is applied via `AugmentedSAMDataset` which uses `apply_transform_with_masks()` to transform the image and all instance masks together, then generates point/box prompts from the augmented mask.

#### YOLO augmentation — Ultralytics built-in (`train_yolo.py` kwargs), matched to shared pipeline:

- `hsv_h=0.0, hsv_s=0.0` (no hue/saturation), `hsv_v=0.2` (mild brightness)
- `degrees=45.0` (rotation ±45°), `translate=0.1` (±10%), `scale=0.3` (0.7-1.3×), `shear=10.0` (±10°)
- `flipud=0.5`, `fliplr=0.5`
- `bgr=0.2` (BGR channel swap, similar to ChannelShuffle)
- `mosaic=0.0`, `mixup=0.0` (disabled for fair comparison)
- Not available in Ultralytics: elastic transform, gaussian blur/noise, gamma, channel dropout

#### Cellpose augmentation — Cellpose built-in (`random_rotate_and_resize()` called internally by `train.train_seg()`):

- Random rotation: 0° to 360° (uniform)
- Random scaling: factors 0.7-1.3 (with `scale_range=0.6`)
- Random horizontal flip: 50% probability
- Random crop to 256×256 patches (default `bsize=256`)
- Not available in Cellpose: elastic transform, brightness/contrast, gaussian blur/noise, gamma, channel dropout/shuffle

### Training Image Size Summary

| Model | Pre-augmentation size | Post-augmentation size | Model trains on |
|-------|----------------------|----------------------|-----------------|
| U-Net++ | 1024×1024 | 1024×1024 | 1024×1024 |
| SAM | Original (varies) | 1024×1024 (A.Resize) | 1024×1024 |
| YOLO | 1024×1024 (PNG) | 1024×1024 | 1024×1024 |
| Cellpose | 512×512 (uint8) | 256×256 (bsize crop) | 256×256 |

---

## Training Configuration (Strategy A Benchmark, Runs 1-5)

### Early Stopping, Checkpointing & Training Outputs

| | YOLO | U-Net++ multilabel | U-Net++ semantic | SAM | Cellpose |
|---|---|---|---|---|---|
| **Early stopping** | Ultralytics internal | PL `EarlyStopping` | PL `EarlyStopping` | Manual counter | None (fixed epochs) |
| **Monitor** | mAP (fitness) | val_loss | val_loss | val_loss | — |
| **Patience** | 15 | 15 | 15 | 15 | — |
| **Best ckpt** | `weights/best.pt` | `checkpoints/best-{epoch}-{val_loss}.ckpt` | `checkpoints/best-{epoch}-{val_loss}.ckpt` | `best.pth` | Cellpose internal |
| **Last ckpt** | `weights/last.pt` | `checkpoints/last.ckpt` | `checkpoints/last.ckpt` | — | — |
| **Periodic ckpt** | Every 10 epochs | `periodic-{epoch}.ckpt` every 10 epochs | `periodic-{epoch}.ckpt` every 10 epochs | `epoch_{N}.pth` every 10 epochs | — |
| **Training history** | `results.csv` (Ultralytics) | `logs/metrics.csv` (CSVLogger) | `logs/metrics.csv` (CSVLogger) | `training_history.json` | `training_history.json` |
| **Loss curve** | `loss_curve.png` | `loss_curve.png` | `loss_curve.png` | `loss_curve.png` | `loss_curve.png` |
| **Other plots** | Confusion matrix, PR curves (Ultralytics auto) | Per-class dice scores | Per-class accuracy | — | — |
| **Hyperparams** | `hparams.yaml` | `hparams.yaml` | `hparams.yaml` | `hparams.yaml` | `hparams.yaml` |

Val_loss and mAP during training are computed on **raw model output** (no post-processing). Custom post-processing (`raw_to_target`, `fill_holes`, etc.) only runs during final evaluation via `evaluate.py`.

### Dated Run Subfolders

Every training run creates a unique dated subfolder: `YYYY-MM-DD_NNN` (auto-incrementing NNN for same-day runs).

```
output/runs/
├── yolo/yolo26m-seg/
│   └── 2026-03-15_001/weights/best.pt, hparams.yaml
├── unet/unetplusplus_resnet34_multilabel_A/
│   └── 2026-03-15_001/checkpoints/best-*.ckpt, hparams.yaml
├── unet/unetplusplus_resnet34_semantic7c_A/
│   └── 2026-03-15_001/checkpoints/best-*.ckpt, hparams.yaml
├── sam/sam_vit_b_c6/
│   └── 2026-03-15_001/best.pth, hparams.yaml
└── cellpose/cellpose_v3_{ClassName}_c6/
    └── 2026-03-16_001/models/*, hparams.yaml
```

Implemented via `make_run_subfolder()` and `save_hparams()` in `src/config.py`.

### Configurable Training Flags

All training scripts support these CLI flags (where applicable):

| Flag | Scripts | Default | Description |
|------|---------|---------|-------------|
| `--epochs` | All | 200 (100 for Cellpose) | Max training epochs (`DEFAULT_EPOCHS=200` in config.py) |
| `--batch-size` | All | 16 (8 for SAM/Cellpose) | Batch size |
| `--lr` | U-Net++, SAM, Cellpose | 1e-4 (U-Net++/SAM), 0.01 (Cellpose) | Learning rate |
| `--backbone-lr` | U-Net++ | 1e-5 | Backbone/encoder LR (differential) |
| `--weight-decay` | U-Net++, SAM, Cellpose | 1e-4 (U-Net++/SAM), 0.01 (Cellpose) | Weight decay |
| `--patience` | YOLO, U-Net++, SAM | 15 | Early stopping patience |
| `--optimizer` | U-Net++, SAM | adamw | Optimizer (adamw/adam/sgd) |
| `--scheduler` | U-Net++, SAM | cosine | LR scheduler (cosine/step/plateau) |
| `--eta-min` | U-Net++, SAM | 1e-7 | Minimum LR for cosine scheduler |
| `--pos-weight` | U-Net++ binary | per-class defaults | BCE pos_weight override (binary mode) |
| `--bce-weight` | U-Net++, SAM | 1.0 | BCE loss weight |
| `--dice-weight` | U-Net++, SAM | 1.0 | Dice loss weight |
| `--save-every` | YOLO, U-Net++, SAM | 50 | Periodic checkpoint interval (epochs) |
| `--num-classes` | YOLO, SAM, Cellpose | 6 | Number of raw annotation classes |
| `--img-size` | All | 1024 (512 for Cellpose) | Input image size |
| `--seed` | All | 42 | Random seed |
| `--version` | Cellpose | 3 | Cellpose version (2 or 3) |
| `--class-id` | U-Net++ binary, Cellpose | None | Target class ID for per-class training |
| `--all-classes` | U-Net++ binary, Cellpose | off | Train separate models for each class |
| `--rescale` | Cellpose | True | Enable diameter-based rescaling |
| `--scale-range` | Cellpose | 0.6 | Random rescaling range (0.6 → factors 0.7-1.3) |
| `--nimg-per-epoch` | Cellpose | None (all images) | Images per epoch (subsample for speed) |
| `--unfreeze-prompt-encoder` | SAM | off | Unfreeze prompt encoder (default: frozen) |

### Evaluation Pipeline

All 4 models are evaluated via the unified `evaluate.py` script after training:

```
python evaluate.py --model {yolo,unet,sam,cellpose} --strategy A --num-classes {4,5} --checkpoint <path> [--no-postprocess] [--no-vis]
```

- **YOLO**: Loads `best.pt`, runs inference directly
- **U-Net++**: Loads `best-*.ckpt`, runs multilabel sigmoid inference; model_tag includes num_classes (`unet_multilabel_c4` vs `unet_multilabel_c5`) to prevent overwriting
- **SAM**: Loads `best.pth`, uses oracle GT bounding boxes + 3 foreground points as prompts
- **Cellpose**: Loads per-class models from checkpoint directory (traverses dated subfolders: `class_dir.glob("*/models/*")`), runs each model independently, merges predictions

**Benchmark evaluation**: Use `--no-postprocess` for fair model comparison without human prior knowledge. Post-processing can be used for deployment/downstream analysis.

**Post-processing pipeline** (`src/postprocessing.py`):
1. `fill_holes` — Fill artifact holes in masks. Ring-aware for endodermis (class 2) and exodermis (class 4): splits ring into outer boundary and central hole, fills only small artifact holes in the ring band, preserves the structural central hole.
2. `cleanup_whole_root` — Morphological close + keep largest connected component (class 0).
3. `clip_aerenchyma` — Clip aerenchyma (class 1) to inside whole root boundary.
4. `raw_to_target` — Endodermis/exodermis ring subtraction (all models trained on raw classes).

**Default post-processing steps per model** (in `DEFAULT_STEPS` dict):
- **YOLO**: fill_holes, cleanup_whole_root, clip_aerenchyma, raw_to_target
- **U-Net++ multilabel**: fill_holes, cleanup_whole_root, clip_aerenchyma, raw_to_target
- **U-Net++ semantic**: fill_holes, cleanup_whole_root, clip_aerenchyma (rings already derived via paint order)
- **SAM**: fill_holes, cleanup_whole_root, clip_aerenchyma, raw_to_target
- **Cellpose**: fill_holes, cleanup_whole_root, clip_aerenchyma, raw_to_target

Output per run: `metrics.json`, `per_sample.csv`, comparison plots (per-class, summary, species+microscope — PNG only), `vis/` overlay PNGs (original image, GT overlay, prediction overlay).

### General Best Practices

- **Learning rate**: Cosine annealing; lower LR for pretrained backbone (1e-5), higher for new heads (1e-3 to 1e-4)
- **Mixed precision (AMP)**: fp16 on A100
- **Batch size**: 8-16 depending on model; maximize within 80GB VRAM
- **1 GPU per job**: All models train on a single A100

---

## Evaluation Metrics

Report all metrics overall AND broken down by species AND by microscope.

- **mAP@0.5** and **mAP@0.5:0.95** (COCO-style) — per class and overall
- **IoU (Jaccard)** and **Dice (F1)** — per class
- **Precision / Recall** per class at IoU=0.5
- Save to CSV/JSON after each evaluation

---

## Explainability Analysis (for paper)

### Embedding Visualization
- Extract image-level embeddings from U-Net++ bottleneck (2048-dim) and SAM cache (256-dim)
- UMAP colored by: species, microscope, aerenchyma ratio (continuous)
- Expectation: species cluster by anatomy, NOT by microscope platform

### Activation Maps
- GradCAM on U-Net++ encoder layer 4, per target class
- Shows what image regions drive segmentation of each structure

### Channel Importance
- Channel occlusion: zero out one channel (TRITC/FITC/DAPI) at a time, measure per-class IoU drop
- Connects to staining biology: TRITC=suberin, FITC=lignin, DAPI=cell walls

### Morphometric Correlation
- Correlate embedding dimensions with GT-derived morphometrics (aerenchyma count, root area, endodermis thickness)
- Validates that learned representations are biologically meaningful

---

## Downstream Analysis Tasks

After segmentation, compute per-sample:

1. **Aerenchyma area ratio**: total aerenchyma area / whole root area
2. **Aerenchyma count**: number of aerenchyma instances
3. **Endodermis channel intensity**: mean TRITC (suberin), FITC (lignin), DAPI per channel in endodermis ring
4. **Vascular channel intensity**: mean per channel in vascular area

Compare automated vs expert measurements: R-squared, Bland-Altman analysis.

---

## Color Convention for Visualization

| Class | Name | Color |
|-------|------|-------|
| 0 | Whole Root | Blue (0, 0, 255) |
| 1 | Aerenchyma | Yellow (255, 255, 0) |
| 2 | Endodermis | Green (0, 255, 0) |
| 3 | Vascular | Red (255, 0, 0) |
| 4 | Exodermis | Cyan (0, 255, 255) |

---

## Polygon Editor (`polygon_editor.py`)

Interactive GUI for visualizing and correcting YOLO polygon annotations.

### Modes
- **Correct GT**: images + annotation/ + prediction/ — edit GT with predictions as reference
- **Correct Pred**: images + prediction/ — edit predictions, save to annotation/
- **Create GT**: images only — draw annotations from scratch

### UI Layout

**Row 1**: Data (QGroupBox) | Navigation (QGroupBox) | ... stretch ... | mode label | Settings gear (⚙)

**Row 2**: Actions (QGroupBox) | QC (QGroupBox) | ... stretch ...

- **Data**: data folder path | Browse (...)
- **Navigation**: Filter: [None / By Missing / By Containing / By QC] [class dropdown] | [sample dropdown] | "N images"
- **Actions** (QStackedWidget): **main page**: Mode: (Node / Brush) | Draw (N) | Edit (Enter) | Delete (Del) | Copy All Pred->GT | Copy Selected (Ctrl+C) | **modal page**: mode label | class radio buttons (short names) | Undo (Ctrl+Z) | Cancel (Esc) | Confirm (Enter)
- **QC**: QC status label | QC | QC All

**Image panels** (QWidget wrapper with stretch=1): `<` prev | [Original | Editable | Reference] splitter | next `>`

**Settings gear**: QToolButton (top-right) with QMenu/QActionGroup for mode selection (Create GT / Correct GT with Pred / Correct Pred). Default mode: Create GT.

**Status bar**: Visibility: Root | Aer | O.Endo | I.Endo | O.Exo | I.Exo | Brightness: [slider] | γ: [slider] | Reset | Home

### Short Class Names (used in UI)

| ID | Full Name | Short |
|----|-----------|-------|
| 0 | Whole Root | Root |
| 1 | Aerenchyma | Aer |
| 2 | Outer Endodermis | O.Endo |
| 3 | Inner Endodermis | I.Endo |
| 4 | Outer Exodermis | O.Exo |
| 5 | Inner Exodermis | I.Exo |

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| Left / Right | Previous / Next sample |
| N | Draw new polygon (Node or Brush per radio) |
| Enter / Return / Space | Confirm action, or edit selected polygon if idle |
| Escape | Cancel action, or deselect polygon |
| S | Save annotations |
| R | Split selected ring polygon (endo/exo) |
| Delete / Backspace | Delete selected polygon/vertices |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |

### PyQt5 Layout Rules (IMPORTANT)
- **`addWidget()` vs `addLayout()` with stretch**: In a QVBoxLayout, `addLayout(layout, stretch=1)` does NOT work reliably for claiming extra space — bare QLayouts have no sizePolicy. Always wrap in a QWidget and use `addWidget(widget, stretch=1)` instead. This is how the image panel works: `panel_widget = QWidget()` with a QHBoxLayout containing prev_btn + splitter + next_btn, added via `main_layout.addWidget(panel_widget, stretch=1)`.
- **Do NOT use `setSizePolicy()` or `setMaximumHeight()` on QGroupBoxes** to fix layout issues — these cause side effects (clipping, space redistribution to wrong places). If QGroupBoxes expand too much, the root cause is usually a missing stretch on the main content widget.
- **Row 1 is flat** (no QGroupBox) — just bare widgets in a QHBoxLayout. Row 2 uses QGroupBoxes with tight `setContentsMargins(4, 2, 4, 2)` on their inner layouts.
- **Do NOT wrap toolbar rows in a container QWidget with maxHeight** — this fights Qt's layout system and creates gaps.

### Brush Mode Behavior
- **Drawing new polygon** (no selection): default = paint/add, Shift = erase
- **Editing existing polygon** (selected): default = erase, Shift = paint/add
- Ctrl+Scroll = change brush size, Scroll = zoom

### Display Adjustments (display only, do not affect saved data)
- **Brightness slider**: -100 to +100 (pixel value shift)
- **Gamma slider**: 0.1 to 3.0 (gamma correction curve)
- Raw image stored in `_raw_image`, adjustments applied on-the-fly for display

### Filter
- **None**: show all samples
- **By Missing**: show samples missing the selected annotation class
- **By Containing**: show samples containing the selected annotation class
- **By QC**: placeholder for future QC flag filtering

### Edit Mode Class Behavior
- When entering edit mode, the class radio defaults to the polygon's current class
- Changing the radio during editing updates the polygon's class in real time

---

## Paper Outline

**Title**: Automated segmentation of root anatomical barriers enables scalable quantification across species and imaging platforms

### Results Structure

1. **Dataset**: Multi-species annotated dataset for root barrier segmentation (Fig 1)
2. **Benchmark**: Unified model segments root barriers across species and microscopes — Strategy A, 4-model comparison (Fig 2, Table 1)
3. **Generalization**: Monocot-only vs dicot-only vs unified model — Strategy B (Fig 3)
4. **Explainability**: Learned representations encode biologically meaningful features — embeddings, GradCAM, channel importance (Fig 4)
5. **Deployment**: Unseen microscope (Zeiss) — zero-shot from best Strategy A model (Fig 5)
6. **Augmentation ablation**: Channel dropout/shuffle enable cross-domain robustness (Fig 6)
7. **Downstream biology**: Automated aerenchyma and barrier composition quantification matches expert annotations (Fig 7)

### Figure Summary

| Figure | Core Message |
|--------|-------------|
| Fig 1 | Root anatomy, dataset diversity, annotation protocol |
| Fig 2 | The unified model works across all species and microscopes |
| Fig 3 | Unified model vs specialist (monocot-only / dicot-only) models |
| Fig 4 | The model learns biologically meaningful features |
| Fig 5 | Deployment to new platforms and species |
| Fig 6 | Channel dropout/shuffle enable cross-domain generalization |
| Fig 7 | Automated measurements match expert annotations |

---

## HPC Commands

**Sync code to HPC** (excludes data, outputs, and build artifacts):

```bash
rsync -avz --exclude='data/' --exclude='output/' --exclude='__pycache__/' --exclude='.git/' --exclude='*.pyc' ~/Documents/Siobhan_Lab/plants/ hpc2:~/plants/
```

**Submit training jobs:**

```bash
sbatch slurm/run_1_yolo.sh
sbatch slurm/run_2_unet_4c.sh
sbatch slurm/run_3_unet_5c.sh
sbatch slurm/run_sam.sh           # Train + eval (auto)
sbatch slurm/run_cellpose.sh      # 5 parallel class jobs + dependent eval job
```

**Submit evaluation-only jobs:**

```bash
sbatch slurm/eval_all.sh           # All 4 models (--no-postprocess)
sbatch slurm/eval_sam.sh           # SAM only
sbatch slurm/eval_cellpose.sh      # Cellpose only (--no-postprocess)
```

**Monitor jobs:**

```bash
# Job queue status
squeue -u $USER
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"

# Job resource usage (after completion)
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,AllocTRES%40

# Follow live output
tail -f logs/sam_<JOB_ID>.out
tail -f logs/cellpose_<JOB_ID>.out
```

**GPU monitoring** (cannot SSH to compute nodes directly):

```bash
srun --jobid=<JOB_ID> --overlap nvidia-smi
```

**HPC details:**
- Cluster: `hpc2`
- Partition: `gpu-qi`
- GPU type: `a100_80` (NVIDIA A100 80GB)
- Conda env: `plants`
- Cannot SSH to compute nodes (publickey denied) — use `srun --overlap` instead

**Transfer results from HPC to local:**

```bash
# All runs and evaluation results at once
rsync -avz hpc2:~/plants/output/ ~/Documents/Siobhan_Lab/plants/output/

# All logs
rsync -avz "hpc2:~/plants/logs/" ~/Documents/Siobhan_Lab/plants/logs/

# Or per-model (quote remote globs to prevent local expansion)
rsync -avz hpc2:~/plants/output/runs/cellpose/ ~/Documents/Siobhan_Lab/plants/output/runs/cellpose/
rsync -avz hpc2:~/plants/output/runs/sam/ ~/Documents/Siobhan_Lab/plants/output/runs/sam/
```

---

## Model Training Notes

### SAM Training

- **1 GPU** — runs with plain `python train_sam.py` (DDP support available but not used)
- **Batch size** — 8
- **DataLoader workers** — 4 (`--num-workers 4`)

### Cellpose Training

- **SLURM structure** — `run_cellpose.sh` submits 5 parallel training jobs (one per class via `--class-id N`) + a 6th dependent eval job (`--dependency=afterok:id1:id2:...`) using `sbatch --parsable`
- **Numpy truthiness fix** — `train_losses`/`test_losses` checked with `is not None and len(...) > 0` instead of bare `if` (Cellpose returns numpy arrays)
- **AP scores fix** — `cellpose.metrics.average_precision()` returns multi-dimensional arrays (samples × IoU thresholds); must use `ap_scores.mean(axis=-1)` for per-sample AP
- **No confidence scores** — Cellpose does not output per-instance confidence; scores are set to 1.0 during evaluation
