# Plant Root Cross-Section Instance Segmentation Project

## Project Overview

Train instance segmentation models for cell-type topology in cereal plant root cross-section fluorescence microscopy images. The goal is robust segmentation across species (millet, rice, sorghum, tomato), genotypes, and microscopes (C10, Olympus, Zeiss), with demonstrated generalization to unseen species and platforms.

**GPU**: 2 or 3* NVIDIA A100 80GB
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
│   ├── train_unet.py
│   ├── train_sam.py
│   ├── train_cellpose.py
│   ├── run_training.py
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
└── output/                            # Training runs, exports
```

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

**Tomato classes (uses 0, 2, 3, 4, 5 — no aerenchyma):**

| Class ID | Annotated Name | Polygon Meaning |
|----------|---------------|-----------------|
| 0 | Whole Root | Outer boundary of root (defined by epidermis edge) |
| 2 | Outer Endodermis | Outer ring of the endodermis layer |
| 3 | Inner Endodermis | Inner ring — encloses the **vascular** region |
| 4 | Outer Exodermis | Outer ring of the exodermis layer |
| 5 | Inner Exodermis | Inner ring of the exodermis layer |

**Note**: Tomato has no aerenchyma (class 1) — aerenchyma is a monocot feature. Tomato adds exodermis annotations (classes 4-5) not present in cereals.

### Actual Semantic Regions (for model training)

| Target Class | Region | How to Derive | Present in |
|-------------|--------|---------------|------------|
| 0 | Whole Root | Use class 0 polygon directly | All species |
| 1 | Aerenchyma | Use class 1 polygons directly | Cereals only |
| 2 | Endodermis | Subtract class 3 polygon from class 2 polygon (ring) | All species |
| 3 | Vascular | Use class 3 polygon directly (area inside inner endodermis) | All species |
| 4 | Exodermis | Subtract class 5 polygon from class 4 polygon (ring) | Tomato only |

### Biology Context

- **Epidermis**: outermost cell layer, defines whole root boundary
- **Cortex**: region between epidermis and outer endodermis; contains aerenchyma in cereals
- **Endodermis**: thin layer with Casparian strip; higher signal in FITC (lignin) and TRITC (suberin) channels; derived by subtraction (outer - inner ring)
- **Exodermis**: barrier layer between cortex and epidermis; annotated in tomato (classes 4-5) as outer/inner ring pair, derived by subtraction like endodermis; not annotated in cereals
- **Vascular cylinder**: innermost region enclosed by inner endodermis
- **Aerenchyma**: irregular air spaces in the cortex only; present in cereals (monocots), absent in tomato (dicot)

### Annotation Rules

- Polygons can overlap but boundaries must NOT intersect
- All aerenchyma polygons are contained within the whole root polygon
- Cereal annotations: typically 1 whole root, many aerenchyma, 1 outer endodermis, 1 inner endodermis
- Tomato annotations: exactly 1 polygon per class (whole root, outer/inner endodermis, outer/inner exodermis); all 5 classes present in every sample

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
- **Classes**: 4 (YOLO) or 5 including exodermis (U-Net++, SAM, Cellpose). Use `--num-classes 5 --mask-missing` for U-Net++ to handle missing annotations per species.

### Strategy B — Leave-one-species-out (generalization)

| Rotation | Train on | Test on | Question |
|----------|----------|---------|----------|
| B1 | Rice + Sorghum | **Millet** | Cereal → unseen cereal |
| B2 | Millet + Sorghum | **Rice** (Olympus+C10) | Cereal → unseen cereal |
| B3 | Millet + Rice | **Sorghum** | Cereal → unseen cereal |
| B4 | Millet + Rice + Sorghum | **Tomato** | Monocot → dicot |

- **No tomato in B1-B3 training** — cereals only, to isolate cross-cereal transfer
- B4 tests the monocot-to-dicot divide
- All use Olympus + C10 only
- Run with both YOLO and U-Net++ (8 runs total)

### Deployment Use Cases (zero-shot, no training)

- **Zeiss**: Apply best Strategy A model to all 35 Zeiss images (unseen microscope)
- **Striga**: Apply best Strategy A model to Striga images (extreme OOD, parasitic plant)

---

## Models to Train

### Training Run Plan

| Run | Model | Strategy | Classes | Masking | Purpose |
|-----|-------|----------|---------|---------|---------|
| 1 | YOLO11m-seg | A | 4 | N/A | Benchmark (YOLO = 4 classes only) |
| 2 | U-Net++ (multilabel) | A | 4 | No | Benchmark (4-class) |
| 3 | U-Net++ (multilabel) | A | 5 | Yes | Benchmark (5-class, masked loss) |
| 4 | SAM 2 (ViT-B) | A | 5 | N/A | Benchmark (prompt-based, natural handling) |
| 5 | Cellpose 3.0 (per-class) | A | 5 | N/A | Benchmark (per-class models, natural handling) |
| 6-9 | YOLO11m-seg | B1-B4 | 4 | N/A | Generalization |
| 10-13 | U-Net++ | B1-B4 | 4 | No | Generalization |
| 14-16 | Best model | A (augmentation ablation) | — | — | Ablation study |

### Benchmark Hyperparameters (Runs 1-5, Strategy A)

| Param | Run 1: YOLO | Run 2: U-Net++ 4c | Run 3: U-Net++ 5c | Run 4: SAM | Run 5: Cellpose |
|-------|-------------|--------------------|--------------------|------------|-----------------|
| Architecture | yolo11m-seg | unetplusplus | unetplusplus | vit_b | cyto3 |
| Encoder | CSPDarknet (COCO) | resnet34 (ImageNet) | resnet34 (ImageNet) | ViT-B (SA-1B) | cyto3 (pretrained) |
| Num classes | 4 | 4 | 5 | 5 | 5 |
| Mask missing | N/A | No | Yes | N/A | N/A |
| Image size | 1024 | 1024 | 1024 | 1024 | 1024 |
| Epochs | 300 | 300 | 300 | 300 | 150 |
| Patience | 15 | 15 | 15 | 15 | — |
| Batch size | 16 | 16 | 16 | 8 | 8 |
| LR (head/decoder) | auto (Ultralytics) | 1e-4 | 1e-4 | 1e-4 | 0.1 |
| LR (backbone) | auto (Ultralytics) | 1e-5 | 1e-5 | frozen | — |
| Weight decay | 5e-4 (Ultralytics) | 1e-4 | 1e-4 | 1e-4 | 1e-4 (Cellpose) |
| Optimizer | SGD (Ultralytics) | AdamW | AdamW | AdamW | AdamW (Cellpose) |
| Scheduler | auto (Ultralytics) | CosineAnnealing (eta_min=1e-7) | CosineAnnealing (eta_min=1e-7) | CosineAnnealing (eta_min=1e-7) | — |
| Loss | Box+Seg+Cls (YOLO) | BCE+Dice (1:1) | BCE(masked)+Dice(masked) (1:1) | BCE+Dice (1:1) | Cellpose flow |
| BCE pos_weight | N/A | [1, 2, 5, 1] | [1, 2, 5, 1, 5] | N/A | N/A |
| Trainable params | 22.4M | 24.4M | 24.4M | 4.1M (93.7M total) | ~13M |
| Precision | fp16 (AMP) | fp16 (AMP) | fp16 (AMP) | fp16 (AMP) | fp32 |
| Frozen layers | None | None | None | Image encoder + prompt encoder | None |

**Notes on 5-class support:**
- YOLO only supports 4 classes (exodermis classes 4/5 are filtered from annotations during export)
- U-Net++ with `--num-classes 5 --mask-missing`: uses validity masking so missing classes (aerenchyma for tomato, exodermis for cereals) contribute zero loss
- SAM and Cellpose handle missing classes naturally (prompt-based / per-class models)

### Model Architecture Details

#### YOLO11m-seg
- **Pretrained on**: COCO
- **Frozen**: Nothing (full fine-tuning; Ultralytics handles warmup internally)
- **Trainable**: Entire model (CSPDarknet backbone + PANet neck + detection head + segmentation head)
- **Embedding extraction**: Forward hook on SPPF layer → global average pool → image-level vector
- **Key setting**: Disable HSV hue/saturation augmentation (meaningless for fluorescence)

#### U-Net++ (multilabel mode — primary)
- **Encoder**: ResNet50 (ImageNet-pretrained) via `segmentation_models_pytorch`
- **Output**: 4 or 5-channel sigmoid (root, aerenchyma, endodermis, vasculature [, exodermis]) — allows overlapping predictions
- **Phase 1** (epochs 1-10): Encoder **frozen**, decoder LR = 1e-3
- **Phase 2** (epochs 11+): Encoder **unfrozen** at LR = 1e-5, decoder LR = 1e-4
- **Loss**: BCE (pos_weight: root=1, aer=2, endo=5, vasc=1, exo=5) + Dice
- **Masked loss** (`--mask-missing`): BCE computed with `reduction='none'`, multiplied by validity mask `(B, C, 1, 1)`, reduced over valid entries only. Dice computed per-channel on valid samples only. Validity determined by `SPECIES_VALID_CLASSES` in config.
- **Embedding extraction**: `model.encoder(x)[-1]` → global average pool → (2048,) vector
- **GradCAM**: On encoder layer 4 for per-class activation maps

#### SAM 2 (ViT-B)
- **Pretrained on**: SA-1B
- **Frozen**: Image encoder (completely) + prompt encoder
- **Trainable**: Mask decoder only
- **Embeddings**: Pre-computed and cached (already in pipeline); global average pool → (256,) vector
- **Prompts**: 3 random foreground points + bounding box with 5% jitter from GT
- **Loss**: BCE + Dice
- **Limitation**: Requires prompts at inference (not fully automatic)

#### Cellpose 3.0
- **Base model**: cyto3 (microscopy-pretrained)
- **Training**: Per-class models (separate model for each target class)
- **Frozen**: Nothing (Cellpose manages internally)
- **Embedding extraction**: Style network → (256,) global vector
- **Limitation**: Black-box flow-based post-processing; harder to interpret

---

## Data Loading & Preprocessing

- Load 3-channel TIF → float32 tensor (R=TRITC, G=FITC, B=DAPI)
- Percentile normalization: 1st-99.5th percentile per channel → [0, 1]
- Resize to 1024x1024 with aspect ratio preservation + zero-padding
- Derive endodermis mask by subtracting inner endodermis from outer endodermis polygon
- Derive exodermis mask (tomato only) by subtracting inner exodermis from outer exodermis polygon

### Augmentation (fluorescence microscopy appropriate)

- Random horizontal/vertical flips, 90-degree rotations
- Affine transforms, elastic deformation (mild)
- Brightness/contrast adjustment, Gaussian blur, Gaussian noise
- **Channel dropout** (p=0.2): zero out 1 of 3 channels — key for cross-microscope robustness
- **Channel shuffle** (p=0.2): randomly permute channel order — key for cross-microscope robustness
- Coarse dropout (spatial patches), gamma adjustment
- Do NOT use color jitter (hue shifts meaningless for fluorescence)

---

## Training Configuration (Strategy A Benchmark, Runs 1-5)

### Early Stopping

| Run | Model | Trigger | Monitor | Patience |
|-----|-------|---------|---------|----------|
| 1 | YOLO | Ultralytics internal | mAP | 15 epochs |
| 2-3 | U-Net++ | PyTorch Lightning `EarlyStopping` | val_loss | 15 epochs |
| 4 | SAM | Manual counter in training loop | val_loss | 15 epochs |
| 5 | Cellpose | None (fixed epochs) | — | — |

### Checkpointing

| Run | Model | Best | Periodic | Last |
|-----|-------|------|----------|------|
| 1 | YOLO | `best.pt` (auto) | Every N epochs (`--save-every`, default 50) | `last.pt` (auto) |
| 2-3 | U-Net++ | `best-*.ckpt` (PL `ModelCheckpoint`) | Every N epochs (`--save-every`, default 50) | `last.ckpt` (auto) |
| 4 | SAM | `best.pth` (manual) | Every N epochs (`--save-every`, default 50) | — |
| 5 | Cellpose | Auto (Cellpose internal) | Auto (Cellpose internal) | — |

### Dated Run Subfolders

Every training run creates a unique dated subfolder: `YYYY-MM-DD_NNN` (auto-incrementing NNN for same-day runs). All hyperparameters are saved to `hparams.yaml` for reproducibility.

```
output/runs/
├── yolo/yolo11m-seg_A/
│   ├── 2026-03-15_001/weights/best.pt, hparams.yaml
│   └── 2026-03-17_001/weights/best.pt, hparams.yaml  (retrain)
├── unet/unetplusplus_resnet34_A_multilabel_c4/
│   └── 2026-03-15_001/checkpoints/best-*.ckpt, hparams.yaml
├── sam/sam_vit_b_A_c5/
│   └── 2026-03-15_001/best.pth, hparams.yaml
└── cellpose/cellpose_v3_{ClassName}_A_c5/
    └── 2026-03-16_001/models/*, hparams.yaml
```

Implemented via `make_run_subfolder()` and `save_hparams()` in `src/config.py`. SLURM eval scripts and `run_grid_training.py` use glob patterns (`*/`) to find the latest checkpoint across dated subfolders.

### Training Outputs

| Run | Model | Output Files |
|-----|-------|-------------|
| 1 | YOLO | `results.csv`, confusion_matrix, PR curves, loss curves, label plots (Ultralytics auto) |
| 2-3 | U-Net++ | `metrics.csv`, `loss_curve.png`, best + last + periodic checkpoints |
| 4 | SAM | `training_history.json`, `loss_curve.png`, best + periodic checkpoints |
| 5 | Cellpose | `training_history.json`, `loss_curve.png`, `test_results.json` |

### Configurable Training Flags

All training scripts support these CLI flags (where applicable):

| Flag | Scripts | Default | Description |
|------|---------|---------|-------------|
| `--epochs` | All | 300 (150 Cellpose) | Max training epochs |
| `--batch-size` | All | 16 (8 for SAM/Cellpose) | Batch size |
| `--lr` | U-Net++, SAM, Cellpose | varies | Learning rate |
| `--backbone-lr` | U-Net++ | 1e-5 | Backbone/encoder LR (differential) |
| `--weight-decay` | U-Net++, SAM | 1e-4 | Weight decay |
| `--patience` | YOLO, U-Net++, SAM | 15 | Early stopping patience |
| `--optimizer` | U-Net++, SAM | adamw | Optimizer (adamw/adam/sgd) |
| `--scheduler` | U-Net++, SAM | cosine | LR scheduler (cosine/step/plateau) |
| `--eta-min` | U-Net++, SAM | 1e-7 | Minimum LR for cosine scheduler |
| `--pos-weight` | U-Net++ | [1,2,5,1] or [1,2,5,1,5] | BCE pos_weight per class |
| `--bce-weight` | U-Net++, SAM | 1.0 | BCE loss weight |
| `--dice-weight` | U-Net++, SAM | 1.0 | Dice loss weight |
| `--save-every` | YOLO, U-Net++, SAM | 50 | Periodic checkpoint interval (epochs) |
| `--num-classes` | All | 4 | Number of target classes (4 or 5) |
| `--mask-missing` | U-Net++ | off | Enable validity masking for missing classes |
| `--img-size` | All | 1024 | Input image size |
| `--seed` | All | 42 | Random seed |

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

Output per run: `metrics.json`, `per_sample.csv`, comparison plots (per-class, summary, species+microscope — PNG only), `vis/` overlay PNGs (original image, GT overlay, prediction overlay).

### General Best Practices

- **Learning rate**: Cosine annealing; lower LR for pretrained backbone (1e-5), higher for new heads (1e-3 to 1e-4)
- **Mixed precision (AMP)**: fp16 on A100
- **Batch size**: 8-16 depending on model; maximize within 80GB VRAM
- **Strategy naming**: CLI uses `--strategy A`/`B`/`C` everywhere (legacy `strategy1`/`strategy2`/`strategy3` also accepted by `get_split()`)

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

## Paper Outline

**Title**: Automated segmentation of root anatomical barriers enables scalable quantification across species and imaging platforms

### Results Structure

1. **Dataset**: Multi-species annotated dataset for root barrier segmentation (Fig 1)
2. **Benchmark**: Unified model segments root barriers across species and microscopes — Strategy A, 4-model comparison (Fig 2, Table 1)
3. **Cereal transfer**: Root anatomy transfers between cereal species — Strategy B1-B3, YOLO + U-Net++ (Fig 3)
4. **Monocot → dicot**: Transfer from cereals to tomato — Strategy B4 (Fig 4)
5. **Explainability**: Learned representations encode biologically meaningful features — embeddings, GradCAM, channel importance (Fig 5)
6. **Deployment**: Unseen microscope (Zeiss) and extreme OOD (Striga) — zero-shot from best Strategy A model (Fig 6)
7. **Augmentation ablation**: Channel dropout/shuffle enable cross-domain robustness (Fig 7)
8. **Downstream biology**: Automated aerenchyma and barrier composition quantification matches expert annotations (Fig 8)

### Figure Summary

| Figure | Core Message |
|--------|-------------|
| Fig 1 | Root anatomy, dataset diversity, annotation protocol |
| Fig 2 | The unified model works across all species and microscopes |
| Fig 3 | Root anatomy is partially conserved across cereals |
| Fig 4 | Transfer across the monocot-dicot divide |
| Fig 5 | The model learns biologically meaningful features |
| Fig 6 | Deployment to new platforms and species |
| Fig 7 | Channel dropout/shuffle enable cross-domain generalization |
| Fig 8 | Automated measurements match expert annotations |

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

- **On-the-fly encoding** — image encoder (frozen) runs every batch instead of pre-computing embeddings, enabling full image augmentation (flips, rotations, noise, channel dropout/shuffle) every epoch
- **Multi-GPU DDP** — supports `torchrun --nproc_per_node=N`; SLURM script requests 2 GPUs by default
- **Prompt augmentation** — point prompts and box jitter are re-randomized every `__getitem__` call
- **Batch size** — 8 per GPU (16 effective with 2 GPUs)
- **DataLoader workers** — 4 per GPU (`--num-workers 4`)
- **Single-GPU fallback** — works with plain `python train_sam.py` (auto-detects DDP)

### Cellpose Training

- **Image size** — 512 (not 1024; 4x faster per sample)
- **Data caching** — images and annotations loaded once via `preload_cellpose_data()`, per-class labels built from cache via `build_class_labels()` (no disk I/O per class)
- **Per-class models** — trains 5 separate models (Whole Root, Aerenchyma, Endodermis, Vascular, Exodermis), 150 epochs each
- **SLURM structure** — `run_cellpose.sh` submits 5 parallel training jobs (one per class via `--class-id N`) + a 6th dependent eval job (`--dependency=afterok:id1:id2:...`) using `sbatch --parsable`
- **`nimg_per_epoch=200`** — subsamples training images per epoch for speed
- **Numpy truthiness fix** — `train_losses`/`test_losses` checked with `is not None and len(...) > 0` instead of bare `if` (Cellpose returns numpy arrays)
- **AP scores fix** — `cellpose.metrics.average_precision()` returns multi-dimensional arrays (samples × IoU thresholds); must use `ap_scores.mean(axis=-1)` for per-sample AP

### Augmentation Summary

| Augmentation | U-Net++ | SAM | YOLO | Cellpose |
|-------------|---------|-----|------|----------|
| Flips / rotations | Albumentations | Albumentations | Ultralytics built-in | Cellpose built-in |
| Affine / elastic | Yes | Yes | Ultralytics built-in | No |
| Brightness / contrast / gamma | Yes | Yes | HSV value only (0.2) | No |
| Gaussian noise / blur | Yes | Yes | No | No |
| Coarse dropout | Yes | Yes | No | No |
| **Channel dropout** (p=0.2) | **Yes** | **Yes** | **No** | **No** |
| **Channel shuffle** (p=0.2) | **Yes** | **Yes** | **No** | **No** |
| Mosaic / Mixup | No | No | Yes | No |
| Applied via | `get_train_transform()` | `AugmentedSAMDataset` | `model.train()` kwargs | `train.train_seg()` internal |
| Re-randomized each epoch | Yes | Yes | Yes | Yes |

**Limitation**: YOLO and Cellpose do **not** support channel dropout/shuffle. Both frameworks use black-box training loops that don't allow injecting custom per-epoch augmentation. This is noted as a limitation in the paper — the ablation study (Fig 7) compares augmentation impact only on U-Net++/SAM.
