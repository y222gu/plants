# Plant Root Cross-Section Instance Segmentation Project

## Project Overview

Train segmentation models for cell-type topology in cereal plant root cross-section fluorescence microscopy images. Robust segmentation across species (millet, rice, sorghum, tomato), genotypes, and microscopes (C10, Olympus, Zeiss), with generalization to unseen species and platforms.

**GPU**: 1 NVIDIA H100 86GB per model | **Dataset**: ~1,600 samples, YOLO polygon format | **Target**: Nature Plants

**Current best model**: DINOv3-S/16 + Meta-exact DPT decoder (Strategy A, full aug, Dice+Focal+wCE+Lovász). See `project_best_model` memory for run path, checkpoint, and ablation results.

---

## Directory Structure

```
plants/
├── data/
│   ├── image/{Species}/{Microscope}/{Exp}/{Sample}/{Sample}_{DAPI|FITC|TRITC}.tif
│   └── annotation/{Species}_{Microscope}_{Exp}_{Sample}.txt  # YOLO polygons
├── train/                    # train_yolo.py, train_unet_semantic.py, train_unet_binary.py,
│                             # train_sam_semantic.py, train_sam_unetpp.py, train_timm_semantic.py,
│                             # train_plantseg.py, train_cellpose_sam.py, train_yolo_ablation.py
├── src/                      # Shared library (dataset, preprocessing, splits, augmentation,
│                             # evaluation, downstream, model_classes, formats/, models/)
├── run_eval_pipeline.py      # Entry point: eval + downstream + correlation plots
├── eval_bio7.py              # Segmentation eval (IoU/Dice)
├── downstream_measure_from_{model,predictions}.py
├── downstream_plot_correlations.py
├── predict.py                # Inference
├── polygon_editor.py         # Interactive annotation GUI
├── slurm/                    # SLURM job scripts
├── figures_for_paper/        # HTML-based figure builders (per-panel + assemblers)
├── output/                   # Training runs, exports
└── data_to_edit/             # Samples needing exodermis annotation
```

**Species**: Millet, Rice, Sorghum (monocots/cereals), Tomato (dicot)
**Microscopes**: Olympus IX83 (widefield), Cytation C10 (plate reader), Zeiss LSM 970 (confocal)
**Channels**: DAPI (blue/cell walls), FITC (green/lignin), TRITC (red/suberin) — 3 grayscale TIFs per sample

---

## Label Definitions

YOLO polygon format: `class_id x1 y1 x2 y2 ... xn yn` (normalized 0-1).

### Raw Annotation Classes (6)

| ID | Name | Present in |
|----|------|-----------|
| 0 | Whole Root | All |
| 1 | Aerenchyma | Cereals only (zero in tomato) |
| 2 | Outer Endodermis | All |
| 3 | Inner Endodermis | All |
| 4 | Outer Exodermis | All |
| 5 | Inner Exodermis | All |

All samples annotated for all 6 classes. Biologically absent classes have zero polygons — real biology, not missing data. Models learn zero area for absent classes.

### 7 Bio Classes (derived)

Epidermis, Cortex, Endodermis, Exodermis, Vascular, Aerenchyma, Whole Root. Endodermis/Exodermis derived from outer-minus-inner ring subtraction. Vascular = inside Inner Endodermis. Cortex = whole root minus all rings.

### Annotation Rules

- Polygons may overlap but boundaries must NOT intersect
- All aerenchyma contained within whole root
- Cereal: 1 root, many aer, 2 endo rings, 0 exo | Tomato: 1 root, 0 aer, 2 endo + 2 exo rings

QC: 415/1,671 flagged but all minor boundary touching (max 2.3% area). No preprocessing needed.

---

## Dataset Splitting

**Critical rule**: Samples from same experiment (`Exp{N}`) must stay together — never split across train/test.

| Species | Microscope | Train | Val | Test | Total |
|---------|------------|------:|----:|-----:|------:|
| Millet | Olympus | 71 | 7 | 32 | 110 |
| Rice | C10 | 38 | 6 | 6 | 50 |
| Rice | Olympus | 402 | 49 | 51 | 502 |
| Rice | Zeiss | 0 | 0 | 35 | 35 |
| Sorghum | C10 | 25 | 11 | 8 | 44 |
| Sorghum | Olympus | 320 | 38 | 51 | 409 |
| Tomato | C10 | 44 | 11 | 10 | 65 |
| Tomato | Olympus | 393 | 60 | 27 | 480 |
| **Total** | | **1293** | **182** | **220** | **1695** |

- **Zeiss (35 samples) fully held out** — zero-shot ("oneshot") evaluation only
- **Strategy A** (unified): all species together; primary benchmark
- **Strategy B-mono / B-dico**: specialist models (monocots only / tomato only) for generalization figure. Note: A / B-mono / B-dico are *internal shorthand* — never use these codes in paper figure labels; in figures call them "Unified Model", "Monocot Specialist", "Dicot Specialist" (Title Case for every word in row/column labels).
- Millet splits were corrected 2026-04-05 to 12 plant groups with no cross-split leakage; runs before that date have data leakage

---

## Models

### Current best (paper headline)
**DINOv3-S/16 + Meta-exact DPT decoder** — 7-class semantic segmentation, Strategy A. See `project_best_model` memory for full recipe, run directory, checkpoint, and 7-run ablation table.

### Architectures trained

| Model | Architecture | Params | Input | Output head | Script |
|-------|-------------|--------|-------|-------------|--------|
| YOLO26m-seg | YOLO26 | 23.6M | 1024 | Instance polygons (6 cls) | `train_yolo.py` |
| U-Net++ multilabel | ResNet34 + smp | 24.4M | 1024 | 6-ch sigmoid | `train_unet_binary.py` |
| U-Net++ semantic | ResNet34 + smp | 24.4M | 1024 | 7-cls softmax | `train_unet_semantic.py` |
| Timm encoder + UNet++/DPT | ConvNeXtV2 / SwinV2 / DINOv2 / DINOv3 | 28-34M | 1024 | 7-cls softmax | `train_timm_semantic.py` |
| SAM UNet++ / SAM UNETR | ViT-B + decoders | ~93M | 1024 | 7-cls softmax | `train_sam_{unetpp,semantic}.py` |
| micro-SAM (per-class) | ViT-B (vit_b_lm) | ~93M | 1024 | Per-class instance masks | (micro-SAM library) |
| PanSeg UNet2D | PanSeg laid-back-lobster | 5.4M | 1024 | 6-ch sigmoid | `train_plantseg.py` |
| Cellpose-SAM | (abandoned — flow pipeline mismatch) | — | — | — | `train_cellpose_sam.py` |

**Shared defaults**: 200 epochs, patience=15, seed=42, AdamW + CosineAnnealing, fp16/bf16, 1 GPU.

**U-Net++ differential LR**: encoder 1e-5, decoder 1e-4.

**CE class_weights (7-cls semantic)**: bg=0.5, epi=1, aer=10 (configurable via `--aer-weight`), endo=5, vasc=1, exo=5, cortex=1. Aerenchyma weight is critical for Millet (aer occupies ~0.5% of image pixels vs 10%+ in rice/sorghum).

**pos_weight (6-ch BCE)**: root=1, aer=10, o.endo=5, i.endo=1, o.exo=5, i.exo=1.

### Key training rules

- **YOLO**: must use `overlap_mask=False`. Default `overlap=True` paints nested instances onto a single canvas sorted by area, so outer containers (Whole Root, Outer Endo) lose overlapping pixels to inner instances — model learns rings instead of filled polygons. Fix gave +0.11 Bio-7 mIoU (0.73→0.84) on outer-containing classes. Contour-fill at inference is a no-op safety net.
- **Lovász loss under AMP**: wrap in `autocast(enabled=False)` + `logits.float()` — smp's `_lovasz_grad` returns fp32 while logits are bf16, causing `torch.dot` mismatch without the cast. Essential for DINOv3 runs.
- **Semantic vs multilabel**: semantic models (7-cls softmax) outperform multilabel (6-ch sigmoid) for anatomical structure segmentation. Rings are derived by paint order.
- **Encoder pretraining**: ImageNet pretraining beats from-scratch by ~6.5% mIoU. ImageNet→MicroNet (materials microscopy) gave no benefit over plain ImageNet. Self-supervised DINOv3 on LVD-1.69B is the current best encoder.
- **Channel augmentation ablation (on DINOv3+DPT)**: Dropout alone ≈ full (Dropout+Shuffle). Shuffle is redundant. No-aug loses ~0.012 oneshot mIoU. See memory.

### Timm implementation notes

- **Swin**: pass `img_size=1024` to `timm.create_model()`; permute `(B,H,W,C)→(B,C,H,W)` after `features_only=True`.
- **DINOv2/v3 ViT** (patch_size=14/16): pad input to multiple of patch size with reflect padding, run encoder, crop back. For p=14 at 1024: pad to 1036.
- **Custom DPT decoder**: smp's built-in DPT had bugs at non-native resolutions. Wrote a Ranftl-2021-style decoder (tap 4 ViT layers → 1×1 proj → progressive bottom-up fusion with residual conv blocks → upsample).
- **UNet++ decoder wiring**: import `smp.UnetPlusPlusDecoder` directly (not `smp.UnetPlusPlus`) to bypass strict encoder validation. Use `depth=4` for timm encoders (4 feature levels vs ResNet's 5).
- All encoders are unmodified from timm.

---

## Data Pipeline

All models: 3 TIFs → `load_sample_normalized()` (percentile 1st-99.5th → [0,1] float32) → augmentation → model.

For downstream intensity analysis use `load_sample_raw()` (unnormalized).

### Augmentation

**Shared albumentations** (`src/augmentation.py`, U-Net++ / SAM / timm): RandomRotate90, H/V Flip, Affine (translate/scale/rotate/shear), ElasticTransform, RandomBrightnessContrast, GaussianBlur, GaussNoise, RandomGamma, ChannelDropout p=0.2, ChannelShuffle p=0.2, Resize.

**YOLO**: Ultralytics built-in matched to shared pipeline; `bgr=0.2` (channel swap); no elastic/blur/noise/gamma/channel-dropout.

No hue/saturation jitter — fluorescence channels have fixed biological meaning.

---

## Evaluation & Downstream Pipeline

**Entry point**: `run_eval_pipeline.py` — runs eval + downstream + correlation plots.

```bash
# Full pipeline (test + oneshot, downstream from saved predictions)
python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/

# Single split / downstream source
python run_eval_pipeline.py ... --split {test,oneshot}
python run_eval_pipeline.py ... --downstream-source {predictions,model}    # model needs GPU
python run_eval_pipeline.py ... --no-downstream --no-eval --force
```

### Output layout

```
{run_dir}/
├── checkpoints/ logs/ tensorboard/ hparams.yaml loss_curve.png
├── eval/
│   ├── test/                # Strategy A test (196)
│   │   ├── metrics_native.csv   metrics_bio7.csv
│   │   ├── predictions/         # YOLO polygon .txt
│   │   └── vis_native/  vis_bio7/
│   └── oneshot/             # Zeiss zero-shot (35)
└── downstream/
    ├── test_from_predictions/ test_from_model/ oneshot_from_predictions/
    │   ├── gt_measurements.csv  pred_measurements.csv  correlation_summary.csv
    │   └── plots/           # correlation_combined.png + per-species
```

### IoU is computed on raw model output — no post-processing, no filling, no polygon conversion

| Model | Prediction format | IoU compared against |
|-------|------------------|----------------------|
| Semantic (U-Net++ / SAM / timm) | `(H,W) int32` argmax (7 cls) | `(pred == cls) vs (gt == cls)` |
| Multilabel | `dict{0..5: (H,W) binary}` from `sigmoid > 0.5` | Per-channel binary |
| YOLO | Per-instance merged via `merge_classes()` | `dict{0..5: (H,W) binary}` |

**Bio-7 conversion** (applied identically to GT and pred):
- Semantic: `unet_semantic_to_bio7()` — pure pixel boolean ops
- YOLO/multilabel: `yolo_overlap_false_to_bio7()` — pixel subtraction to derive rings

### Prediction saving (YOLO polygon .txt)

After IoU, predictions are saved as YOLO polygons via `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` with `min_area=50`. Roundtrip drops interior holes and simplifies vertices. Empirically changes mIoU by +0.0008 (paired t-test p=0.34, not significant). **Does not affect reported metrics** — only saved `.txt` files.

### Downstream measurements

Per-sample on **raw unnormalized** images:
- Aerenchyma ratio (aer / whole root area)
- Exodermis / Endodermis / Vascular TRITC + FITC mean intensity

`--downstream-source predictions` rasterizes saved polygons (subject to roundtrip artifacts). `--downstream-source model` re-runs inference (no polygon artifacts; needs GPU).

### Metrics
IoU, Dice — per class (native + bio-7), per species, per microscope. Downstream R² and slope from linear regression of predicted vs GT measurements.

---

## Paper Outline

**Title**: Automated segmentation of root anatomical barriers enables scalable quantification across species and imaging platforms

| Figure | Content | Models |
|--------|---------|--------|
| Fig 1 | Dataset diversity, annotation protocol | None |
| Fig 2 | Benchmark: unified model across species/microscopes (Strategy A) | All generalists |
| Fig 3 | Generalization: unified vs specialist (B-mono, B-dico) | Best model only |
| Fig 4 | Explainability: embeddings, GradCAM, channel importance | Best model |
| Fig 5 | Deployment: zero-shot on Zeiss | Best model |
| Fig 6 | Augmentation ablation (channel dropout/shuffle) | Best model only |
| Fig 7 | Downstream: automated vs expert measurements | Best model |

**Strategy**: Strategy B (specialist vs generalist) and augmentation ablation run on **best model only** — they test data/augmentation hypotheses, not architecture hypotheses.

### Explainability
UMAP of embeddings colored by species/microscope/aer-ratio. GradCAM on encoder per class. Channel occlusion: zero out TRITC/FITC/DAPI, measure IoU drop.

---

## Color Conventions

### Raw classes (`CLASS_COLORS_RGB` — YOLO, multilabel diagnostics)
Whole Root=Blue, Aer=Yellow, Outer Endo=Green, Inner Endo=Red, Outer Exo=Orange, Inner Exo=Purple.

### Target classes (`TARGET_CLASS_COLORS_RGB` — downstream diagnostics)
Whole Root=Blue, Aer=Yellow, Endo=Green, Vascular=Red, Exo=Cyan.

### Diagnostic Bio-7 palette (`BIO_7_COLORS_RGB` in `src/model_classes.py`)
Used in `eval/vis_bio7/` during evaluation. **Do not use for paper figures.**

---

## Paper Figure Conventions

Applies to every figure in the Nature Plants manuscript. Don't invent palettes or fonts per-figure.

### Bio-7 anatomy palette (Figure 1a canonical)

| Class | Hex | Role |
|-------|-----|-----|
| Epidermis | `#0a9396` | outer teal ring |
| Exodermis | `#f4a261` | orange ring (tomato only) |
| Cortex | `#94d2bd` | light teal mid region |
| Aerenchyma | `#264653` | dark teal (paint last, over cortex) |
| Endodermis | `#f6e48e` | yellow ring |
| Vascular | `#e76f61` | salmon center |
| Whole Root | (not painted) | bg color |

Paint order (outer → inner): Whole Root → Epidermis → Exodermis → Cortex → Endodermis → Vascular → **Aerenchyma** (last, overlaps Cortex).

### Channel composite (Figure 1c canonical)

DAPI→cyan (G+B), FITC→yellow (R+G), TRITC→red (R). Additively blended, clipped to [0,1]. Each channel percentile-normalised (1st/99.5th).

```python
comp[..., 1] += dapi;  comp[..., 2] += dapi      # DAPI  → cyan
comp[..., 0] += fitc;  comp[..., 1] += fitc      # FITC  → yellow
comp[..., 0] += tritc                            # TRITC → red
comp = np.clip(comp, 0, 1)
```

Do **not** use the `(R=TRITC, G=FITC, B=DAPI)` mapping from `load_sample_normalized()` for paper figures — that's the training convention.

### Panel backgrounds
GT and prediction mask panels use **black background** outside the root. Horizontal-concat gaps are black. Never white.

### Canvas & sizing
- **180 mm wide** (Nature 2-column), height capped at **170 mm**
- Per-panel sized in mm via CSS variables (`--w-mm`, `--h-mm`) in each builder's `:root`
- One HTML file per panel with SVG inside mm-sized `<svg>` + Save PNG (600 dpi) / Save SVG buttons
- Assembler HTMLs place exported SVGs via `<img>` in absolute-positioned `<div class="panel">`, driven by a `PANEL_SIZES` table at the top

### Typography (all figures, all panels)
**Font: Helvetica everywhere.** Fall back Arial → DejaVu Sans only if Helvetica missing. No serif, no novelty fonts.
- HTML/SVG: `font-family: Helvetica, Arial, sans-serif` on `body` and `svg text`
- Python PIL: prefer `/System/Library/Fonts/Helvetica.ttc` (index 0=Regular, 1=Bold)

Sizes (mm, viewBox-relative):
- Tick labels 2.0 mm (~5.7 pt) | Axis titles 2.5 mm (~7 pt) | Legend text 1.8-2.0 mm
- Panel letters (a/b/c/d): **10 pt bold Helvetica** at `top: 2mm; left: 2mm` inside the panel. In SVG export, set `font-size` as a **unitless** number (= viewBox-mm); appending `mm` causes some renderers to re-convert via 1mm = 3.78px and the letter prints ~4× too big.
- Tick marks 0.8 mm (stroke 0.2 mm) | Axis line stroke 0.25 mm | Legend swatch radius 0.9-1.0 mm
- **Axis-label → axis-title gap: 0.5 mm** (matches `header_bottom_pad_mm` in `make_fig2d.py`):
  - x-title baseline ≈ axis + 5.6 mm (horizontal labels)
  - y-title centre ≈ axis − 7 mm (4-char labels) / −6 mm (3-char)

### Builder workflow
1. Build each panel as HTML + SVG + Save buttons (not matplotlib — see `feedback_html_figure_workflow` memory).
2. Serve via `python3 -m http.server 8000` in `figures_for_paper/`; always cache-bust with `?t=Date.now()` and `cache: "no-store"`.
3. Export SVG from each panel, drop into `figureN/`.
4. Open `figureN/assemble_figureN.html` to preview and export combined PNG/SVG.

---

## Polygon Editor (`polygon_editor.py`)

Interactive GUI for visualizing and correcting YOLO polygon annotations.

### Modes
Create GT (draw from scratch) | Correct GT (edit GT with predictions as reference) | Correct Pred (edit predictions, save to annotation/)

### Short class names
Root, Aer, O.Endo, I.Endo, O.Exo, I.Exo

### Shortcuts
N=Draw, Enter/Space=Confirm/Edit, Esc=Cancel, S=Save, R=Split ring, Del=Delete, Ctrl+Z/Shift+Z=Undo/Redo, Left/Right=Navigate

### Brush mode
Drawing new (no selection): default=paint, Shift=erase. Editing existing (selected): default=erase, Shift=paint. Ctrl+Scroll=brush size. Scroll=zoom.

### Display adjustments (display only)
Brightness (-100 to +100) and Gamma (0.1 to 3.0) sliders. Raw image in `_raw_image`, adjustments on-the-fly.

### PyQt5 layout rules (IMPORTANT)
- `addWidget(widget, stretch=1)` NOT `addLayout(layout, stretch=1)` for claiming space
- Do NOT `setSizePolicy()` / `setMaximumHeight()` on QGroupBoxes — add stretch to content widget instead
- Do NOT wrap toolbar rows in container QWidget with maxHeight
- Row 1 flat (no QGroupBox); Row 2 uses QGroupBoxes with tight margins `(4, 2, 4, 2)`

---

## HPC Commands

```bash
# Sync to HPC
rsync -avz --progress --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='output/' --exclude='logs/' --exclude='.DS_Store' --exclude='annotation_copy/' --exclude='*.pt' ~/Documents/Siobhan_Lab/plants/ hpc2:~/plants/

# Environment
module load conda3/4.13.0 && conda activate plants
# First-time: conda create -n plants python=3.11 -y && conda activate plants && pip install -r ~/plants/requirements.txt

# Submit / interactive / monitor
sbatch slurm/run_grid.sh
srun --job-name=grid_train --partition=gpu-qi --gres=gpu:a100_80:1 --cpus-per-task=8 --mem=64G --time=72:00:00 --pty bash
squeue -u $USER
tail -f logs/grid_<JOB_ID>.out
srun --jobid=<JOB_ID> --overlap nvidia-smi

# Transfer results back
rsync -avz hpc2:~/plants/output/ ~/Documents/Siobhan_Lab/plants/output/
rsync -avz hpc2:~/plants/logs/ ~/Documents/Siobhan_Lab/plants/logs/
```

**HPC**: Cluster `hpc2`, partition `gpu-qi`, GPU `a100_80`. Cannot SSH to compute nodes — use `srun --overlap`.

### ⚠️ CRITICAL: Avoid CUDA init races on shared GPU nodes

Every SLURM array script running PyTorch on a multi-GPU node MUST include these two guards, or it can **permanently brick a GPU** (driver state corruption only fixable by admin `nvidia-smi --gpu-reset` or reboot):

```bash
#!/bin/bash
#SBATCH ...
set -e   # Propagate python crashes so SLURM marks the job FAILED, not silent COMPLETED

cd ~/plants
module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

# 1) Stagger CUDA init by task ID — prevents concurrent-init races
#    that corrupt the NVIDIA driver state on the shared node.
sleep $((SLURM_ARRAY_TASK_ID * 30))

# 2) CUDA smoke test with retry + self-requeue, in case we land on an
#    already-stuck GPU left by someone else.
for attempt in 1 2 3 4 5; do
    if python -c "import torch; torch.cuda.init(); torch.randn(32,32,device='cuda')@torch.randn(32,32,device='cuda')"; then
        break
    fi
    [ "$attempt" -eq 5 ] && { scontrol requeue "$SLURM_JOB_ID"; sleep 5; exit 1; }
    sleep 60
done
```

**Why it matters**: 2026-04-22 — 3-task array on diamond-0 fired `torch._C._cuda_init()` within seconds of each other, collided on NVIDIA driver global state, corrupted one A100 (UUID `c986b32c-...`). That GPU returns `CUDA unknown error` on every subsequent job. SLURM keeps assigning it (looks "free" to the scheduler). No user-level recovery. **This is the #1 operational priority when writing SLURM scripts.**
