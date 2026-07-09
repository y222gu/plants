# RADIX: Root Anatomy Deep Identification across Species

Semantic segmentation of plant root cross-section fluorescence microscopy. Trained on Millet, Rice, Sorghum, and Tomato roots imaged on Olympus IX83, Cytation C10, and Zeiss LSM 970 microscopes. Produces per-pixel labels for 7 anatomical classes (Background, Epidermis, Aerenchyma, Endodermis, Vascular, Exodermis, Cortex) and downstream measurements (aerenchyma ratio, per-region channel intensities).

Headline model: DINOv3-S/16 encoder with a Meta-exact DPT decoder.

---

## Data Format

Each sample is a folder containing three grayscale 16-bit TIFs, one per fluorescence channel:

```
{sample}/
├── {sample}_DAPI.tif    # cell walls
├── {sample}_FITC.tif    # lignin
└── {sample}_TRITC.tif   # suberin
```

The filename prefix must match the folder name, and all three channels must share the same prefix. Both flat (`data_dir/sample/*.tif`) and two-level (`data_dir/group/sample/*.tif`) layouts are accepted; in the two-level case sample IDs become `{group}/{sample}`.

Images are normalized on the fly (1st-99.5th percentile per channel) and resized to 1024×1024 for inference; original resolution is preserved for downstream intensity measurements.

Optional ground-truth annotations use YOLO polygon format (`class_id x1 y1 x2 y2 ...`, normalized 0-1) with 6 raw classes: `0` Whole Root, `1` Aerenchyma, `2` Outer Endodermis, `3` Inner Endodermis, `4` Outer Exodermis, `5` Inner Exodermis. Rings are derived at load time by subtracting inner from outer polygons.

---

## Running RADIX on New Data

The deployment entry point is `predict.py`. It runs inference with the trained DINOv3+DPT checkpoint and writes per-sample masks, visualizations, and a measurements CSV.

1. Open `predict.py` and set the two paths at the top:

   ```python
   DATA_DIR  = Path("/path/to/samples/")   # folder of per-sample subfolders
   MODEL_DIR = Path("/path/to/run/")       # run dir (contains checkpoints/best-*.ckpt) or a .ckpt file
   ```

2. Run:

   ```bash
   python predict.py
   ```

3. Outputs land in `{DATA_DIR}/predictions_dinov3_dpt/` (or a custom `OUTPUT_DIR`):

   ```
   predictions/{sample}.npy   # uint8 semantic argmax (0=bg, 1=epi, 2=aer, 3=endo, 4=vasc, 5=exo, 6=cortex)
   vis/{sample}.png           # input + Bio-7 overlay
   measurements.csv           # aerenchyma_ratio + per-region DAPI/FITC/TRITC means
   ```

Requires a CUDA GPU. Environment: `pip install -r requirements.txt` in a Python 3.11 conda env.

### Evaluating against ground truth

If you have annotations and want IoU/Dice + paired GT-vs-prediction downstream comparisons, use `run_eval_pipeline.py`:

```bash
python run_eval_pipeline.py \
    --model-key timm_semantic \
    --checkpoint path/to/best.ckpt \
    --run-dir path/to/run/
```

This runs eval on the test and Zeiss oneshot splits, saves predictions as YOLO `.txt` files, then computes downstream measurements from both GT and predictions with correlation plots.

---

## Polygon Editor

Interactive GUI for creating, reviewing, and correcting YOLO polygon annotations.

```bash
python polygon_editor.py --data-dir path/to/data/
python polygon_editor.py                             # no args: use Browse button
```

### Modes (Mode dropdown)

| Mode                | Panels                       | Required folders                       | Purpose                                             |
|---------------------|------------------------------|----------------------------------------|-----------------------------------------------------|
| Correct GT          | 3 (Original, GT, Prediction) | `image/`, `annotation/`, `prediction/` | Edit ground truth with predictions as reference     |
| Correct Predictions | 2 (Original, Editable)       | `image/`, `prediction/`                | Edit predictions, save into `annotation/`           |
| Create GT           | 2 (Original, Editable)       | `image/`                               | Draw annotations from scratch, save to `annotation/` |

### Class labels

Short names in the UI: Root, Aer, O.Endo, I.Endo, O.Exo, I.Exo.

### Controls

| Key                       | Action                                                              |
|---------------------------|---------------------------------------------------------------------|
| `N`                       | Start drawing new polygon (click to add nodes)                      |
| `B`                       | Enter brush mode (default paint, Shift=erase, Ctrl+Scroll=size)     |
| `E`                       | Edit selected polygon with brush (default erase, Shift=paint)       |
| `Enter` / `Space`         | Confirm drawing or edits                                            |
| `Escape`                  | Cancel drawing or edits                                             |
| `Delete` / `Backspace`    | Delete selected vertex (edit mode) or polygon                       |
| `R`                       | Split a ring polygon into outer/inner                               |
| `S`                       | Save annotations                                                    |
| `Ctrl+C`                  | Copy selected reference polygon to editable panel                   |
| `C`                       | Copy all reference polygons to editable panel                       |
| `1`-`6`                   | Set class for new polygon                                           |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo                                                         |
| `A` / `D` / `Left` / `Right` | Previous / Next sample                                           |
| `H`                       | Reset zoom, center all panels                                       |
| Mouse wheel               | Zoom                                                                |
| Middle/Right drag         | Pan                                                                 |

### Vertex editing

Drag vertices to move. Hover an edge to see a green "+" marker; click to insert a vertex. Select a vertex and press Delete to remove it.

### Display sliders

Brightness (-100 to +100) and Gamma (0.1 to 3.0) affect the on-screen image only; the underlying raw TIFs are never modified.

### Saving

Press `S` to save. Annotations are written to `{data-dir}/annotation/` in YOLO polygon format, one file per sample: `{Species}_{Microscope}_{Exp}_{Sample}.txt` for the structured layout, or `{sample}.txt` for generic layouts. The folder is created if it does not exist.
