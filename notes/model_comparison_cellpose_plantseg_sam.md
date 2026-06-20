# Comparison: PlantSeg/PanSeg vs Cellpose-SAM vs SAM/micro-SAM

## Three Completely Independent Projects

### 1. PlantSeg / PanSeg (Kreshuk Lab, EMBL/Heidelberg)

**Paper**: Wolny et al., "Accurate and versatile 3D segmentation of plant tissues at cellular resolution", *eLife* 2020 (doi:10.7554/eLife.57613)

**What it is**: A two-stage pipeline for plant cell segmentation:
- **Stage 1**: A 3D or 2D U-Net predicts **boundary probability maps** (bright = cell wall, outputs a single-channel probability image)
- **Stage 2**: Graph partitioning algorithms (GASP, multicut, mutex watershed, or distance-transform watershed) convert boundaries into instance masks

**PlantSeg** = the original v1.x software. **PanSeg** = the rebranded v2.x with napari GUI integration. Same pipeline, same authors, just renamed.

**The bioimage.io models are just the U-Net weights from Stage 1:**

| Model ID | Name | Data | Dim |
|----------|------|------|-----|
| `laid-back-lobster` | Stem cells UNet | Arabidopsis apical stem, confocal | 2D |
| `pioneering-rhino` | Ovules UNet | Arabidopsis ovules, confocal | 2D |
| `thoughtful-turtle` | Lateral root UNet | Arabidopsis lateral root, lightsheet | 3D |
| `passionate-t-rex` | Ovules UNet 3D | Arabidopsis ovules, confocal | 3D |
| `wild-whale` | Epithelial affinity model | Epithelial tissue (not plant) | 2D |

These are **not different architectures** — they're all the same U-Net architecture trained on different Arabidopsis tissues. They only predict boundaries; you still need the PanSeg segmentation step (dt_watershed etc.) to get instance masks.

---

### 2. Cellpose / Cellpose-SAM (Stringer & Pachitariu, HHMI Janelia)

**Papers**: 
- Cellpose 1.0: Stringer et al., *Nature Methods* 2021
- Cellpose-SAM: Pachitariu et al., *bioRxiv* 2025

**What it is**: A single-stage model that directly outputs **instance masks** via flow fields:
- Each pixel predicts (flow_x, flow_y, cell_probability) — a vector pointing toward the cell's center
- Pixels are grouped into instances by following the flow vectors
- No separate segmentation step needed

**Cellpose-SAM specifically**: Takes SAM's ViT-L image encoder architecture, modifies the patch size (16→8) and replaces the mask decoder with Cellpose's flow readout head. Trained from scratch on ~2M cell annotations (not fine-tuned from SAM weights — the code comment says `default to not loading SAM`). The SAM architecture is reused but the weights are independently trained.

---

### 3. SAM / micro-SAM (Meta AI / Archit et al.)

**Papers**:
- SAM: Kirillov et al., *ICCV* 2023 (Meta AI)
- micro-SAM: Archit et al., *Nature Methods* 2025

**What it is**: Prompt-based segmentation — give it points/boxes, it outputs masks. micro-SAM fine-tunes SAM on microscopy data and adds a UNETR decoder for automatic (no-prompt) segmentation.

---

## How They Relate

```
                    ┌─────────────────────────────┐
                    │     SAM (Meta AI, 2023)      │
                    │  ViT encoder + mask decoder   │
                    │  Trained on 1B masks          │
                    └──────────┬──────────┬────────┘
                               │          │
              Architecture reused    Fine-tuned on
              (weights NOT shared)    microscopy
                               │          │
                    ┌──────────▼──┐  ┌────▼──────────┐
                    │ Cellpose-SAM│  │  micro-SAM     │
                    │ (HHMI 2025) │  │ (Archit 2025)  │
                    │ ViT-L + flow│  │ ViT-B + UNETR  │
                    │ 2M cells    │  │ prompt-based    │
                    └─────────────┘  └────────────────┘

    ┌──────────────────────────────────┐
    │  PlantSeg / PanSeg (EMBL 2020)   │
    │  Completely independent project   │
    │  U-Net (NOT SAM/ViT) + watershed  │
    │  Trained on Arabidopsis only      │
    └──────────────────────────────────┘
```

**None of these are fine-tuned on top of each other** (except micro-SAM which fine-tunes SAM). Cellpose-SAM borrows SAM's architecture but trains its own weights. PlantSeg/PanSeg is entirely separate — different architecture (U-Net vs ViT), different output (boundaries vs flows vs prompted masks), different training data.

---

## Key Differences at a Glance

| | PlantSeg/PanSeg | Cellpose-SAM | micro-SAM |
|---|---|---|---|
| **Architecture** | 2D/3D U-Net | ViT-L (SAM arch) | ViT-B (SAM weights) |
| **Output** | Boundary probability map | Flow fields → instances | Prompted masks |
| **Pipeline** | 2-stage (UNet + watershed) | 1-stage (end-to-end) | 1-stage (prompted) |
| **Training data** | Arabidopsis only (~100s images) | ~2M cell annotations, diverse | Microscopy fine-tune of SAM |
| **Designed for** | Plant tissue (confocal/lightsheet) | Any cells (generalist) | Any microscopy (generalist) |
| **Needs prompts?** | No | No | Yes (or auto via UNETR) |
| **Input** | Single-channel (cell walls) | 3-channel RGB | 3-channel RGB |
| **SAM relationship** | None | Reuses architecture only | Fine-tunes SAM weights |
