% Study Notes: ViT, SAM, DINO, and Self-Supervised Learning
% Yifei Gu
% 2026-04-17

# 1. How ViT Pretrained Models Are Trained (DINO Paradigm)

Both **DINOv2** and **DINOv3** are trained **self-supervised** — no labels at all. Same family of techniques, just scaled differently.

## Paradigm: Self-Distillation with Masked Image Modeling

Two copies of the network — a **student** and a **teacher**. Teacher = EMA (exponential moving average) of student weights.

For each image:

- Two augmented global views + several local crops are fed through both networks.
- **DINO loss** (image-level): student's `[CLS]` token on one view is trained to match teacher's `[CLS]` on another view via cross-entropy over a prototype codebook.
- **iBOT loss** (patch-level): random patches in the student's input are masked; the student must predict the teacher's patch-token outputs at those positions (analogous to MAE/BEiT, but in feature space, not pixel space).
- **KoLeo regularization**: spreads feature vectors apart to prevent representational collapse.
- **DINOv3 adds a Gram-anchoring loss** to stop dense features degrading during long training runs.

No reconstruction, no contrastive pairs, no classification — just teacher-student feature matching.

## Datasets: Curated, Not Labeled

| Model  | Dataset    | Size         | How it was built |
|--------|------------|--------------|------------------|
| DINOv2 | LVD-142M   | 142M images  | Scraped ~1.2B web images → dedup → retrieval-based filtering using seed images from ImageNet-22k, Google Landmarks, etc. |
| DINOv3 | LVD-1689M  | 1.69B images | Same pipeline, much bigger pool, stricter curation |

**Curation matters.** Random web scrapes give worse features than this retrieval-balanced mix. The labels from ImageNet, etc. are used **only to pick which images to keep** — never as a classification target.

## Why This Matters for Our Task

The ViT encoder has never seen a class label, but it has learned patch-level features rich enough that a tiny decoder (a linear head in SegDINO, DPT in our runs) can read cell-wall, endodermis, and vascular structure out of them. That's why "freeze encoder + light decoder" is viable for DINOv2/v3 but *not* for an ImageNet-supervised ViT, whose features are squeezed toward the 1000 ImageNet classes.

\newpage

# 2. ViT in SAM vs. ViT in DINO

They share the name "ViT" but differ in almost every dimension.

|                          | **SAM ViT-B** (ours: `vit_b_lm`)                                  | **DINOv2/v3 ViT-S** (ours)                          |
|--------------------------|-------------------------------------------------------------------|-----------------------------------------------------|
| Params                   | ~89M (encoder only)                                               | 22M (v2) / 27M (v3)                                 |
| In our runs              | Frozen                                                            | Fully trainable                                     |
| Pretraining paradigm     | MAE → supervised mask prediction on 1.1B masks → supervised microscopy fine-tune (`_lm`) | Self-supervised (DINO + iBOT losses), never sees labels |
| Pretraining data         | ImageNet-1k → SA-1B (11M natural imgs, 1.1B masks) → LiveCell + TissueNet etc. for `_lm` | LVD-142M / LVD-1689M (curated web)                  |
| Native resolution        | 1024×1024 (architected for it)                                    | 224 for DINOv2, 256 for DINOv3                      |
| Patch size               | 16 → 64×64 tokens at 1024                                         | 14 (v2) / 16 (v3)                                   |
| Attention                | **Windowed** (14×14 local) + 4 periodic global blocks             | **Global self-attention** at every layer            |
| Positional encoding      | Absolute + learned 2D relative bias in global blocks              | Learned (v2) / RoPE (v3)                            |
| What it learned          | Object / region boundaries                                        | Semantic + structural similarity                    |
| Output                   | Single dense feature map, 256-dim, 1/16 resolution                | Per-layer patch tokens + CLS                        |

## Practical Upshots

- **SAM's encoder is designed around high-resolution dense masks.** Windowed attention + 1024 native resolution = tractable compute on microscopy-scale images. DINOv2 at 1024 is expensive because every layer does full O(n²) attention over 5,329 tokens (1036²/14²).
- **SAM's features already encode "where are the boundaries".** That's why we freeze it — the prompt encoder + mask decoder just need to be told which object to extract.
- **DINO's features encode "what is this region structurally, semantically".** They don't know mask boundaries per se, but they cluster tissue types. With a light decoder (DPT, SegDINO MLP, ms_linear) you can read anatomical classes out of them.
- **Why freeze SAM but train DINO end-to-end?** SAM was already fine-tuned on microscopy, so its features transfer directly; fine-tuning the whole 89M risks destroying SA-1B's mask prior. DINO-S is smaller (22–27M) and untouched by microscopy, so unfreezing pays off.
- **Neither is strictly better; they're trained for different questions.** SAM asks *"what are the objects here?"*; DINO asks *"what is this patch like?"*. Our task (structure-level semantic segmentation of nested anatomical layers) is closer to the DINO question.

\newpage

# 3. SAM Architecture: Progression over ViT

## Vanilla ViT (Baseline)

```
image → patch embed → N transformer blocks → [CLS] + patch tokens → linear classifier
```

Trained with **image-level classification** (ImageNet) or **self-supervised feature matching** (DINO). Output is a single vector (or patch grid) meant for downstream tasks.

## SAM = ViT + Prompt Encoder + Mask Decoder + SA-1B Training

```
                       ┌───────────────────────────────┐
image (1024×1024) ───▶ │  1. IMAGE ENCODER (heavy)    │ ───▶ 64×64×256 feature map
                       │     modified ViT-B/L/H       │    (computed once per image)
                       └───────────────────────────────┘
                                       │
                                       ▼
user input  ───▶  ┌──────────────────┐ │  ┌──────────────────────────────┐
(point/box/       │  2. PROMPT ENC.  │ │  │  3. MASK DECODER (tiny,      │
 mask/text)       │  (few k params)  ├─┼─▶│     ~4M params, 2 layers)    │ ──▶ masks + IoU scores
                  └──────────────────┘ │  │     two-way transformer      │
                                       └─▶│     image feat ⇄ prompt tok  │
                                          └──────────────────────────────┘
```

## Step-by-Step Progression

**Step 1 — Re-engineer the ViT for dense 1024×1024 input.**

- Vanilla ViT at 1024 with global attention is prohibitive: ~4,096 tokens, O(n²) = 16M attention entries per layer.
- SAM replaces most blocks with **windowed attention** (14×14 local windows), then puts **4 global-attention blocks** at evenly spaced depths.
- Adds **relative positional bias** in global blocks (learned 2D).
- Output is now a **feature map** (64×64×256), not a `[CLS]` token. That grid *is* the "image embedding".

**Step 2 — Pretrain the encoder with MAE on ImageNet.**

- Same masked-autoencoder objective as Kaiming He 2022. Purely for encoder initialization — no segmentation yet.

**Step 3 — Add a prompt encoder (new module).**

- **Points & boxes** → sinusoidal positional encoding + learned "foreground / background / top-left / bottom-right" embeddings → sparse tokens.
- **Masks** → downsampled through a small conv stack → added element-wise to the image feature map.
- **Text** → CLIP text embedding (paper-only, not released).
- Size: a few thousand parameters — trivial.

**Step 4 — Add a mask decoder (the key new invention).**

- Takes: image feature map + prompt tokens + a set of **learned "output" tokens** (1 IoU token + 3 mask tokens).
- Core is a **two-way transformer**:
  - Prompt/output tokens attend to image features, then image features attend back to tokens.
  - Two such rounds — total ~4M params.
- Each mask token → upsampled via transposed convs + an MLP → a full-resolution mask.
- The IoU token → MLP → predicted quality of each mask.
- **Multiple masks per prompt (usually 3) to resolve ambiguity** — "did you mean the whole dog, its head, or its ear?" At training time only the mask best matching GT is used for the mask loss; the others stay alive via the IoU head.

**Step 5 — Train end-to-end on SA-1B with mask prediction.**

- 11M images, **1.1B masks** — the largest segmentation dataset ever released.
- For each training step: sample a GT mask, sample a random prompt derived from it (a point inside, a bounding box, a partial mask), and predict the mask.
- Loss: **focal + dice** on masks + **MSE** on predicted IoU.
- Iterative "predict → sample error region → refine" lets SAM learn to refine predictions.

**Step 6 — At inference, bypass prompts if you want.**

- "Automatic mask generation": tile a dense grid of point prompts over the image, run the decoder at every point, NMS the resulting masks. That's how SAM does "segment everything" with no user input.
- The encoder runs once; the decoder runs thousands of times — but it's tiny, so this is fast.

## Key Additions over ViT

| Layer                 | **ViT**                          | **SAM**                                  |
|-----------------------|----------------------------------|------------------------------------------|
| Input resolution      | 224                              | 1024                                     |
| Attention             | Global everywhere                | Windowed + 4 global blocks               |
| Output                | `[CLS]` + tokens                 | 64×64×256 feature map                    |
| New: prompt encoder   | —                                | Points, boxes, masks, text → tokens      |
| New: mask decoder     | —                                | Two-way transformer, 3 masks + IoU       |
| Training objective    | Classification / self-supervised | Supervised mask prediction               |
| Training data         | ImageNet-1k / 22k / LVD          | SA-1B: 11M images, 1.1B masks            |

## What micro-SAM Adds on Top of SAM

- Further fine-tunes the encoder on microscopy datasets (LiveCell, TissueNet, PlantSeg, etc.) — the `_lm` suffix = "light microscopy generalist".
- Adds a **UNETR decoder** head next to the mask decoder. Takes the same image feature map and predicts center-distance + boundary-distance + foreground maps directly — no prompts needed. Lets micro-SAM do automatic instance segmentation from just an image.
- In our project: **encoder frozen**, only prompt encoder + mask decoder + UNETR decoder are trained. Per-class models (one per annotation class 0–5) because the prompt/decoder stack is cheap to retrain but tied to a specific object definition.

So the progression is: **ViT** = general visual backbone → **SAM** = ViT re-engineered for dense high-res + prompt + mask modules, trained on 1.1B masks → **micro-SAM** = SAM further specialized to microscopy + UNETR for promptless instance segmentation.

\newpage

# 4. Timeline and Connections

All of these except ViT itself come from **Meta AI** (formerly FAIR) — that's why they interoperate so cleanly.

## Timeline

```
2017 ─ Transformer (Vaswani et al., Google)
       "Attention is all you need" — NLP only at this point
       
2020 ─ ViT (Dosovitskiy et al., Google Brain) ─ Oct 2020
       "An Image is Worth 16×16 Words"
       Images → patches → transformer → classification. Supervised on JFT-300M.
       
2021 ─ DINO v1 (Caron et al., Meta) ─ Apr 2021
       "Emerging Properties in Self-Supervised Vision Transformers"
       First to show self-distillation on ViT produces surprisingly rich features
       (attention maps literally segment objects with no labels).
       
2021 ─ MAE (He et al., Meta) ─ Nov 2021
       Masked Autoencoder — mask 75% of patches, reconstruct pixels.
       Becomes the standard ViT pretraining recipe. SAM later uses this.
       
2023 ─ DINOv2 (Oquab et al., Meta) ─ Apr 2023
       Scaled DINO + added iBOT loss + KoLeo. Curated LVD-142M dataset.
       Off-the-shelf features that rival/beat supervised.
       
2023 ─ SAM (Kirillov et al., Meta) ─ Apr 2023   ← same month as DINOv2
       "Segment Anything" — MAE-pretrained ViT + prompt encoder + mask decoder.
       Released SA-1B: 11M images, 1.1B masks. Supervised, task-specific.
       
2023 ─ micro-SAM (Archit et al., preprint Aug 2023 → Nature Methods 2025)
       Fine-tuned SAM's encoder on ~20 microscopy datasets ("vit_b_lm").
       Added UNETR decoder for automatic instance seg without prompts.
       
2024 ─ SAM 2 (Ravi et al., Meta) ─ Jul 2024
       Extended SAM to video with a memory bank. Hiera backbone instead of ViT.
       
2024 ─ DINOv3 (Siméoni et al., Meta)
       Scaled to LVD-1689M (12× v2 data), bigger models up to 7B params,
       patch 14 → 16, added Gram-anchoring loss to stabilize dense features.
```

## Connection Diagram

```
                     ┌─────────┐
                     │   ViT    │         ← shared foundation
                     │  (2020)  │
                     └─────────┘
                     /           \
                    /             \
   Self-supervised /               \   Task-specific supervised
   feature line   /                 \  segmentation line
                 ▼                   ▼
         ┌──────────┐         ┌──────────┐
         │  DINO v1 │         │   MAE    │
         │  (2021)  │         │  (2021)  │
         └──────────┘         └──────────┘
              │                     │
              ▼                     │  (MAE provides encoder init)
         ┌──────────┐                ▼
         │  DINOv2  │         ┌──────────────┐
         │  (2023)  │         │    SAM       │  trained on 1.1B masks
         └──────────┘         │   (2023)     │
              │                └──────────────┘
              ▼                       │
         ┌──────────┐                 ▼
         │  DINOv3  │         ┌──────────────┐      ┌──────────┐
         │  (2024)  │         │  micro-SAM   │      │  SAM 2   │
         └──────────┘         │  (2023/25)   │      │  (2024)  │
                              └──────────────┘      └──────────┘
                              (microscopy FT          (video +
                              + UNETR head)           memory)
```

## Key Connections

- **ViT → everything.** The patch-embedding + transformer-blocks architecture is the common ancestor. DINO and SAM both wrap ViT in different training regimes.
- **MAE → SAM.** SAM doesn't train its image encoder from scratch; it starts from an MAE-pretrained ViT, then continues training on SA-1B with mask prediction. Without MAE you'd need vastly more supervised data.
- **DINO v1 → DINOv2 → DINOv3.** Pure lineage. Same self-distillation idea, progressively more data, bigger models, and additional losses (iBOT in v2, Gram anchoring in v3).
- **SAM → micro-SAM.** micro-SAM kept SAM's three-module architecture but replaced the pretraining data distribution (natural → microscopy) and added a UNETR decoder so you can run segmentation without having to click a prompt.
- **SAM and DINOv2 are siblings**, released the same month (Apr 2023) by the same lab, but built to answer different questions:
  - **SAM** = "segment this object I'm pointing at" (needs labels)
  - **DINOv2** = "give me features that cluster semantically" (needs no labels)
- **DINOv3 and SAM 2 are the current frontier** (both 2024). SAM 2 went toward video / memory; DINOv3 went toward scale and denser features. They optimize different axes.

\newpage

# 5. Self-Supervised vs. Supervised: MAE and SAM

## Definitions

- **Supervised:** training signal comes from human-provided labels.
- **Self-supervised:** training signal comes from the data itself (no labels needed).
- **Semi-supervised:** mix of labeled and unlabeled.

## MAE — Self-Supervised

No labels involved.

- Take an image, randomly mask out ~75% of its patches.
- Feed only the visible 25% through the ViT encoder.
- Decoder tries to **reconstruct the raw pixels** of the masked patches.
- Loss = MSE between predicted and true pixels.

The "supervision" is the image itself — specifically, pixels the model hasn't seen yet. No human annotation. This is **self-supervised learning** (same category as DINO, BEiT, MoCo, SimCLR). "Unsupervised" in the loose sense, but the field has settled on "self-supervised" as the precise term.

## SAM — Supervised (on masks)

SAM's training has two phases, often confused:

**Phase 1: encoder initialization** — MAE on ImageNet. **Self-supervised.** Off-the-shelf recipe to get a good ViT.

**Phase 2: SAM training on SA-1B** — **Supervised.** The model is given image + prompt and must predict the **ground-truth mask**. Loss = focal + dice on masks, MSE on the IoU prediction. Every training example has a mask as its target. Textbook supervised learning.

## The Asterisk on SA-1B

The **dataset itself** was built through a bootstrapping loop (Meta calls it the "data engine"):

1. **Assisted-manual stage** — humans annotate ~120k masks; train SAM on those.
2. **Semi-automatic** — SAM proposes masks on a pool of images; humans correct / add missed objects; retrain.
3. **Fully automatic** — final SAM runs on 11M images at a dense grid of point prompts; resulting masks become training targets for the released checkpoint — **no human in this loop**.

So the ~1.1B masks in SA-1B's final release were mostly generated by an earlier SAM and kept if they passed a quality / IoU filter. Some call this **semi-supervised** or **self-training** (iterative pseudo-labeling). But once those masks exist in SA-1B, training against them is still supervised learning — the target signal comes from labels (even if those labels were model-generated and filtered).

## Summary Table

| Model                       | Training paradigm               | Source of training signal                                  |
|-----------------------------|---------------------------------|------------------------------------------------------------|
| ViT                         | Supervised                      | Human-labeled classifications (ImageNet-21k, JFT)          |
| DINO / DINOv2 / DINOv3      | Self-supervised                 | Student–teacher feature matching, no labels                |
| MAE                         | Self-supervised                 | Pixel reconstruction from masked input                     |
| SAM (encoder init)          | Self-supervised                 | MAE pixel reconstruction                                   |
| SAM (full model)            | Supervised                      | Ground-truth masks (human + model-generated)               |
| micro-SAM                   | Supervised fine-tune of SAM     | Microscopy masks from published datasets                   |

\newpage

# 6. DINOv2 + DPT — Detailed Model Walkthrough

The best model in our paper is a **DINOv2-Small (ViT-S/14) encoder + custom DPT decoder**, fine-tuned on the root cross-section dataset. This section walks through the full forward pass dimension-by-dimension, then zooms into a single Transformer block.

## 6.1 Full pipeline (input → logits)

| Stage | Operation | Output shape |
|-------|-----------|--------------|
| Input | Raw normalised image, channel order `[TRITC, FITC, DAPI]` | `B × 3 × 1024 × 1024` |
| Patch-align pad | `F.pad(..., mode='reflect')`, +12 on H and W so size divides 14 | `B × 3 × 1036 × 1036` |
| Patch embed | `Conv2d(3, 384, kernel=14, stride=14)` → flatten → transpose | `B × 5476 × 384` (74×74 patch tokens) |
| Prepend `[CLS]` + add pos-embed | 5476 patches + 1 `[CLS]` | `B × 5477 × 384` |
| 12 × Transformer block | LN → MHSA → residual → LN → MLP → residual (see §6.2) | `B × 5477 × 384` |
| Tap 4 layers {2, 5, 8, 11} | `forward_intermediates(..., return_prefix_tokens=False)` — drop `[CLS]`, apply final `LayerNorm` | 4 × `B × 5476 × 384` |
| Tokens → BCHW | Reshape `(B, 5476, 384) → (B, 384, 74, 74)` | 4 × `B × 384 × 74 × 74` |
| DPT 1×1 project | `Conv2d(384→256) → BN → GELU` (×4, one per tap) | 4 × `B × 256 × 74 × 74` |
| DPT bottom-up fusion | For i ∈ {2,1,0}: `x = fusions[i](x + projected[i])` | `B × 256 × 74 × 74` |
| DPT upsample head | Conv3×3 → BN → GELU → Upsample×2 → Conv → Upsample×2 → Conv | `B × 64 × 296 × 296` |
| 1×1 classification head | `Conv2d(64 → 7)` | `B × 7 × 296 × 296` |
| Bilinear upsample | to padded input size | `B × 7 × 1036 × 1036` |
| Crop pad | `logits[:, :, :1024, :1024]` | `B × 7 × 1024 × 1024` |

### Key architectural facts

- **Encoder**: `vit_small_patch14_dinov2.lvd142m` via `timm` — ViT-S/14, embed dim 384, depth 12, 6 heads, MLP ratio 4 (hidden 1536), pretrained on LVD-142M.
- **Decoder**: custom DPT (Ranftl et al. 2021) — simplified because smp's DPT had bugs at non-native resolutions.
- **Feature taps**: 4 evenly-spaced layers `[n/4 − 1, n/2 − 1, 3n/4 − 1, n − 1] = [2, 5, 8, 11]`. All taps live at the same spatial resolution (74×74) because plain ViT has no down-sampling — so the DPT fusion just becomes 3 residual refinement passes over the same 74×74 grid.
- **Total params**: 28.5 M (23.6 M encoder + ~4.9 M decoder + head).
- **Loss**: `Dice + Focal + weighted-CE + Lovász` with class weights `[bg=0.5, epi=1, aer=10, endo=5, vasc=1, exo=5, cortex=1]`.
- **Optimiser**: AdamW with differential LR — encoder 1 × 10⁻⁵, decoder + head 1 × 10⁻⁴; CosineAnnealingLR; patience 15 early stop.

## 6.2 Inside one Transformer block

Each of the 12 blocks has the structure:

```
def forward(x):                              # x: (B, 5477, 384)
    y = x + LS1(MHSA(LayerNorm1(x)))
    z = y + LS2(MLP(LayerNorm2(y)))
    return z
```

Two sub-layers (MHSA and MLP), each with its own LayerNorm, LayerScale, and residual.

### Sub-layer 1 — Multi-Head Self-Attention

| Step | Operation | Shape |
|------|-----------|-------|
| LayerNorm 1 | per-token normalise 384 dims, learned γ, β | `B × 5477 × 384` |
| QKV projection | `Linear(384 → 3·384)`, split into Q, K, V | each `B × 5477 × 384` |
| Reshape for heads | 384 = 6 × 64; transpose to `(B, heads, N, d_h)` | each `B × 6 × 5477 × 64` |
| Attention scores | `Q · Kᵀ / √64` | `B × 6 × 5477 × 5477` |
| Softmax (last dim) | 6 independent attention distributions | `B × 6 × 5477 × 5477` |
| Weighted V | `α · V` | `B × 6 × 5477 × 64` |
| Merge heads | transpose + reshape | `B × 5477 × 384` |
| Output projection | `Linear(384 → 384)` | `B × 5477 × 384` |
| LayerScale 1 | multiply by learned `γ₁` (shape 384), init 1 × 10⁻⁵ | `B × 5477 × 384` |
| Residual add | `y = x + LS1_out` | `B × 5477 × 384` |

### Sub-layer 2 — MLP (this is where GELU lives)

| Step | Operation | Shape |
|------|-----------|-------|
| LayerNorm 2 | | `B × 5477 × 384` |
| fc1 — expand | `Linear(384 → 1536)` | `B × 5477 × 1536` |
| **GELU** | **element-wise** `x · Φ(x)` | `B × 5477 × 1536` |
| fc2 — contract | `Linear(1536 → 384)` | `B × 5477 × 384` |
| LayerScale 2 | multiply by learned `γ₂` | `B × 5477 × 384` |
| Residual add | `z = y + LS2_out` | `B × 5477 × 384` |

GELU is the **only** non-linearity in the block (apart from softmax inside attention). It's element-wise — each of the `B × 5477 × 1536` scalars is fed through `GELU(x) = x · Φ(x)` independently, where Φ is the standard-normal CDF.

## 6.3 What is GELU?

**Gaussian Error Linear Unit** (Hendrycks & Gimpel, 2016) — the default activation in Transformers.

$$\text{GELU}(x) = x \cdot \Phi(x)$$

Compared to ReLU:

- ReLU is a **hard gate** (0 if negative, else x).
- GELU is a **soft gate** — multiplies x by "the probability of keeping x" under a standard-normal prior.
- GELU is **smooth everywhere** and slightly **non-monotonic** near 0 (dips below zero for small negative x before going to 0 as x → −∞).

Fast approximation used in practice:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \big(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715 x^3))\big)$$

## 6.4 Why multi-head attention (the "reshape" is not cosmetic)

A common confusion: if we just did `Q · Kᵀ` on the full 384-d Q and K, wouldn't that already encode multi-head behaviour via the embedding dimension?

**No — because of the softmax.**

### Single-head vs. multi-head

- **Single-head (dim 384)**: one dot-product per query–key pair → **one** `N × N` attention matrix → **one softmax distribution** per query. All 384 dims collapse into a single "which tokens matter most" vote.
- **Multi-head (6 × 64)**: reshape Q, K, V to `(B, 6, N, 64)`, then `Q · Kᵀ` gives **six** independent `N × N` attention matrices. Head *h*'s Q only dots with head *h*'s K — **heads never cross**. Six independent softmaxes in six independent 64-d subspaces.

### Why independent softmaxes matter

Imagine a query token needs two patterns simultaneously:

- **Pattern A**: attend to my 4 spatial neighbours (local texture)
- **Pattern B**: attend to the central vascular tissue 50 patches away (global context)

With single-head, the single softmax has to compromise — it produces one blended distribution (maybe 50/50), so the output is a weighted average. You can never have both patterns at full confidence simultaneously.

With 6 heads, one head's softmax can peak at 99% on local neighbours while another's peaks at 99% on vascular tissue. Both patterns coexist in the concatenated `(B, N, 384)` output, then the output projection `Linear(384 → 384)` linearly combines them.

### Low-rank factorisation

- Single-head with dim 384 = one rank-384 attention pattern.
- 6 heads with dim 64 = six rank-64 attention patterns.

Multi-head is a **low-rank factorisation trick**: sacrifice per-pattern expressiveness (64-d subspace instead of 384-d) to get **multiple simultaneous patterns for the same parameter cost**. The `Linear(384 → 1152)` that produces QKV has the same parameter count whether interpreted as one head of 384 or six heads of 64 — the reshape is how we tell the softmax to apply in block-structured chunks rather than as one giant pool.

Empirically, multiple diverse attention patterns beat one rich one. That's why every Transformer from GPT-1 to DINOv2 uses multi-head, and ablating to single-head consistently hurts performance.

## 6.5 How do heads decide what to specialise in?

**Nobody tells them.** There is no supervision or loss term encouraging heads to be different. Specialisation emerges from gradient descent minimising the segmentation loss.

### How it emerges

1. **At initialisation, heads are interchangeable** — random Xavier init gives each head's Q/K/V projections slightly different weights, but there is no architectural bias saying "head 1 does local".
2. **Slight initial asymmetries get amplified**. If head 1's random weights happen to be marginally better at local texture recognition, its gradient points along that direction, and it gets even better at local texture.
3. **Redundancy is wasteful**. Two identical heads contribute no more than one, so gradient descent naturally pushes heads toward diverse specialisations — diverse heads reduce loss more per step than redundant ones.
4. **Specialisation is stochastic**. Same model + different random seed → different head assignments. Seed A might end up with head 3 = local; seed B might end up with head 7 = local. The *set* of patterns learned is usually similar, but the indices are not fixed.

### How we know what a head learned — post-hoc

Interpretability methods are run **after** training:

| Method | What it reveals |
|--------|-----------------|
| **Attention visualisation** — plot attention weights for a query as a heatmap over the image | "Head 4 consistently attends to the 4 nearest patches" → local head |
| **Head ablation** — zero out one head's output and measure task degradation | "Removing head 7 hurts texture classification" → head 7 does texture |
| **Probing classifiers** — fit a linear model on one head's outputs to predict hand-crafted features | "Head 12's activations linearly encode distance to image centre" → positional head |

Labels like "local head" or "global head" are **descriptive names humans assign after inspecting a trained model**, not design decisions.

### General principle

This is true across neural nets — CNN filters are never told "be an edge detector"; LLM neurons are never told "encode sentiment". Feature specialisation emerges automatically from gradient descent on the task loss when architectural capacity is sufficient. Hand-designing what each head should do would just impose a prior that's almost certainly worse than what data-driven specialisation finds.

---

# 7. The DPT-Meta Decoder — What Happens After the ViT Taps

After extracting intermediate features from blocks 3, 6, 9, 12 of the ViT, the DPT decoder performs five learned transformations to produce a dense segmentation map. This section traces every step with exact dimensions for **DINOv3-S/16 at 1024×1024 input** (the model we're currently using). All batch dimensions omitted.

## 7.1 Starting state — what each tap looks like

Each transformer block preserves its input sequence shape. The patch embedding converts the image to a token grid and all 12 blocks operate on that same grid:

```
input            (3, 1024, 1024)
patch embed      (384, 64, 64)        # 64×64 = 4096 patches
add prefix       (4101, 384)          # 1 CLS + 4 registers + 4096 patches
block 1..12      (4101, 384)          # shape preserved at every depth
```

At each tap (blocks 3, 6, 9, 12):

1. Drop the 5 prefix tokens → `(4096, 384)`
2. Reshape to spatial → `(384, 64, 64)`

**All four taps enter the decoder as identical-shape tensors `(384, 64, 64)`.** They differ only in semantic depth — tap 3 encodes local textures and edges (closer to raw patch embeddings), tap 12 encodes semantic categories ("this is cortex", "this is vascular").

(For DINOv2-S/14 the grid is 74×74 after padding 1024→1036; the rest is identical.)

## 7.2 Step 1 — Reassemble channel projection (1×1 Conv, learned)

Per-tap 1×1 convolution projects each tap to a different channel count (Meta's pyramid):

| Tap | In | 1×1 Conv | Out |
|---|---|---|---|
| 0 (block 3)  | (384, 64, 64) | 384 → 96  | (96, 64, 64) |
| 1 (block 6)  | (384, 64, 64) | 384 → 192 | (192, 64, 64) |
| 2 (block 9)  | (384, 64, 64) | 384 → 384 | (384, 64, 64) |
| 3 (block 12) | (384, 64, 64) | 384 → 768 | (768, 64, 64) |

**What it does.** A 1×1 conv at each position applies a learned `W ∈ R^{ppc[i] × 384}` matrix — per-position linear remix across channels.

**Why the pyramid.** Shallow taps encode local texture which doesn't need high dimensionality; deep taps carry semantic categories that benefit from more channels. Meta's `[96, 192, 384, 768]` allocates compute where it matters instead of uniformly.

## 7.3 Step 2 — Reassemble spatial resample (learned)

DPT artificially recreates a CNN pyramid from the ViT's uniform grid using learned resampling:

| Tap | In | Resample op | Out |
|---|---|---|---|
| 0 (shallowest) | (96, 64, 64)  | ConvTranspose 4× | **(96, 256, 256)** |
| 1              | (192, 64, 64) | ConvTranspose 2× | **(192, 128, 128)** |
| 2              | (384, 64, 64) | Identity         | **(384, 64, 64)** |
| 3 (deepest)    | (768, 64, 64) | Conv 3×3 stride-2| **(768, 32, 32)** |

**Why learned instead of bilinear.** `ConvTranspose(k=s=4)` learns how to *distribute* a single patch embedding across its 16×16 pixel footprint — the filter can sharpen edges or recover specific spatial patterns that bilinear upsampling cannot. This is how DPT recovers fine spatial detail from a 1/16-resolution ViT output.

**Why downsample the deepest tap.** Creates a compact 32×32 semantic bottleneck — DPT starts fusion from the most compressed representation (like U-Net's bottleneck) and progressively upsamples.

## 7.4 Step 3 — Post-process: unify to 256 channels (learned)

```python
self.post_process[i] = Sequential(
    Conv2d(ppc[i], 256, kernel_size=3, padding=1, bias=False),
    BatchNorm2d(256),
)
```

| Tap | In | Out |
|---|---|---|
| 0 | (96, 256, 256)  | **(256, 256, 256)** |
| 1 | (192, 128, 128) | **(256, 128, 128)** |
| 2 | (384, 64, 64)   | **(256, 64, 64)** |
| 3 | (768, 32, 32)   | **(256, 32, 32)** |

**Why**: fusion is element-wise addition — requires matching channels. 256 is the classic DPT width (Ranftl 2021). The 3×3 receptive field smooths out any artifacts from the learned upsample before cross-level mixing. BN standardizes per-channel scale so features from different depths (which may differ by orders of magnitude) can be safely added.

## 7.5 Step 4 — Progressive fusion (the heart of DPT)

This is where the decoder actually combines multi-level features. **Crucial: fusion is element-wise ADD, not concat. Every conv inside is learned; bilinear upsample is the only non-learned op.**

### Structure of one fusion block

```python
class FeatureFusionBlockMeta:
    res_skip = ResidualConvUnit(256)   # 2 × Conv3×3 + BN + residual add
    res_out  = ResidualConvUnit(256)   # 2 × Conv3×3 + BN + residual add
```

Each **ResidualConvUnit (RCU)**:

```
input ─┬──────────────────────────────────────────────────┐
       │                                                  │
       └→ ReLU → Conv 3×3 → BN → ReLU → Conv 3×3 → BN ──⊕→ output
                 (learned)         (learned)         (element-wise add)
```

Parameters per RCU at 256 channels: ~1.18M (two 3×3 convs at 590k each plus BN scale/bias).
Each fusion block has 2 RCUs → ~2.36M learned params. Four fusion blocks → **~9.4M learned params total**, ~22% of the whole model.

### One fusion step, annotated

```python
def forward(skip, prev):
    # (A) Match prev's resolution to skip's (non-learned, fixed bilinear)
    prev = F.interpolate(prev, size=skip.shape[2:], mode='bilinear')

    # (B) Refine skip through learned residual convs
    refined_skip = self.res_skip(skip)      # ~1.18M learned params used

    # (C) Element-wise ADD — NOT concat. No learned weights.
    x = refined_skip + prev

    # (D) Refine the mixed tensor through another learned RCU
    x = self.res_out(x)                      # ~1.18M learned params used

    # (E) Upsample 2× for the next fusion level (non-learned bilinear)
    x = F.interpolate(x, scale_factor=2, mode='bilinear')

    return x
```

### Full fusion chain for our model

| Fusion | Skip | Prev (after bilinear match) | After res_out | After 2× upsample |
|---|---|---|---|---|
| 3 (deepest, no skip) | — | (256, 32, 32) | (256, 32, 32) | **(256, 64, 64)** |
| 2 | (256, 64, 64) | (256, 64, 64) | (256, 64, 64) | **(256, 128, 128)** |
| 1 | (256, 128, 128) | (256, 128, 128) | (256, 128, 128) | **(256, 256, 256)** |
| 0 (shallowest) | (256, 256, 256) | (256, 256, 256) | (256, 256, 256) | **(256, 512, 512)** |

### Why add instead of concat?

Architecturally both work. DPT picked add for two reasons:

1. **Lower compute.** Concat doubles channels, forcing a subsequent Conv 1×1 (512→256) to get back to 256. Add skips that.
2. **Residual-learning pressure.** Add forces skip and prev tensors to live in the same feature space. `res_skip` learns to *translate* the raw shallow skip into whatever coordinate system the deeper `prev` represents, and their sum becomes a refinement rather than a concatenation of two independent feature views. The decoder learns "iteratively refine at higher resolution" (U-Net dynamic) instead of "concat and re-mix" (which would need more capacity to merge).

The asymmetry — `res_skip` runs only on the skip, not on `prev` — is the key. `prev` is already a fused, refined tensor from the previous fusion block; `skip` is a raw projected feature that needs translation into `prev`'s coordinate system before addition. The learned 3×3 convs in `res_skip` *are* that skip→prev translator.

## 7.6 Step 5 — Head

```python
out_conv = Sequential(
    Conv2d(256, 128, 3, bias=False), BN(128), ReLU,   # learned
    Conv2d(128, 32,  3, bias=False), BN(32),  ReLU,   # learned
)
head = Conv2d(32, 7, kernel_size=1)   # learned 1×1 classifier
```

| Stage | Input | Op | Output |
|---|---|---|---|
| out_conv layer 1 | (256, 512, 512) | 3×3 → BN → ReLU | (128, 512, 512) |
| out_conv layer 2 | (128, 512, 512) | 3×3 → BN → ReLU | (32, 512, 512) |
| head             | (32, 512, 512)  | 1×1 (32 → 7)    | (7, 512, 512) |
| bilinear upsample| (7, 512, 512)   | → (1024, 1024)  | **(7, 1024, 1024)** |

**Why squeeze 256 → 32 before full-res**: 4× less activation memory than upsampling a 256-channel tensor. Information-wise free since we only need 7 output classes. Bilinear upsample on 7 channels is 32× cheaper than on 32 channels and gives bit-identical logits (bilinear and 1×1 conv commute).

## 7.7 What's learned vs what's not — summary

| Operation | Learned? | Why |
|---|---|---|
| ReLU | No | Fixed `max(0, x)` |
| 1×1 Conv (Reassemble proj) | **Yes** | Per-position channel remix |
| ConvTranspose / Conv stride-2 (Reassemble resize) | **Yes** | Learned up/downsample with spatial structure |
| 3×3 Conv (Post-process + RCU) | **Yes** | Local spatial filtering |
| BatchNorm | **Yes** | Per-channel scale γ + shift β |
| Residual add (inside RCU) | No | Element-wise tensor + |
| Fusion add (`refined_skip + prev`) | No | Element-wise tensor + |
| Bilinear upsample | No | Fixed interpolation formula |

Total learned params in the decoder: ~25M out of the 40.9M model (the other 15M is the frozen-during-loading-but-fine-tuned DINOv3 encoder).

## 7.8 Big-picture intuitions

1. **Feature specialization across depth.** Shallow taps encode local texture; deep taps encode semantic category. Using all four as skip connections lets the decoder pick the right abstraction level per pixel.

2. **Pyramid via learned resampling.** ViT gives you one resolution; the Reassemble step fakes a CNN pyramid so fusion has genuine multi-scale inputs.

3. **U-Net-style iterative refinement.** Progressively combine semantic + spatial information while upsampling. Residual convs let the decoder refine without blocking gradient flow — essential for stable fine-tuning on 1293 training images.

4. **Compute allocated by pyramid channels.** [96, 192, 384, 768] gives capacity to levels that carry semantic load, keeps shallow-but-high-res levels cheap.

5. **Add beats concat here.** Forces skip features into prev's space (learning pressure) and saves compute. The residual-convs-on-skip-only asymmetry is the mechanism.

---

*Last updated: 2026-04-23. Add new topics below as you learn them.*
