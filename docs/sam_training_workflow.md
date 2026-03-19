# SAM Training Workflow

## Overview

SAM (Segment Anything Model) is trained as a single model that handles all classes for all species. Unlike Cellpose (which trains 5 separate per-class models), SAM learns a general "segment the prompted object" ability. The class label is never fed to the network — SAM only sees prompts (points + bounding box) that indicate what to segment.

## Architecture

- Image encoder (ViT-B): FROZEN — pretrained on SA-1B (11M images)
- Prompt encoder: FROZEN — encodes point coordinates + bounding box into tokens
- Mask decoder: TRAINABLE (~4.1M params) — predicts binary mask for the prompted instance
- Total: 93.7M params, only 4.1M trainable

## Dataset Construction

The dataset builds a flat index of every annotation instance across all samples:

For each sample:
  - Count all instances: whole root (1) + aerenchyma (N) + endodermis ring (1) + vascular (1) + exodermis (1, tomato only)
  - Each instance becomes a separate training item

Example instance counts per sample:
  - Rice sample with 20 aerenchyma: 23 items (1 root + 20 aer + 1 endo + 1 vasc)
  - Tomato sample: 4 items (1 root + 1 endo + 1 vasc + 1 exo, no aerenchyma)
  - Millet sample with 4 aerenchyma: 7 items (1 root + 4 aer + 1 endo + 1 vasc)

Total training items are dominated by aerenchyma:
  - Aerenchyma: ~23,342 instances
  - Whole Root: ~1,717 instances
  - Endodermis: ~1,717 instances
  - Vascular: ~1,717 instances
  - Exodermis: ~545 instances (tomato only)

## Training Loop (per batch item)

Step 1: Load image + all annotation masks for one sample
Step 2: Select one instance (instance_idx) → get its binary mask
Step 3: Apply augmentation to image + all masks together
         - Spatial: flips, rotations, affine, elastic deformation
         - Photometric: brightness/contrast, Gaussian noise/blur
         - Channel dropout (p=0.2): zero out 1 of 3 channels
         - Channel shuffle (p=0.2): randomly permute channel order
         - Re-randomized every epoch (on-the-fly encoding)
Step 4: Generate prompts from the augmented GT mask
         - 3 random foreground points (sampled from mask pixels)
         - Bounding box with 5% random jitter
         - Prompts are re-randomized every __getitem__ call
Step 5: Forward pass
         - Frozen image encoder: (3, 1024, 1024) → (256, 64, 64) embeddings
         - Frozen prompt encoder: points + box → prompt tokens
         - Trainable mask decoder: embeddings + tokens → predicted binary mask
Step 6: Loss = BCE + Dice between predicted mask and GT mask
Step 7: Backpropagate through mask decoder only

## Inference (Evaluation)

For each test sample:
  For each class:
    - Generate prompts from GT masks (oracle prompts)
    - 3 random foreground points + bounding box from GT
    - SAM predicts binary mask for each prompted instance
  Merge all predicted masks across classes
  Compare with GT for metrics (IoU, Dice, mAP)

Note: SAM requires prompts at inference. For evaluation, we use GT-derived prompts
(oracle setting). For deployment, prompts would come from another source (e.g.,
a detection model or user clicks).

## Key Properties

- Single model for all 5 classes and all 4 species
- Class-agnostic: the model never sees class labels; it learns "segment what I point to"
- Prompt-based: behavior is controlled by input prompts, not by the architecture
- The same model segments aerenchyma (small holes), endodermis (thin ring), whole root (large blob), etc. — the prompts tell it which object to focus on
- Training is heavily biased toward aerenchyma instances due to their abundance (~23K vs ~1.7K per other class)

## Comparison with Other Models

                    SAM              U-Net++          YOLO             Cellpose
Architecture:       Prompt-based     Pixel multilabel Detect+segment   Flow field
Num models:         1                1                1                5 (per-class)
Knows class:        No               Yes (channels)   Yes (det head)   Implicitly
Needs prompts:      Yes (pts+box)    No               No               No
Trainable params:   4.1M             24.4M            22.4M            ~13M per model
Frozen layers:      Encoder+prompt   None             None             None
Augmentation:       Full pipeline    Full pipeline    Ultralytics      Built-in only
