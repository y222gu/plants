"""SAM ViT-B encoder + U-Net++ decoder for 7-class semantic segmentation.

Uses SAM's ViT-B encoder (pretrained on light microscopy) with smp's
UnetPlusPlus decoder (dense nested skip connections) for a fair comparison
against the ResNet34 + UnetPlusPlus baseline.

The key challenge: ViT-B outputs all features at 1/16 spatial resolution,
but smp's UnetPlusPlus decoder expects hierarchical multi-scale features.
We solve this with an FPN-style adapter that projects ViT intermediate
block outputs into a multi-scale feature pyramid via learned upsampling.

Feature extraction points (same as UNETR): blocks 2, 5, 8, 11
  -> FPN adapter produces 5 feature maps at 1/2, 1/4, 1/8, 1/16, 1/32

Three encoder modes (same as SAM UNETR):
    1. Frozen (default): encoder fixed, only FPN adapter + decoder train
    2. LoRA adapter: encoder frozen + LoRA in attention layers
    3. Full fine-tune: all parameters trainable

Usage:
    # Frozen encoder
    python train/train_sam_unetpp.py

    # LoRA adapter
    python train/train_sam_unetpp.py --adapter lora --lora-rank 4

    # Compare with UNETR decoder
    python train/train_sam_semantic.py --adapter lora --lora-rank 4
"""

import sys
import types
import importlib.abc
import importlib.machinery
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock nifty/vigra/affogato if not available (only needed for elf post-processing)
class _NiftyVigraMockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _PREFIXES = ("nifty", "vigra", "affogato")
    def find_spec(self, fullname, path, target=None):
        for p in self._PREFIXES:
            if fullname == p or fullname.startswith(p + "."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        return mod
    def exec_module(self, module):
        pass

try:
    import nifty  # noqa: F401
except ImportError:
    sys.meta_path.insert(0, _NiftyVigraMockFinder())

import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from micro_sam.util import get_sam_model

from src.augmentation import get_train_transform, get_val_transform
from src.config import (
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
    make_run_subfolder,
    save_hparams,
)
from src.unet_dataset import UNetSemanticDataset
from src.splits import get_split, print_split_summary

NUM_CLASSES = 7
CLASS_NAMES = ["bg", "epidermis", "aerenchyma", "endo_ring", "vascular", "exo_ring", "cortex"]

# ViT-B intermediate block indices to extract features from
EXTRACT_BLOCKS = [2, 5, 8, 11]


# ── LoRA (reused from train_sam_semantic.py) ─────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
        return base + lora


def apply_lora_to_encoder(image_encoder, rank=4, alpha=1.0):
    """Apply LoRA to all attention QKV and output projections in ViT encoder."""
    lora_params = 0
    for block in image_encoder.blocks:
        old_qkv = block.attn.qkv
        block.attn.qkv = LoRALinear(old_qkv, rank=rank, alpha=alpha)
        lora_params += old_qkv.in_features * rank + rank * old_qkv.out_features

        old_proj = block.attn.proj
        block.attn.proj = LoRALinear(old_proj, rank=rank, alpha=alpha)
        lora_params += old_proj.in_features * rank + rank * old_proj.out_features

    return lora_params


# ── FPN Adapter: ViT features → multi-scale pyramid ─────────────────────────

class ViTToFPN(nn.Module):
    """Convert ViT intermediate block features into a multi-scale feature pyramid.

    ViT-B outputs (B, H/16, W/16, 768) at all blocks. We extract features from
    4 intermediate blocks and produce 5 feature maps at different spatial scales
    to match smp's encoder output format:

        Level 0: (B, C0, H, W)       — 1/1 scale (input image, passed through)
        Level 1: (B, C1, H/2, W/2)   — 1/2 scale
        Level 2: (B, C2, H/4, W/4)   — 1/4 scale
        Level 3: (B, C3, H/8, W/8)   — 1/8 scale
        Level 4: (B, C4, H/16, W/16) — 1/16 scale (native ViT resolution)
        Level 5: (B, C5, H/32, W/32) — 1/32 scale (downsampled)

    The channel dims match ResNet34: [3, 64, 64, 128, 256, 512]
    """

    def __init__(self, embed_dim=768, out_channels=(3, 64, 64, 128, 256, 512)):
        super().__init__()
        self.out_channels = out_channels

        # Project each ViT block output (768-dim) to target channel dims
        # Block 2 (shallowest) → upsampled most → levels 1,2
        # Block 11 (deepest) → kept at native or downsampled → levels 4,5

        # Level 5 (1/32): from block 11, downsample 2x
        self.proj_5 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[5], 1),
            nn.BatchNorm2d(out_channels[5]),
            nn.ReLU(inplace=True),
        )

        # Level 4 (1/16): from block 11, native resolution
        self.proj_4 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[4], 1),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU(inplace=True),
        )

        # Level 3 (1/8): from block 8, upsample 2x
        self.proj_3 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True),
        )

        # Level 2 (1/4): from block 5, upsample 4x
        self.proj_2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[2], 1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True),
        )

        # Level 1 (1/2): from block 2, upsample 8x
        self.proj_1 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[1], 1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_input, block_features):
        """
        Args:
            x_input: (B, 3, H, W) original input image (for level 0)
            block_features: list of 4 tensors from ViT blocks [2, 5, 8, 11],
                each (B, H/16, W/16, embed_dim)

        Returns:
            list of 6 feature maps matching smp encoder output format
        """
        # Reshape ViT features: (B, h, w, C) -> (B, C, h, w)
        feats = [f.permute(0, 3, 1, 2) for f in block_features]
        # feats[0]=block2, feats[1]=block5, feats[2]=block8, feats[3]=block11
        # All are (B, 768, H/16, W/16)

        h16, w16 = feats[0].shape[2], feats[0].shape[3]

        # Level 0: input image (B, 3, H, W)
        level_0 = x_input

        # Level 1 (1/2): from block 2, upsample 8x from 1/16
        level_1 = self.proj_1(feats[0])
        level_1 = F.interpolate(level_1, scale_factor=8, mode='bilinear', align_corners=False)

        # Level 2 (1/4): from block 5, upsample 4x from 1/16
        level_2 = self.proj_2(feats[1])
        level_2 = F.interpolate(level_2, scale_factor=4, mode='bilinear', align_corners=False)

        # Level 3 (1/8): from block 8, upsample 2x from 1/16
        level_3 = self.proj_3(feats[2])
        level_3 = F.interpolate(level_3, scale_factor=2, mode='bilinear', align_corners=False)

        # Level 4 (1/16): from block 11, native resolution
        level_4 = self.proj_4(feats[3])

        # Level 5 (1/32): from block 11, downsample 2x
        level_5 = self.proj_5(feats[3])
        level_5 = F.avg_pool2d(level_5, kernel_size=2, stride=2)

        return [level_0, level_1, level_2, level_3, level_4, level_5]


# ── Model ────────────────────────────────────────────────────────────────────

class SAMUNetPPModel(nn.Module):
    """SAM ViT-B encoder + FPN adapter + U-Net++ decoder.

    For fair comparison with ResNet34 + U-Net++: same decoder architecture,
    different encoder.
    """

    def __init__(self, model_type="vit_b_lm", freeze_encoder=True,
                 adapter=None, lora_rank=4, lora_alpha=1.0):
        super().__init__()

        # Load SAM encoder
        _, sam = get_sam_model(model_type=model_type, return_sam=True)
        self.image_encoder = sam.image_encoder
        self.embed_dim = 768  # ViT-B

        # Freeze encoder
        if freeze_encoder or adapter == "lora":
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
        self.adapter = adapter
        if adapter == "lora":
            n_lora = apply_lora_to_encoder(
                self.image_encoder, rank=lora_rank, alpha=lora_alpha,
            )
            print(f"LoRA applied: rank={lora_rank}, alpha={lora_alpha}, "
                  f"added {n_lora:,} params to {len(list(self.image_encoder.blocks))} blocks")

        # FPN adapter: ViT features → multi-scale pyramid
        # Match ResNet34 channel dims: [3, 64, 64, 128, 256, 512]
        encoder_channels = (3, 64, 64, 128, 256, 512)
        self.fpn = ViTToFPN(embed_dim=self.embed_dim, out_channels=encoder_channels)

        # U-Net++ decoder from smp (same as ResNet34 baseline)
        from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(16, NUM_CLASSES, kernel_size=1)

        # SAM normalization
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
        )

    def _extract_block_features(self, x_norm):
        """Run ViT encoder and extract intermediate block features.

        Args:
            x_norm: (B, 3, 1024, 1024) normalized input

        Returns:
            list of 4 tensors, each (B, H/16, W/16, embed_dim)
        """
        enc = self.image_encoder

        # Patch embedding
        x = enc.patch_embed(x_norm)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed

        block_features = []
        for i, block in enumerate(enc.blocks):
            x = block(x)
            if i in EXTRACT_BLOCKS:
                block_features.append(x)

        return block_features

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) float32 in [0, 1]

        Returns:
            (B, 7, H, W) logits
        """
        B, _, H, W = x.shape

        # SAM preprocessing: scale to [0, 255] and normalize
        x_255 = x * 255.0
        x_norm = (x_255 - self.pixel_mean) / self.pixel_std

        # Pad to 1024 if needed (SAM expects 1024x1024)
        if H != 1024 or W != 1024:
            x_norm = F.interpolate(x_norm, size=(1024, 1024), mode='bilinear', align_corners=False)

        # Extract ViT intermediate features
        block_feats = self._extract_block_features(x_norm)

        # FPN: convert to multi-scale pyramid
        pyramid = self.fpn(x, block_feats)

        # U-Net++ decoder
        decoder_out = self.decoder(pyramid)

        # Segmentation head
        logits = self.seg_head(decoder_out)

        # Resize to input resolution if needed
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits


# ── Lightning Module ─────────────────────────────────────────────────────────

class SAMUNetPPModule(pl.LightningModule):
    """PyTorch Lightning module for SAM + U-Net++ semantic segmentation."""

    def __init__(
        self,
        model_type="vit_b_lm",
        freeze_encoder=True,
        adapter=None,
        lora_rank=4,
        lora_alpha=1.0,
        aer_weight=2.0,
        equal_weights=False,
        lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="cosine",
        eta_min=1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.eta_min = eta_min

        self.model = SAMUNetPPModel(
            model_type=model_type,
            freeze_encoder=freeze_encoder,
            adapter=adapter,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Losses (same as U-Net++ semantic and SAM UNETR)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")

        if equal_weights:
            weights = [1.0] * NUM_CLASSES
        else:
            weights = [0.5, 1.0, aer_weight, 5.0, 1.0, 5.0, 1.0]
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        dice = self.dice_loss(logits, masks)
        focal = self.focal_loss(logits, masks)
        wce = F.cross_entropy(logits, masks, weight=self.class_weights)
        lovasz = self.lovasz_loss(logits, masks)
        loss = dice + focal + wce + lovasz

        preds = logits.argmax(dim=1)
        correct = (preds == masks).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_dice_loss", dice, sync_dist=True)
        self.log(f"{stage}_focal_loss", focal, sync_dist=True)
        self.log(f"{stage}_pixel_acc", correct, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            for c in range(1, NUM_CLASSES):
                cls_mask = masks == c
                if cls_mask.sum() > 0:
                    cls_acc = (preds[cls_mask] == c).float().mean()
                    self.log(f"{stage}_acc_{CLASS_NAMES[c]}", cls_acc, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Separate encoder params (LoRA or full) from FPN + decoder params
        encoder_params = [p for p in self.model.image_encoder.parameters() if p.requires_grad]
        other_params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and not name.startswith("image_encoder"):
                other_params.append(p)

        param_groups = [
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": other_params, "lr": self.lr},
        ]

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM ViT-B encoder + U-Net++ decoder for 7-class semantic segmentation")

    parser.add_argument("--model-type", default="vit_b_lm")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--adapter", default=None, choices=[None, "lora"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-7)
    parser.add_argument("--aer-weight", type=float, default=2.0)
    parser.add_argument("--equal-weights", action="store_true",
                        help="Use uniform [1,1,...,1] CE class weights (overrides --aer-weight)")
    parser.add_argument("--channel-dropout", type=float, default=0.2)
    parser.add_argument("--channel-shuffle", type=float, default=0.2)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="A")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    split = get_split(args.strategy)
    print_split_summary(split)

    train_transform = get_train_transform(
        args.img_size, p_channel_dropout=args.channel_dropout,
        p_channel_shuffle=args.channel_shuffle,
    )
    val_transform = get_val_transform(args.img_size)

    train_ds = UNetSemanticDataset(split["train"], transform=train_transform, img_size=args.img_size)
    val_ds = UNetSemanticDataset(split["val"], transform=val_transform, img_size=args.img_size)
    test_ds = UNetSemanticDataset(split["test"], transform=val_transform, img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    freeze_encoder = not args.unfreeze_encoder
    model = SAMUNetPPModule(
        model_type=args.model_type,
        freeze_encoder=freeze_encoder,
        adapter=args.adapter,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        aer_weight=args.aer_weight,
        equal_weights=args.equal_weights,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
    )

    # Print param summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

    # Run name: model_pretrain_classweight_augmentation_loss_strategy
    if args.adapter == "lora":
        encoder_tag = f"sam_vit_b_lm_lora_r{args.lora_rank}"
    elif args.unfreeze_encoder:
        encoder_tag = "sam_vit_b_lm_finetune"
    else:
        encoder_tag = "sam_vit_b_lm_frozen"

    if args.equal_weights:
        weight_tag = "equalw"
    else:
        weight_tag = f"aer{int(args.aer_weight)}" if args.aer_weight != 2.0 else "defaultw"

    aug_parts = []
    if args.channel_dropout > 0: aug_parts.append("drop")
    if args.channel_shuffle > 0: aug_parts.append("shuf")
    aug_tag = "_".join(aug_parts) if aug_parts else "noaug"

    run_name = f"unetpp_{encoder_tag}_{weight_tag}_{aug_tag}_dfcel_semantic7c_{args.strategy}"
    base_run_dir = OUTPUT_DIR / "runs" / "sam_unetpp" / run_name
    run_dir = make_run_subfolder(base_run_dir)
    save_hparams(run_dir, args)

    # Epoch logger (same as other training scripts)
    class EpochLogger(Callback):
        def __init__(self, val_dl, val_iou_every=10, run_dir=None):
            self.val_dl = val_dl
            self.val_iou_every = val_iou_every
            self.run_dir = run_dir
            self.iou_csv_written = False

        def on_train_epoch_end(self, trainer, pl_module):
            import csv as csv_mod
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch
            parts = [f"Epoch {epoch:3d}"]
            for key in ["train_loss", "val_loss", "train_pixel_acc", "val_pixel_acc"]:
                if key in metrics:
                    parts.append(f"{key}={metrics[key]:.4f}")
            print(" | ".join(parts), flush=True)

            if self.val_iou_every > 0 and (epoch + 1) % self.val_iou_every == 0:
                print(f"  Computing val IoU/Dice (epoch {epoch + 1})...")
                pl_module.eval()
                class_inter = [0] * NUM_CLASSES
                class_union = [0] * NUM_CLASSES
                class_pred_sum = [0] * NUM_CLASSES
                class_gt_sum = [0] * NUM_CLASSES

                with torch.no_grad():
                    for batch in self.val_dl:
                        images = batch["image"].to(pl_module.device)
                        masks = batch["mask"].to(pl_module.device)
                        preds = pl_module(images).argmax(dim=1)
                        for c in range(NUM_CLASSES):
                            gt_c = (masks == c)
                            pred_c = (preds == c)
                            class_inter[c] += int((gt_c & pred_c).sum())
                            class_union[c] += int((gt_c | pred_c).sum())
                            class_pred_sum[c] += int(pred_c.sum())
                            class_gt_sum[c] += int(gt_c.sum())

                pl_module.train()

                row = {"epoch": epoch + 1}
                for c in range(NUM_CLASSES):
                    name = CLASS_NAMES[c]
                    iou = class_inter[c] / class_union[c] if class_union[c] > 0 else float("nan")
                    denom = class_pred_sum[c] + class_gt_sum[c]
                    dice = 2 * class_inter[c] / denom if denom > 0 else float("nan")
                    row[f"{name}_IoU"] = round(iou, 4)
                    row[f"{name}_Dice"] = round(dice, 4)
                    print(f"    {name:15s}  IoU={iou:.4f}  Dice={dice:.4f}")
                    if trainer.logger:
                        for lg in (trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]):
                            if hasattr(lg, 'experiment') and hasattr(lg.experiment, 'add_scalar'):
                                lg.experiment.add_scalar(f"val_IoU/{name}", iou, epoch)
                                lg.experiment.add_scalar(f"val_Dice/{name}", dice, epoch)

                fg_ious = [class_inter[c] / class_union[c] for c in range(1, NUM_CLASSES)
                           if class_union[c] > 0]
                mean_iou = sum(fg_ious) / len(fg_ious) if fg_ious else float("nan")
                row["mean_IoU"] = round(mean_iou, 4)
                print(f"    Mean (no bg): IoU={mean_iou:.4f}")

                if self.run_dir:
                    iou_path = self.run_dir / "val_iou_dice.csv"
                    with open(iou_path, "a", newline="") as f:
                        writer = csv_mod.DictWriter(f, fieldnames=row.keys())
                        if not self.iou_csv_written:
                            writer.writeheader()
                            self.iou_csv_written = True
                        writer.writerow(row)

    callbacks = [
        EpochLogger(val_dl=val_dl, val_iou_every=args.save_every, run_dir=run_dir),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
    ]
    if args.save_every > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="periodic-{epoch}",
            every_n_epochs=args.save_every,
            save_top_k=-1,
        ))

    logger = [
        CSVLogger(str(run_dir), name="logs"),
        TensorBoardLogger(str(run_dir), name="tensorboard"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
    )

    trainer.fit(model, train_dl, val_dl)

    # Plot loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_logger = logger[0]
    metrics_file = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "train_loss" in df.columns:
            train = df[df["train_loss"].notna()]
            axes[0].plot(train["epoch"], train["train_loss"], label="Train Loss")
        if "val_loss" in df.columns:
            val = df[df["val_loss"].notna()]
            axes[0].plot(val["epoch"], val["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"SAM ViT-B + U-Net++ ({adapter_tag.strip('_')}) — Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "train_pixel_acc" in df.columns:
            train = df[df["train_pixel_acc"].notna()]
            axes[1].plot(train["epoch"], train["train_pixel_acc"], label="Train Acc")
        if "val_pixel_acc" in df.columns:
            val = df[df["val_pixel_acc"].notna()]
            axes[1].plot(val["epoch"], val["val_pixel_acc"], label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Pixel Accuracy")
        axes[1].set_title(f"SAM ViT-B + U-Net++ ({adapter_tag.strip('_')}) — Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Test
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    trainer.logger = False
    test_results = trainer.test(model, test_dl, ckpt_path="best")
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
