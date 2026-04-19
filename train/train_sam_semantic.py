"""SAM encoder + UNETR decoder for 7-class semantic segmentation.

Uses micro-SAM's ViT-B encoder (pretrained on light microscopy) with a UNETR
decoder for 7-class semantic segmentation.

Three encoder modes:
    1. Frozen (default): encoder weights fixed, only decoder trains (~9.6M params)
    2. LoRA adapter: encoder frozen + small trainable matrices in attention (~9.9M params)
    3. Full fine-tune: all parameters trainable (~99M params)

Architecture:
    Encoder: SAM ViT-B (vit_b_lm, ~89M params)
        - Extracts features at blocks 2, 5, 8, 11
    Decoder: UNETR (ConvTranspose + skip connections, ~9.6M params, trainable)
        - Takes multi-scale ViT features → upsamples → 7-class output

7 semantic classes (same as U-Net++ semantic):
    0 = background
    1 = epidermis
    2 = aerenchyma
    3 = endodermis ring
    4 = vascular
    5 = exodermis ring
    6 = cortex

Usage:
    # Frozen encoder (baseline)
    python train/train_sam_semantic.py

    # LoRA adapter (rank=4, adapts attention layers)
    python train/train_sam_semantic.py --adapter lora --lora-rank 4

    # Full fine-tune
    python train/train_sam_semantic.py --unfreeze-encoder --backbone-lr 1e-5
"""

import sys
import types
import importlib.abc
import importlib.machinery
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock nifty/vigra if not available (only needed for elf post-processing, not training)
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
from micro_sam.instance_segmentation import get_unetr

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


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Keeps the original weight W frozen and adds a trainable low-rank
    decomposition: output = Wx + (BA)x, where A is (in, rank) and
    B is (rank, out). Only A and B are trained.

    Args:
        original: The original nn.Linear layer (will be frozen).
        rank: Rank of the low-rank matrices (default 4).
        alpha: Scaling factor. Output is scaled by alpha/rank.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices: A projects down, B projects back up
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        # Original frozen path + trainable LoRA path
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
        return base + lora


def apply_lora_to_encoder(image_encoder: nn.Module, rank: int = 4, alpha: float = 1.0):
    """Inject LoRA adapters into all attention QKV and output projections.

    Targets: blocks[i].attn.qkv and blocks[i].attn.proj in SAM's ViT.
    The original weights are frozen; only LoRA A/B matrices are trainable.

    Args:
        image_encoder: SAM's image encoder (ViT).
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Number of LoRA parameters added.
    """
    lora_params = 0
    for block in image_encoder.blocks:
        # Replace attn.qkv (nn.Linear)
        old_qkv = block.attn.qkv
        block.attn.qkv = LoRALinear(old_qkv, rank=rank, alpha=alpha)
        lora_params += old_qkv.in_features * rank + rank * old_qkv.out_features

        # Replace attn.proj (nn.Linear)
        old_proj = block.attn.proj
        block.attn.proj = LoRALinear(old_proj, rank=rank, alpha=alpha)
        lora_params += old_proj.in_features * rank + rank * old_proj.out_features

    return lora_params


# ── Model ─────────────────────────────────────────────────────────────────────

class SAMSemanticModel(nn.Module):
    """SAM ViT encoder + UNETR decoder for 7-class semantic segmentation.

    The UNETR from micro-SAM expects input as image embeddings (from
    predictor.features), not raw images. We wrap the full pipeline:
        raw image → SAM preprocessing → ViT encoder → UNETR decoder → logits
    """

    def __init__(self, model_type="vit_b_lm", freeze_encoder=True,
                 adapter=None, lora_rank=4, lora_alpha=1.0):
        super().__init__()

        # Load SAM model to get the encoder
        _, sam = get_sam_model(model_type=model_type, return_sam=True)
        self.image_encoder = sam.image_encoder

        # Freeze encoder first (LoRA will add trainable params on top)
        if freeze_encoder or adapter == "lora":
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA adapters if requested
        self.adapter = adapter
        if adapter == "lora":
            n_lora = apply_lora_to_encoder(
                self.image_encoder, rank=lora_rank, alpha=lora_alpha,
            )
            print(f"LoRA applied: rank={lora_rank}, alpha={lora_alpha}, "
                  f"added {n_lora:,} params to {len(list(self.image_encoder.blocks))} blocks")

        # Build UNETR decoder with 7 output channels, no final activation
        # (we apply softmax in the loss, not in the model)
        self.unetr = get_unetr(
            image_encoder=self.image_encoder,
            out_channels=NUM_CLASSES,
            final_activation=None,
        )

        # SAM's image normalization (same as used during pretraining)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) float32 in [0, 1] (from UNetSemanticDataset)

        Returns:
            (B, 7, H, W) logits (no softmax applied)
        """
        # Scale to [0, 255] and normalize with SAM's ImageNet stats
        x_255 = x * 255.0
        x_norm = (x_255 - self.pixel_mean) / self.pixel_std

        # UNETR handles the full pipeline: encoder + decoder
        # It calls self.encoder internally via its forward method
        logits = self.unetr(x_norm)
        return logits


class SAMSemanticModule(pl.LightningModule):
    """PyTorch Lightning module for SAM + UNETR semantic segmentation."""

    def __init__(
        self,
        model_type="vit_b_lm",
        freeze_encoder=True,
        adapter=None,
        lora_rank=4,
        lora_alpha=1.0,
        aer_weight=10.0,
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
        self.freeze_encoder = freeze_encoder
        self.adapter = adapter

        self.model = SAMSemanticModel(
            model_type=model_type,
            freeze_encoder=freeze_encoder,
            adapter=adapter,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Same loss as U-Net++ semantic: Dice + Focal + weighted CE + Lovász
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")

        # Class weights: bg=0.5, epi=1, aer=configurable, endo=5, vasc=1, exo=5, cortex=1
        weights = [0.5, 1.0, aer_weight, 5.0, 1.0, 5.0, 1.0]
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]    # (B, 3, H, W) float32 [0, 1]
        masks = batch["mask"]      # (B, H, W) int64

        logits = self(images)      # (B, 7, H, W)

        # Resize logits to match mask if needed (UNETR may output different size)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )

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
        # Collect UNETR decoder params (always trained)
        decoder_params = [
            p for n, p in self.model.unetr.named_parameters()
            if not n.startswith("encoder") and p.requires_grad
        ]

        if self.adapter == "lora":
            # LoRA params (in encoder attention layers) + decoder params
            lora_params = [
                p for n, p in self.model.image_encoder.named_parameters()
                if p.requires_grad  # only LoRA A/B are unfrozen
            ]
            param_groups = [
                {"params": lora_params, "lr": self.backbone_lr},
                {"params": decoder_params, "lr": self.lr},
            ]
        elif self.freeze_encoder:
            # Frozen encoder: only decoder params
            param_groups = [{"params": decoder_params, "lr": self.lr}]
        else:
            # Full fine-tune: differential LR
            encoder_params = [
                p for p in self.model.image_encoder.parameters()
                if p.requires_grad
            ]
            param_groups = [
                {"params": encoder_params, "lr": self.backbone_lr},
                {"params": decoder_params, "lr": self.lr},
            ]

        optimizer = torch.optim.AdamW(
            param_groups, lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(
        description="SAM encoder + UNETR decoder for 7-class semantic segmentation")

    # Model
    parser.add_argument("--model-type", default="vit_b_lm",
                        help="SAM model type: vit_b, vit_b_lm, vit_l_lm")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Unfreeze SAM encoder for full fine-tuning")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Freeze encoder for this many epochs before unfreezing "
                             "(only used with --unfreeze-encoder)")
    parser.add_argument("--adapter", default=None, choices=["lora"],
                        help="Adapter type to apply to encoder")
    parser.add_argument("--lora-rank", type=int, default=4,
                        help="LoRA rank (default: 4)")
    parser.add_argument("--lora-alpha", type=float, default=1.0,
                        help="LoRA scaling factor (default: 1.0)")

    # Training
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aer-weight", type=float, default=10.0,
                        help="CE class weight for aerenchyma (default: 10)")
    parser.add_argument("--eta-min", type=float, default=1e-7)

    # Augmentation
    parser.add_argument("--channel-dropout", type=float, default=0.2)
    parser.add_argument("--channel-shuffle", type=float, default=0.2)

    # Infrastructure
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="A")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    split = get_split(args.strategy)
    print_split_summary(split)

    train_transform = get_train_transform(
        args.img_size,
        p_channel_dropout=args.channel_dropout,
        p_channel_shuffle=args.channel_shuffle,
    )
    val_transform = get_val_transform(args.img_size)

    train_ds = UNetSemanticDataset(split["train"], transform=train_transform,
                                    img_size=args.img_size)
    val_ds = UNetSemanticDataset(split["val"], transform=val_transform,
                                  img_size=args.img_size)
    test_ds = UNetSemanticDataset(split["test"], transform=val_transform,
                                   img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.adapter and args.unfreeze_encoder:
        parser.error("Cannot use --adapter with --unfreeze-encoder")

    # With warm-up: start frozen, unfreeze later via callback
    warmup = args.warmup_epochs > 0 and args.unfreeze_encoder
    if warmup:
        # Start with encoder frozen; callback will unfreeze after warmup
        freeze_encoder = True
    else:
        freeze_encoder = not args.unfreeze_encoder

    model = SAMSemanticModule(
        model_type=args.model_type,
        freeze_encoder=freeze_encoder,
        adapter=args.adapter,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        aer_weight=args.aer_weight,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params:    {total_params - trainable_params:,}")
    print(f"Encoder mode:     ", end="")
    if args.adapter == "lora":
        print(f"LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    elif warmup:
        print(f"frozen → full fine-tune after {args.warmup_epochs} warmup epochs")
    elif freeze_encoder:
        print("frozen")
    else:
        print("full fine-tune")

    # ── Run directory ─────────────────────────────────────────────────────────
    # Run name: model_pretrain_classweight_augmentation_loss_strategy
    if args.adapter == "lora":
        encoder_tag = f"sam_{args.model_type}_lora_r{args.lora_rank}"
    elif warmup:
        encoder_tag = f"sam_{args.model_type}_warmup{args.warmup_epochs}_finetune"
    elif freeze_encoder:
        encoder_tag = f"sam_{args.model_type}_frozen"
    else:
        encoder_tag = f"sam_{args.model_type}_finetune"

    weight_tag = "equalw" if args.aer_weight == 1.0 else f"aer{int(args.aer_weight)}" if args.aer_weight != 2.0 else "defaultw"

    aug_parts = []
    if args.channel_dropout > 0: aug_parts.append("drop")
    if args.channel_shuffle > 0: aug_parts.append("shuf")
    aug_tag = "_".join(aug_parts) if aug_parts else "noaug"

    run_name = f"unetr_{encoder_tag}_{weight_tag}_{aug_tag}_dfcel_semantic7c_{args.strategy}"
    base_run_dir = OUTPUT_DIR / "runs" / "sam_semantic" / run_name
    run_dir = make_run_subfolder(base_run_dir)
    save_hparams(run_dir, args)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    class EpochLogger(Callback):
        """Print epoch summary + compute val IoU/Dice periodically."""
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
                        logits = pl_module(images)
                        if logits.shape[-2:] != masks.shape[-2:]:
                            logits = F.interpolate(
                                logits, size=masks.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )
                        preds = logits.argmax(dim=1)
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

    # Warm-up callback: unfreeze encoder after N epochs
    class EncoderWarmup(Callback):
        """Freeze encoder for warmup_epochs, then unfreeze with low LR."""
        def __init__(self, warmup_epochs, backbone_lr):
            self.warmup_epochs = warmup_epochs
            self.backbone_lr = backbone_lr
            self.unfrozen = False

        def on_train_epoch_start(self, trainer, pl_module):
            if not self.unfrozen and trainer.current_epoch >= self.warmup_epochs:
                print(f"\n{'='*60}")
                print(f"WARM-UP COMPLETE (epoch {trainer.current_epoch}): "
                      f"unfreezing encoder with lr={self.backbone_lr}")
                print(f"{'='*60}\n")

                # Unfreeze encoder
                for param in pl_module.model.image_encoder.parameters():
                    param.requires_grad = True

                # Update internal state so configure_optimizers works correctly
                pl_module.freeze_encoder = False

                # Count new trainable params
                trainable = sum(p.numel() for p in pl_module.parameters()
                                if p.requires_grad)
                print(f"Trainable params after unfreeze: {trainable:,}")

                # Add encoder params to existing optimizer with low LR
                encoder_params = list(pl_module.model.image_encoder.parameters())
                trainer.optimizers[0].add_param_group(
                    {"params": encoder_params, "lr": self.backbone_lr}
                )

                self.unfrozen = True

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
    if warmup:
        callbacks.append(EncoderWarmup(args.warmup_epochs, args.backbone_lr))
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

    # ── Train ─────────────────────────────────────────────────────────────────
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

    # ── Loss curve ────────────────────────────────────────────────────────────
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
        axes[0].set_title(f"SAM UNETR ({args.model_type}, {freeze_tag}) — Loss")
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
        axes[1].set_title(f"SAM UNETR ({args.model_type}, {freeze_tag}) — Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    trainer.logger = False
    test_results = trainer.test(model, test_dl, ckpt_path="best")
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
