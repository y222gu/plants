"""U-Net / U-Net++ segmentation training with SMP + PyTorch Lightning.

Supports two modes:
    semantic:   Mutually exclusive classes (softmax, CE + Dice + Focal + Lovasz)
    multilabel: Independent binary channels (sigmoid, BCE + Dice)

Usage:
    python train_unet.py --mode semantic --strategy A
    python train_unet.py --mode multilabel --strategy A
    python train_unet.py --arch unetplusplus --encoder resnet101 --epochs 150
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.augmentation import get_train_transform, get_val_transform
from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
    make_run_subfolder,
    save_hparams,
)
from src.dataset import SampleRegistry
from src.models.unet_dataset import UNetDataset
from src.splits import get_split, print_split_summary


# ── Optimizer / scheduler factories ──────────────────────────────────────────

def _build_optimizer(name: str, param_groups: list, lr: float, weight_decay: float):
    """Build optimizer by name."""
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay,
                               momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(name: str, optimizer, max_epochs: int, eta_min: float):
    """Build LR scheduler by name."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=eta_min)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(max_epochs // 3, 1), gamma=0.1)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


class SegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for semantic segmentation (softmax/CE)."""

    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet34",
        lr: float = DEFAULT_LR,
        backbone_lr: float = 1e-5,
        num_classes: int = 4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        eta_min: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.eta_min = eta_min
        num_semantic = num_classes + 1  # 0=bg + N target classes

        model_cls = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
        }[arch]

        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_semantic,
        )

        # Dice + Focal + weighted CE + Lovász loss
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            classes=num_semantic,
        )
        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass",
        )
        self.lovasz_loss = smp.losses.LovaszLoss(
            mode="multiclass",
        )
        # Class weights: bg=0.5, root=1, aer=2, endo=5, vasc=1 [, exo=5]
        weights = [0.5, 1.0, 2.0, 5.0, 1.0]
        if num_classes >= 5:
            weights.append(5.0)  # Exodermis (thin ring, like endodermis)
        self.register_buffer(
            "class_weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)  # (B, C, H, W)
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
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Different LR for encoder (pretrained) vs decoder (new)
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = [p for p in self.model.parameters()
                          if not any(p is ep for ep in encoder_params)]

        param_groups = [
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ]
        optimizer = _build_optimizer(self.optimizer_name, param_groups, self.lr, self.weight_decay)
        scheduler = _build_scheduler(self.scheduler_name, optimizer, self.trainer.max_epochs, self.eta_min)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class MultiLabelSegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for multi-label segmentation (sigmoid/BCE).

    Each pixel has independent binary labels (whole root, aerenchyma,
    endodermis, vascular, [exodermis]). Overlapping classes are handled naturally.
    """

    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet34",
        lr: float = DEFAULT_LR,
        backbone_lr: float = 1e-5,
        num_classes: int = 4,
        mask_missing: bool = False,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        eta_min: float = 1e-7,
        pos_weight: list = None,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.eta_min = eta_min
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.mask_missing = mask_missing

        model_cls = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
        }[arch]

        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

        # BCEWithLogitsLoss with class-specific pos_weight
        if pos_weight is not None:
            pw = pos_weight
        else:
            # Default: root=1 (large area), aer=2 (many small), endo=5 (thin ring), vasc=1 [, exo=5]
            pw = [1.0, 2.0, 5.0, 1.0]
            if num_classes >= 5:
                pw.append(5.0)  # Exodermis (thin ring, like endodermis)
        self.register_buffer(
            "pos_weight",
            torch.tensor(pw, dtype=torch.float32),
        )

        # Dice loss in multilabel mode
        self.dice_loss = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]          # (B, 3, H, W)
        masks = batch["mask"]            # (B, C, H, W) float32

        logits = self(images)            # (B, C, H, W)

        if self.mask_missing and "valid_mask" in batch:
            # Masked loss: only compute on channels that are annotated for each sample
            valid = batch["valid_mask"]  # (B, C)
            vm = valid.view(valid.shape[0], valid.shape[1], 1, 1)  # (B, C, 1, 1)

            # BCE with reduction='none', then mask
            bce_raw = F.binary_cross_entropy_with_logits(
                logits, masks,
                pos_weight=self.pos_weight.view(1, -1, 1, 1),
                reduction="none",
            )  # (B, C, H, W)
            bce_masked = bce_raw * vm
            n_valid = vm.sum() * masks.shape[2] * masks.shape[3]
            bce = bce_masked.sum() / n_valid.clamp(min=1.0)

            # Dice per-channel with validity weighting
            probs = torch.sigmoid(logits)
            dice_loss = 0.0
            n_valid_ch = 0
            for c in range(self.num_classes):
                ch_valid = valid[:, c]  # (B,)
                if ch_valid.sum() == 0:
                    continue
                idx = ch_valid.bool()
                ch_dice = 1.0 - self._channel_dice(probs[idx, c], masks[idx, c])
                dice_loss = dice_loss + ch_dice
                n_valid_ch += 1
            dice = dice_loss / max(n_valid_ch, 1)
        else:
            # Standard unmasked loss
            bce = F.binary_cross_entropy_with_logits(
                logits, masks,
                pos_weight=self.pos_weight.view(1, -1, 1, 1),
            )
            dice = self.dice_loss(logits, masks)

        loss = self.bce_weight * bce + self.dice_weight * dice

        # Metrics: per-channel accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == masks).float().mean()

        # Per-class dice (for monitoring)
        class_names = ["root", "aer", "endo", "vasc", "exo"][:self.num_classes]
        with torch.no_grad():
            for c, name in enumerate(class_names):
                p = preds[:, c]
                t = masks[:, c]
                inter = (p * t).sum()
                denom = p.sum() + t.sum()
                ch_dice = (2 * inter / denom) if denom > 0 else torch.tensor(1.0)
                self.log(f"{stage}_dice_{name}", ch_dice, sync_dist=True)

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_bce_loss", bce, sync_dist=True)
        self.log(f"{stage}_dice_loss", dice, sync_dist=True)
        self.log(f"{stage}_pixel_acc", correct, prog_bar=True, sync_dist=True)
        return loss

    @staticmethod
    def _channel_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Compute Dice score for a single channel across a batch subset."""
        inter = (pred * target).sum()
        denom = pred.sum() + target.sum()
        return (2 * inter + smooth) / (denom + smooth)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = [p for p in self.model.parameters()
                          if not any(p is ep for ep in encoder_params)]

        param_groups = [
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ]
        optimizer = _build_optimizer(self.optimizer_name, param_groups, self.lr, self.weight_decay)
        scheduler = _build_scheduler(self.scheduler_name, optimizer, self.trainer.max_epochs, self.eta_min)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(description="U-Net segmentation training")
    parser.add_argument("--mode", default="semantic",
                        choices=["semantic", "multilabel"],
                        help="Segmentation mode: semantic (softmax) or multilabel (sigmoid)")
    parser.add_argument("--arch", default="unet", choices=["unet", "unetplusplus"])
    parser.add_argument("--encoder", default="resnet34",
                        help="Encoder backbone (e.g. resnet50, resnet101, efficientnet-b4)")
    parser.add_argument("--strategy", default="A",
                        choices=["A", "B", "C"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau"])
    parser.add_argument("--eta-min", type=float, default=1e-7,
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("--pos-weight", type=float, nargs="+", default=None,
                        help="Per-class pos_weight for BCE loss (multilabel mode). "
                             "Default: 1 2 5 1 [5] for 4 [5] classes")
    parser.add_argument("--bce-weight", type=float, default=1.0,
                        help="Weight for BCE loss term (multilabel mode)")
    parser.add_argument("--dice-weight", type=float, default=1.0,
                        help="Weight for Dice loss term (multilabel mode)")
    parser.add_argument("--num-classes", type=int, default=4, choices=[4, 5],
                        help="Number of target classes (4=standard, 5=with exodermis)")
    parser.add_argument("--mask-missing", action="store_true",
                        help="Use validity masking for missing classes (e.g. no aerenchyma in tomato)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (uses DDP strategy when > 1)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save periodic checkpoint every N epochs (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Setup
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    # Datasets
    train_transform = get_train_transform(args.img_size)
    val_transform = get_val_transform(args.img_size)

    train_ds = UNetDataset(split["train"], transform=train_transform,
                           img_size=args.img_size, mode=args.mode,
                           num_classes=args.num_classes, mask_missing=args.mask_missing)
    val_ds = UNetDataset(split["val"], transform=val_transform,
                         img_size=args.img_size, mode=args.mode,
                         num_classes=args.num_classes, mask_missing=args.mask_missing)
    test_ds = UNetDataset(split["test"], transform=val_transform,
                          img_size=args.img_size, mode=args.mode,
                          num_classes=args.num_classes, mask_missing=args.mask_missing)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # Model
    if args.mode == "multilabel":
        model = MultiLabelSegmentationModule(
            arch=args.arch,
            encoder=args.encoder,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            num_classes=args.num_classes,
            mask_missing=args.mask_missing,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            eta_min=args.eta_min,
            pos_weight=args.pos_weight,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )
    else:
        model = SegmentationModule(
            arch=args.arch,
            encoder=args.encoder,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            num_classes=args.num_classes,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            eta_min=args.eta_min,
        )

    # Callbacks
    run_name = f"{args.arch}_{args.encoder}_{args.strategy}_{args.mode}_c{args.num_classes}"
    if args.mask_missing:
        run_name += "_masked"
    if args.species:
        run_name += f"_{args.species}"
    base_run_dir = OUTPUT_DIR / "runs" / "unet" / run_name
    run_dir = make_run_subfolder(base_run_dir)
    save_hparams(run_dir, args)

    callbacks = [
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

    logger = CSVLogger(str(run_dir), name="logs")

    # Train
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_dl, val_dl)

    # Plot loss curves from CSVLogger output
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_file = Path(logger.log_dir) / "metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        if "train_loss" in df.columns:
            train = df[df["train_loss"].notna()]
            axes[0].plot(train["epoch"], train["train_loss"], label="Train Loss")
        if "val_loss" in df.columns:
            val = df[df["val_loss"].notna()]
            axes[0].plot(val["epoch"], val["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{args.arch} ({args.encoder}) [{args.mode}] — Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Pixel accuracy
        if "train_pixel_acc" in df.columns:
            train = df[df["train_pixel_acc"].notna()]
            axes[1].plot(train["epoch"], train["train_pixel_acc"], label="Train Acc")
        if "val_pixel_acc" in df.columns:
            val = df[df["val_pixel_acc"].notna()]
            axes[1].plot(val["epoch"], val["val_pixel_acc"], label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Pixel Accuracy")
        axes[1].set_title(f"{args.arch} ({args.encoder}) [{args.mode}] — Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = run_dir / "loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to {plot_path}")

    # Test
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    test_results = trainer.test(model, test_dl, ckpt_path="best")
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
