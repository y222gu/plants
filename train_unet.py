"""U-Net / U-Net++ segmentation training with SMP + PyTorch Lightning.

Supports two modes:
    semantic:   Mutually exclusive classes (softmax, CE + Dice + Focal + Lovasz)
    multilabel: Independent binary channels (sigmoid, BCE + Dice)

Usage:
    python train_unet.py --mode semantic --strategy strategy1
    python train_unet.py --mode multilabel --strategy strategy1
    python train_unet.py --arch unetplusplus --encoder resnet101 --epochs 150
"""

import argparse
from pathlib import Path

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
    NUM_CLASSES,
    OUTPUT_DIR,
)
from src.dataset import SampleRegistry
from src.models.unet_dataset import UNetDataset
from src.splits import get_split, print_split_summary


NUM_SEMANTIC_CLASSES = NUM_CLASSES + 1  # 0=bg + 4 target classes


class SegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for semantic segmentation (softmax/CE)."""

    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet34",
        lr: float = DEFAULT_LR,
        backbone_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.backbone_lr = backbone_lr

        model_cls = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
        }[arch]

        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_SEMANTIC_CLASSES,
        )

        # Dice + Focal + weighted CE + Lovász loss
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            classes=NUM_SEMANTIC_CLASSES,
        )
        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass",
        )
        # Lovász loss: differentiable IoU surrogate that penalizes
        # fragmentation and holes at the mask level, not per-pixel
        self.lovasz_loss = smp.losses.LovaszLoss(
            mode="multiclass",
        )
        # Class weights: bg=0.5, root=1, aer=2, endo=5, vasc=1
        # Endodermis (thin ring) gets 5x weight to compensate for few pixels
        self.register_buffer(
            "class_weights",
            torch.tensor([0.5, 1.0, 2.0, 5.0, 1.0], dtype=torch.float32),
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

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class MultiLabelSegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for multi-label segmentation (sigmoid/BCE).

    Each pixel has 4 independent binary labels (whole root, aerenchyma,
    endodermis, vascular). Overlapping classes are handled naturally.
    """

    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet34",
        lr: float = DEFAULT_LR,
        backbone_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.backbone_lr = backbone_lr

        model_cls = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
        }[arch]

        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,  # 4 channels, no background class
        )

        # BCEWithLogitsLoss with class-specific pos_weight
        # root=1 (large area), aer=2 (many small), endo=5 (thin ring), vasc=1
        self.register_buffer(
            "pos_weight",
            torch.tensor([1.0, 2.0, 5.0, 1.0], dtype=torch.float32),
        )

        # Dice loss in multilabel mode
        self.dice_loss = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]          # (B, 3, H, W)
        masks = batch["mask"]            # (B, 4, H, W) float32

        logits = self(images)            # (B, 4, H, W)

        # BCE loss with pos_weight (broadcast: weight shape [4] → [1, 4, 1, 1])
        bce = F.binary_cross_entropy_with_logits(
            logits, masks,
            pos_weight=self.pos_weight.view(1, -1, 1, 1),
        )
        dice = self.dice_loss(logits, masks)
        loss = bce + dice

        # Metrics: per-channel accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == masks).float().mean()

        # Per-class dice (for monitoring)
        with torch.no_grad():
            for c, name in enumerate(["root", "aer", "endo", "vasc"]):
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

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(description="U-Net segmentation training")
    parser.add_argument("--mode", default="semantic",
                        choices=["semantic", "multilabel"],
                        help="Segmentation mode: semantic (softmax) or multilabel (sigmoid)")
    parser.add_argument("--arch", default="unet", choices=["unet", "unetplusplus"])
    parser.add_argument("--encoder", default="resnet34",
                        help="Encoder backbone (e.g. resnet50, resnet101, efficientnet-b4)")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
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
                           img_size=args.img_size, mode=args.mode)
    val_ds = UNetDataset(split["val"], transform=val_transform,
                         img_size=args.img_size, mode=args.mode)
    test_ds = UNetDataset(split["test"], transform=val_transform,
                          img_size=args.img_size, mode=args.mode)

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
        )
    else:
        model = SegmentationModule(
            arch=args.arch,
            encoder=args.encoder,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
        )

    # Callbacks
    run_name = f"{args.arch}_{args.encoder}_{args.strategy}_{args.mode}"
    if args.species:
        run_name += f"_{args.species}"
    run_dir = OUTPUT_DIR / "runs" / "unet" / run_name

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

    logger = CSVLogger(str(run_dir), name="logs")

    # Train
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
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
