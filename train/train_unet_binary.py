"""U-Net/U-Net++ multilabel segmentation — 6 raw annotation classes.

Single model with 6 sigmoid output channels predicting all raw annotation
classes simultaneously. Channels can overlap (e.g., outer endodermis polygon
contains inner endodermis area). Post-processing derives 5 target classes
by subtracting inner from outer rings via raw_to_target.

Raw annotation classes (output channels):
    0 = Whole Root
    1 = Aerenchyma
    2 = Outer Endodermis
    3 = Inner Endodermis
    4 = Outer Exodermis
    5 = Inner Exodermis

Usage:
    python train/train_unet_binary.py
    python train/train_unet_binary.py --arch unetplusplus --encoder resnet50
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
    ANNOTATED_CLASSES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
    make_run_subfolder,
    save_hparams,
)
from src.models.unet_dataset import UNetMultilabelDataset
from src.splits import get_split, print_split_summary

NUM_CLASSES = 6  # raw annotation classes 0-5

# Per-class pos_weight: higher for thin/rare structures
DEFAULT_POS_WEIGHTS = [1.0, 2.0, 5.0, 1.0, 5.0, 1.0]
#                      Root  Aer  O.Endo I.Endo O.Exo I.Exo


# ── Optimizer / scheduler factories ──────────────────────────────────────────

def _build_optimizer(name: str, param_groups: list, lr: float, weight_decay: float):
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay,
                               momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(name: str, optimizer, max_epochs: int, eta_min: float):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=eta_min)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(max_epochs // 3, 1), gamma=0.1)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10)
    raise ValueError(f"Unknown scheduler: {name}")


class MultilabelSegModule(pl.LightningModule):
    """6-channel multilabel segmentation (BCE + Dice per channel)."""

    def __init__(
        self,
        arch: str = "unetplusplus",
        encoder: str = "resnet34",
        lr: float = DEFAULT_LR,
        backbone_lr: float = 1e-5,
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

        model_cls = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}[arch]
        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )

        pw = pos_weight if pos_weight is not None else DEFAULT_POS_WEIGHTS
        self.register_buffer("pos_weight", torch.tensor(pw, dtype=torch.float32))
        self.dice_loss = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]   # (B, 3, H, W)
        masks = batch["mask"]     # (B, 6, H, W)

        logits = self(images)     # (B, 6, H, W)

        bce = F.binary_cross_entropy_with_logits(
            logits, masks, pos_weight=self.pos_weight.view(1, -1, 1, 1))
        dice = self.dice_loss(logits, masks)
        loss = self.bce_weight * bce + self.dice_weight * dice

        preds = (torch.sigmoid(logits) > 0.5).float()

        # Per-channel dice scores
        with torch.no_grad():
            for c in range(NUM_CLASSES):
                inter = (preds[:, c] * masks[:, c]).sum()
                denom = preds[:, c].sum() + masks[:, c].sum()
                dice_c = (2 * inter / denom) if denom > 0 else torch.tensor(1.0)
                self.log(f"{stage}_dice_{ANNOTATED_CLASSES[c]}", dice_c, sync_dist=True)

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_bce_loss", bce, sync_dist=True)
        self.log(f"{stage}_dice_loss", dice, sync_dist=True)
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
        param_groups = [
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ]
        optimizer = _build_optimizer(self.optimizer_name, param_groups, self.lr, self.weight_decay)
        scheduler = _build_scheduler(self.scheduler_name, optimizer, self.trainer.max_epochs, self.eta_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(
        description="U-Net 6-channel multilabel segmentation (raw annotation classes)")
    parser.add_argument("--arch", default="unetplusplus", choices=["unet", "unetplusplus"])
    parser.add_argument("--encoder", default="resnet34",
                        help="Encoder backbone (e.g. resnet50, resnet101, efficientnet-b4)")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau"])
    parser.add_argument("--eta-min", type=float, default=1e-7)
    parser.add_argument("--pos-weight", type=float, nargs=6, default=None,
                        help="BCE pos_weight for each of 6 channels (default: [1,2,5,1,5,1])")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="A", help="Split strategy")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    split = get_split(args.strategy)
    print_split_summary(split)

    train_transform = get_train_transform(args.img_size)
    val_transform = get_val_transform(args.img_size)

    train_ds = UNetMultilabelDataset(split["train"], transform=train_transform,
                                      img_size=args.img_size)
    val_ds = UNetMultilabelDataset(split["val"], transform=val_transform,
                                    img_size=args.img_size)
    test_ds = UNetMultilabelDataset(split["test"], transform=val_transform,
                                     img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    model = MultilabelSegModule(
        arch=args.arch,
        encoder=args.encoder,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
        pos_weight=args.pos_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )

    run_name = f"{args.arch}_{args.encoder}_multilabel_{args.strategy}"
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

    # Plot loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_file = Path(logger.log_dir) / "metrics.csv"
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
        axes[0].set_title(f"{args.arch} ({args.encoder}) [multilabel 6c] — Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Per-class dice scores
        for c in range(NUM_CLASSES):
            col = f"val_dice_{ANNOTATED_CLASSES[c]}"
            if col in df.columns:
                vals = df[df[col].notna()]
                axes[1].plot(vals["epoch"], vals[col], label=ANNOTATED_CLASSES[c])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_title(f"{args.arch} ({args.encoder}) [multilabel 6c] — Dice")
        axes[1].legend(fontsize=8)
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
