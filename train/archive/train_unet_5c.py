"""U-Net++ 5-class multilabel training WITHOUT validity masking.

All 5 target classes (Whole Root, Aerenchyma, Endodermis, Vascular, Exodermis)
contribute to loss for ALL species. This assumes all classes are fully annotated:

- Cereals now have exodermis annotations (classes 4-5 in annotation files)
- Tomato has zero aerenchyma annotations — this is biologically correct
  (tomato has no aerenchyma), NOT missing data. The model learns that
  aerenchyma channel should be all-zeros for tomato.

Usage:
    python train/train_unet_5c.py
    python train/train_unet_5c.py --arch unetplusplus --encoder resnet50 --epochs 300
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
from src.models.unet_dataset import UNetDataset
from src.splits import get_split, print_split_summary

NUM_CLASSES = 5
CLASS_NAMES = ["root", "aer", "endo", "vasc", "exo"]


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


class MultiLabel5CModule(pl.LightningModule):
    """5-class multilabel segmentation — all classes contribute to loss for all species.

    No validity masking. Tomato samples have all-zeros for aerenchyma channel
    (biologically correct), and the model learns this directly.
    """

    def __init__(
        self,
        arch: str = "unet",
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

        model_cls = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
        }[arch]

        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )

        # BCEWithLogitsLoss with class-specific pos_weight
        # root=1 (large area), aer=2 (many small), endo=5 (thin ring), vasc=1, exo=5 (thin ring)
        pw = pos_weight if pos_weight is not None else [1.0, 2.0, 5.0, 1.0, 5.0]
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
        masks = batch["mask"]            # (B, 5, H, W) float32

        logits = self(images)            # (B, 5, H, W)

        # Standard unmasked loss — all 5 classes, all species
        bce = F.binary_cross_entropy_with_logits(
            logits, masks,
            pos_weight=self.pos_weight.view(1, -1, 1, 1),
        )
        dice = self.dice_loss(logits, masks)

        loss = self.bce_weight * bce + self.dice_weight * dice

        # Metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == masks).float().mean()

        # Per-class dice (for monitoring)
        with torch.no_grad():
            for c, name in enumerate(CLASS_NAMES):
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

        param_groups = [
            {"params": encoder_params, "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ]
        optimizer = _build_optimizer(self.optimizer_name, param_groups, self.lr, self.weight_decay)
        scheduler = _build_scheduler(self.scheduler_name, optimizer, self.trainer.max_epochs, self.eta_min)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(
        description="U-Net++ 5-class multilabel training (no validity masking)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory containing image/ and annotation/ subdirectories. "
                             "Defaults to project data/train, data/val, data/test structure.")
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
    parser.add_argument("--eta-min", type=float, default=1e-7,
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("--pos-weight", type=float, nargs="+", default=None,
                        help="Per-class pos_weight for BCE loss. "
                             "Default: 1 2 5 1 5 (root, aer, endo, vasc, exo)")
    parser.add_argument("--bce-weight", type=float, default=1.0,
                        help="Weight for BCE loss term")
    parser.add_argument("--dice-weight", type=float, default=1.0,
                        help="Weight for Dice loss term")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (uses DDP strategy when > 1)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save periodic checkpoint every N epochs (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint (.ckpt) to load model weights from "
                             "(warm-start, resets optimizer/epoch)")
    parser.add_argument("--no-val", action="store_true",
                        help="Train on ALL samples with no validation set. "
                             "No early stopping — runs for exactly --epochs epochs. "
                             "Useful for human-in-the-loop fine-tuning with few samples.")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Setup
    train_transform = get_train_transform(args.img_size)
    val_transform = get_val_transform(args.img_size)

    if args.data_dir:
        # Custom data dir: all samples go to train, no val/test
        from src.dataset import SampleRegistry
        data_dir = Path(args.data_dir)
        registry = SampleRegistry(data_dir=data_dir)
        split = {"train": registry.samples, "val": [], "test": []}
        print(f"\nCustom data dir: {len(registry.samples)} samples for training")
    else:
        split = get_split()

    if args.no_val:
        # No validation: train on all samples, fixed epochs, no early stopping
        all_samples = split["train"] + split["val"] + split["test"]
        print(f"\n--no-val: Training on ALL {len(all_samples)} samples, "
              f"no validation, {args.epochs} epochs")
        split = {"train": all_samples, "val": [], "test": []}

    print_split_summary(split)

    # Datasets
    train_ds = UNetDataset(split["train"], transform=train_transform,
                           img_size=args.img_size, mode="multilabel",
                           num_classes=NUM_CLASSES, mask_missing=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=len(split["train"]) > args.batch_size)

    val_dl = None
    test_dl = None
    if split["val"]:
        val_ds = UNetDataset(split["val"], transform=val_transform,
                             img_size=args.img_size, mode="multilabel",
                             num_classes=NUM_CLASSES, mask_missing=False)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    if split["test"]:
        test_ds = UNetDataset(split["test"], transform=val_transform,
                              img_size=args.img_size, mode="multilabel",
                              num_classes=NUM_CLASSES, mask_missing=False)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    model = MultiLabel5CModule(
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

    # Load pretrained weights (warm-start: loads model weights only, resets optimizer/epoch)
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        print(f"\nLoading pretrained weights from: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # Extract model weights, skip non-model keys (pos_weight may differ)
        state_dict = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(state_dict)} weight tensors from checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Callbacks
    run_name = f"{args.arch}_{args.encoder}_multilabel_c5_full"
    base_run_dir = OUTPUT_DIR / "runs" / "unet" / run_name
    run_dir = make_run_subfolder(base_run_dir)
    save_hparams(run_dir, args)

    callbacks = []
    if val_dl is not None:
        # With validation: monitor val_loss for best checkpoint + early stopping
        callbacks.append(ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ))
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ))
    else:
        # No validation: save best by train_loss + save last, no early stopping
        callbacks.append(ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="best-{epoch}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ))

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
        log_every_n_steps=min(10, len(train_dl)),
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
        axes[0].set_title(f"{args.arch} ({args.encoder}) [5-class full] — Loss")
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
        axes[1].set_title(f"{args.arch} ({args.encoder}) [5-class full] — Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = run_dir / "loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to {plot_path}")

    # Test
    if test_dl is not None:
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        test_results = trainer.test(model, test_dl, ckpt_path="best")
        print(f"Test results: {test_results}")
    else:
        print("\n" + "=" * 60)
        print("No test set — skipping evaluation")
        print(f"Final checkpoint saved to: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
