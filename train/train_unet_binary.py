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
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
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
from src.unet_dataset import UNetMultilabelDataset
from src.splits import get_split, print_split_summary

NUM_CLASSES = 6  # raw annotation classes 0-5

# Per-class pos_weight: higher for thin/rare structures
DEFAULT_POS_WEIGHTS = [1.0, 10.0, 5.0, 1.0, 5.0, 1.0]
#                      Root  Aer   O.Endo I.Endo O.Exo I.Exo


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
    parser.add_argument("--batch-size", type=int, default=8)
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

    # Run name: model_pretrain_classweight_augmentation_loss_strategy
    pw = args.pos_weight if args.pos_weight else [1.0, 10.0, 5.0, 1.0, 5.0, 1.0]
    weight_tag = "equalw" if all(w == 1.0 for w in pw) else "defaultw"

    run_name = f"{args.arch}_{args.encoder}_imagenet_{weight_tag}_fullaug_bcedice_multilabel_{args.strategy}"
    base_run_dir = OUTPUT_DIR / "runs" / "unet" / run_name
    run_dir = make_run_subfolder(base_run_dir)
    save_hparams(run_dir, args)

    class EpochLogger(Callback):
        """Print epoch summary + compute val IoU/Dice every N epochs."""
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
            for key in ["train_loss", "val_loss", "train_bce_loss", "train_dice_loss"]:
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
                        preds = (torch.sigmoid(pl_module(images)) > 0.5).float()
                        for c in range(NUM_CLASSES):
                            gt_c = masks[:, c].bool()
                            pred_c = preds[:, c].bool()
                            class_inter[c] += int((gt_c & pred_c).sum())
                            class_union[c] += int((gt_c | pred_c).sum())
                            class_pred_sum[c] += int(pred_c.sum())
                            class_gt_sum[c] += int(gt_c.sum())

                pl_module.train()

                row = {"epoch": epoch + 1}
                for c in range(NUM_CLASSES):
                    name = ANNOTATED_CLASSES[c]
                    iou = class_inter[c] / class_union[c] if class_union[c] > 0 else float("nan")
                    denom = class_pred_sum[c] + class_gt_sum[c]
                    dice = 2 * class_inter[c] / denom if denom > 0 else float("nan")
                    row[f"{name}_IoU"] = round(iou, 4)
                    row[f"{name}_Dice"] = round(dice, 4)
                    print(f"    {name:25s}  IoU={iou:.4f}  Dice={dice:.4f}")
                    if trainer.logger:
                        for lg in (trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]):
                            if hasattr(lg, 'experiment') and hasattr(lg.experiment, 'add_scalar'):
                                lg.experiment.add_scalar(f"val_IoU/{name}", iou, epoch)
                                lg.experiment.add_scalar(f"val_Dice/{name}", dice, epoch)

                valid_ious = [class_inter[c] / class_union[c] for c in range(NUM_CLASSES)
                              if class_union[c] > 0]
                mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")
                row["mean_IoU"] = round(mean_iou, 4)
                print(f"    Mean IoU={mean_iou:.4f}")

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

    csv_logger = logger[0] if isinstance(logger, list) else logger
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

    # Test — evaluate without logging to TensorBoard (avoid polluting training curves)
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    trainer.logger = False
    test_results = trainer.test(model, test_dl, ckpt_path="best")
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
