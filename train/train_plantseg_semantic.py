"""Fine-tune PanSeg/PlantSeg UNet2D for 7-class semantic segmentation.

Same architecture as train_plantseg.py (multilabel) but with semantic
(softmax, mutually exclusive classes) formulation. Uses the same pretrained
UNet2D from bioimage.io (laid-back-lobster) with 3-channel input and
weight replication.

7 semantic classes (same as U-Net++ semantic):
    0 = background
    1 = epidermis
    2 = aerenchyma
    3 = endodermis ring
    4 = vascular
    5 = exodermis ring
    6 = cortex

Loss: Dice + Focal + weighted CE + Lovász (same as U-Net++ semantic)

Usage:
    python train/train_plantseg_semantic.py
    python train/train_plantseg_semantic.py --batch-size 8 --aer-weight 10
    python train/train_plantseg_semantic.py --no-pretrained
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from pytorch3dunet.unet3d.model import UNet2D

from src.augmentation import get_train_transform, get_val_transform
from src.config import (
    ANNOTATED_CLASSES,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
    make_run_subfolder,
    save_hparams,
)
from src.unet_dataset import UNetSemanticDataset
from src.splits import get_split, print_split_summary

import segmentation_models_pytorch as smp

NUM_CLASSES = 7  # bg + 6 anatomical regions
CLASS_NAMES = ["bg", "epidermis", "aerenchyma", "endo_ring", "vascular", "exo_ring", "cortex"]

PRETRAINED_MODEL_ID = "laid-back-lobster"


def load_pretrained_weights(model, model_id):
    """Load pretrained PanSeg UNet weights with channel replication.

    Pretrained: in_channels=1, out_channels=1.
    Our model: in_channels=3, out_channels=7.
    First conv: replicated 3x ÷ 3. Last conv: skipped (random init).
    All other layers: loaded exactly.
    """
    from bioimageio.core import predict as _unused
    from bioimageio.spec import load_model_description
    import glob, os

    md = load_model_description(model_id)

    cache_dir = os.path.expanduser("~/.cache/bioimageio")
    ts_files = sorted(glob.glob(f"{cache_dir}/**/torchscript_tracing.pt", recursive=True))
    pt_files = sorted(glob.glob(f"{cache_dir}/**/*.pytorch", recursive=True))

    pretrained_dict = None
    if pt_files:
        print(f"  Loading pytorch weights from {pt_files[-1]}")
        pretrained_dict = torch.load(pt_files[-1], map_location="cpu", weights_only=True)
    elif ts_files:
        print(f"  Loading torchscript weights from {ts_files[-1]}")
        ts_model = torch.jit.load(ts_files[-1], map_location="cpu")
        pretrained_dict = ts_model.state_dict()
    else:
        import numpy as np, xarray as xr
        dummy = xr.DataArray(np.zeros((1, 1, 64, 64), dtype=np.float32),
                             dims=["batch", "channel", "y", "x"])
        from bioimageio.core import predict as biio_predict
        biio_predict(model=model_id, inputs=dummy)
        ts_files = sorted(glob.glob(f"{cache_dir}/**/torchscript_tracing.pt", recursive=True))
        pt_files = sorted(glob.glob(f"{cache_dir}/**/*.pytorch", recursive=True))
        if pt_files:
            pretrained_dict = torch.load(pt_files[-1], map_location="cpu", weights_only=True)
        elif ts_files:
            ts_model = torch.jit.load(ts_files[-1], map_location="cpu")
            pretrained_dict = ts_model.state_dict()

    if pretrained_dict is None:
        print("  WARNING: Could not load pretrained weights")
        return

    model_dict = model.state_dict()

    loaded, replicated, skipped_missing, skipped_shape = 0, 0, 0, 0
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            skipped_missing += 1
            continue

        target = model_dict[name]
        if param.shape == target.shape:
            model_dict[name] = param
            loaded += 1
        elif (param.dim() >= 2 and target.dim() >= 2
              and param.shape[0] == target.shape[0]
              and param.shape[1] == 1 and target.shape[1] == 3):
            model_dict[name] = param.repeat(1, 3, 1, 1) / 3.0
            replicated += 1
            print(f"  Replicated: {name} {param.shape} → {target.shape} (÷3)")
        else:
            skipped_shape += 1
            print(f"  Skipped (shape mismatch): {name} {param.shape} → {target.shape}")

    model.load_state_dict(model_dict)
    print(f"  Loaded {loaded}, replicated {replicated}, "
          f"skipped {skipped_missing} missing + {skipped_shape} shape mismatch")


class PlantSegSemanticModule(pl.LightningModule):
    """7-class semantic segmentation using PanSeg UNet2D architecture."""

    def __init__(
        self,
        pretrained: bool = True,
        aer_weight: float = 10.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        eta_min: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.eta_min = eta_min

        # PanSeg UNet2D: 3 input channels, 7 output channels (softmax), no final activation
        self.model = UNet2D(
            in_channels=3,
            out_channels=NUM_CLASSES,
            final_sigmoid=False,
            f_maps=64,
            layer_order="gcr",
            num_groups=8,
            num_levels=4,
            is_segmentation=True,
            conv_padding=1,
            conv_upscale=2,
            dropout_prob=0.1,
        )

        if pretrained:
            print("Loading pretrained PanSeg weights...")
            load_pretrained_weights(self.model, PRETRAINED_MODEL_ID)

        # Same loss as U-Net++ semantic: Dice + Focal + weighted CE + Lovász
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")

        weights = [0.5, 1.0, aer_weight, 5.0, 1.0, 5.0, 1.0]
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)  # raw logits (B, 7, H, W)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                       weight_decay=self.weight_decay)
        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.eta_min)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(self.trainer.max_epochs // 3, 1), gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class EpochLogger(Callback):
    """Print epoch summary + compute val IoU every N epochs."""

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


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune PanSeg UNet2D for 7-class semantic segmentation")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aer-weight", type=float, default=10.0,
                        help="CE class weight for aerenchyma (default: 10)")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step"])
    parser.add_argument("--eta-min", type=float, default=1e-7)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="A")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    split = get_split(args.strategy)
    print_split_summary(split)

    train_transform = get_train_transform(args.img_size)
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

    model = PlantSegSemanticModule(
        pretrained=not args.no_pretrained,
        aer_weight=args.aer_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total_params:,}")

    pretrained_tag = "scratch" if args.no_pretrained else "plantseg"
    run_name = f"plantseg_unet2d_{pretrained_tag}_3ch_semantic7c_{args.strategy}"
    base_run_dir = OUTPUT_DIR / "runs" / "plantseg" / run_name
    run_dir = make_run_subfolder(base_run_dir)
    save_hparams(run_dir, args)

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
        axes[0].set_title("PanSeg UNet2D (3ch, semantic) — Loss")
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
        axes[1].set_title("PanSeg UNet2D (3ch, semantic) — Accuracy")
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
