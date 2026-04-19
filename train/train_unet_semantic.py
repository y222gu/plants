"""U-Net/U-Net++ semantic segmentation — 7-class (bg + 6 anatomical regions).

Single model predicts per-pixel class from 7 mutually exclusive regions derived
by painting raw annotation polygons from largest to smallest:

    0 = background
    1 = epidermis (whole root area not occupied by inner structures)
    2 = aerenchyma (cortex holes)
    3 = endodermis ring (outer endo - inner endo via paint order)
    4 = vascular (inner endo area)
    5 = exodermis ring (outer exo - inner exo via paint order)
    6 = cortex (between inner exo and outer endo)

Post-processing derives 5 target classes:
    Target 0 (Whole Root) = union of all non-background pixels
    Target 1 (Aerenchyma) = semantic class 2
    Target 2 (Endodermis ring) = semantic class 3
    Target 3 (Vascular) = semantic class 4
    Target 4 (Exodermis ring) = semantic class 5

Usage:
    python train/train_unet_semantic.py
    python train/train_unet_semantic.py --arch unetplusplus --encoder resnet50 --epochs 300
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
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
    make_run_subfolder,
    save_hparams,
)
from src.unet_dataset import UNetSemanticDataset
from src.splits import get_split, print_split_summary

NUM_CLASSES = 7  # bg + 6 anatomical regions

# Class names for logging
CLASS_NAMES = ["bg", "epidermis", "aerenchyma", "endo_ring", "vascular", "exo_ring", "cortex"]


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


class SemanticSegModule(pl.LightningModule):
    """7-class semantic segmentation (softmax, Dice+Focal+CE+Lovász)."""

    def __init__(
        self,
        arch: str = "unetplusplus",
        encoder: str = "resnet34",
        encoder_weights: str = "imagenet",
        aer_weight: float = 10.0,
        equal_weights: bool = False,
        no_lovasz: bool = False,
        lr: float = DEFAULT_LR,
        backbone_lr: float = 1e-5,
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

        # NASA MicroNet pretrained weight URLs (no package dependency needed)
        _MICRONET_URLS = {
            "micronet": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/{encoder}_pretrained_microscopynet_v1.0.pth.tar",
            "imagenet-micronet": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/{encoder}_pretrained_imagenet-microscopynet_v1.0.pth.tar",
        }

        model_cls = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "dpt": smp.DPT,
        }[arch]

        if encoder_weights in _MICRONET_URLS:
            self.model = model_cls(
                encoder_name=encoder, encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES,
            )
            url = _MICRONET_URLS[encoder_weights].format(encoder=encoder)
            try:
                pretrained_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                self.model.encoder.load_state_dict(pretrained_dict, strict=False)
                print(f"Loaded {encoder_weights} pretrained weights for {encoder}")
            except Exception as e:
                print(f"Skipping {encoder_weights} download ({e}). Weights will be loaded from checkpoint.")
        else:
            smp_weights = encoder_weights if encoder_weights != "none" else None
            self.model = model_cls(
                encoder_name=encoder, encoder_weights=smp_weights,
                in_channels=3, classes=NUM_CLASSES,
            )

        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.use_lovasz = not no_lovasz
        if self.use_lovasz:
            self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")

        if equal_weights:
            weights = [1.0] * NUM_CLASSES
        else:
            # Class weights: bg=0.5, epidermis=1, aer=configurable, endo=5, vasc=1, exo=5, cortex=1
            weights = [0.5, 1.0, aer_weight, 5.0, 1.0, 5.0, 1.0]
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)  # (B, 7, H, W)
        dice = self.dice_loss(logits, masks)
        focal = self.focal_loss(logits, masks)
        wce = F.cross_entropy(logits, masks, weight=self.class_weights)
        loss = dice + focal + wce
        if self.use_lovasz:
            lovasz = self.lovasz_loss(logits, masks)
            loss = loss + lovasz

        preds = logits.argmax(dim=1)
        correct = (preds == masks).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_dice_loss", dice, sync_dist=True)
        self.log(f"{stage}_focal_loss", focal, sync_dist=True)
        self.log(f"{stage}_pixel_acc", correct, prog_bar=True, sync_dist=True)

        # Per-class accuracy (skip background)
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
        description="U-Net 7-class semantic segmentation (bg + 6 anatomical regions)")
    parser.add_argument("--arch", default="unetplusplus", choices=["unet", "unetplusplus"])
    parser.add_argument("--encoder", default="resnet34",
                        help="Encoder backbone (e.g. resnet50, resnet101, efficientnet-b4)")
    parser.add_argument("--encoder-weights", default="imagenet",
                        help="Encoder pretrained weights: imagenet, micronet, imagenet-micronet, none")
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
    parser.add_argument("--aer-weight", type=float, default=10.0,
                        help="CE class weight for aerenchyma (default: 10)")
    parser.add_argument("--equal-weights", action="store_true",
                        help="Use equal CE weights (1.0) for all classes")
    parser.add_argument("--no-lovasz", action="store_true",
                        help="Disable Lovász loss (use only Dice+Focal+wCE)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--channel-dropout", type=float, default=0.2,
                        help="ChannelDropout probability (0 to disable)")
    parser.add_argument("--channel-shuffle", type=float, default=0.2,
                        help="ChannelShuffle probability (0 to disable)")
    parser.add_argument("--hue-sat", type=float, default=0.0,
                        help="HueSaturationValue probability (0 to disable)")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="A", help="Split strategy")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    split = get_split(args.strategy)
    print_split_summary(split)

    train_transform = get_train_transform(args.img_size, p_channel_dropout=args.channel_dropout, p_channel_shuffle=args.channel_shuffle, p_hue_sat=args.hue_sat)
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

    model = SemanticSegModule(
        arch=args.arch,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
        aer_weight=args.aer_weight,
        equal_weights=args.equal_weights,
        no_lovasz=args.no_lovasz,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
    )

    # Run name: model_pretrain_classweight_augmentation_loss_strategy
    # Model + pretrain
    pretrain_tag = args.encoder_weights.replace("-", "")  # imagenet, imagenetmicronet, none

    # Class weights
    if args.equal_weights:
        weight_tag = "equalw"
    else:
        weight_tag = f"aer{int(args.aer_weight)}" if args.aer_weight != 2.0 else "defaultw"

    # Augmentation
    aug_parts = []
    if args.channel_dropout > 0:
        aug_parts.append("drop")
    if args.channel_shuffle > 0:
        aug_parts.append("shuf")
    if args.hue_sat > 0:
        aug_parts.append("hue")
    aug_tag = "_".join(aug_parts) if aug_parts else "noaug"

    # Loss
    loss_tag = "dfce" if args.no_lovasz else "dfcel"  # d=dice, f=focal, ce=wCE, l=lovasz

    run_name = f"{args.arch}_{args.encoder}_{pretrain_tag}_{weight_tag}_{aug_tag}_{loss_tag}_semantic7c_{args.strategy}"
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
            for key in ["train_loss", "val_loss", "train_pixel_acc", "val_pixel_acc"]:
                if key in metrics:
                    parts.append(f"{key}={metrics[key]:.4f}")
            print(" | ".join(parts), flush=True)

            # Compute per-class IoU/Dice every N epochs
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
                    # Log to TensorBoard
                    if trainer.logger:
                        for lg in (trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]):
                            if hasattr(lg, 'experiment') and hasattr(lg.experiment, 'add_scalar'):
                                lg.experiment.add_scalar(f"val_IoU/{name}", iou, epoch)
                                lg.experiment.add_scalar(f"val_Dice/{name}", dice, epoch)

                # Mean (skip background)
                fg_ious = [class_inter[c] / class_union[c] for c in range(1, NUM_CLASSES)
                           if class_union[c] > 0]
                mean_iou = sum(fg_ious) / len(fg_ious) if fg_ious else float("nan")
                row["mean_IoU"] = round(mean_iou, 4)
                print(f"    Mean (no bg): IoU={mean_iou:.4f}")

                # Save to CSV
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
        deterministic=False,  # cross_entropy has no deterministic CUDA impl
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
        axes[0].set_title(f"{args.arch} ({args.encoder}) [semantic 7c] — Loss")
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
        axes[1].set_title(f"{args.arch} ({args.encoder}) [semantic 7c] — Accuracy")
        axes[1].legend()
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
