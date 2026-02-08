"""U-Net / U-Net++ semantic segmentation training with SMP + PyTorch Lightning.

Usage:
    python train_unet.py --strategy strategy1
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
    """PyTorch Lightning module for semantic segmentation."""

    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet50",
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

        # Dice + Focal loss
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            classes=NUM_SEMANTIC_CLASSES,
        )
        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass",
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)  # (B, C, H, W)
        dice = self.dice_loss(logits, masks)
        focal = self.focal_loss(logits, masks)
        loss = dice + focal

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


def main():
    parser = argparse.ArgumentParser(description="U-Net semantic segmentation training")
    parser.add_argument("--arch", default="unet", choices=["unet", "unetplusplus"])
    parser.add_argument("--encoder", default="resnet50",
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

    train_ds = UNetDataset(split["train"], transform=train_transform, img_size=args.img_size)
    val_ds = UNetDataset(split["val"], transform=val_transform, img_size=args.img_size)
    test_ds = UNetDataset(split["test"], transform=val_transform, img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # Model
    model = SegmentationModule(
        arch=args.arch,
        encoder=args.encoder,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
    )

    # Callbacks
    run_name = f"{args.arch}_{args.encoder}_{args.strategy}"
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

    # Test
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    test_results = trainer.test(model, test_dl, ckpt_path="best")
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
