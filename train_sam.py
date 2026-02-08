"""SAM (Segment Anything Model) fine-tuning script.

Freezes the image encoder, fine-tunes mask decoder (and optionally prompt encoder).
Uses point and box prompts generated from GT masks.

Usage:
    python train_sam.py --strategy strategy1
    python train_sam.py --strategy strategy1 --unfreeze-prompt-encoder --epochs 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
)
from src.dataset import SampleRegistry
from src.models.sam_dataset import SAMDataset
from src.splits import get_split, print_split_summary

# SAM imports
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(-2, -1))
        total = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2 * intersection + 1) / (total + 1)
        return 1 - dice.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dice_loss_fn: DiceLoss,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["image"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        boxes = batch["box"].to(device)

        # SAM forward (per-sample since SAM processes one image at a time)
        losses = []
        for i in range(images.shape[0]):
            with torch.no_grad():
                image_embedding = model.image_encoder(images[i:i+1])

            # Prompt encoding
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point_coords[i:i+1], point_labels[i:i+1]),
                boxes=boxes[i:i+1, None, :] if boxes is not None else None,
                masks=None,
            )

            # Mask prediction
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Upscale masks to input resolution
            pred_mask = F.interpolate(
                low_res_masks,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (1, H, W)

            target = gt_masks[i:i+1]

            bce = F.binary_cross_entropy_with_logits(pred_mask, target)
            dice = dice_loss_fn(pred_mask, target)
            loss = bce + dice
            losses.append(loss)

        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dice_loss_fn: DiceLoss,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val", leave=False):
        images = batch["image"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        boxes = batch["box"].to(device)

        for i in range(images.shape[0]):
            image_embedding = model.image_encoder(images[i:i+1])

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point_coords[i:i+1], point_labels[i:i+1]),
                boxes=boxes[i:i+1, None, :] if boxes is not None else None,
                masks=None,
            )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_mask = F.interpolate(
                low_res_masks,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            target = gt_masks[i:i+1]
            bce = F.binary_cross_entropy_with_logits(pred_mask, target)
            dice = dice_loss_fn(pred_mask, target)
            total_loss += (bce + dice).item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="SAM fine-tuning")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--sam-type", default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint. Auto-downloaded if None.")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (SAM is memory-heavy)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--unfreeze-prompt-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    train_ds = SAMDataset(split["train"], img_size=args.img_size)
    val_ds = SAMDataset(split["val"], img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Load SAM
    if args.checkpoint is None:
        checkpoint_map = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
        }
        ckpt_name = checkpoint_map[args.sam_type]
        ckpt_path = OUTPUT_DIR / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            print(f"SAM checkpoint not found at {ckpt_path}")
            print(f"Please download from https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return
    else:
        ckpt_path = Path(args.checkpoint)

    model = sam_model_registry[args.sam_type](checkpoint=str(ckpt_path))
    model.to(device)

    # Freeze image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Optionally freeze prompt encoder
    if not args.unfreeze_prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7,
    )
    dice_loss_fn = DiceLoss()
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    run_name = f"sam_{args.sam_type}_{args.strategy}"
    if args.species:
        run_name += f"_{args.species}"
    run_dir = OUTPUT_DIR / "runs" / "sam" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_dl, optimizer, device, dice_loss_fn)
        val_loss = validate(model, val_dl, device, dice_loss_fn)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best.pth")
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Save training history
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {run_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
