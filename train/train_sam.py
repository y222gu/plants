"""SAM (Segment Anything Model) fine-tuning script.

Freezes the image encoder, pre-computes embeddings, fine-tunes mask decoder
(and optionally prompt encoder). Uses point and box prompts from GT masks.

Usage:
    python train_sam.py --strategy strategy1
    python train_sam.py --strategy strategy1 --unfreeze-prompt-encoder --epochs 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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


class EmbeddingDataset(Dataset):
    """Lightweight dataset that serves pre-computed image embeddings."""

    def __init__(self, embeddings, gt_masks, point_coords, point_labels, boxes):
        self.embeddings = embeddings
        self.gt_masks = gt_masks
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.boxes = boxes

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "gt_mask": self.gt_masks[idx],
            "point_coords": self.point_coords[idx],
            "point_labels": self.point_labels[idx],
            "box": self.boxes[idx],
        }


@torch.no_grad()
def precompute_embeddings(model, dataset, device, batch_size=4):
    """Run frozen image encoder once on all images, cache embeddings."""
    model.eval()
    # Use a DataLoader just for image loading
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    embeddings = []
    gt_masks = []
    point_coords_list = []
    point_labels_list = []
    boxes_list = []

    print(f"Pre-computing image embeddings for {len(dataset)} instances...")
    for batch in tqdm(loader, desc="Encoding"):
        images = batch["image"].to(device)
        # Encode each image in the batch
        for i in range(images.shape[0]):
            with torch.amp.autocast("cuda"):
                emb = model.image_encoder(images[i:i+1])
            embeddings.append(emb.cpu())

        gt_masks.append(batch["gt_mask"])
        point_coords_list.append(batch["point_coords"])
        point_labels_list.append(batch["point_labels"])
        boxes_list.append(batch["box"])

    embeddings = torch.cat(embeddings, dim=0)
    gt_masks = torch.cat(gt_masks, dim=0)
    point_coords = torch.cat(point_coords_list, dim=0)
    point_labels = torch.cat(point_labels_list, dim=0)
    boxes = torch.cat(boxes_list, dim=0)

    print(f"  Embeddings shape: {embeddings.shape}, cached {embeddings.element_size() * embeddings.nelement() / 1e9:.1f} GB")
    return EmbeddingDataset(embeddings, gt_masks, point_coords, point_labels, boxes)


def train_one_epoch(model, dataloader, optimizer, device, dice_loss_fn, scaler):
    model.train()
    # Keep image encoder in eval mode (frozen)
    model.image_encoder.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        embeddings = batch["embedding"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        boxes = batch["box"].to(device)

        losses = []
        with torch.amp.autocast("cuda"):
            for i in range(embeddings.shape[0]):
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(point_coords[i:i+1], point_labels[i:i+1]),
                    boxes=boxes[i:i+1, None, :],
                    masks=None,
                )

                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=embeddings[i:i+1],
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(gt_masks.shape[-2], gt_masks.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                target = gt_masks[i:i+1]
                bce = F.binary_cross_entropy_with_logits(pred_mask, target)
                dice = dice_loss_fn(pred_mask, target)
                losses.append(bce + dice)

            batch_loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device, dice_loss_fn):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val", leave=False):
        embeddings = batch["embedding"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        boxes = batch["box"].to(device)

        for i in range(embeddings.shape[0]):
            with torch.amp.autocast("cuda"):
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(point_coords[i:i+1], point_labels[i:i+1]),
                    boxes=boxes[i:i+1, None, :],
                    masks=None,
                )

                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=embeddings[i:i+1],
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(gt_masks.shape[-2], gt_masks.shape[-1]),
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
    parser.add_argument("--img-size", type=int, default=1024,
                        help="SAM requires 1024x1024 input (fixed positional embeddings)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (embeddings are small, can use larger batches)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--unfreeze-prompt-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. SAM training requires a GPU.")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Check SAM checkpoint exists BEFORE loading data
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

    # Load SAM
    model = sam_model_registry[args.sam_type](checkpoint=str(ckpt_path))
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # Freeze image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Optionally freeze prompt encoder
    if not args.unfreeze_prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")

    # Setup data
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    train_ds = SAMDataset(split["train"], img_size=args.img_size)
    val_ds = SAMDataset(split["val"], img_size=args.img_size)

    # Pre-compute image embeddings (one-time cost, skips encoder during training)
    train_emb_ds = precompute_embeddings(model, train_ds, device, batch_size=4)
    val_emb_ds = precompute_embeddings(model, val_ds, device, batch_size=4)

    train_dl = DataLoader(train_emb_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_emb_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    # Optimizer
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

        train_loss = train_one_epoch(model, train_dl, optimizer, device, dice_loss_fn, scaler)
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

    # Plot loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"SAM ({args.sam_type}) Training Loss — {args.strategy}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {run_dir / 'loss_curve.png'}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {run_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
