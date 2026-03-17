"""SAM (Segment Anything Model) fine-tuning script.

Runs frozen image encoder on-the-fly (no pre-computation) so that full
image augmentation (flips, rotations, noise, channel dropout) is applied
every epoch.  Supports multi-GPU via DistributedDataParallel (DDP).

Usage:
    # Single GPU
    python train_sam.py --strategy A

    # Multi-GPU (e.g. 2 GPUs)
    torchrun --nproc_per_node=2 train_sam.py --strategy A
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.augmentation import get_train_transform, get_val_transform, apply_transform_with_masks
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
from src.models.sam_dataset import SAMDataset
from src.splits import get_split, print_split_summary

# SAM imports
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(-2, -1))
        total = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2 * intersection + 1) / (total + 1)
        return 1 - dice.mean()


# ── Augmented SAM Dataset ────────────────────────────────────────────────────

class AugmentedSAMDataset(SAMDataset):
    """SAMDataset with on-the-fly albumentations augmentation.

    Applies spatial + photometric augmentations to the image and all instance
    masks together, then picks the requested instance.
    """

    def __init__(self, samples, img_size=1024, points_per_mask=3,
                 use_box_prompt=True, use_point_prompt=True,
                 cache_size=64, num_classes=4, augment=True):
        super().__init__(
            samples, img_size=img_size, points_per_mask=points_per_mask,
            use_box_prompt=use_box_prompt, use_point_prompt=use_point_prompt,
            cache_size=cache_size, num_classes=num_classes,
        )
        self.augment = augment
        self._train_tf = get_train_transform(img_size) if augment else None
        self._val_tf = get_val_transform(img_size)

    def __getitem__(self, idx):
        sample_idx, instance_idx = self._index[idx]
        img, ann = self._load_sample(sample_idx)
        h, w = img.shape[:2]

        all_masks = ann["masks"]  # (N, H, W) uint8
        gt_mask_full = all_masks[instance_idx]

        # Apply augmentation to image + all masks together (keeps spatial consistency)
        if self.augment and self._train_tf is not None:
            img_aug, masks_aug = apply_transform_with_masks(
                self._train_tf, img, all_masks,
            )
            gt_mask = masks_aug[instance_idx]
        else:
            result = self._val_tf(image=img, mask=gt_mask_full)
            img_aug = result["image"]
            gt_mask = result["mask"]

        result = {
            "image": torch.from_numpy(img_aug.copy()).permute(2, 0, 1).float(),
            "gt_mask": torch.from_numpy(gt_mask.copy()).float(),
            "class_id": torch.tensor(ann["labels"][instance_idx], dtype=torch.long),
        }

        # Generate random point prompts from augmented mask
        if self.use_point_prompt:
            ys, xs = np.where(gt_mask > 0)
            if len(xs) >= self.points_per_mask:
                indices = np.random.choice(len(xs), self.points_per_mask, replace=False)
                points = np.stack([xs[indices], ys[indices]], axis=-1).astype(np.float32)
            elif len(xs) > 0:
                points = np.stack([xs, ys], axis=-1).astype(np.float32)
                while len(points) < self.points_per_mask:
                    points = np.concatenate([points, points[:1]])
                points = points[:self.points_per_mask]
            else:
                points = np.zeros((self.points_per_mask, 2), dtype=np.float32)
            point_labels = np.ones(self.points_per_mask, dtype=np.float32)
            result["point_coords"] = torch.from_numpy(points)
            result["point_labels"] = torch.from_numpy(point_labels)

        # Generate random box with jitter from augmented mask
        if self.use_box_prompt:
            ys, xs = np.where(gt_mask > 0)
            if len(xs) > 0:
                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                bw, bh = x2 - x1, y2 - y1
                jitter = 0.05
                x1 = max(0, x1 - random.uniform(0, jitter * bw))
                y1 = max(0, y1 - random.uniform(0, jitter * bh))
                x2 = min(self.img_size, x2 + random.uniform(0, jitter * bw))
                y2 = min(self.img_size, y2 + random.uniform(0, jitter * bh))
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
            else:
                box = np.zeros(4, dtype=np.float32)
            result["box"] = torch.from_numpy(box)

        return result


# ── Training / Validation ────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, dice_loss_fn, scaler,
                    bce_weight=1.0, dice_weight=1.0):
    model.train()
    # Keep image encoder in eval mode (frozen, batch norm etc.)
    mod = model.module if isinstance(model, DDP) else model
    mod.image_encoder.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False,
                      disable=not is_main_process()):
        images = batch["image"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        boxes = batch["box"].to(device)

        losses = []
        with torch.amp.autocast("cuda"):
            # Run frozen encoder on the batch (no grad flows, but forward runs)
            with torch.no_grad():
                embeddings = mod.image_encoder(images)

            for i in range(embeddings.shape[0]):
                sparse_emb, dense_emb = mod.prompt_encoder(
                    points=(point_coords[i:i+1], point_labels[i:i+1]),
                    boxes=boxes[i:i+1, None, :],
                    masks=None,
                )
                low_res_masks, _ = mod.mask_decoder(
                    image_embeddings=embeddings[i:i+1],
                    image_pe=mod.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )

                # Upsample prediction to match GT (1024x1024)
                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(gt_masks.shape[-2], gt_masks.shape[-1]),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                target = gt_masks[i:i+1]
                bce = F.binary_cross_entropy_with_logits(pred_mask, target)
                dice = dice_loss_fn(pred_mask, target)
                losses.append(bce_weight * bce + dice_weight * dice)

            batch_loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device, dice_loss_fn,
             bce_weight=1.0, dice_weight=1.0):
    mod = model.module if isinstance(model, DDP) else model
    mod.eval()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Val", leave=False,
                      disable=not is_main_process()):
        images = batch["image"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        boxes = batch["box"].to(device)

        with torch.amp.autocast("cuda"):
            embeddings = mod.image_encoder(images)

            for i in range(embeddings.shape[0]):
                sparse_emb, dense_emb = mod.prompt_encoder(
                    points=(point_coords[i:i+1], point_labels[i:i+1]),
                    boxes=boxes[i:i+1, None, :],
                    masks=None,
                )
                low_res_masks, _ = mod.mask_decoder(
                    image_embeddings=embeddings[i:i+1],
                    image_pe=mod.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(gt_masks.shape[-2], gt_masks.shape[-1]),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                target = gt_masks[i:i+1]
                bce = F.binary_cross_entropy_with_logits(pred_mask, target)
                dice = dice_loss_fn(pred_mask, target)
                total_loss += (bce_weight * bce + dice_weight * dice).item()
                n_samples += 1

    return total_loss / max(n_samples, 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM fine-tuning")
    parser.add_argument("--strategy", default="A",
                        choices=["A", "B", "C"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--sam-type", default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint. Auto-downloaded if None.")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="SAM requires 1024x1024 input (fixed positional embeddings)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau"])
    parser.add_argument("--eta-min", type=float, default=1e-7,
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("--bce-weight", type=float, default=1.0,
                        help="Weight for BCE loss term")
    parser.add_argument("--dice-weight", type=float, default=1.0,
                        help="Weight for Dice loss term")
    parser.add_argument("--num-classes", type=int, default=4, choices=[4, 5],
                        help="Number of target classes (4=standard, 5=with exodermis)")
    parser.add_argument("--unfreeze-prompt-encoder", action="store_true")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save periodic checkpoint every N epochs (0 to disable)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers per GPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── DDP setup ──
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        local_rank = 0
        world_size = 1

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. SAM training requires a GPU.")

    if is_main_process():
        print(f"Using {world_size} GPU(s)")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA version: {torch.version.cuda}")

    # ── Load SAM ──
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.1f}%)")

    # Wrap in DDP (only trainable params need gradient sync)
    if distributed:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)

    # ── Data ──
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    if is_main_process():
        print_split_summary(split)

    train_ds = AugmentedSAMDataset(
        split["train"], img_size=args.img_size, num_classes=args.num_classes,
        augment=True,
    )
    val_ds = AugmentedSAMDataset(
        split["val"], img_size=args.img_size, num_classes=args.num_classes,
        augment=False,
    )

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Optimizer / Scheduler ──
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=0.9)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10)
    dice_loss_fn = DiceLoss()
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ──
    run_name = f"sam_{args.sam_type}_{args.strategy}_c{args.num_classes}"
    if args.species:
        run_name += f"_{args.species}"
    base_run_dir = OUTPUT_DIR / "runs" / "sam" / run_name
    if is_main_process():
        run_dir = make_run_subfolder(base_run_dir)
        save_hparams(run_dir, args)
    else:
        run_dir = base_run_dir / "placeholder"

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_dl, optimizer, device, dice_loss_fn, scaler,
                                     bce_weight=args.bce_weight, dice_weight=args.dice_weight)
        val_loss = validate(model, val_dl, device, dice_loss_fn,
                            bce_weight=args.bce_weight, dice_weight=args.dice_weight)
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if is_main_process():
            print(f"  Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

            # Checkpointing (only save from rank 0)
            mod = model.module if isinstance(model, DDP) else model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(mod.state_dict(), run_dir / "best.pth")
                print(f"  Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(mod.state_dict(), run_dir / f"epoch_{epoch+1}.pth")
                print(f"  Saved periodic checkpoint (epoch {epoch+1})")

    # ── Save history + plot (rank 0 only) ──
    if is_main_process():
        with open(run_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

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

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
