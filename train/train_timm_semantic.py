"""Semantic segmentation with timm/HF encoders + custom decoders.

Supports:
  - Hierarchical encoders (Swin, ConvNeXt) via timm + UNet++ decoder
  - Plain ViT encoders (DINOv2) via timm + DPT / ms-linear / SegDINO-MLP decoders
  - DINOv3 via HuggingFace transformers (patch-16, with register tokens)

Usage:
    # DINOv2-S + custom DPT
    python train/train_timm_semantic.py --encoder vit_small_patch14_dinov2.lvd142m --decoder dpt

    # DINOv2-S + Meta official +ms linear head
    python train/train_timm_semantic.py --encoder vit_small_patch14_dinov2.lvd142m --decoder ms_linear

    # DINOv3-S + custom DPT
    python train/train_timm_semantic.py --encoder hf:facebook/dinov3-vits16-pretrain-lvd1689m --decoder dpt

    # DINOv3-S + SegDINO-style MLP
    python train/train_timm_semantic.py --encoder hf:facebook/dinov3-vits16-pretrain-lvd1689m --decoder segdino_mlp
"""
import argparse, os, csv, sys
from pathlib import Path
from datetime import datetime

# Add project root to path (for running as `python train/train_timm_semantic.py`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.config import OUTPUT_DIR
from src.splits import get_split
from src.augmentation import get_train_transform, get_val_transform
from src.unet_dataset import UNetSemanticDataset

import segmentation_models_pytorch as smp

NUM_CLASSES = 7
DEFAULT_LR = 1e-4


def _tokens_to_bchw(feat):
    """Convert (B, N, C), (B, H, W, C), or (B, C, H, W) to (B, C, H, W).

    Heuristic for 4D inputs with H == W (common case): channels-last has
    shape[-1] = C which is usually much larger than the spatial dims, while
    channels-first has shape[1] = C large. So "last-dim is the biggest" ⇒
    channels-last. Fall-through default: assume already channels-first.
    """
    if feat.dim() == 3:
        B, N, C = feat.shape
        h = w = int(N ** 0.5)
        return feat.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
    if feat.dim() == 4:
        _, d1, d2, d3 = feat.shape
        # channels-last iff last dim > middle dims (assuming square spatial)
        if d3 > d1 and d3 > d2:
            return feat.permute(0, 3, 1, 2).contiguous()
    return feat


# ══════════════════════════════════════════════════════════════════════
# HuggingFace DINOv3 backbone wrapper (timm-compatible API)
# ══════════════════════════════════════════════════════════════════════

class HFViTBackbone(nn.Module):
    """Wrap a HuggingFace ViT-style model to expose a timm-like
    `forward_intermediates(x, indices=..., norm=True)` returning a list of
    patch-token tensors at the requested block indices."""

    def __init__(self, hf_name, pretrained=True):
        super().__init__()
        from transformers import AutoModel
        kwargs = {"output_hidden_states": True}
        if not pretrained:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(hf_name)
            self.model = AutoModel.from_config(cfg)
        else:
            self.model = AutoModel.from_pretrained(hf_name, **kwargs)
        cfg = self.model.config
        self.embed_dim = cfg.hidden_size
        self.num_blocks = cfg.num_hidden_layers
        # Number of prefix tokens (CLS + register tokens)
        self.num_register = getattr(cfg, "num_register_tokens", 0)
        self.num_prefix = 1 + self.num_register  # CLS + registers
        # patch size
        ps = getattr(cfg, "patch_size", 16)
        self.patch_size = (ps, ps) if isinstance(ps, int) else tuple(ps)
        # Fake `blocks` and `patch_embed.patch_size` for code that reads them
        self.blocks = self.model.encoder.layer if hasattr(self.model, "encoder") else []

    def forward_intermediates(self, x, indices, return_prefix_tokens=False,
                              intermediates_only=True, norm=True):
        """Return a list of (B, N, C) patch-token tensors at block `indices`.

        Matches timm's API contract for the subset of args we use.
        """
        out = self.model(pixel_values=x, output_hidden_states=True)
        hidden = out.hidden_states  # tuple, length = num_blocks + 1
        feats = []
        for i in indices:
            h = hidden[i + 1]  # after block i (0-indexed)
            # Strip CLS + register tokens
            if h.shape[1] > self.num_prefix:
                h = h[:, self.num_prefix:, :]
            feats.append(h)
        return feats

    @property
    def parameters_module(self):
        return self.model


# ══════════════════════════════════════════════════════════════════════
# Decoders
# ══════════════════════════════════════════════════════════════════════

class DPTDecoder(nn.Module):
    """Simple DPT-style decoder for plain ViT encoders.

    Takes multi-layer ViT features (all at 1/patch_size resolution) and
    progressively upsamples to full resolution.

    Reference: Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021.
    """

    def __init__(self, embed_dim, feature_indices, patch_size=14, out_channels=256):
        super().__init__()
        self.patch_size = patch_size
        self.feature_indices = feature_indices
        n_features = len(feature_indices)

        # Project each ViT layer output to common channel dim
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ) for _ in range(n_features)
        ])

        # Progressive fusion: bottom-up, each level fuses with the one below
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ) for _ in range(n_features - 1)
        ])

        # Final upsample to full resolution
        self.up = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels // 2, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU(),
        )
        self.out_channels = out_channels // 4

    def forward(self, features):
        """
        Args:
            features: list of (B, N, C) or (B, C, H, W) tensors from ViT blocks
        Returns:
            (B, out_channels//4, H, W) tensor at ~full resolution
        """
        projected = [self.projects[i](_tokens_to_bchw(f))
                     for i, f in enumerate(features)]

        # Progressive fusion (bottom-up: deepest first)
        x = projected[-1]
        for i in range(len(self.fusions) - 1, -1, -1):
            x = F.interpolate(x, size=projected[i].shape[2:], mode='bilinear', align_corners=False)
            x = x + projected[i]
            x = self.fusions[i](x)

        # Upsample to near-full resolution
        x = self.up(x)
        return x


class MSLinearDecoder(nn.Module):
    """Meta DINOv2 official +ms linear head.

    Concatenates the 4 specified ViT layers' patch features along channels,
    applies BatchNorm2d, and outputs via the final 1x1 classifier (added by the
    wrapping model). Intended to be paired with tapping the LAST 4 layers.

    Reference: facebookresearch/dinov2, dinov2/eval/segmentation/models/decode_heads/bn_head.py
    """

    def __init__(self, embed_dim, feature_indices, **_ignored):
        super().__init__()
        n = len(feature_indices)
        self.total_dim = embed_dim * n
        self.bn = nn.BatchNorm2d(self.total_dim)
        # Output channels for the wrapper's 1x1 head to consume
        self.out_channels = self.total_dim

    def forward(self, features):
        feats = [_tokens_to_bchw(f) for f in features]
        x = torch.cat(feats, dim=1)
        x = self.bn(x)
        return x


class SegDINOMLPDecoder(nn.Module):
    """SegDINO-style lightweight MLP decoder.

    Tap 4 ViT layers, project each to `proj_channels` via 1x1 conv, upsample
    to a common spatial resolution (all ViT intermediates have the same patch
    grid, so this is a no-op for plain ViT), concatenate along channels, and
    apply a 3-stage pointwise MLP.

    Reference: Yang et al. 2025, SegDINO (arXiv 2509.00833).
    """

    def __init__(self, embed_dim, feature_indices, proj_channels=256,
                 hidden_channels=512, **_ignored):
        super().__init__()
        n = len(feature_indices)
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, proj_channels, 1) for _ in range(n)
        ])
        total = proj_channels * n
        self.mlp = nn.Sequential(
            nn.Conv2d(total, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, proj_channels, 1),
            nn.GELU(),
        )
        self.out_channels = proj_channels

    def forward(self, features):
        projected = [p(_tokens_to_bchw(f)) for p, f in zip(self.projects, features)]
        x = torch.cat(projected, dim=1)
        x = self.mlp(x)
        return x


# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════

class TimmSemanticModel(nn.Module):
    """Flexible semantic segmentation model using any timm encoder.

    For hierarchical encoders (Swin, ConvNeXt): uses UNet++ decoder from smp.
    For plain ViT encoders (DINOv2): uses custom DPT decoder.
    """

    def __init__(self, encoder_name, decoder_type="unetplusplus",
                 pretrained=True, img_size=1024, num_classes=7):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_type = decoder_type
        self.img_size = img_size
        self.is_hf = encoder_name.startswith("hf:")
        self.is_vit = self.is_hf or ('vit' in encoder_name) or ('dino' in encoder_name)

        if self.is_vit:
            if self.is_hf:
                hf_name = encoder_name[len("hf:"):]
                self.encoder = HFViTBackbone(hf_name, pretrained=pretrained)
                embed_dim = self.encoder.embed_dim
                n_blocks = self.encoder.num_blocks
                patch_size = self.encoder.patch_size[0]
            else:
                # timm plain ViT (DINOv2 via `vit_small_patch14_dinov2.lvd142m` etc.)
                patch_size_guess = 14 if 'patch14' in encoder_name else 16
                padded_size = img_size + (patch_size_guess - img_size % patch_size_guess) % patch_size_guess
                self.encoder = timm.create_model(
                    encoder_name, pretrained=pretrained,
                    img_size=padded_size, num_classes=0,
                )
                embed_dim = self.encoder.embed_dim
                n_blocks = len(self.encoder.blocks)
                patch_size = self.encoder.patch_embed.patch_size[0]

            # Which 4 blocks to tap depends on the decoder:
            #   dpt / segdino_mlp : 4 evenly-spaced layers (captures low→high)
            #   ms_linear         : last 4 layers (Meta's +ms recipe)
            if decoder_type == "ms_linear":
                self.feature_indices = [n_blocks - 4, n_blocks - 3,
                                        n_blocks - 2, n_blocks - 1]
            else:
                self.feature_indices = [
                    n_blocks // 4 - 1,
                    n_blocks // 2 - 1,
                    3 * n_blocks // 4 - 1,
                    n_blocks - 1,
                ]

            if decoder_type == "dpt":
                self.decoder = DPTDecoder(
                    embed_dim=embed_dim,
                    feature_indices=self.feature_indices,
                    patch_size=patch_size,
                    out_channels=256,
                )
            elif decoder_type == "ms_linear":
                self.decoder = MSLinearDecoder(
                    embed_dim=embed_dim, feature_indices=self.feature_indices,
                )
            elif decoder_type == "segdino_mlp":
                self.decoder = SegDINOMLPDecoder(
                    embed_dim=embed_dim, feature_indices=self.feature_indices,
                    proj_channels=256, hidden_channels=512,
                )
            else:
                raise ValueError(
                    f"decoder={decoder_type} not supported for plain-ViT encoders")

            self.head = nn.Conv2d(self.decoder.out_channels, num_classes, 1)
            self.patch_size = patch_size
        else:
            # Hierarchical encoder: use features_only + UNet++ decoder
            # Pass img_size for models that need it (e.g., Swin)
            encoder_kwargs = dict(
                pretrained=pretrained,
                features_only=True, out_indices=(0, 1, 2, 3),
            )
            if 'swin' in encoder_name:
                encoder_kwargs['img_size'] = img_size
            self.encoder = timm.create_model(encoder_name, **encoder_kwargs)

            # Get feature channels (handle both BCHW and BHWC formats)
            dummy = torch.randn(1, 3, img_size, img_size)
            with torch.no_grad():
                feats = self.encoder(dummy)
            # Some encoders (Swin) output (B,H,W,C), others (ConvNeXt) output (B,C,H,W)
            self.channels_last = feats[0].dim() == 4 and feats[0].shape[-1] < feats[0].shape[1]
            if self.channels_last:
                feature_channels = [f.shape[-1] for f in feats]
            else:
                feature_channels = [f.shape[1] for f in feats]

            if decoder_type == "unetplusplus":
                from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
                # encoder_channels format: [input_channels, *feature_channels]
                # For 4-level features, we need depth=4
                encoder_channels = [3] + feature_channels
                decoder_channels = [256, 128, 64, 32]
                self.decoder = UnetPlusPlusDecoder(
                    encoder_channels=encoder_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=4,
                )
                self.head = nn.Sequential(
                    nn.Conv2d(decoder_channels[-1], num_classes, 3, padding=1),
                )
            else:
                raise ValueError(f"Decoder {decoder_type} not supported for hierarchical encoders")

            self.patch_size = None

    def _pad_to_patch(self, x):
        """Pad input to be divisible by patch_size."""
        if self.patch_size is None:
            return x, None
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (h, w)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]

        if self.is_vit:
            # Pad for patch alignment
            x, orig_size = self._pad_to_patch(x)

            # Extract intermediate features
            intermediates = self.encoder.forward_intermediates(
                x, indices=self.feature_indices, return_prefix_tokens=False,
                intermediates_only=True, norm=True,
            )

            # Decode
            decoded = self.decoder(intermediates)

            # Apply head (reduce to num_classes) BEFORE upsampling to save memory.
            # Bilinear interpolation is linear so result is identical to upsample-then-head,
            # but we avoid materializing a high-channel tensor at full resolution.
            logits = self.head(decoded)
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

            if orig_size is not None:
                logits = logits[:, :, :orig_size[0], :orig_size[1]]
        else:
            # Hierarchical encoder
            features = self.encoder(x)

            # Convert channels-last (Swin) to channels-first
            if self.channels_last:
                features = [f.permute(0, 3, 1, 2) for f in features]

            # Build feature list matching UNet++ decoder format:
            # [input, stage1, stage2, stage3, stage4]
            input_feat = x
            decoder_features = [features[-1]] + features[:-1][::-1]
            decoder_features.append(input_feat)
            decoder_features = decoder_features[::-1]

            decoded = self.decoder(decoder_features)

            # Upsample to original size if needed
            if decoded.shape[2] != orig_h or decoded.shape[3] != orig_w:
                decoded = F.interpolate(decoded, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

            logits = self.head(decoded)

        return logits


# ══════════════════════════════════════════════════════════════════════
# Lightning Module
# ══════════════════════════════════════════════════════════════════════

class TimmSemanticModule(pl.LightningModule):
    """Lightning module for timm-based semantic segmentation."""

    def __init__(
        self,
        encoder_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        decoder_type: str = "unetplusplus",
        pretrained: bool = True,
        img_size: int = 1024,
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

        self.model = TimmSemanticModel(
            encoder_name=encoder_name,
            decoder_type=decoder_type,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=NUM_CLASSES,
        )

        # Losses (same as train_unet_semantic.py)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.use_lovasz = not no_lovasz
        if self.use_lovasz:
            self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")

        if equal_weights:
            weights = [1.0] * NUM_CLASSES
        else:
            weights = [0.5, 1.0, aer_weight, 5.0, 1.0, 5.0, 1.0]
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, logits, masks):
        dice = self.dice_loss(logits, masks)
        focal = self.focal_loss(logits, masks)
        wce = F.cross_entropy(logits, masks, weight=self.class_weights)
        loss = dice + focal + wce
        if self.use_lovasz:
            lovasz = self.lovasz_loss(logits, masks)
            loss = loss + lovasz
        return loss

    def training_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        preds = logits.argmax(dim=1)
        acc = (preds == masks).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_pixel_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        preds = logits.argmax(dim=1)
        acc = (preds == masks).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_pixel_acc", acc, prog_bar=True, sync_dist=True)
        return {"val_loss": loss, "preds": preds, "masks": masks}

    def configure_optimizers(self):
        # Differential LR: encoder gets lower LR
        encoder_params = list(self.model.encoder.parameters())
        encoder_ids = set(id(p) for p in encoder_params)
        decoder_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]

        param_groups = [
            {"params": [p for p in encoder_params if p.requires_grad], "lr": self.backbone_lr},
            {"params": decoder_params, "lr": self.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.eta_min
        )
        return [optimizer], [scheduler]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def make_run_subfolder(base_dir, run_name):
    """Create dated run subfolder: base_dir/run_name/YYYY-MM-DD_NNN/"""
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    existing = sorted(run_dir.glob(f"{today}_*"))
    n = len(existing) + 1
    sub = run_dir / f"{today}_{n:03d}"
    sub.mkdir()
    return sub


def main():
    parser = argparse.ArgumentParser(description="Train timm encoder + decoder for semantic segmentation")
    parser.add_argument("--encoder", default="convnextv2_tiny.fcmae_ft_in22k_in1k",
                        help="timm encoder name")
    parser.add_argument("--decoder", default="unetplusplus",
                        choices=["unetplusplus", "dpt", "ms_linear", "segdino_mlp"],
                        help="Decoder: unetplusplus (hierarchical) | dpt / ms_linear / segdino_mlp (plain ViT)")
    parser.add_argument("--strategy", default="A")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--equal-weights", action="store_true")
    parser.add_argument("--aer-weight", type=float, default=10.0)
    parser.add_argument("--no-lovasz", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel-dropout", type=float, default=0.2)
    parser.add_argument("--channel-shuffle", type=float, default=0.2)
    parser.add_argument("--hue-sat", type=float, default=0.0,
                        help="HueSaturationValue probability (0 to disable)")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Run naming
    enc_raw = args.encoder[len("hf:"):] if args.encoder.startswith("hf:") else args.encoder
    encoder_short = enc_raw.split(".")[0].replace("/", "_")
    weight_tag = "equalw" if args.equal_weights else "defaultw"
    aug_parts = []
    if args.channel_dropout > 0: aug_parts.append("drop")
    if args.channel_shuffle > 0: aug_parts.append("shuf")
    if args.hue_sat > 0: aug_parts.append("hue")
    aug_tag = "_".join(aug_parts) if aug_parts else "noaug"
    loss_tag = "dfce" if args.no_lovasz else "dfcel"
    run_name = f"{args.decoder}_{encoder_short}_{weight_tag}_{aug_tag}_{loss_tag}_semantic7c_{args.strategy}"

    run_dir = make_run_subfolder(OUTPUT_DIR / "runs" / "timm", run_name)
    print(f"Run: {run_name}")
    print(f"Output: {run_dir}")

    # Data
    split = get_split(strategy=args.strategy)
    train_transform = get_train_transform(
        args.img_size, p_channel_dropout=args.channel_dropout,
        p_channel_shuffle=args.channel_shuffle, p_hue_sat=args.hue_sat)
    val_transform = get_val_transform(args.img_size)

    train_ds = UNetSemanticDataset(split["train"], transform=train_transform)
    val_ds = UNetSemanticDataset(split["val"], transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"\nTRAIN: {len(train_ds)} samples")
    print(f"VAL: {len(val_ds)} samples")

    # Model
    module = TimmSemanticModule(
        encoder_name=args.encoder,
        decoder_type=args.decoder,
        img_size=args.img_size,
        equal_weights=args.equal_weights,
        aer_weight=args.aer_weight,
        no_lovasz=args.no_lovasz,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f} M")
    print(f"Trainable params: {trainable_params / 1e6:.1f} M")

    # Callbacks
    checkpoint_best = pl.callbacks.ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    checkpoint_periodic = pl.callbacks.ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="periodic-{epoch}",
        every_n_epochs=args.save_every, save_top_k=-1,
    )
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_best, checkpoint_periodic, early_stop, lr_monitor],
        default_root_dir=str(run_dir),
        log_every_n_steps=10,
    )

    # Save hparams
    import yaml
    with open(run_dir / "hparams.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    trainer.fit(module, train_loader, val_loader)

    print(f"\nTraining complete. Best checkpoint: {checkpoint_best.best_model_path}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
