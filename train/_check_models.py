"""Diagnostic: verify each backbone + decoder combo end-to-end.

For each of the 4 planned configurations:
  1. DINOv2 ViT-S/14 + DPT       (current baseline)
  2. DINOv2 ViT-S/14 + ms_linear (Meta +ms)
  3. DINOv3 ViT-S/16 + DPT       (backbone swap)
  4. DINOv3 ViT-S/16 + segdino_mlp (SegDINO recipe)

Check:
  - encoder loads; report embed_dim, num_blocks, patch_size, num_prefix_tokens
  - tapped feature_indices are within range and give sensible shapes
  - decoder accepts those features and outputs a tensor of expected shape
  - full forward produces (B, 7, 1024, 1024) logits with finite values
  - rough parameter counts

Uses 1024×1024 input on CPU (or GPU if available). Batch 1 for speed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from train.train_timm_semantic import TimmSemanticModel


CONFIGS = [
    ("DINOv3 ViT-S/16 + DPT",          "hf:facebook/dinov3-vits16-pretrain-lvd1689m",
        "dpt",          [2, 5, 8, 11]),
    ("DINOv3 ViT-S/16 + segdino_mlp",  "hf:facebook/dinov3-vits16-pretrain-lvd1689m",
        "segdino_mlp",  [2, 5, 8, 11]),
]


def check(label, encoder, decoder, expected_indices, img_size=1024):
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)

    try:
        m = TimmSemanticModel(encoder, decoder_type=decoder,
                              pretrained=True, img_size=img_size, num_classes=7)
    except Exception as e:
        print(f"  FAIL: model construction: {type(e).__name__}: {e}")
        return False
    m.eval()

    # 1. Encoder metadata
    enc = m.encoder
    embed_dim = getattr(enc, "embed_dim", "?")
    num_blocks = getattr(enc, "num_blocks", None) or len(getattr(enc, "blocks", []))
    ps = getattr(enc, "patch_size", None)
    if ps is None and hasattr(enc, "patch_embed"):
        ps = enc.patch_embed.patch_size
    num_prefix = getattr(enc, "num_prefix", None)
    print(f"  encoder: embed_dim={embed_dim}, num_blocks={num_blocks}, patch_size={ps}")
    if num_prefix is not None:
        print(f"           num_prefix_tokens (CLS+reg) = {num_prefix}")
    print(f"  feature_indices (model): {m.feature_indices}  (expected {expected_indices})")
    assert m.feature_indices == expected_indices, "feature_indices mismatch"

    # 2. Tap sizes: raw forward_intermediates output
    x = torch.randn(1, 3, img_size, img_size)
    if torch.cuda.is_available():
        m = m.cuda(); x = x.cuda()
    x_padded, _ = m._pad_to_patch(x)
    with torch.no_grad():
        feats = m.encoder.forward_intermediates(
            x_padded, indices=m.feature_indices, return_prefix_tokens=False,
            intermediates_only=True, norm=True,
        )
    expected_N = (x_padded.shape[-1] // m.patch_size) ** 2
    print(f"  padded input: {tuple(x_padded.shape)}  (patch grid {int(expected_N**0.5)}²)")
    for i, f in enumerate(feats):
        stats = f"min={f.min().item():+.3f}, max={f.max().item():+.3f}, mean={f.mean().item():+.3f}"
        print(f"    tap {m.feature_indices[i]:>2}: shape={tuple(f.shape)}  {stats}")
        # Shape check: must be (1, expected_N, embed_dim) or (1, C, H, W) with H*W == expected_N
        if f.dim() == 3:
            assert f.shape[0] == 1 and f.shape[2] == embed_dim, f"bad 3D shape {f.shape}"
            assert f.shape[1] == expected_N, f"tokens {f.shape[1]} != expected {expected_N}"
        elif f.dim() == 4:
            # channels-first or channels-last
            dims = list(f.shape[1:])
            assert embed_dim in dims, f"embed_dim {embed_dim} not in {dims}"
            spatial_total = [d for d in dims if d != embed_dim]
            assert spatial_total[0] * spatial_total[1] == expected_N, \
                f"spatial {spatial_total} doesn't match expected N={expected_N}"

    # 3. Full forward
    with torch.no_grad():
        out = m(x)
    print(f"  forward(x): input {tuple(x.shape)} → output {tuple(out.shape)}")
    assert out.shape == (1, 7, img_size, img_size), \
        f"output shape {out.shape} != expected (1, 7, {img_size}, {img_size})"
    assert torch.isfinite(out).all(), "output has non-finite values"
    print(f"    logits: min={out.min().item():+.3f}, max={out.max().item():+.3f}, "
          f"mean={out.mean().item():+.3f}, std={out.std().item():.3f}")
    probs = F.softmax(out, dim=1)
    print(f"    probs:  min={probs.min().item():.4f}, max={probs.max().item():.4f}  (should be [0,1])")
    assert 0 <= probs.min().item() and probs.max().item() <= 1 + 1e-5

    # 4. Parameter count breakdown
    total = sum(p.numel() for p in m.parameters())
    enc_total = sum(p.numel() for p in m.encoder.parameters())
    dec_total = sum(p.numel() for p in m.decoder.parameters())
    head_total = sum(p.numel() for p in m.head.parameters())
    print(f"  params: total={total/1e6:.1f}M  "
          f"encoder={enc_total/1e6:.1f}M  decoder={dec_total/1e6:.2f}M  head={head_total/1e3:.1f}K")
    print("  PASS")
    return True


if __name__ == "__main__":
    passed = 0
    for label, enc, dec, idx in CONFIGS:
        try:
            if check(label, enc, dec, idx):
                passed += 1
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        print()
    print(f"{passed}/{len(CONFIGS)} configs passed")
