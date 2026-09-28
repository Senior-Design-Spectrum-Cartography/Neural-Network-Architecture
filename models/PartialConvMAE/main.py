"""
main.py
=======
Instantiates the PartialConvMAE model and runs a dummy forward pass with
a randomly generated sparse RF tensor and binary mask.

Tensor layout
-------------
  Batch size  B = 2
  Channels    C = 1   (single-channel power map in dBm)
  Depth       D = 8   (frequency / time slices)
  Height      H = 32  (spatial grid y)
  Width       W = 32  (spatial grid x)

Sparsity
--------
  ~85 % of voxels are masked (unobserved), matching the expected
  operational density of the UGV traversing the UCF campus grid.

Usage
-----
  python main.py [--device cuda|cpu] [--sparsity 0.85]
"""

import argparse
import sys
import time

import torch

# Add the module directory to path (needed when running from outside the pkg)
sys.path.insert(0, ".")

from model import PartialConvMAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_dummy_data(
    batch:    int   = 2,
    channels: int   = 1,
    depth:    int   = 8,
    height:   int   = 32,
    width:    int   = 32,
    sparsity: float = 0.85,
    device:   str   = "cpu",
) -> tuple:
    """
    Generate a synthetic sparse RF tensor and its binary mask.

    The raw RF power map values are drawn from a normal distribution
    centred at –80 dBm (typical indoor received power) with σ = 10 dB.

    Parameters
    ----------
    sparsity : float  – fraction of voxels that are *unobserved* (mask=0)

    Returns
    -------
    x_sparse : (B, C, D, H, W) – zeroed at unobserved locations
    mask     : (B, 1, D, H, W) – 1=measured, 0=missing
    target   : (B, C, D, H, W) – full ground-truth map (for loss)
    """
    shape = (batch, channels, depth, height, width)
    dev   = torch.device(device)

    # Ground truth: realistic dBm values centred around –80 dBm
    target = torch.randn(*shape, device=dev) * 10.0 - 80.0

    # Binary mask: Bernoulli draw – each voxel observed with prob (1-sparsity)
    mask_shape = (batch, 1, depth, height, width)
    mask = torch.bernoulli(
        torch.full(mask_shape, 1.0 - sparsity, device=dev)
    )

    # Guard: ensure at least one measured voxel per sample to avoid
    # the degenerate all-invisible case during the demo
    for b in range(batch):
        if mask[b].sum() == 0:
            mask[b, 0, 0, 0, 0] = 1.0

    # Sparse input: zero out unmeasured locations
    x_sparse = target * mask  # mask broadcasts over channels

    return x_sparse, mask, target


def pretty_print_params(param_dict: dict) -> None:
    print("\n┌─────────────────────────────────────┐")
    print("│        PartialConvMAE Parameters      │")
    print("├──────────────────────┬───────────────┤")
    for name, count in param_dict.items():
        label = f"  {name:<18}"
        value = f"{count:>12,}  "
        sep   = "│" if name != "total" else "╞"
        print(f"{sep}{label}│{value}│")
    print("└──────────────────────┴───────────────┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PartialConvMAE dummy forward pass")
    parser.add_argument("--device",   default="cpu",  choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    parser.add_argument("--sparsity", default=0.85, type=float,
                        help="Fraction of unmeasured voxels (default: 0.85)")
    parser.add_argument("--batch",    default=2,    type=int)
    parser.add_argument("--depth",    default=8,    type=int,
                        help="Number of frequency / time slices")
    parser.add_argument("--height",   default=32,   type=int)
    parser.add_argument("--width",    default=32,   type=int)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"

    print("=" * 55)
    print("  PartialConvMAE – Intelligent Dynamic Spectrum Cartography")
    print("  UCF / US Army DEVCOM ARL  |  Jetson Orin Nano Target")
    print("=" * 55)
    print(f"\n  Device   : {device.upper()}")
    print(f"  Sparsity : {args.sparsity * 100:.0f}% unmeasured")
    print(f"  Tensor   : B={args.batch}, C=1, D={args.depth}, "
          f"H={args.height}, W={args.width}")

    # ── Instantiate model ───────────────────────────────────────────────
    print("\n[1/4] Building PartialConvMAE …")
    model = PartialConvMAE(
        in_channels   = 1,
        stem_dim      = 64,
        embed_dim     = 256,
        dec_dim       = 128,
        enc_depth     = 6,
        dec_depth     = 4,
        num_heads_enc = 8,
        num_heads_dec = 8,
        mlp_ratio     = 4.0,
        patch_stride  = 4,
        dropout       = 0.1,
        lambda_full   = 0.1,
    ).to(device)
    model.eval()

    pretty_print_params(model.count_parameters())

    # ── Generate dummy data ─────────────────────────────────────────────
    print("\n[2/4] Generating sparse RF data …")
    x_sparse, mask, target = generate_dummy_data(
        batch    = args.batch,
        channels = 1,
        depth    = args.depth,
        height   = args.height,
        width    = args.width,
        sparsity = args.sparsity,
        device   = device,
    )

    observed = mask.sum().item()
    total    = mask.numel()
    print(f"         Observed voxels : {int(observed):,} / {total:,} "
          f"({100 * observed / total:.1f}%)")
    print(f"         Missing  voxels : {total - int(observed):,} "
          f"({100 * (1 - observed / total):.1f}%)")

    # ── Forward pass (inference mode) ───────────────────────────────────
    print("\n[3/4] Running forward pass …")
    with torch.no_grad():
        t0 = time.perf_counter()
        recon, loss = model(x_sparse, mask, target)
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1_000

    # ── Report results ───────────────────────────────────────────────────
    print("\n[4/4] Results")
    print(f"  Input  shape : {list(x_sparse.shape)}")
    print(f"  Output shape : {list(recon.shape)}")
    print(f"  Masked MSE   : {loss.item():.6f}")
    print(f"  Latency      : {latency_ms:.2f} ms  "
          f"({'✓ real-time capable' if latency_ms < 500 else '⚠ may need optimisation'})")

    # Sanity checks
    assert recon.shape == x_sparse.shape, \
        f"Shape mismatch: got {recon.shape}, expected {x_sparse.shape}"
    assert not torch.isnan(recon).any(), "NaN values detected in reconstruction!"
    assert not torch.isinf(recon).any(), "Inf values detected in reconstruction!"

    print("\n  ✅  All sanity checks passed.")
    print("\n  Reconstruction statistics (hole region only):")
    hole_mask   = (1.0 - mask).bool()
    hole_target = target[hole_mask]
    hole_recon  = recon[hole_mask]
    print(f"    Target  – mean: {hole_target.mean():.2f} dBm, "
          f"std: {hole_target.std():.2f} dB")
    print(f"    Recon   – mean: {hole_recon.mean():.2f} dBm, "
          f"std: {hole_recon.std():.2f} dB")
    mae = (hole_recon - hole_target).abs().mean().item()
    print(f"    MAE     : {mae:.4f} dB  (untrained model – expected to be high)")

    print("\n" + "=" * 55)
    print("  Forward pass complete. Model is ready for training.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
