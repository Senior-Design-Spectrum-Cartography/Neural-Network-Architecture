"""
loss.py
=======
Phase 3b – Custom Reconstruction Loss for PartialConvMAE.

Implements a masked MSE loss that computes the error **strictly over the
initially unmeasured regions** (where mask == 0), i.e. only the pixels
the model had to hallucinate / infer.

This design choice follows the MAE philosophy: there is no reward for
accurately copying already-observed measurements; the model must learn
the underlying RF propagation structure to reconstruct missing areas.

Optionally a weighted combination with a *full-map* MSE term can be
enabled (via `lambda_full`) to stabilise training in early epochs.

Functions
---------
masked_mse_loss(pred, target, mask, lambda_full=0.0, reduction='mean')

    pred        : (B, C, D, H, W)  – model output
    target      : (B, C, D, H, W)  – ground-truth full RF tensor
    mask        : (B, 1, D, H, W)  – binary mask  (1=measured, 0=missing)
    lambda_full : float             – weight for the auxiliary full-map term
    reduction   : 'mean' | 'sum'   – how to reduce the hole loss

    Returns scalar loss tensor.
"""

import torch
import torch.nn.functional as F


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_full: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE loss over the *unobserved* (hole) region only.

    The hole region is defined by  (1 - mask),  i.e. locations that
    were zero in the original sparse measurement.

    Parameters
    ----------
    pred        : (B, C, D, H, W)  – reconstructed tensor
    target      : (B, C, D, H, W)  – ground-truth full RF tensor
    mask        : (B, 1, D, H, W)  – binary mask (1=measured, 0=missing)
                  Automatically broadcast over channels.
    lambda_full : float  ≥ 0       – if > 0, adds λ × MSE_full to the
                                      loss for auxiliary supervision
    reduction   : 'mean' | 'sum'   – aggregation over hole voxels

    Returns
    -------
    loss : scalar Tensor
    """
    # Broadcast mask to match channel dimension
    hole_mask = (1.0 - mask)                # (B, 1, D, H, W)  1=hole, 0=measured
    # Expand to C channels
    if hole_mask.shape[1] != pred.shape[1]:
        hole_mask = hole_mask.expand_as(pred)

    # ── Squared error everywhere ────────────────────────────────────────
    sq_err = (pred - target) ** 2           # (B, C, D, H, W)

    # ── Hole loss ──────────────────────────────────────────────────────
    hole_sq_err = sq_err * hole_mask        # zero out measured locations

    n_hole = hole_mask.sum().clamp(min=1.0) # number of hole voxels (avoid /0)

    if reduction == "mean":
        loss_hole = hole_sq_err.sum() / n_hole
    else:  # 'sum'
        loss_hole = hole_sq_err.sum()

    # ── Optional full-map auxiliary loss ──────────────────────────────
    if lambda_full > 0.0:
        loss_full = F.mse_loss(pred, target, reduction="mean")
        return loss_hole + lambda_full * loss_full

    return loss_hole


# ---------------------------------------------------------------------------
# Convenience class-based wrapper (compatible with nn.Module training loops)
# ---------------------------------------------------------------------------

class MaskedReconstructionLoss(torch.nn.Module):
    """
    nn.Module wrapper around masked_mse_loss for use in training loops.

    Parameters
    ----------
    lambda_full : float – auxiliary full-map MSE weight  (default 0.1)
    reduction   : str   – 'mean' or 'sum'
    """

    def __init__(
        self,
        lambda_full: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_full = lambda_full
        self.reduction   = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return masked_mse_loss(
            pred, target, mask,
            lambda_full=self.lambda_full,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (f"lambda_full={self.lambda_full}, "
                f"reduction='{self.reduction}'")
