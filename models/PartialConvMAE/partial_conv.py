"""
partial_conv.py
===============
Phase 1 – Partial Convolutional Embedding for PartialConvMAE.

A 3-D Partial Convolution layer that:
  • Conditions the convolution strictly on *valid* (measured) pixels
    (locations where the binary mask M == 1).
  • Scales activations to compensate for the fraction of valid pixels
    inside each receptive field.
  • Propagates an updated mask forward: an output location is valid (1)
    if at least one input pixel in its receptive field was valid.

Reference:
  Liu et al., "Image Inpainting for Irregular Holes Using Partial
  Convolutions", ECCV 2018.  Extended here to 3-D (spatial-x,
  spatial-y, frequency/time) to match the spatio-temporal RF tensor.

Input tensor layout  : (B, C_in,  D, H, W)
                         B  = batch
                         C  = channels (1 for raw power map)
                         D  = frequency / time slices
                         H,W = spatial grid
Binary mask layout   : (B,  1,    D, H, W)  –  1 = measured, 0 = missing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv3d(nn.Module):
    """
    3-D Partial Convolution layer.

    Parameters
    ----------
    in_channels  : int  – number of input feature channels
    out_channels : int  – number of output feature channels
    kernel_size  : int or 3-tuple
    stride       : int or 3-tuple
    padding      : int or 3-tuple
    bias         : bool – include a bias term (default True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Main convolution – applied to masked-zeroed input
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,          # bias handled manually after scaling
        )

        # Mask update kernel: all-ones, single-channel, no learning
        self.register_buffer(
            "mask_kernel",
            torch.ones(1, 1, *self._to_tuple(kernel_size)),
        )

        self.stride = self._to_tuple(stride)
        self.padding = self._to_tuple(padding)

        # Receptive field size (used for normalisation)
        k = self._to_tuple(kernel_size)
        self.rf_size = float(k[0] * k[1] * k[2]) * in_channels

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self._init_weights()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_tuple(v):
        return (v, v, v) if isinstance(v, int) else tuple(v)

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out",
                                nonlinearity="relu")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x    : (B, C_in, D, H, W)  – raw input (zeros at missing sites)
        mask : (B,  1,   D, H, W)  – binary mask (1=valid, 0=missing)

        Returns
        -------
        out       : (B, C_out, D', H', W')  – partial-conv activations
        mask_out  : (B,  1,    D', H', W')  – propagated binary mask
        """
        # ── Step 1: zero-out invalid pixels ──────────────────────────
        # mask is broadcast across channels
        x_masked = x * mask                      # (B, C, D, H, W)

        # ── Step 2: standard convolution on zeroed input ─────────────
        raw_out = self.conv(x_masked)            # (B, C_out, D', H', W')

        # ── Step 3: count valid pixels in each receptive field ───────
        # Use the all-ones mask kernel; sum gives |M ∩ RF|
        with torch.no_grad():
            # mask must be single-channel for this conv
            mask_sum = F.conv3d(
                mask,
                self.mask_kernel,
                stride=self.stride,
                padding=self.padding,
            )                                    # (B, 1, D', H', W')

        # ── Step 4: scale activations ────────────────────────────────
        # ratio = (total RF size) / (number of valid pixels)
        # Clamp denominator to avoid division by zero at fully-masked
        # locations (their output will later be zeroed by mask_out).
        scale = self.rf_size / (mask_sum + 1e-8)  # (B, 1, D', H', W')
        out = raw_out * scale                    # broadcast over C_out

        # ── Step 5: add bias ─────────────────────────────────────────
        if self.bias is not None:
            # bias shape: (C_out,) → (1, C_out, 1, 1, 1)
            out = out + self.bias.view(1, -1, 1, 1, 1)

        # ── Step 6: update mask ──────────────────────────────────────
        # Output location is valid iff mask_sum > 0
        with torch.no_grad():
            mask_out = (mask_sum > 0).float()   # (B, 1, D', H', W')

        # Zero activations at fully-masked output locations
        out = out * mask_out

        return out, mask_out


# ---------------------------------------------------------------------------
# Patch-Embedding stem built from stacked Partial Convolutions
# ---------------------------------------------------------------------------
class PartialConvStem(nn.Module):
    """
    A two-layer 3-D Partial Convolution stem that converts a raw sparse
    RF tensor into a dense feature map, while propagating the validity
    mask through each layer.

    Architecture
    ------------
    Input  (B, 1,        D,    H,    W)
    ↓  PartialConv3d(1  → stem_dim, k=3, s=1, p=1)
    ↓  BatchNorm3d + ReLU
    ↓  PartialConv3d(stem_dim → embed_dim, k=3, s=patch_stride, p=1)
    ↓  BatchNorm3d + ReLU
    Output (B, embed_dim, D//ps, H//ps, W//ps)

    Parameters
    ----------
    in_channels  : int – raw signal channels (typically 1)
    stem_dim     : int – intermediate channel width
    embed_dim    : int – final embedding dimension (= encoder hidden dim)
    patch_stride : int – spatial downsampling factor (patch size)
    """

    def __init__(
        self,
        in_channels: int = 1,
        stem_dim: int = 64,
        embed_dim: int = 256,
        patch_stride: int = 4,
    ) -> None:
        super().__init__()

        self.pconv1 = PartialConv3d(in_channels, stem_dim,
                                    kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(stem_dim)

        self.pconv2 = PartialConv3d(stem_dim, embed_dim,
                                    kernel_size=3,
                                    stride=patch_stride,
                                    padding=1)
        self.bn2 = nn.BatchNorm3d(embed_dim)

        self.act = nn.ReLU(inplace=True)
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        x    : (B, C_in, D, H, W)
        mask : (B, 1,    D, H, W)

        Returns
        -------
        feat      : (B, embed_dim, D', H', W')
        mask_down : (B, 1,         D', H', W')
        """
        x, mask = self.pconv1(x, mask)
        x = self.act(self.bn1(x))

        x, mask = self.pconv2(x, mask)
        x = self.act(self.bn2(x))

        return x, mask
