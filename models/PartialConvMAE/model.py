"""
model.py
========
PartialConvMAE – top-level model that wires together all three phases:

  Phase 1  PartialConvStem   (partial_conv.py)
  Phase 2  AsymmetricEncoder (encoder.py)
  Phase 3  PartialConvDecoder + MaskedReconstructionLoss
           (decoder.py / loss.py)

The model takes a sparse RF tensor and its binary measurement mask and
returns a fully reconstructed spatio-temporal spectrum map of the same
spatial size.

Input / Output
--------------
  x     : (B, C_in, D, H, W)   sparse RF power map  (zeros at holes)
  mask  : (B, 1,    D, H, W)   binary mask  (1 = measured, 0 = missing)

  → recon : (B, C_in, D, H, W) reconstructed full map

Default configuration targets the Jetson Orin Nano:
  - embed_dim  256  (enc_dim)
  - dec_dim    128
  - enc_depth   6 blocks
  - dec_depth   4 blocks
  - patch_stride 4   (64×64 map → 16×16 grid)
  - stem_dim   64
"""

import torch
import torch.nn as nn

from partial_conv import PartialConvStem
from encoder      import AsymmetricEncoder
from decoder      import PartialConvDecoder
from loss         import MaskedReconstructionLoss


class PartialConvMAE(nn.Module):
    """
    Partial Convolutional Masked Autoencoder for spatio-temporal
    RF spectrum map reconstruction.

    Parameters
    ----------
    in_channels   : int   – raw input channels (1 for power map)
    stem_dim      : int   – PartialConvStem intermediate channels
    embed_dim     : int   – encoder (and stem output) token dimension
    dec_dim       : int   – decoder hidden dimension
    enc_depth     : int   – number of encoder Transformer blocks  (L_e)
    dec_depth     : int   – number of decoder cross-attention blocks (L_d)
    num_heads_enc : int   – attention heads in encoder
    num_heads_dec : int   – attention heads in decoder
    mlp_ratio     : float – FFN expansion ratio
    patch_stride  : int   – spatial downsampling in the stem (patch size)
    dropout       : float – dropout probability in attention / FFN
    lambda_full   : float – auxiliary full-map loss weight (0 = disabled)
    """

    def __init__(
        self,
        in_channels:   int   = 1,
        stem_dim:      int   = 64,
        embed_dim:     int   = 256,
        dec_dim:       int   = 128,
        enc_depth:     int   = 6,
        dec_depth:     int   = 4,
        num_heads_enc: int   = 8,
        num_heads_dec: int   = 8,
        mlp_ratio:     float = 4.0,
        patch_stride:  int   = 4,
        dropout:       float = 0.1,
        lambda_full:   float = 0.1,
    ) -> None:
        super().__init__()

        # ── Phase 1: Partial Convolutional Stem ────────────────────────
        self.stem = PartialConvStem(
            in_channels  = in_channels,
            stem_dim     = stem_dim,
            embed_dim    = embed_dim,
            patch_stride = patch_stride,
        )

        # ── Phase 2: Asymmetric Transformer Encoder ────────────────────
        self.encoder = AsymmetricEncoder(
            embed_dim = embed_dim,
            depth     = enc_depth,
            num_heads = num_heads_enc,
            mlp_ratio = mlp_ratio,
            dropout   = dropout,
        )

        # ── Phase 3: Reconstructive Decoder ───────────────────────────
        self.decoder = PartialConvDecoder(
            enc_dim      = embed_dim,
            dec_dim      = dec_dim,
            depth        = dec_depth,
            num_heads    = num_heads_dec,
            patch_stride = patch_stride,
            in_channels  = in_channels,
            mlp_ratio    = mlp_ratio,
            dropout      = dropout,
        )

        # ── Loss ───────────────────────────────────────────────────────
        self.criterion = MaskedReconstructionLoss(lambda_full=lambda_full)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Apply Xavier / truncated-normal initialisation to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x:      torch.Tensor,   # (B, C, D, H, W)
        mask:   torch.Tensor,   # (B, 1, D, H, W)
        target: torch.Tensor = None,  # (B, C, D, H, W) optional for loss
    ):
        """
        Forward pass.

        Parameters
        ----------
        x      : sparse input RF tensor  (zeros at unmeasured locations)
        mask   : binary measurement mask (1=measured, 0=missing)
        target : ground-truth full RF tensor (required for loss computation)

        Returns
        -------
        If target is provided:
            (recon, loss)  – reconstructed tensor + scalar loss
        Else:
            recon          – reconstructed tensor only
        """
        original_shape = x.shape           # (B, C, D, H, W)

        # ── Phase 1 ────────────────────────────────────────────────────
        feat, mask_down = self.stem(x, mask)
        # feat:      (B, embed_dim, D', H', W')
        # mask_down: (B, 1,         D', H', W')

        # ── Phase 2 ────────────────────────────────────────────────────
        encoded, pos_all, vis_any, grid_shape = self.encoder(feat, mask_down)
        # encoded:    (B, N_vis, embed_dim)
        # pos_all:    (B, N,     embed_dim)
        # vis_any:    (N,) bool
        # grid_shape: (D', H', W')

        # ── Phase 3 ────────────────────────────────────────────────────
        recon = self.decoder(
            encoded,
            pos_all,
            vis_any,
            grid_shape,
            original_shape,
        )
        # recon: (B, C, D, H, W)

        if target is not None:
            loss = self.criterion(recon, target, mask)
            return recon, loss

        return recon

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        """Return parameter counts per sub-module."""
        def _count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "stem":    _count(self.stem),
            "encoder": _count(self.encoder),
            "decoder": _count(self.decoder),
            "total":   _count(self),
        }
