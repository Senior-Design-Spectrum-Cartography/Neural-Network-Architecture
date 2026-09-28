"""
decoder.py
==========
Phase 3a – Reconstructive Decoder for PartialConvMAE.

Pipeline
--------
1.  Receive encoded visible tokens  (B, N_vis, C_enc)  from the encoder.
2.  Allocate a full token sequence of length N = D'·H'·W' initialised
    with a shared learned *mask token* at invisible positions.
3.  Fill visible positions with the encoded tokens (after a linear
    projection from C_enc → C_dec).
4.  Add sinusoidal positional encoding to the full sequence.
5.  Run L_d *cross-attention* Transformer blocks where:
      Q  = full sequence (mask tokens + visible tokens)
      K,V = encoded visible tokens                   ← information flow
    This forces mask tokens to query information from visible tokens
    rather than attending to each other's zeros.
6.  Project each token (C_dec → patch_vol) to recover the raw signal
    values for every voxel in the patch.
7.  Rearrange the output into the original spatial volume
    (B, 1, D, H, W)  via a pixel-shuffle–style fold.

Parameters that must match the encoder / stem
----------------------------------------------
  embed_dim   (C_enc)  – encoder hidden dim
  dec_dim     (C_dec)  – decoder hidden dim (can be smaller for efficiency)
  patch_stride         – downsampling factor used in the stem
  in_channels          – raw signal channels (1 for power map)
  grid_shape           – (D', H', W') returned by the encoder
"""

import torch
import torch.nn as nn
from encoder import sinusoidal_3d


# ---------------------------------------------------------------------------
# Cross-Attention Transformer Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Pre-LayerNorm cross-attention block.

    Q  comes from the *full* sequence (mask + visible tokens).
    K,V come from the *encoded visible* tokens only.

    After cross-attention the block applies a standard FFN to each token.
    """

    def __init__(
        self,
        dec_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Cross-attention
        self.norm_q  = nn.LayerNorm(dec_dim)
        self.norm_kv = nn.LayerNorm(dec_dim)
        self.cross_attn = nn.MultiheadAttention(
            dec_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Self-attention on full sequence
        self.norm_self = nn.LayerNorm(dec_dim)
        self.self_attn = nn.MultiheadAttention(
            dec_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(dec_dim)
        hidden = int(dec_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dec_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dec_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, N,     C_dec)  full sequence
        context: torch.Tensor,    # (B, N_vis, C_dec)  visible tokens
    ) -> torch.Tensor:
        # 1. Cross-attention: full sequence queries encoded visible tokens
        q   = self.norm_q(x)
        kv  = self.norm_kv(context)
        ca_out, _ = self.cross_attn(q, kv, kv)
        x = x + ca_out

        # 2. Self-attention over full sequence
        sa_normed = self.norm_self(x)
        sa_out, _ = self.self_attn(sa_normed, sa_normed, sa_normed)
        x = x + sa_out

        # 3. FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full Decoder
# ---------------------------------------------------------------------------

class PartialConvDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs the full RF tensor from the
    encoded visible patches plus learned mask tokens.

    Parameters
    ----------
    enc_dim      : int   – encoder output dimension
    dec_dim      : int   – decoder hidden dimension
    depth        : int   – number of cross-attention blocks (L_d)
    num_heads    : int
    patch_stride : int   – must match the stem's patch_stride
    in_channels  : int   – raw signal channels (1)
    mlp_ratio    : float
    dropout      : float
    """

    def __init__(
        self,
        enc_dim: int = 256,
        dec_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        patch_stride: int = 4,
        in_channels: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dec_dim      = dec_dim
        self.patch_stride = patch_stride
        self.in_channels  = in_channels

        # Each patch reconstructs (patch_stride)^3 voxels × in_channels
        self.patch_vol = (patch_stride ** 3) * in_channels

        # Project encoder dim → decoder dim
        self.enc_proj = nn.Linear(enc_dim, dec_dim)

        # Learned mask token (shared across all invisible positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Cross-attention decoder blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dec_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dec_dim)

        # Prediction head: token → raw voxel values for one patch
        self.pred_head = nn.Linear(dec_dim, self.patch_vol)

    # ------------------------------------------------------------------
    def forward(
        self,
        encoded: torch.Tensor,        # (B, N_vis, enc_dim)
        pos_all: torch.Tensor,        # (B, N,     enc_dim)
        vis_any: torch.Tensor,        # (N,) bool  – which positions are visible
        grid_shape: tuple,            # (D', H', W')
        original_shape: tuple,        # (B, C, D, H, W) – target volume shape
    ) -> torch.Tensor:
        """
        Returns
        -------
        recon : (B, C, D, H, W)  – reconstructed full RF tensor
        """
        B      = encoded.shape[0]
        Dp, Hp, Wp = grid_shape
        N      = Dp * Hp * Wp

        # ── Project encoded tokens to decoder dim ──────────────────────
        ctx = self.enc_proj(encoded)                  # (B, N_vis, dec_dim)

        # ── Build full token sequence ───────────────────────────────────
        # Initialise all N positions with the mask token
        full_tokens = self.mask_token.expand(B, N, -1).clone()
        # (B, N, dec_dim)

        # Overwrite visible positions with projected encoder output
        full_tokens[:, vis_any, :] = ctx              # in-place fill

        # ── Add positional encoding (projected to dec_dim) ─────────────
        # pos_all is (B, N, enc_dim); we project it too
        pos_dec = nn.functional.linear(
            pos_all,
            self.enc_proj.weight,
            self.enc_proj.bias,
        )                                             # (B, N, dec_dim)
        full_tokens = full_tokens + pos_dec

        # ── Cross-attention decoder blocks ─────────────────────────────
        for blk in self.blocks:
            full_tokens = blk(full_tokens, ctx)

        full_tokens = self.norm(full_tokens)          # (B, N, dec_dim)

        # ── Prediction head: (B, N, patch_vol) ────────────────────────
        pred = self.pred_head(full_tokens)            # (B, N, patch_vol)

        # ── Fold patch predictions back into spatial volume ────────────
        recon = self._fold(pred, grid_shape, original_shape)
        return recon

    # ------------------------------------------------------------------
    def _fold(
        self,
        pred: torch.Tensor,    # (B, N, patch_vol)
        grid_shape: tuple,     # (D', H', W')
        original_shape: tuple, # (B, C, D, H, W)
    ) -> torch.Tensor:
        """
        Rearrange flat patch predictions into the full 3-D volume.

        Each patch covers (patch_stride × patch_stride × patch_stride)
        voxels in (D, H, W).  We simply reshape and permute.
        """
        B, C_orig, D, H, W = original_shape
        Dp, Hp, Wp = grid_shape
        ps = self.patch_stride

        # pred: (B, N, ps^3 * C)
        # Reshape to (B, D', H', W', ps, ps, ps, C)
        pred = pred.view(B, Dp, Hp, Wp, ps, ps, ps, self.in_channels)

        # Permute to (B, C, D'*ps, H'*ps, W'*ps)
        #   axes:  0   7   1  4   2  5   3  6
        pred = pred.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        pred = pred.view(B, self.in_channels, Dp * ps, Hp * ps, Wp * ps)

        # Crop / pad to exact original spatial size
        pred = pred[:, :, :D, :H, :W]
        return pred
