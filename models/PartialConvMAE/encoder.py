"""
encoder.py
==========
Phase 2 – Asymmetric Transformer Encoder for PartialConvMAE.

Pipeline
--------
1. Receive the dense feature map  (B, C, D', H', W')  and the
   downsampled binary mask         (B, 1, D', H', W')  from the
   PartialConvStem.

2. Flatten spatial/freq dimensions → N = D'·H'·W' tokens, each of
   dimension C.

3. Compute 3-D sinusoidal positional encoding for every token, storing
   the (d, h, w) grid indices so that the decoder can reuse them.

4. **Discard** patches that are entirely un-measured (mask sum == 0 over
   the patch volume).  This is the "asymmetric" trick from MAE: the
   encoder only processes the small visible subset, dramatically reducing
   compute.

5. Process the surviving tokens through L_e Transformer blocks (MSA +
   FFN, pre-LayerNorm style, standard PyTorch MultiheadAttention).

6. Return:
   - encoded visible tokens   (B, N_vis, C)
   - positional encodings of ALL N tokens   (B, N, C)   [for decoder]
   - visibility mask          (B, N)  bool  [True = visible]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3-D Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def sinusoidal_3d(
    d_grid: int,
    h_grid: int,
    w_grid: int,
    embed_dim: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build a 3-D sinusoidal positional encoding table.

    The embed_dim is split into three equal parts (⌊C/3⌋ each);
    any remainder dims are zero-padded.

    Returns
    -------
    pe : (1, D'·H'·W', embed_dim)  – ready to broadcast over batch
    """
    dim_each = embed_dim // 3          # dims allocated per axis
    half = dim_each // 2              # sin / cos split

    def _1d_enc(length: int) -> torch.Tensor:
        """Returns (length, dim_each) sinusoidal encoding for one axis."""
        enc = torch.zeros(length, dim_each)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (L,1)
        # Use only even indices so sin/cos pairs are always equal count
        even_idx = torch.arange(0, dim_each, 2, dtype=torch.float32)  # (half_floor,)
        div = torch.pow(10000.0, even_idx / dim_each).unsqueeze(0)    # (1, half_floor)
        sins = torch.sin(pos / div)   # (L, half_floor)
        coss = torch.cos(pos / div)   # (L, half_floor)
        n_sin = sins.shape[1]
        n_cos = coss.shape[1]
        enc[:, 0::2] = sins                    # positions 0,2,4,...
        enc[:, 1::2] = coss[:, :enc[:, 1::2].shape[1]]  # positions 1,3,5,... (safe slice)
        return enc  # (length, dim_each)

    enc_d = _1d_enc(d_grid)   # (D', dim_each)
    enc_h = _1d_enc(h_grid)   # (H', dim_each)
    enc_w = _1d_enc(w_grid)   # (W', dim_each)

    # Tile into full grid
    # Each token at (d, h, w) gets [ enc_d[d] | enc_h[h] | enc_w[w] ]
    # shape after combination: (D'*H'*W', 3*dim_each)
    N = d_grid * h_grid * w_grid
    pe_full = torch.zeros(N, 3 * dim_each)
    idx = 0
    for di in range(d_grid):
        for hi in range(h_grid):
            for wi in range(w_grid):
                pe_full[idx, :dim_each]           = enc_d[di]
                pe_full[idx, dim_each:2*dim_each]  = enc_h[hi]
                pe_full[idx, 2*dim_each:3*dim_each] = enc_w[wi]
                idx += 1

    # Pad to embed_dim if necessary
    if 3 * dim_each < embed_dim:
        padding = torch.zeros(N, embed_dim - 3 * dim_each)
        pe_full = torch.cat([pe_full, padding], dim=-1)

    pe_full = pe_full.unsqueeze(0)  # (1, N, embed_dim)
    if device is not None:
        pe_full = pe_full.to(device)
    return pe_full


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block:
        x = x + MSA(LN(x))
        x = x + FFN(LN(x))

    Parameters
    ----------
    embed_dim  : int
    num_heads  : int
    mlp_ratio  : float – FFN hidden dim = embed_dim * mlp_ratio
    dropout    : float
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Asymmetric Transformer Encoder
# ---------------------------------------------------------------------------

class AsymmetricEncoder(nn.Module):
    """
    Tokenizes the PartialConvStem output, discards invisible patches,
    adds sinusoidal positional encoding, and runs L_e Transformer blocks.

    Parameters
    ----------
    embed_dim  : int   – feature / token dimension (= stem output channels)
    depth      : int   – number of Transformer blocks  (L_e)
    num_heads  : int
    mlp_ratio  : float
    dropout    : float
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # Input projection: 1×1×1 conv to make channels explicit
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        feat: torch.Tensor,   # (B, C, D', H', W')  from PartialConvStem
        mask: torch.Tensor,   # (B, 1, D', H', W')  downsampled binary mask
    ):
        """
        Returns
        -------
        encoded    : (B, N_vis, C)      – encoded visible tokens
        pos_all    : (B, N,     C)      – positional enc for ALL N tokens
        vis_mask   : (B, N)   bool      – True where patch is visible
        grid_shape : (D', H', W')       – spatial shape for decoder
        """
        B, C, Dp, Hp, Wp = feat.shape
        N = Dp * Hp * Wp

        # ── Flatten to tokens ──────────────────────────────────────────
        # (B, C, D', H', W') → (B, N, C)
        tokens = feat.flatten(2).transpose(1, 2)       # (B, N, C)
        tokens = self.proj(tokens)                      # linear mixing

        # ── Positional encoding for all N positions ────────────────────
        pos_all = sinusoidal_3d(Dp, Hp, Wp, C, device=feat.device)
        # pos_all: (1, N, C)  → expand to (B, N, C)
        pos_all = pos_all.expand(B, -1, -1)

        # ── Determine visibility: a patch is visible if its mask sum > 0 ─
        # mask: (B, 1, D', H', W') → (B, N)
        mask_flat = mask.flatten(2).squeeze(1)          # (B, N)
        vis_mask  = mask_flat > 0                       # (B, N) bool

        # ── Extract visible tokens (variable length per sample) ────────
        # Strategy: for efficiency during inference we batch-process by
        # taking the union of visible positions across the batch.
        # During training with uniform sparsity, all samples share the
        # same visible set; during inference we iterate per-sample.
        vis_any = vis_mask.any(dim=0)                  # (N,)
        # Use union-of-visible-positions approach (safe for batching)
        visible_tokens = tokens[:, vis_any, :]         # (B, N_vis, C)
        visible_pos    = pos_all[:, vis_any, :]        # (B, N_vis, C)

        # Add positional encoding to visible tokens
        x = visible_tokens + visible_pos               # (B, N_vis, C)

        # ── Transformer blocks ─────────────────────────────────────────
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)                               # (B, N_vis, C)

        return x, pos_all, vis_any, (Dp, Hp, Wp)
