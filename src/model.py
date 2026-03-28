"""
model.py
--------
Defines the conditional U-Net–style denoising network used by the DDPM.

Architecture
------------
* Input: noisy return sequence  x_t  of shape (B, C, L)
          where B = batch, C = n_assets, L = seq_len
* Conditioning: scalar diffusion timestep t  +  binary tail label y
* Output: predicted noise ε of the same shape as x_t

The network follows a 1-D temporal U-Net pattern:
    Encoder  → [ResBlock + optional self-attention] × n_levels
    Bottleneck → ResBlock + SelfAttention + ResBlock
    Decoder  → [ResBlock + optional self-attention] × n_levels
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional / timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (as in the original DDPM paper)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb  # (B, dim)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Residual block for 1-D temporal sequences with time + cond injection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the temporal dimension."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Conditional U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet1D(nn.Module):
    """
    1-D Conditional U-Net denoising network.

    Parameters
    ----------
    in_channels : number of asset return channels (C)
    hidden_dim  : base channel width; doubled at each encoder level
    num_res_blocks : ResBlocks per encoder / decoder level
    attn_resolutions : sequence lengths at which to add self-attention
    dropout     : dropout rate inside ResBlocks
    cond_dim    : dimension of the label conditioning embedding
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 64,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[List[int]] = None,
        dropout: float = 0.1,
        cond_dim: int = 32,
    ) -> None:
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = []

        # ---------- embeddings ----------
        time_emb_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Binary label → embedding (0 = normal, 1 = tail)
        self.label_emb = nn.Embedding(2, cond_dim)
        self.label_proj = nn.Linear(cond_dim, time_emb_dim)

        total_emb_dim = time_emb_dim  # label is *added* to time emb

        # ---------- encoder ----------
        self._num_res_blocks = num_res_blocks
        ch = hidden_dim
        self.input_conv = nn.Conv1d(in_channels, ch, kernel_size=3, padding=1)

        self.encoder_blocks: nn.ModuleList = nn.ModuleList()
        self.encoder_attns: nn.ModuleList = nn.ModuleList()
        self.down_samples: nn.ModuleList = nn.ModuleList()

        ch_list = [ch]
        n_levels = 3
        for level in range(n_levels):
            out_ch = hidden_dim * (2 ** level)
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock1D(ch, out_ch, total_emb_dim, dropout))
                ch = out_ch
                ch_list.append(ch)
            self.encoder_attns.append(
                SelfAttention1D(ch) if (level in attn_resolutions) else nn.Identity()
            )
            if level < n_levels - 1:
                self.down_samples.append(nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())

        # ---------- bottleneck ----------
        self.mid_block1 = ResBlock1D(ch, ch, total_emb_dim, dropout)
        self.mid_attn = SelfAttention1D(ch)
        self.mid_block2 = ResBlock1D(ch, ch, total_emb_dim, dropout)

        # ---------- decoder ----------
        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        self.decoder_attns: nn.ModuleList = nn.ModuleList()
        self.up_samples: nn.ModuleList = nn.ModuleList()

        for level in reversed(range(n_levels)):
            out_ch = hidden_dim * (2 ** level)
            for _ in range(num_res_blocks):
                skip_ch = ch_list.pop()
                self.decoder_blocks.append(
                    ResBlock1D(ch + skip_ch, out_ch, total_emb_dim, dropout)
                )
                ch = out_ch
            self.decoder_attns.append(
                SelfAttention1D(ch) if (level in attn_resolutions) else nn.Identity()
            )
            if level > 0:
                self.up_samples.append(
                    nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())

        # ---------- output ----------
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv1d(ch, in_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : (B, C, L) noisy input
        t     : (B,)      diffusion timestep (integer)
        label : (B,)      binary tail label (0 or 1)

        Returns
        -------
        predicted noise ε of shape (B, C, L)
        """
        # Embedding
        emb = self.time_mlp(t) + self.label_proj(self.label_emb(label))

        # Encoder
        h = self.input_conv(x)
        skips = [h]

        blk_idx = 0
        for level in range(len(self.down_samples)):
            for _ in range(self._resblocks_per_level):
                h = self.encoder_blocks[blk_idx](h, emb)
                blk_idx += 1
                skips.append(h)
            h = self.encoder_attns[level](h)
            h = self.down_samples[level](h)

        # Bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder
        dec_idx = 0
        up_idx = 0
        for level in range(len(self.up_samples)):
            for _ in range(self._resblocks_per_level):
                skip = skips.pop()
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-1])
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[dec_idx](h, emb)
                dec_idx += 1
            h = self.decoder_attns[level](h)
            h = self.up_samples[up_idx](h)
            up_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))

    @property
    def _resblocks_per_level(self) -> int:
        """Returns the number of ResBlocks per encoder/decoder level."""
        return self._num_res_blocks


def build_model(cfg) -> ConditionalUNet1D:
    """Instantiate model from OmegaConf model sub-config."""
    return ConditionalUNet1D(
        in_channels=cfg.in_channels,
        hidden_dim=cfg.hidden_dim,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=list(cfg.attn_resolutions),
        dropout=cfg.dropout,
        cond_dim=cfg.cond_dim,
    )
