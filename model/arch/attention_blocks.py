"""Attention-based decoder blocks for separator architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .common import RMSNorm, SwiGLU


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: (B, T_q, d_model) query
            k: (B, T_k, d_model) key
            v: (B, T_k, d_model) value
            mask: optional attention mask
        Returns:
            out: (B, T_q, d_model)
        """
        B, T_q, D = q.shape
        T_k = k.shape[1]

        # Project
        q = self.q_proj(q).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's scaled dot-product attention implementation.
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, T_q, D)
        out = self.out_proj(out)

        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for decoder to query encoder.

    Allows the decoder to retrieve information from encoder representations.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) decoder input
            encoder_memory: (B, T_enc, d_model) encoder output to attend to
        Returns:
            out: (B, T, d_model)
        """
        q = self.norm_q(x)
        kv = self.norm_kv(encoder_memory)
        attn_out = self.cross_attn(q, kv, kv)
        return x + attn_out


class BandTimeCrossAttention(nn.Module):
    """
    Factorized cross-attention over time and band axes.

    The decoder first attends over time within each band using pooled encoder
    tokens, then attends across bands using a short time-conditioned summary of
    the encoder state. This keeps the computation structured around the
    ``(bands, time)`` layout without flattening everything into one sequence.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        n_global_tokens: int = 64,
        time_context_radius: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.n_global_tokens = n_global_tokens
        self.time_context_radius = time_context_radius

        # Intra-band cross-attention (along time)
        self.intra_cross_attn = CrossAttentionBlock(d_model, n_heads, dropout)

        # Inter-band cross-attention (along bands)
        self.inter_cross_attn = CrossAttentionBlock(d_model, n_heads, dropout)

        # Global tokens pooling for intra-band
        self.global_pool = nn.AdaptiveAvgPool1d(n_global_tokens)

    def _get_time_indices(self, T_dec: int, T_enc: int, device: torch.device) -> torch.Tensor:
        """Map decoder time indices to encoder time indices."""
        if T_dec == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        dec_times = torch.arange(T_dec, device=device, dtype=torch.float32)
        enc_times = (dec_times * (T_enc - 1) / (T_dec - 1)).round().long()
        return enc_times.clamp(0, T_enc - 1)

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model) decoder features
            encoder_memory: (B, n_bands, T_enc, d_model) encoder features
        Returns:
            out: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape
        _, _, T_enc, _ = encoder_memory.shape

        # Stage 1: each band attends over pooled encoder time tokens.
        enc_for_pool = encoder_memory.permute(0, 1, 3, 2).reshape(B * N, D, T_enc)
        global_mem = self.global_pool(enc_for_pool).permute(0, 2, 1)
        x_flat = x.reshape(B * N, T, D)
        x_flat = self.intra_cross_attn(x_flat, global_mem)
        x = x_flat.reshape(B, N, T, D)

        # Stage 2: each time frame attends across bands using a local encoder
        # summary aligned to the decoder timeline.
        time_indices = self._get_time_indices(T, T_enc, x.device)
        time_pad = self.time_context_radius
        window_size = 2 * time_pad + 1
        enc_time_padded = F.pad(encoder_memory, (0, 0, time_pad, time_pad), mode='constant', value=0)
        offsets = torch.arange(window_size, device=x.device)
        all_indices = time_indices.unsqueeze(1) + offsets.unsqueeze(0)
        time_windows = enc_time_padded[:, :, all_indices, :]
        enc_time_pooled = time_windows.mean(dim=3)

        x_band = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        enc_time_flat = enc_time_pooled.permute(0, 2, 1, 3).reshape(B * T, N, D)

        x_band = self.inter_cross_attn(x_band, enc_time_flat)
        x = x_band.reshape(B, T, N, D).permute(0, 2, 1, 3)

        return x


class MultiScaleMemoryBank(nn.Module):
    """
    Pooled multi-scale memory bank from encoder levels.

    Pools each encoder level to fixed token counts, then concatenates
    for efficient cross-attention from any decoder layer.
    """

    def __init__(
        self,
        d_model: int,
        k1: int = 64,  # Tokens from level 1 (finest)
        k2: int = 48,  # Tokens from level 2
        k3: int = 32,  # Tokens from level 3
        kb: int = 16,  # Tokens from bottleneck
    ):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.kb = kb
        self.total_tokens = k1 + k2 + k3 + kb

        # Adaptive pooling for each level
        self.pool1 = nn.AdaptiveAvgPool1d(k1)
        self.pool2 = nn.AdaptiveAvgPool1d(k2)
        self.pool3 = nn.AdaptiveAvgPool1d(k3)
        self.poolb = nn.AdaptiveAvgPool1d(kb)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build memory bank from encoder levels.

        Args:
            e1: (B, N, T1, D) finest encoder level
            e2: (B, N, T2, D)
            e3: (B, N, T3, D)
            b: (B, N, Tb, D) bottleneck

        Returns:
            memory: (B, N, K_total, D) pooled memory bank
        """
        B, N, _, D = e1.shape

        def pool_level(x, pool_fn, k):
            # x: (B, N, T, D) -> pool over time
            x = x.permute(0, 1, 3, 2).reshape(B * N, D, -1)  # (B*N, D, T)
            x = pool_fn(x)  # (B*N, D, k)
            return x.reshape(B, N, D, k).permute(0, 1, 3, 2)  # (B, N, k, D)

        m1 = pool_level(e1, self.pool1, self.k1)
        m2 = pool_level(e2, self.pool2, self.k2)
        m3 = pool_level(e3, self.pool3, self.k3)
        mb = pool_level(b, self.poolb, self.kb)

        # Concatenate along token dimension
        memory = torch.cat([m1, m2, m3, mb], dim=2)  # (B, N, K_total, D)

        return memory


class GatedSkipFusion(nn.Module):
    """
    Gated skip connection fusion.

    Instead of simple concatenation, uses a learnable gate to control
    how much of the skip connection to incorporate. This prevents
    reintroducing mixture noise while keeping detail available.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm_x = RMSNorm(d_model)
        self.norm_skip = RMSNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, D) decoder features
            skip: (*, D) skip connection from encoder

        Returns:
            out: (*, D) fused features
        """
        x_norm = self.norm_x(x)
        skip_norm = self.norm_skip(skip)

        # Compute gate from both inputs
        gate = self.gate(torch.cat([x_norm, skip_norm], dim=-1))

        # Gated fusion
        return x + gate * self.proj(skip_norm)


class DecoderBlock(nn.Module):
    """
    Decoder block with:
    - Cross-attention to encoder memory (retrieval)
    - Gated skip connection fusion (at all layers)
    - Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Cross-attention to encoder
        self.cross_attn = BandTimeCrossAttention(d_model, n_heads, dropout)

        # Gated skip fusion (replaces simple concat+proj)
        self.skip_fusion = GatedSkipFusion(d_model)

        # Feed-forward
        self.ff = SwiGLU(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model) decoder features
            encoder_memory: (B, n_bands, T_enc, d_model) encoder to attend to
            skip: (B, n_bands, T, d_model) optional skip from encoder
        Returns:
            out: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape

        # Cross-attention to encoder
        x = self.cross_attn(x, encoder_memory)

        # Add skip connection if provided (gated fusion)
        if skip is not None:
            # Match temporal dimension using crop/pad (not interpolation)
            T_skip = skip.shape[2]
            if T_skip != T:
                if T_skip > T:
                    # Crop center
                    start = (T_skip - T) // 2
                    skip = skip[:, :, start:start + T, :]
                else:
                    # Pad symmetrically
                    pad_total = T - T_skip
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    skip = F.pad(skip, (0, 0, pad_left, pad_right), mode='constant', value=0)

            # Gated skip fusion
            x = self.skip_fusion(x, skip)

        # Feed-forward
        x_flat = x.reshape(B * N, T, D)
        x_flat = self.ff(x_flat)
        x = x_flat.reshape(B, N, T, D)

        return x


class BottleneckAttention(nn.Module):
    """
    Self-attention at the bottleneck for global context.

    Applied at the lowest resolution for full song-level understanding.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        Returns:
            out: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape

        # Flatten bands and time for full attention
        x_flat = x.reshape(B, N * T, D)

        # Self-attention
        x_norm = self.norm(x_flat)
        attn_out = self.self_attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        x = x_flat.reshape(B, N, T, D)

        # Feed-forward
        x_flat = x.reshape(B * N, T, D)
        x_flat = self.ff(x_flat)
        x = x_flat.reshape(B, N, T, D)

        return x
