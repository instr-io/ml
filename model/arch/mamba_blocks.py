"""
Mamba-based blocks for the encoder.

Uses bidirectional Mamba for non-causal audio processing.
Includes dual-path processing (intra-band + inter-band).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, SwiGLU

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "mamba_ssm is required for the separator architecture. "
        "Install project dependencies before importing model.arch.mamba_blocks."
    ) from exc


class BiMamba(nn.Module):
    """
    Bidirectional Mamba - process forward and backward, then combine.

    For non-causal tasks like source separation where we have the full audio.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)

        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Combine forward and backward
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        x_norm = self.norm(x)

        # Forward pass
        fwd = self.mamba_fwd(x_norm)

        # Backward pass (flip, process, flip back)
        bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])

        # Combine
        combined = self.proj(torch.cat([fwd, bwd], dim=-1))

        return x + combined


class ConvModule(nn.Module):
    """Depthwise separable convolution for local patterns."""

    def __init__(self, d_model: int, kernel_size: int = 7):
        super().__init__()
        self.norm = RMSNorm(d_model)
        # Depthwise conv
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        # Pointwise
        self.pwconv = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # Conv needs (B, C, T)
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)

        x = self.act(x)
        x = self.pwconv(x)

        return residual + x


class DualPathBiMambaBlock(nn.Module):
    """
    Dual-path block: intra-band (time) then inter-band (frequency).

    This applies temporal and spectral mixing in separate passes.
    """

    def __init__(self, d_model: int, d_state: int = 32):
        super().__init__()

        # Intra-band: BiMamba along time axis
        self.intra_mamba = BiMamba(d_model, d_state=d_state)

        # Inter-band: BiMamba along band axis
        self.inter_mamba = BiMamba(d_model, d_state=d_state)

        # Feed-forward
        self.ff = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        Returns:
            out: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape

        # Intra-band: process time for each band
        # Reshape: (B * n_bands, T, d_model)
        x_time = x.reshape(B * N, T, D)
        x_time = self.intra_mamba(x_time)
        x = x_time.reshape(B, N, T, D)

        # Inter-band: process bands for each time frame
        # Reshape: (B * T, n_bands, d_model)
        x_band = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x_band = self.inter_mamba(x_band)
        x = x_band.reshape(B, T, N, D).permute(0, 2, 1, 3)

        # Feed-forward (applied to flattened)
        x_flat = x.reshape(B * N, T, D)
        x_flat = self.ff(x_flat)
        x = x_flat.reshape(B, N, T, D)

        return x


class EncoderBlock(nn.Module):
    """
    Encoder block combining conv (local) + dual-path BiMamba (sequential).
    """

    def __init__(self, d_model: int, use_conv: bool = True, d_state: int = 32):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.conv = ConvModule(d_model)

        self.dual_path = DualPathBiMambaBlock(d_model, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape

        # Conv along time for each band (local patterns)
        if self.use_conv:
            x_flat = x.reshape(B * N, T, D)
            x_flat = self.conv(x_flat)
            x = x_flat.reshape(B, N, T, D)

        # Dual-path BiMamba
        x = self.dual_path(x)

        return x


class DownSample(nn.Module):
    """Reduce temporal resolution by 2x."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        Returns:
            out: (B, n_bands, T//2, d_model)
        """
        B, N, T, D = x.shape

        # Process each band
        x = x.reshape(B * N, T, D)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B*N, D, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B*N, T//2, D)

        T_new = x.shape[1]
        x = x.reshape(B, N, T_new, D)

        return x


class UpSample(nn.Module):
    """Increase temporal resolution by 2x."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        Returns:
            out: (B, n_bands, T*2, d_model)
        """
        B, N, T, D = x.shape

        x = x.reshape(B * N, T, D)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B*N, D, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B*N, T*2, D)

        T_new = x.shape[1]
        x = x.reshape(B, N, T_new, D)

        return x
