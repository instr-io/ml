"""
Selective SSM blocks for the encoder.

Supports bidirectional Mamba and Mamba-3 backends for non-causal audio
processing, plus dual-path processing (intra-band + inter-band).
"""

import torch
import torch.nn as nn

from .common import RMSNorm, SwiGLU

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False

try:
    from mamba_ssm import Mamba3
    MAMBA3_AVAILABLE = True
except ImportError:
    Mamba3 = None
    MAMBA3_AVAILABLE = False

VALID_SSM_VARIANTS = ("mamba", "mamba3")


def _normalize_ssm_variant(ssm_variant: str) -> str:
    """Return a validated lowercase backend name."""
    variant = (ssm_variant or "mamba").lower()
    if variant not in VALID_SSM_VARIANTS:
        raise ValueError(
            f"Unsupported ssm_variant '{ssm_variant}'. "
            f"Expected one of: {', '.join(VALID_SSM_VARIANTS)}"
        )
    return variant


def _create_ssm_module(
    d_model: int,
    d_state: int = 32,
    d_conv: int = 4,
    expand: int = 2,
    ssm_variant: str = "mamba",
    mamba3_headdim: int = 64,
    mamba3_is_mimo: bool = False,
    mamba3_mimo_rank: int = 4,
    mamba3_chunk_size: int = 32,
    mamba3_is_outproj_norm: bool = False,
) -> nn.Module:
    """Build the configured selective SSM backend."""
    variant = _normalize_ssm_variant(ssm_variant)

    if variant == "mamba":
        if not MAMBA_AVAILABLE:
            raise RuntimeError(
                "ssm_variant='mamba' requires mamba_ssm.Mamba. "
                "Install project dependencies before creating the separator."
            )
        return Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    if not MAMBA3_AVAILABLE:
        raise RuntimeError(
            "ssm_variant='mamba3' requires mamba_ssm.Mamba3, which is not "
            "available in the current environment. Install state-spaces/mamba "
            "from source to enable official Mamba-3 support."
        )

    return Mamba3(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=mamba3_headdim,
        is_mimo=mamba3_is_mimo,
        mimo_rank=mamba3_mimo_rank,
        chunk_size=mamba3_chunk_size,
        is_outproj_norm=mamba3_is_outproj_norm,
    )


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
        ssm_variant: str = "mamba",
        mamba3_headdim: int = 64,
        mamba3_is_mimo: bool = False,
        mamba3_mimo_rank: int = 4,
        mamba3_chunk_size: int = 32,
        mamba3_is_outproj_norm: bool = False,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ssm_variant = _normalize_ssm_variant(ssm_variant)

        common_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ssm_variant=self.ssm_variant,
            mamba3_headdim=mamba3_headdim,
            mamba3_is_mimo=mamba3_is_mimo,
            mamba3_mimo_rank=mamba3_mimo_rank,
            mamba3_chunk_size=mamba3_chunk_size,
            mamba3_is_outproj_norm=mamba3_is_outproj_norm,
        )
        self.mamba_fwd = _create_ssm_module(**common_kwargs)
        self.mamba_bwd = _create_ssm_module(**common_kwargs)

        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        x_norm = self.norm(x)

        fwd = self.mamba_fwd(x_norm)
        bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        combined = self.proj(torch.cat([fwd, bwd], dim=-1))

        return x + combined


class ConvModule(nn.Module):
    """Depthwise separable convolution for local patterns."""

    def __init__(self, d_model: int, kernel_size: int = 7):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.pwconv = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

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

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        ssm_variant: str = "mamba",
        mamba3_headdim: int = 64,
        mamba3_is_mimo: bool = False,
        mamba3_mimo_rank: int = 4,
        mamba3_chunk_size: int = 32,
        mamba3_is_outproj_norm: bool = False,
    ):
        super().__init__()

        common_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ssm_variant=ssm_variant,
            mamba3_headdim=mamba3_headdim,
            mamba3_is_mimo=mamba3_is_mimo,
            mamba3_mimo_rank=mamba3_mimo_rank,
            mamba3_chunk_size=mamba3_chunk_size,
            mamba3_is_outproj_norm=mamba3_is_outproj_norm,
        )

        self.intra_mamba = BiMamba(**common_kwargs)
        self.inter_mamba = BiMamba(**common_kwargs)
        self.ff = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        Returns:
            out: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape

        x_time = x.reshape(B * N, T, D)
        x_time = self.intra_mamba(x_time)
        x = x_time.reshape(B, N, T, D)

        x_band = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x_band = self.inter_mamba(x_band)
        x = x_band.reshape(B, T, N, D).permute(0, 2, 1, 3)

        x_flat = x.reshape(B * N, T, D)
        x_flat = self.ff(x_flat)
        x = x_flat.reshape(B, N, T, D)

        return x


class EncoderBlock(nn.Module):
    """
    Encoder block combining conv (local) + dual-path BiMamba (sequential).
    """

    def __init__(
        self,
        d_model: int,
        use_conv: bool = True,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        ssm_variant: str = "mamba",
        mamba3_headdim: int = 64,
        mamba3_is_mimo: bool = False,
        mamba3_mimo_rank: int = 4,
        mamba3_chunk_size: int = 32,
        mamba3_is_outproj_norm: bool = False,
    ):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.conv = ConvModule(d_model)

        self.dual_path = DualPathBiMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ssm_variant=ssm_variant,
            mamba3_headdim=mamba3_headdim,
            mamba3_is_mimo=mamba3_is_mimo,
            mamba3_mimo_rank=mamba3_mimo_rank,
            mamba3_chunk_size=mamba3_chunk_size,
            mamba3_is_outproj_norm=mamba3_is_outproj_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_bands, T, d_model)
        """
        B, N, T, D = x.shape

        if self.use_conv:
            x_flat = x.reshape(B * N, T, D)
            x_flat = self.conv(x_flat)
            x = x_flat.reshape(B, N, T, D)

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

        x = x.reshape(B * N, T, D)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

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
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        T_new = x.shape[1]
        x = x.reshape(B, N, T_new, D)

        return x
