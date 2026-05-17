"""
Vocal-focused band splitting and merging.

Divides the frequency spectrum into non-uniform bands with more resolution
in the vocal frequency range (80-4000 Hz).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .common import RMSNorm


def compute_band_edges(n_fft: int = 2048, sr: int = 44100) -> List[Tuple[int, int]]:
    """
    Compute non-uniform band boundaries focused on vocal frequencies.

    Returns:
        List of (start_bin, end_bin) tuples for each band
    """
    freq_per_bin = sr / n_fft  # ~21.5 Hz per bin

    # Define frequency ranges and number of bands for each
    # (start_hz, end_hz, num_bands)
    ranges = [
        (0, 80, 2),           # Sub-bass: minimal resolution
        (80, 250, 12),        # Bass/low vocals: vocal fundamentals
        (250, 500, 12),       # Low-mid: vocal body
        (500, 2000, 20),      # Mid: formants F1, F2 (critical for vowels)
        (2000, 4000, 12),     # Upper-mid: formant F3, vocal clarity
        (4000, 8000, 8),      # Presence: sibilants, consonants
        (8000, sr // 2, 6),   # Air: harmonics, less critical
    ]

    band_edges = []

    for start_hz, end_hz, num_bands in ranges:
        start_bin = int(start_hz / freq_per_bin)
        end_bin = min(int(end_hz / freq_per_bin), n_fft // 2 + 1)

        if end_bin <= start_bin:
            continue

        # Distribute bands within this range
        bins_per_band = (end_bin - start_bin) / num_bands

        for i in range(num_bands):
            b_start = start_bin + int(i * bins_per_band)
            b_end = start_bin + int((i + 1) * bins_per_band)
            if b_start < b_end:
                band_edges.append((b_start, b_end))

    return band_edges


class BandSplit(nn.Module):
    """
    Split complex STFT into vocal-focused frequency bands and embed.

    Input: L and R complex STFT (B, F, T)
    Output: Band embeddings (B, n_bands, T, d_model)
    """

    def __init__(
        self,
        n_fft: int = 2048,
        sr: int = 44100,
        d_model: int = 384,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.d_model = d_model

        # Compute band boundaries
        self.band_edges = compute_band_edges(n_fft, sr)
        self.n_bands = len(self.band_edges)

        # Per-band embedding layers
        # Input: real + imag for L and R = 4 values per freq bin
        self.band_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear((end - start) * 4, d_model),
                RMSNorm(d_model),
            )
            for start, end in self.band_edges
        ])

        # Band positional embedding (learnable)
        self.band_pos = nn.Parameter(torch.randn(1, self.n_bands, 1, d_model) * 0.02)

    def forward(self, L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L, R: (B, F, T) complex STFT
        Returns:
            x: (B, n_bands, T, d_model) band embeddings
        """
        B, F, T = L.shape

        # Stack L and R: (B, F, T, 4) for [L_real, L_imag, R_real, R_imag]
        x = torch.stack([L.real, L.imag, R.real, R.imag], dim=-1)

        # Embed each band
        band_features = []
        for i, (start, end) in enumerate(self.band_edges):
            # Extract band: (B, band_width, T, 4)
            band = x[:, start:end, :, :]

            # Reshape for linear: (B, T, band_width * 4)
            band = band.permute(0, 2, 1, 3).reshape(B, T, -1)

            # Embed: (B, T, d_model)
            band_emb = self.band_embeds[i](band)
            band_features.append(band_emb)

        # Stack bands: (B, n_bands, T, d_model)
        x = torch.stack(band_features, dim=1)

        # Add band positional embedding
        x = x + self.band_pos

        return x


class BandMerge(nn.Module):
    """
    Merge band features back to full frequency spectrum and estimate complex masks.

    Input: Band features (B, n_bands, T, d_model)
    Output: Complex masks for L and R (B, F, T) each
    """

    def __init__(
        self,
        n_fft: int = 2048,
        sr: int = 44100,
        d_model: int = 384,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.d_model = d_model

        # Get band boundaries
        self.band_edges = compute_band_edges(n_fft, sr)
        self.n_bands = len(self.band_edges)

        # Per-band projection to frequency bins
        # Output 4 values per freq bin: L_mag, L_phase, R_mag, R_phase
        self.band_projections = nn.ModuleList([
            nn.Sequential(
                RMSNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, (end - start) * 4),
            )
            for start, end in self.band_edges
        ])

    def forward(
        self,
        x: torch.Tensor,
        input_L: torch.Tensor,
        input_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_bands, T, d_model) band features
            input_L, input_R: (B, F, T) complex input STFT
        Returns:
            out_L, out_R: (B, F, T) complex masked STFT
        """
        B, N, T, D = x.shape
        F = self.n_freqs

        # Initialize output tensors
        mask_raw = torch.zeros(B, F, T, 4, device=x.device, dtype=x.dtype)

        # Project each band and place in full spectrum
        for i, (start, end) in enumerate(self.band_edges):
            band_feat = x[:, i, :, :]  # (B, T, d_model)
            band_out = self.band_projections[i](band_feat)  # (B, T, width * 4)

            # Reshape: (B, T, width, 4)
            width = end - start
            band_out = band_out.view(B, T, width, 4)

            # Place in full spectrum: (B, width, T, 4)
            band_out = band_out.permute(0, 2, 1, 3)
            mask_raw[:, start:end, :, :] = band_out

        # Convert to complex masks
        # Magnitude mask: sigmoid for [0, 1]
        L_mag = torch.sigmoid(mask_raw[..., 0])
        R_mag = torch.sigmoid(mask_raw[..., 2])

        # Phase adjustment: tanh * pi for [-pi, pi]
        L_phase = torch.tanh(mask_raw[..., 1]) * torch.pi
        R_phase = torch.tanh(mask_raw[..., 3]) * torch.pi

        # Build complex masks
        mask_L = L_mag * torch.exp(1j * L_phase)
        mask_R = R_mag * torch.exp(1j * R_phase)

        # Apply masks to input
        out_L = input_L * mask_L
        out_R = input_R * mask_R

        return out_L, out_R
