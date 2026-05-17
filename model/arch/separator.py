"""
Main vocal separator model.

Architecture:
- Mamba encoder (efficient, sequential patterns)
- Attention decoder (retrieval via cross-attention)
- U-Net structure with skip connections
- Band-split with vocal-focused frequency bands
"""

import torch
import torch.nn as nn
from typing import Tuple

from .band_split import BandSplit, BandMerge
from .mamba_blocks import EncoderBlock, DownSample, UpSample, DualPathBiMambaBlock
from .attention_blocks import DecoderBlock, BottleneckAttention, MultiScaleMemoryBank
from .spectrogram import SpectrogramTransform, match_time_length


class VocalSeparator(nn.Module):
    """
    Mamba Encoder + Attention Decoder + U-Net for vocal separation.

    Processes stereo audio to remove vocals and output instrumentals.

    Architecture:
        Input: Stereo waveform (L, R)
            ↓
        STFT: Complex spectrograms
            ↓
        Band Split: 72 vocal-focused bands
            ↓
        Encoder (BiMamba): Multi-scale encoding with skip connections
            ↓
        Bottleneck: Global attention
            ↓
        Decoder (Cross-Attention): Query encoder, use skips
            ↓
        Band Merge: Back to full spectrum
            ↓
        Complex Mask: Magnitude + phase adjustment
            ↓
        ISTFT: Back to waveform
            ↓
        Output: Instrumental waveform (L, R)
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        sr: int = 44100,
        d_model: int = 384,
        n_heads: int = 8,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        n_bottleneck_layers: int = 4,
        dropout: float = 0.0,
        use_mid_side: bool = False,
        d_state: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.d_model = d_model
        self.use_mid_side = use_mid_side

        # Band split/merge
        self.band_split = BandSplit(n_fft, sr, d_model)
        self.band_merge = BandMerge(n_fft, sr, d_model)
        self.n_bands = self.band_split.n_bands

        # ===== ENCODER (Mamba) =====
        # Level 1: Fine temporal patterns
        self.enc1 = nn.ModuleList([
            EncoderBlock(d_model, use_conv=True, d_state=d_state)
            for _ in range(n_encoder_layers)
        ])
        self.down1 = DownSample(d_model)

        # Level 2: Note-level patterns
        self.enc2 = nn.ModuleList([
            EncoderBlock(d_model, use_conv=True, d_state=d_state)
            for _ in range(n_encoder_layers)
        ])
        self.down2 = DownSample(d_model)

        # Level 3: Phrase-level patterns
        self.enc3 = nn.ModuleList([
            EncoderBlock(d_model, use_conv=False, d_state=d_state)  # No conv at deep levels
            for _ in range(n_encoder_layers)
        ])
        self.down3 = DownSample(d_model)

        # ===== BOTTLENECK =====
        self.bottleneck_mamba = nn.ModuleList([
            DualPathBiMambaBlock(d_model, d_state=d_state)
            for _ in range(n_bottleneck_layers // 2)
        ])
        self.bottleneck_attn = nn.ModuleList([
            BottleneckAttention(d_model, n_heads, dropout)
            for _ in range(n_bottleneck_layers // 2)
        ])

        # ===== MEMORY BANK =====
        # Pooled multi-scale memory for efficient cross-attention
        self.memory_bank = MultiScaleMemoryBank(d_model, k1=64, k2=48, k3=32, kb=16)

        # ===== DECODER (Attention) =====
        self.up3 = UpSample(d_model)
        self.dec3 = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])

        self.up2 = UpSample(d_model)
        self.dec2 = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])

        self.up1 = UpSample(d_model)
        self.dec1 = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])

        self.spectrogram = SpectrogramTransform(n_fft, hop_length)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT."""
        return self.spectrogram.stft(x)

    def istft(self, x: torch.Tensor, length: int) -> torch.Tensor:
        """Compute inverse STFT."""
        return self.spectrogram.istft(x, length)

    def forward(
        self,
        audio_L: torch.Tensor,
        audio_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end separation: waveform in, waveform out.

        Args:
            audio_L: (B, n_samples) left channel waveform
            audio_R: (B, n_samples) right channel waveform

        Returns:
            inst_L: (B, n_samples) left instrumental
            inst_R: (B, n_samples) right instrumental
        """
        n_samples = audio_L.shape[-1]

        # Optional Mid/Side conversion
        # Mid = (L+R)/2 contains center-panned content (vocals)
        # Side = (L-R)/2 contains stereo-spread content (instruments)
        if self.use_mid_side:
            audio_mid = (audio_L + audio_R) * 0.5
            audio_side = (audio_L - audio_R) * 0.5
            input_1, input_2 = audio_mid, audio_side
        else:
            input_1, input_2 = audio_L, audio_R

        # STFT
        stft_1 = self.stft(input_1)  # (B, F, T) complex
        stft_2 = self.stft(input_2)

        # Separate in frequency domain
        out_1, out_2 = self.forward_stft(stft_1, stft_2)

        # ISTFT
        inst_1 = self.istft(out_1, n_samples)
        inst_2 = self.istft(out_2, n_samples)

        # Convert back from Mid/Side to L/R
        if self.use_mid_side:
            inst_L = inst_1 + inst_2  # Mid + Side = L
            inst_R = inst_1 - inst_2  # Mid - Side = R
        else:
            inst_L, inst_R = inst_1, inst_2

        return inst_L, inst_R

    def forward_stft(
        self,
        stft_L: torch.Tensor,
        stft_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on STFT spectrograms.

        Args:
            stft_L, stft_R: (B, F, T) complex STFT

        Returns:
            out_L, out_R: (B, F, T) complex masked STFT
        """
        # Band split
        x = self.band_split(stft_L, stft_R)  # (B, n_bands, T, d_model)

        # ===== ENCODE =====
        # Level 1
        e1 = x
        for layer in self.enc1:
            e1 = layer(e1)

        # Level 2
        e2 = self.down1(e1)
        for layer in self.enc2:
            e2 = layer(e2)

        # Level 3
        e3 = self.down2(e2)
        for layer in self.enc3:
            e3 = layer(e3)

        # ===== BOTTLENECK =====
        b = self.down3(e3)

        # Alternate Mamba and Attention at bottleneck
        for mamba, attn in zip(self.bottleneck_mamba, self.bottleneck_attn):
            b = mamba(b)
            b = attn(b)

        # ===== BUILD MEMORY BANK =====
        # Pool all encoder levels for efficient cross-attention
        memory = self.memory_bank(e1, e2, e3, b)  # (B, N, K_total, D)

        # ===== DECODE =====
        # Level 3: pass skip to ALL layers (gated fusion handles it)
        d3 = self.up3(b)
        d3 = match_time_length(d3, e3.shape[2])
        for layer in self.dec3:
            d3 = layer(d3, encoder_memory=memory, skip=e3)

        # Level 2
        d2 = self.up2(d3)
        d2 = match_time_length(d2, e2.shape[2])
        for layer in self.dec2:
            d2 = layer(d2, encoder_memory=memory, skip=e2)

        # Level 1
        d1 = self.up1(d2)
        d1 = match_time_length(d1, e1.shape[2])
        for layer in self.dec1:
            d1 = layer(d1, encoder_memory=memory, skip=e1)

        # Band merge and mask estimation
        out_L, out_R = self.band_merge(d1, stft_L, stft_R)

        return out_L, out_R

def create_model(
    n_fft: int = 2048,
    hop_length: int = 512,
    sr: int = 44100,
    d_model: int = 384,
    n_heads: int = 8,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    n_bottleneck_layers: int = 4,
    dropout: float = 0.0,
    use_mid_side: bool = False,
    d_state: int = 32,
) -> VocalSeparator:
    """Factory function to create the separator model."""
    return VocalSeparator(
        n_fft=n_fft,
        hop_length=hop_length,
        sr=sr,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_bottleneck_layers=n_bottleneck_layers,
        dropout=dropout,
        use_mid_side=use_mid_side,
        d_state=d_state,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_model(d_model=384)
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")

    # Test forward pass
    B, T = 2, 44100 * 5  # 5 seconds
    audio_L = torch.randn(B, T)
    audio_R = torch.randn(B, T)

    with torch.no_grad():
        inst_L, inst_R = model(audio_L, audio_R)

    print(f"Input shape: ({B}, {T})")
    print(f"Output shape: {inst_L.shape}")
