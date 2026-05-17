"""Shared spectrogram helpers for separator architectures."""

import torch
import torch.nn as nn


class SpectrogramTransform(nn.Module):
    """Wrap STFT and inverse STFT with a shared analysis window."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute a complex STFT for a mono waveform batch or vector."""
        return torch.stft(
            audio,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )

    def istft(self, spectrogram: torch.Tensor, length: int) -> torch.Tensor:
        """Invert a complex STFT back to the waveform domain."""
        return torch.istft(
            spectrogram,
            self.n_fft,
            self.hop_length,
            window=self.window,
            length=length,
        )


def match_time_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Crop or pad the time axis of ``x`` to match ``target_length``."""
    current_length = x.shape[2]

    if current_length == target_length:
        return x

    if current_length > target_length:
        start = (current_length - target_length) // 2
        return x[:, :, start:start + target_length, :]

    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return torch.nn.functional.pad(
        x,
        (0, 0, pad_left, pad_right),
        mode="constant",
        value=0,
    )
