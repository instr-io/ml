"""Small architecture smoke tests."""

import numpy as np
import torch

from model.arch.band_split import compute_band_edges
from model.arch.separator import create_model


def test_compute_band_edges_cover_spectrum():
    edges = compute_band_edges(n_fft=2048, sr=44100)

    assert edges[0][0] == 0
    assert edges[-1][1] == 1025

    for i in range(len(edges) - 1):
        assert edges[i][1] == edges[i + 1][0]


def test_stft_roundtrip_preserves_signal():
    model = create_model(d_model=64, n_encoder_layers=1, n_decoder_layers=1, n_bottleneck_layers=2)

    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = torch.from_numpy(np.sin(2 * np.pi * 440 * t).astype(np.float32))

    stft_out = model.stft(audio)
    reconstructed = model.istft(stft_out, len(audio))

    error = (reconstructed - audio).abs().mean().item()
    assert error < 0.01


def test_model_forward_preserves_waveform_shape():
    model = create_model(
        n_fft=2048,
        hop_length=256,
        sr=44100,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_bottleneck_layers=2,
    )
    model.eval()

    batch_size = 1
    n_samples = 22050
    audio_left = torch.randn(batch_size, n_samples)
    audio_right = torch.randn(batch_size, n_samples)

    with torch.no_grad():
        inst_left, inst_right = model(audio_left, audio_right)

    assert inst_left.shape == (batch_size, n_samples)
    assert inst_right.shape == (batch_size, n_samples)
    assert torch.isfinite(inst_left).all()
    assert torch.isfinite(inst_right).all()
