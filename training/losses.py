"""
Loss functions for vocal separation training.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagnitudeLoss(nn.Module):
    """L1 loss on STFT magnitude."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred.abs(), target.abs())


class BandWiseLoss(nn.Module):
    """
    Band-wise log-magnitude L1 loss.

    Computes an L1 loss in log-magnitude space per frequency band so quiet and
    loud regions are both represented in the objective.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_bands: int = 4,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.eps = eps
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        if pred.dim() == 3:
            batch, channels, steps = pred.shape
            pred = pred.reshape(batch * channels, steps)
            target = target.reshape(batch * channels, steps)

        pred_stft = torch.stft(pred, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        target_stft = torch.stft(target, self.n_fft, self.hop_length, window=self.window, return_complex=True)

        return self.forward_from_stft(pred_stft, target_stft)

    def forward_from_stft(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
        """Compute band-wise loss from precomputed STFTs."""
        pred_log = torch.log(pred_stft.abs() + self.eps)
        target_log = torch.log(target_stft.abs() + self.eps)

        n_freqs = pred_log.shape[1]
        band_size = n_freqs // self.n_bands
        trim_freqs = band_size * self.n_bands

        pred_bands = pred_log[:, :trim_freqs, :].reshape(-1, self.n_bands, band_size, pred_log.shape[2])
        target_bands = target_log[:, :trim_freqs, :].reshape(-1, self.n_bands, band_size, target_log.shape[2])

        return F.l1_loss(pred_bands, target_bands)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss with three fixed analysis sizes."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w512", torch.hann_window(512), persistent=False)
        self.register_buffer("w1024", torch.hann_window(1024), persistent=False)
        self.register_buffer("w2048", torch.hann_window(2048), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])

        eps = 1e-7

        p512 = torch.stft(pred, 512, 128, window=self.w512, return_complex=True).abs()
        t512 = torch.stft(target, 512, 128, window=self.w512, return_complex=True).abs()
        loss = F.l1_loss(p512, t512) + F.l1_loss(torch.log(p512 + eps), torch.log(t512 + eps))

        p1024 = torch.stft(pred, 1024, 256, window=self.w1024, return_complex=True).abs()
        t1024 = torch.stft(target, 1024, 256, window=self.w1024, return_complex=True).abs()
        loss = loss + F.l1_loss(p1024, t1024) + F.l1_loss(torch.log(p1024 + eps), torch.log(t1024 + eps))

        p2048 = torch.stft(pred, 2048, 512, window=self.w2048, return_complex=True).abs()
        t2048 = torch.stft(target, 2048, 512, window=self.w2048, return_complex=True).abs()
        loss = loss + F.l1_loss(p2048, t2048) + F.l1_loss(torch.log(p2048 + eps), torch.log(t2048 + eps))

        return loss / 3.0


class SISDRLoss(nn.Module):
    """Scale-invariant signal-to-distortion ratio loss."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = pred.float()
        target = target.float()

        if pred.dim() == 3:
            batch, channels, steps = pred.shape
            pred = pred.reshape(batch, channels * steps)
            target = target.reshape(batch, channels * steps)

        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        dot = (pred * target).sum(dim=-1, keepdim=True)
        s_target_sq = (target ** 2).sum(dim=-1, keepdim=True) + self.eps
        s_target = dot * target / s_target_sq
        e_noise = pred - s_target

        signal_power = (s_target ** 2).sum(dim=-1) + self.eps
        noise_power = (e_noise ** 2).sum(dim=-1) + self.eps

        si_sdr = 10 * torch.log10((signal_power / noise_power) + 1e-8)
        loss = torch.clamp((40 - si_sdr) / 20, min=0, max=2)

        return loss.mean(), si_sdr.mean()


class SpectralSDRLoss(nn.Module):
    """
    SDR computed on STFT magnitudes.

    This objective is more tolerant of small timing mismatches than waveform
    SDR losses because it compares magnitude spectra instead of raw samples.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, eps: float = 1e-6):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = pred.float()
        target = target.float()

        if pred.dim() == 3:
            batch, channels, steps = pred.shape
            pred = pred.reshape(batch * channels, steps)
            target = target.reshape(batch * channels, steps)

        pred_stft = torch.stft(pred, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        target_stft = torch.stft(target, self.n_fft, self.hop_length, window=self.window, return_complex=True)

        return self.forward_from_stft(pred_stft, target_stft)

    def forward_from_stft(
        self,
        pred_stft: torch.Tensor,
        target_stft: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spectral SDR from precomputed STFTs."""
        pred_flat = pred_stft.abs().reshape(pred_stft.shape[0], -1)
        target_flat = target_stft.abs().reshape(target_stft.shape[0], -1)

        signal_power = (target_flat ** 2).sum(dim=-1) + self.eps
        noise_power = ((target_flat - pred_flat) ** 2).sum(dim=-1) + self.eps

        sdr = 10 * torch.log10(signal_power / noise_power)
        loss = torch.clamp((40 - sdr) / 40, min=0, max=1)

        return loss.mean(), sdr.mean()


class SeparationLoss(nn.Module):
    """Combined training loss for stereo source separation."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        mag_weight: float = 0.5,
        mr_stft_weight: float = 1.0,
        si_sdr_weight: float = 0.0,
        spectral_sdr_weight: float = 1.0,
        band_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag_weight = mag_weight
        self.mr_stft_weight = mr_stft_weight
        self.si_sdr_weight = si_sdr_weight
        self.spectral_sdr_weight = spectral_sdr_weight
        self.band_weight = band_weight

        self.mag_loss = MagnitudeLoss()
        self.mr_stft_loss = MultiResolutionSTFTLoss()
        self.si_sdr_loss = SISDRLoss()
        self.spectral_sdr_loss = SpectralSDRLoss(n_fft, hop_length)
        self.band_loss = BandWiseLoss(n_fft, hop_length)

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    @staticmethod
    def _stack_stereo(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.stack([left, right], dim=1)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)

    def forward(
        self,
        pred_L: torch.Tensor,
        pred_R: torch.Tensor,
        target_L: torch.Tensor,
        target_R: torch.Tensor,
        original_L: torch.Tensor = None,
        original_R: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        loss_dict = {}
        pred_stereo = None
        target_stereo = None
        pred_stft = None
        target_stft = None

        def get_stereo_pair() -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal pred_stereo, target_stereo
            if pred_stereo is None:
                pred_stereo = self._stack_stereo(pred_L, pred_R)
                target_stereo = self._stack_stereo(target_L, target_R)
            return pred_stereo, target_stereo

        def get_stft_pair() -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal pred_stft, target_stft
            if pred_stft is None:
                pred_stereo_local, target_stereo_local = get_stereo_pair()
                pred_stft = self.stft(pred_stereo_local.reshape(-1, pred_stereo_local.shape[-1]))
                target_stft = self.stft(target_stereo_local.reshape(-1, target_stereo_local.shape[-1]))
            return pred_stft, target_stft

        if original_L is not None:
            rms = ((original_L ** 2).mean() + (original_R ** 2).mean()).sqrt()
        else:
            rms = ((target_L ** 2).mean() + (target_R ** 2).mean()).sqrt()
        rms_weight = 1.0 + rms.clamp(0, 0.5) * 2.0
        loss_dict["rms_weight"] = rms_weight.item()

        if self.mag_weight > 0:
            pred_stft_local, target_stft_local = get_stft_pair()
            mag_loss = self.mag_loss(pred_stft_local, target_stft_local)
            loss_dict["magnitude"] = mag_loss.item()
        else:
            mag_loss = 0.0

        if self.mr_stft_weight > 0:
            pred_stereo, target_stereo = get_stereo_pair()
            mr_stft_loss = self.mr_stft_loss(pred_stereo, target_stereo)
            loss_dict["mr_stft"] = mr_stft_loss.item()
        else:
            mr_stft_loss = 0.0

        if self.si_sdr_weight > 0:
            pred_stereo, target_stereo = get_stereo_pair()
            si_sdr_loss, si_sdr_value = self.si_sdr_loss(pred_stereo, target_stereo)
            loss_dict["si_sdr"] = si_sdr_value.item()
        else:
            si_sdr_loss = 0.0

        if self.spectral_sdr_weight > 0:
            pred_stft_local, target_stft_local = get_stft_pair()
            spectral_sdr_loss, spectral_sdr_value = self.spectral_sdr_loss.forward_from_stft(
                pred_stft_local,
                target_stft_local,
            )
            loss_dict["spectral_sdr"] = spectral_sdr_value.item()
        else:
            spectral_sdr_loss = 0.0

        if self.band_weight > 0:
            pred_stft_local, target_stft_local = get_stft_pair()
            band_loss = self.band_loss.forward_from_stft(pred_stft_local, target_stft_local)
            loss_dict["band"] = band_loss.item()
        else:
            band_loss = 0.0

        total_loss = (
            self.mag_weight * mag_loss
            + self.mr_stft_weight * mr_stft_loss
            + self.si_sdr_weight * si_sdr_loss
            + self.spectral_sdr_weight * spectral_sdr_loss
            + self.band_weight * band_loss
        )

        total_loss = total_loss * rms_weight
        total_loss = torch.clamp(total_loss, max=20.0)
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict
