"""
Inference entrypoint for vocal removal.

Hi-res mask reuse is the default behavior: when the input sample rate is
higher than the model sample rate, the script preserves the original sample
rate in the output instead of forcing a 44.1kHz export.

Usage:
    python -m inference.infer input.wav output.wav --checkpoint path/to/checkpoint.pt
"""

import argparse
from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from model.checkpoint import load_separator
from model.config import load_runtime_config

logger = logging.getLogger(__name__)


class VocalRemover:
    """Inference wrapper with hi-res output as the default behavior."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        loaded = load_separator(checkpoint_path, device=device, config_path=config_path)
        self.device = loaded.device
        self.model = loaded.model
        self.config = loaded.config
        self.model_sr = self.config.audio.sample_rate
        self.n_fft = self.config.audio.n_fft
        self.hop_length = self.config.audio.hop_length

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info("Inference precision: fp32")
        logger.info(
            "Config: sr=%s d_model=%s layers=%s/%s/%s",
            self.model_sr,
            self.config.model.d_model,
            self.config.model.n_encoder_layers,
            self.config.model.n_decoder_layers,
            self.config.model.n_bottleneck_layers,
        )

    @staticmethod
    def _validate_chunking(chunk_seconds: float, overlap_seconds: float):
        if chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds must be > 0, got {chunk_seconds}")
        if overlap_seconds < 0:
            raise ValueError(f"overlap_seconds must be >= 0, got {overlap_seconds}")
        if overlap_seconds >= chunk_seconds:
            raise ValueError(
                f"overlap_seconds ({overlap_seconds}) must be smaller than "
                f"chunk_seconds ({chunk_seconds})"
            )

    def _precision_context(self):
        # Keep inference in fp32 to match training stability and avoid silent AMP use.
        return nullcontext()

    def process(
        self,
        audio_path: str,
        output_path: str,
        chunk_seconds: Optional[float] = None,
        overlap_seconds: float = 1.0,
        match_loudness: bool = True,
        preserve_hires: bool = True,
        comparison_dir: Optional[str] = None,
    ):
        """Process an audio file and save the instrumental."""
        if chunk_seconds is None:
            chunk_seconds = self.config.audio.chunk_seconds
            logger.info(f"Using chunk_seconds={chunk_seconds} from config")
        self._validate_chunking(chunk_seconds, overlap_seconds)

        audio_orig, orig_sr = torchaudio.load(audio_path)
        audio_orig = self._ensure_stereo(audio_orig)
        logger.info(f"Loaded audio: {audio_orig.shape} at {orig_sr}Hz")

        if preserve_hires and orig_sr != self.model_sr:
            logger.info("Using hi-res mask reuse output")
            audio_model = torchaudio.transforms.Resample(orig_sr, self.model_sr)(audio_orig)
            output = self._process_hires_waveform(
                audio_model,
                audio_orig,
                orig_sr,
                chunk_seconds,
                overlap_seconds,
            )
            output_sr = orig_sr

            if comparison_dir:
                self._write_comparison_outputs(
                    comparison_dir,
                    Path(audio_path).stem,
                    audio_model,
                    audio_orig,
                    orig_sr,
                    chunk_seconds,
                    overlap_seconds,
                )
        else:
            if orig_sr != self.model_sr:
                logger.info(
                    "Hi-res output disabled; resampling from %sHz to %sHz",
                    orig_sr,
                    self.model_sr,
                )
                audio_input = torchaudio.transforms.Resample(orig_sr, self.model_sr)(audio_orig)
            else:
                audio_input = audio_orig

            output = self._process_standard_waveform(
                audio_input,
                self.model_sr,
                chunk_seconds,
                overlap_seconds,
            )
            output_sr = self.model_sr

        input_for_loudness = audio_orig if output_sr == orig_sr else (
            torchaudio.transforms.Resample(orig_sr, output_sr)(audio_orig)
            if orig_sr != output_sr
            else audio_orig
        )

        if match_loudness:
            output = self._match_loudness(input_for_loudness, output, output_sr)

        output = self._limit_peak(output)
        self._save_audio(output_path, output, output_sr)

    def _ensure_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        return audio

    def _limit_peak(self, audio: torch.Tensor, max_peak: float = 0.99) -> torch.Tensor:
        peak = audio.abs().max()
        if peak > max_peak:
            audio = audio * (max_peak / peak)
            logger.info(f"Scaled output to prevent clipping (peak was {peak:.2f})")
        return audio

    def _save_audio(self, output_path: str, audio: torch.Tensor, sample_rate: int):
        logger.info(f"Saving to: {output_path}")
        sf.write(output_path, audio.T.numpy(), sample_rate, subtype="FLOAT")
        logger.info("Done!")

    def _match_loudness(
        self,
        input_audio: torch.Tensor,
        output_audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sample_rate)
            input_np = input_audio.T.numpy()
            output_np = output_audio.T.numpy()
            input_lufs = meter.integrated_loudness(input_np)
            output_lufs = meter.integrated_loudness(output_np)

            if output_lufs > -70:
                output_np = pyln.normalize.loudness(output_np, output_lufs, input_lufs)
                logger.info(f"LUFS matched: {output_lufs:.1f} -> {input_lufs:.1f} LUFS")
                return torch.from_numpy(output_np.T).float()
        except ImportError:
            pass

        input_rms = (input_audio ** 2).mean().sqrt()
        output_rms = (output_audio ** 2).mean().sqrt()
        if output_rms > 1e-8:
            scale = (input_rms / output_rms).clamp(0.5, 2.0)
            output_audio = output_audio * scale
            logger.info(f"RMS matched (install pyloudnorm for LUFS): scaled {scale:.2f}x")
        return output_audio

    def _process_standard_waveform(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
    ) -> torch.Tensor:
        original_length = audio.shape[1]
        chunk_outputs, padded_length, overlap_samples, chunk_samples, hop_samples = self._run_chunked_model(
            audio,
            sample_rate,
            chunk_seconds,
            overlap_seconds,
        )

        output = self._crossfade_chunks(
            chunk_outputs,
            padded_length,
            chunk_samples,
            overlap_samples,
            hop_samples,
            apply_gain_correction=True,
        )

        output = output[:, :original_length]
        return self._limit_peak(output)

    def _run_chunked_model(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
    ):
        chunk_samples = int(chunk_seconds * sample_rate)
        overlap_samples = int(overlap_seconds * sample_rate)
        hop_samples = chunk_samples - overlap_samples

        original_length = audio.shape[1]
        n_chunks = max(1, (original_length + hop_samples - 1) // hop_samples)
        padded_length = (n_chunks - 1) * hop_samples + chunk_samples
        if padded_length > original_length:
            audio = torch.nn.functional.pad(audio, (0, padded_length - original_length))

        chunk_outputs = []
        logger.info(f"Processing {n_chunks} chunks ({chunk_seconds}s each, {overlap_seconds}s overlap)...")

        with torch.inference_mode():
            for i in range(n_chunks):
                start = i * hop_samples
                end = start + chunk_samples

                chunk = audio[:, start:end].to(self.device)
                with self._precision_context():
                    inst_L, inst_R = self.model(chunk[0].unsqueeze(0), chunk[1].unsqueeze(0))
                chunk_outputs.append(torch.stack([inst_L.squeeze(0).cpu(), inst_R.squeeze(0).cpu()]))

                progress = (i + 1) / n_chunks * 100
                print(f"  Chunk {i + 1}/{n_chunks} ({progress:.0f}%)", end="\r")

        print()
        return chunk_outputs, padded_length, overlap_samples, chunk_samples, hop_samples

    def _crossfade_chunks(
        self,
        chunk_outputs,
        padded_length: int,
        chunk_samples: int,
        overlap_samples: int,
        hop_samples: int,
        apply_gain_correction: bool,
    ) -> torch.Tensor:
        output = torch.zeros(2, padded_length)
        weights = torch.zeros(padded_length)

        if apply_gain_correction:
            gains = self._solve_overlap_gains(chunk_outputs, overlap_samples)
        else:
            gains = [1.0] * len(chunk_outputs)

        fade_in = torch.linspace(0, 1, overlap_samples) if overlap_samples > 0 else None
        fade_out = torch.linspace(1, 0, overlap_samples) if overlap_samples > 0 else None

        for i, chunk_out in enumerate(chunk_outputs):
            start = i * hop_samples
            end = start + chunk_samples
            chunk_weight = torch.ones(chunk_samples)
            if overlap_samples > 0 and i > 0:
                chunk_weight[:overlap_samples] *= fade_in
            if overlap_samples > 0 and i < len(chunk_outputs) - 1:
                chunk_weight[-overlap_samples:] *= fade_out

            chunk_out = chunk_out * gains[i]
            output[0, start:end] += chunk_out[0] * chunk_weight
            output[1, start:end] += chunk_out[1] * chunk_weight
            weights[start:end] += chunk_weight

        weights = weights.clamp(min=1e-8)
        return output / weights

    def _solve_overlap_gains(self, chunk_outputs, overlap_samples: int):
        if len(chunk_outputs) <= 1 or overlap_samples <= 0:
            return [1.0] * len(chunk_outputs)

        ratios = []
        for i in range(len(chunk_outputs) - 1):
            end_overlap = chunk_outputs[i][:, -overlap_samples:]
            start_overlap = chunk_outputs[i + 1][:, :overlap_samples]
            end_rms = (end_overlap ** 2).mean().sqrt().item()
            start_rms = (start_overlap ** 2).mean().sqrt().item()
            ratios.append(end_rms / start_rms if start_rms > 1e-8 and end_rms > 1e-8 else 1.0)

        log_gains = [0.0]
        for ratio in ratios:
            log_gains.append(log_gains[-1] + np.log(ratio))

        median_lg = np.median(log_gains)
        max_log_dev = np.log(1.25)
        log_gains = [max(-max_log_dev, min(max_log_dev, lg - median_lg)) for lg in log_gains]
        gains = [np.exp(lg) for lg in log_gains]

        if max(gains) / min(gains) > 1.05:
            logger.info(f"Overlap gain correction: {min(gains):.2f}x - {max(gains):.2f}x")

        return gains

    def _process_hires_waveform(
        self,
        audio_model: torch.Tensor,
        audio_orig: torch.Tensor,
        orig_sr: int,
        chunk_seconds: float,
        overlap_seconds: float,
    ) -> torch.Tensor:
        sr_ratio = orig_sr / self.model_sr
        hires_n_fft = round(self.n_fft * sr_ratio)
        if hires_n_fft % 2 != 0:
            hires_n_fft += 1
        hires_hop = round(self.hop_length * sr_ratio)
        hires_n_freqs = hires_n_fft // 2 + 1
        model_n_freqs = self.n_fft // 2 + 1

        chunk_samples_model = int(chunk_seconds * self.model_sr)
        chunk_samples_orig = int(chunk_seconds * orig_sr)
        overlap_model = int(overlap_seconds * self.model_sr)
        overlap_orig = int(overlap_seconds * orig_sr)
        hop_model = chunk_samples_model - overlap_model
        hop_orig = chunk_samples_orig - overlap_orig

        original_length = audio_orig.shape[1]
        n_chunks = max(1, (audio_model.shape[1] + hop_model - 1) // hop_model)

        padded_model = (n_chunks - 1) * hop_model + chunk_samples_model
        padded_orig = (n_chunks - 1) * hop_orig + chunk_samples_orig

        if padded_model > audio_model.shape[1]:
            audio_model = torch.nn.functional.pad(audio_model, (0, padded_model - audio_model.shape[1]))
        if padded_orig > audio_orig.shape[1]:
            audio_orig = torch.nn.functional.pad(audio_orig, (0, padded_orig - audio_orig.shape[1]))

        window_hires = torch.hann_window(hires_n_fft)
        output = torch.zeros_like(audio_orig)
        weights = torch.zeros(audio_orig.shape[1])
        fade_in = torch.linspace(0, 1, overlap_orig) if overlap_orig > 0 else None
        fade_out = torch.linspace(1, 0, overlap_orig) if overlap_orig > 0 else None

        logger.info(
            "Hi-res STFT params: n_fft=%s hop=%s freqs=%s",
            hires_n_fft,
            hires_hop,
            hires_n_freqs,
        )
        logger.info(f"Processing {n_chunks} chunks for hi-res mask reuse...")

        with torch.inference_mode():
            for i in range(n_chunks):
                start_m = i * hop_model
                end_m = start_m + chunk_samples_model
                start_o = i * hop_orig
                end_o = start_o + chunk_samples_orig

                chunk_model = audio_model[:, start_m:end_m].to(self.device)
                chunk_orig = audio_orig[:, start_o:end_o]

                chunk_L = chunk_model[0].unsqueeze(0)
                chunk_R = chunk_model[1].unsqueeze(0)

                with self._precision_context():
                    stft_in_L = self.model.stft(chunk_L)
                    stft_in_R = self.model.stft(chunk_R)

                    if self.model.use_mid_side:
                        mid = (chunk_L + chunk_R) * 0.5
                        side = (chunk_L - chunk_R) * 0.5
                        stft_1 = self.model.stft(mid)
                        stft_2 = self.model.stft(side)
                        out_1, out_2 = self.model.forward_stft(stft_1, stft_2)
                        stft_out_L = out_1 + out_2
                        stft_out_R = out_1 - out_2
                        stft_in_L = stft_1 + stft_2
                        stft_in_R = stft_1 - stft_2
                    else:
                        stft_out_L, stft_out_R = self.model.forward_stft(stft_in_L, stft_in_R)

                eps = 1e-10
                mask_L = stft_out_L / (stft_in_L + eps)
                mask_R = stft_out_R / (stft_in_R + eps)

                mask_L_mag = mask_L.abs().clamp(0, 1.5)
                mask_R_mag = mask_R.abs().clamp(0, 1.5)
                mask_L = (mask_L_mag * torch.exp(1j * mask_L.angle())).squeeze(0).cpu()
                mask_R = (mask_R_mag * torch.exp(1j * mask_R.angle())).squeeze(0).cpu()

                stft_orig_L = torch.stft(
                    chunk_orig[0].unsqueeze(0),
                    hires_n_fft,
                    hires_hop,
                    window=window_hires,
                    return_complex=True,
                ).squeeze(0)
                stft_orig_R = torch.stft(
                    chunk_orig[1].unsqueeze(0),
                    hires_n_fft,
                    hires_hop,
                    window=window_hires,
                    return_complex=True,
                ).squeeze(0)

                hires_mask_L = self._interpolate_mask(mask_L, model_n_freqs, hires_n_freqs, stft_orig_L.shape[1])
                hires_mask_R = self._interpolate_mask(mask_R, model_n_freqs, hires_n_freqs, stft_orig_R.shape[1])

                inst_L = torch.istft(
                    (stft_orig_L * hires_mask_L).unsqueeze(0),
                    hires_n_fft,
                    hires_hop,
                    window=window_hires,
                    length=chunk_samples_orig,
                ).squeeze(0)
                inst_R = torch.istft(
                    (stft_orig_R * hires_mask_R).unsqueeze(0),
                    hires_n_fft,
                    hires_hop,
                    window=window_hires,
                    length=chunk_samples_orig,
                ).squeeze(0)

                chunk_weight = torch.ones(chunk_samples_orig)
                if overlap_orig > 0 and i > 0:
                    chunk_weight[:overlap_orig] *= fade_in
                if overlap_orig > 0 and i < n_chunks - 1:
                    chunk_weight[-overlap_orig:] *= fade_out

                output[0, start_o:end_o] += inst_L * chunk_weight
                output[1, start_o:end_o] += inst_R * chunk_weight
                weights[start_o:end_o] += chunk_weight

                print(f"  Chunk {i + 1}/{n_chunks} ({(i + 1) / n_chunks * 100:.0f}%)", end="\r")

        print()
        weights = weights.clamp(min=1e-8)
        return self._limit_peak((output / weights)[:, :original_length])

    def _interpolate_mask(
        self,
        mask: torch.Tensor,
        model_freq_bins: int,
        hires_freq_bins: int,
        hires_time_bins: int,
    ) -> torch.Tensor:
        mag = mask.abs().float()
        phase = mask.angle().float()

        mag_interp = torch.nn.functional.interpolate(
            mag.unsqueeze(0).unsqueeze(0),
            size=(model_freq_bins, hires_time_bins),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        phase_interp = torch.nn.functional.interpolate(
            phase.unsqueeze(0).unsqueeze(0),
            size=(model_freq_bins, hires_time_bins),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        full_mag = torch.ones(hires_freq_bins, hires_time_bins)
        full_phase = torch.zeros(hires_freq_bins, hires_time_bins)
        full_mag[:model_freq_bins, :] = mag_interp
        full_phase[:model_freq_bins, :] = phase_interp

        return full_mag * torch.exp(1j * full_phase)

    def _write_comparison_outputs(
        self,
        comparison_dir: str,
        stem: str,
        audio_model: torch.Tensor,
        audio_orig: torch.Tensor,
        orig_sr: int,
        chunk_seconds: float,
        overlap_seconds: float,
    ):
        comparison_path = Path(comparison_dir)
        comparison_path.mkdir(parents=True, exist_ok=True)

        standard = self._process_standard_waveform(
            audio_model,
            self.model_sr,
            chunk_seconds,
            overlap_seconds,
        )
        self._save_audio(
            str(comparison_path / f"{stem}_standard_{self.model_sr}hz.wav"),
            standard,
            self.model_sr,
        )

        upsampled = torchaudio.transforms.Resample(self.model_sr, orig_sr)(standard)
        self._save_audio(
            str(comparison_path / f"{stem}_standard_upsampled_{orig_sr}hz.wav"),
            upsampled,
            orig_sr,
        )

        hires = self._process_hires_waveform(
            audio_model,
            audio_orig,
            orig_sr,
            chunk_seconds,
            overlap_seconds,
        )
        self._save_audio(
            str(comparison_path / f"{stem}_hires_{orig_sr}hz.wav"),
            hires,
            orig_sr,
        )


def main():
    parser = argparse.ArgumentParser(description="Remove vocals from audio")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("output", type=str, help="Output instrumental file")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON (overrides checkpoint config)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Device to use")
    parser.add_argument("--chunk-seconds", type=float, default=None, help="Chunk length in seconds")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="Overlap between chunks in seconds")
    parser.add_argument("--no-loudness-match", action="store_true", help="Disable matching output loudness to input")
    parser.add_argument("--disable-hires", action="store_true", help="Disable hi-res output and force model sample rate output")
    parser.add_argument("--comparison-dir", type=str, default=None, help="Optional directory for standard/upsampled/hi-res comparison exports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runtime = load_runtime_config()
    checkpoint_path = args.checkpoint or runtime.checkpoint_path
    if not checkpoint_path:
        parser.error("--checkpoint is required unless INSTR_CHECKPOINT_PATH is set")

    remover = VocalRemover(checkpoint_path, device=args.device, config_path=args.config)
    remover.process(
        args.input,
        args.output,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        match_loudness=not args.no_loudness_match,
        preserve_hires=not args.disable_hires,
        comparison_dir=args.comparison_dir,
    )


if __name__ == "__main__":
    main()
