"""Helpers for loading model config and weights from a checkpoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .arch import create_model
from .config import Config


@dataclass
class LoadedSeparator:
    """Loaded model bundle for inference."""

    model: torch.nn.Module
    config: Config
    device: torch.device
    checkpoint: dict


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick a runtime device, defaulting to CUDA/MPS/CPU."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def load_config_from_checkpoint(
    checkpoint_path: str,
    checkpoint: dict,
    config_path: Optional[str] = None,
) -> Config:
    """Load config via explicit path, sibling config.json, embedded config, or defaults."""
    default_config_path = Path(checkpoint_path).parent / "config.json"
    ckpt_config = checkpoint.get("config", {})

    if config_path and Path(config_path).exists():
        return Config.load(config_path)

    if default_config_path.exists():
        return Config.load(str(default_config_path))

    if ckpt_config:
        config = Config()
        for key in ["n_fft", "hop_length", "sample_rate", "chunk_seconds"]:
            if key in ckpt_config:
                setattr(config.audio, key, ckpt_config[key])
        for key in [
            "d_model",
            "n_heads",
            "n_encoder_layers",
            "n_decoder_layers",
            "n_bottleneck_layers",
            "d_state",
            "use_mid_side",
        ]:
            if key in ckpt_config:
                setattr(config.model, key, ckpt_config[key])
        return config

    return Config()


def load_separator(
    checkpoint_path: str,
    device: Optional[str] = None,
    config_path: Optional[str] = None,
) -> LoadedSeparator:
    """Load a separator model, config, checkpoint, and runtime device."""
    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = load_config_from_checkpoint(checkpoint_path, checkpoint, config_path=config_path)

    model = create_model(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        sr=config.audio.sample_rate,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_encoder_layers=config.model.n_encoder_layers,
        n_decoder_layers=config.model.n_decoder_layers,
        n_bottleneck_layers=config.model.n_bottleneck_layers,
        dropout=0.0,
        use_mid_side=getattr(config.model, "use_mid_side", False),
        d_state=getattr(config.model, "d_state", 32),
    ).to(resolved_device)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model_state_dict' or 'model' key")

    model.load_state_dict(state_dict)
    model.eval()

    return LoadedSeparator(
        model=model,
        config=config,
        device=resolved_device,
        checkpoint=checkpoint,
    )
