"""
Training/model configuration.

Runtime environment loading lives in `model.runtime`.
"""

from dataclasses import asdict, dataclass, field
import json
import tempfile
from pathlib import Path
from typing import Optional

from .runtime import DEFAULT_DATA_DIRS, DEFAULT_OUTPUT_DIR, RuntimeConfig, load_runtime_config


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 256
    chunk_seconds: float = 3.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_bottleneck_layers: int = 6
    dropout: float = 0.0
    d_state: int = 32
    use_mid_side: bool = False
    ssm_variant: str = "mamba"
    d_conv: int = 4
    expand: int = 2
    mamba3_headdim: int = 64
    mamba3_is_mimo: bool = False
    mamba3_mimo_rank: int = 4
    mamba3_chunk_size: int = 32
    mamba3_is_outproj_norm: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""

    mag_weight: float = 0.5
    mr_stft_weight: float = 0.0
    si_sdr_weight: float = 0.5
    spectral_sdr_weight: float = 0.0
    band_weight: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.005
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation: int = 16
    max_grad_norm: float = 1.0
    save_every_steps: int = 50
    eval_every_steps: int = 50
    log_every_steps: int = 100
    use_amp: bool = True
    # Backward compatible with bool configs:
    # false -> "none", true -> "full".
    # String modes let us trade speed for memory more gradually.
    gradient_checkpointing: bool | str = False
    num_workers: int = 0
    prefetch_factor: int = 2


@dataclass
class Config:
    """Full training configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_dirs: str = DEFAULT_DATA_DIRS
    output_dir: str = DEFAULT_OUTPUT_DIR
    checkpoint_path: Optional[str] = None
    experiment_name: str = "vocal_separator"
    seed: int = 42

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def apply_runtime_defaults(self, runtime: Optional[RuntimeConfig] = None) -> "Config":
        runtime = runtime or load_runtime_config()

        if self.data_dirs in {"./data", DEFAULT_DATA_DIRS, ""}:
            self.data_dirs = runtime.data_dirs
        if self.output_dir in {DEFAULT_OUTPUT_DIR, ""}:
            self.output_dir = runtime.output_dir
        if not self.checkpoint_path and runtime.checkpoint_path:
            self.checkpoint_path = runtime.checkpoint_path

        return self

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            model=ModelConfig(**data.get("model", {})),
            loss=LossConfig(**data.get("loss", {})),
            training=TrainingConfig(**data.get("training", {})),
            data_dirs=data.get("data_dirs", data.get("data_dir", DEFAULT_DATA_DIRS)),
            output_dir=data.get("output_dir", DEFAULT_OUTPUT_DIR),
            checkpoint_path=data.get("checkpoint_path"),
            experiment_name=data.get("experiment_name", "vocal_separator"),
            seed=data.get("seed", 42),
        )


def base_config() -> Config:
    """Default training configuration."""
    return Config(
        audio=AudioConfig(),
        model=ModelConfig(),
        loss=LossConfig(),
        training=TrainingConfig(),
        experiment_name="vocal_separator_base",
    )


def mamba3_5060ti_prodlike_config() -> Config:
    """Near-production Mamba-3 preset sized for a 16 GB RTX 5060 Ti."""
    return Config(
        audio=AudioConfig(
            chunk_seconds=3.0,
        ),
        model=ModelConfig(
            d_model=256,
            n_heads=4,
            n_encoder_layers=4,
            n_decoder_layers=4,
            n_bottleneck_layers=4,
            dropout=0.0,
            d_state=32,
            use_mid_side=False,
            ssm_variant="mamba3",
            d_conv=4,
            expand=2,
            mamba3_headdim=64,
            mamba3_is_mimo=False,
            mamba3_mimo_rank=4,
            mamba3_chunk_size=32,
            mamba3_is_outproj_norm=False,
        ),
        training=TrainingConfig(
            batch_size=1,
            gradient_accumulation=8,
            gradient_checkpointing=True,
            num_workers=0,
            prefetch_factor=2,
        ),
        experiment_name="vocal_separator_mamba3_5060ti_prodlike",
    )


def mamba3_5060ti_reduced_config() -> Config:
    """Fallback Mamba-3 preset if the near-production depth does not fit."""
    return Config(
        audio=AudioConfig(
            chunk_seconds=3.0,
        ),
        model=ModelConfig(
            d_model=256,
            n_heads=4,
            n_encoder_layers=3,
            n_decoder_layers=3,
            n_bottleneck_layers=4,
            dropout=0.0,
            d_state=32,
            use_mid_side=False,
            ssm_variant="mamba3",
            d_conv=4,
            expand=2,
            mamba3_headdim=64,
            mamba3_is_mimo=False,
            mamba3_mimo_rank=4,
            mamba3_chunk_size=32,
            mamba3_is_outproj_norm=False,
        ),
        training=TrainingConfig(
            batch_size=1,
            gradient_accumulation=8,
            gradient_checkpointing=True,
            num_workers=0,
            prefetch_factor=2,
        ),
        experiment_name="vocal_separator_mamba3_5060ti_reduced",
    )


if __name__ == "__main__":
    config = base_config()
    runtime = load_runtime_config()
    print("Base config:")
    print(json.dumps(asdict(config), indent=2))
    print("\nRuntime config:")
    print(json.dumps(asdict(runtime), indent=2))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        test_path = f.name
    config.save(test_path)
    loaded = Config.load(test_path)
    print("\nLoaded config matches:", asdict(config) == asdict(loaded))
    Path(test_path).unlink()
