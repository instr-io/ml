"""Runtime environment loading shared across workflows."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_DATA_DIRS = "./data"
DEFAULT_OUTPUT_DIR = "./outputs"


def _load_dotenv_file(env_path: Optional[str] = None) -> Optional[Path]:
    path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if not path.exists():
        return None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value

    return path


_load_dotenv_file()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_prefix(value: Optional[str], default: str) -> str:
    clean = (value or default).strip().strip("/")
    return clean or default


def require_runtime_value(value: Optional[str], env_name: str) -> str:
    if value:
        return value
    raise RuntimeError(f"{env_name} is required. Set it in .env or your environment.")


@dataclass
class RuntimeConfig:
    """Environment-driven settings shared by every workflow."""

    data_dirs: str = field(default_factory=lambda: os.getenv("INSTR_DATA_DIRS", DEFAULT_DATA_DIRS))
    output_dir: str = field(default_factory=lambda: os.getenv("INSTR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    checkpoint_path: Optional[str] = field(default_factory=lambda: _optional_env("INSTR_CHECKPOINT_PATH"))
    aws_region: str = field(default_factory=lambda: os.getenv("INSTR_AWS_REGION", "us-east-1"))
    enable_s3: bool = field(default_factory=lambda: _env_bool("INSTR_ENABLE_S3", False))
    s3_bucket: Optional[str] = field(default_factory=lambda: _optional_env("INSTR_S3_BUCKET"))
    s3_checkpoint_prefix: str = field(default_factory=lambda: _normalize_prefix(os.getenv("INSTR_S3_CHECKPOINT_PREFIX"), "checkpoints"))
    enable_wandb: bool = field(default_factory=lambda: _env_bool("INSTR_ENABLE_WANDB", False))
    wandb_project: Optional[str] = field(default_factory=lambda: _optional_env("INSTR_WANDB_PROJECT"))
    wandb_entity: Optional[str] = field(default_factory=lambda: _optional_env("INSTR_WANDB_ENTITY"))
    wandb_api_key: Optional[str] = field(default_factory=lambda: _optional_env("INSTR_WANDB_API_KEY"))


def load_runtime_config(env_path: Optional[str] = None) -> RuntimeConfig:
    """Load `.env` if present, then return current runtime settings."""
    if env_path:
        _load_dotenv_file(env_path)
    return RuntimeConfig()
