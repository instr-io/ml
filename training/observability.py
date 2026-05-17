"""Logging helpers for training metrics."""

from typing import Dict


def _loss_average(loss_sums: Dict[str, float], key: str) -> float:
    count = max(int(loss_sums.get("_count", 1)), 1)
    return loss_sums.get(key, 0.0) / count


def _display_key(key: str) -> str:
    aliases = {
        "magnitude": "mag",
        "mr_stft": "mrstft",
        "spectral_sdr": "spec_sdr",
    }
    return aliases.get(key, key)


def _format_metric(key: str, value: float) -> str:
    if "sdr" in key:
        return f"{value:.1f}dB"
    return f"{value:.4f}"


def build_progress_postfix(loss_sums: Dict[str, float], lr: float) -> Dict[str, str]:
    """Build a compact tqdm postfix from the currently accumulated losses."""
    postfix = {
        "loss": f"{_loss_average(loss_sums, 'total'):.4f}",
        "lr": f"{lr:.2e}",
    }

    ordered_keys = ["si_sdr", "spectral_sdr", "magnitude", "mr_stft", "band"]
    dynamic_keys = [
        key for key in sorted(loss_sums)
        if key not in {"_count", "total", "rms_weight"} and key not in ordered_keys
    ]

    for key in ordered_keys + dynamic_keys:
        if key in loss_sums:
            postfix[_display_key(key)] = _format_metric(key, _loss_average(loss_sums, key))

    return postfix


def build_train_metrics(loss_sums: Dict[str, float], lr: float) -> Dict[str, float]:
    """Build train metrics for observability backends such as W&B."""
    metrics = {
        "train/loss": _loss_average(loss_sums, "total"),
        "train/lr": lr,
    }

    if "rms_weight" in loss_sums:
        metrics["train/rms_weight"] = _loss_average(loss_sums, "rms_weight")

    for key, value in sorted(loss_sums.items()):
        if key in {"_count", "total", "rms_weight"}:
            continue
        metrics[f"train/{key}"] = _loss_average(loss_sums, key)

    return metrics
