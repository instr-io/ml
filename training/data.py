"""Dataset helpers for training."""

from typing import Dict, Tuple

from model.config import Config
from model.dataset import VocalSeparatorDataset, discover_dataset_splits


def build_train_val_datasets(config: Config) -> Tuple[VocalSeparatorDataset, VocalSeparatorDataset, Dict[str, int]]:
    """Build train/val datasets from explicit splits or stable hash fallback."""
    split_pairs = discover_dataset_splits(config.data_dirs)
    train_pairs = split_pairs["train"]
    val_pairs = split_pairs["val"]
    test_pairs = split_pairs["test"]

    if not train_pairs:
        raise ValueError("No training audio pairs found")
    if not val_pairs:
        raise ValueError("No validation audio pairs found")

    train_dataset = VocalSeparatorDataset(
        data_dirs=config.data_dirs,
        pairs=train_pairs,
        sr=config.audio.sample_rate,
        chunk_seconds=config.audio.chunk_seconds,
        augment=True,
    )
    val_dataset = VocalSeparatorDataset(
        data_dirs=config.data_dirs,
        pairs=val_pairs,
        sr=config.audio.sample_rate,
        chunk_seconds=config.audio.chunk_seconds,
        augment=False,
    )

    return train_dataset, val_dataset, {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }
