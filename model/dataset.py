"""
Dataset for vocal separation training.

Loads paired audio files (original with vocals, instrumental).
Unicode-safe for international filenames.
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torchaudio
import torch
from torch.utils.data import Dataset
import unicodedata

logger = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test")

def _normalize_path(path: Path) -> Path:
    """Normalize unicode in path (NFC normalization for consistency)."""
    try:
        normalized = unicodedata.normalize('NFC', str(path))
        return Path(normalized)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return path


def parse_data_dirs(data_dirs: str) -> List[Path]:
    """Parse a comma-delimited list of dataset roots."""
    return [
        _normalize_path(Path(d.strip()))
        for d in data_dirs.split(",")
        if d.strip()
    ]


def _find_pairs_in_root(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Find direct child folders that contain `in.wav` and `out.wav`."""
    pairs: List[Tuple[Path, Path]] = []

    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return pairs

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue

        in_file = _normalize_path(subdir / "in.wav")
        out_file = _normalize_path(subdir / "out.wav")

        if in_file.exists() and out_file.exists():
            pairs.append((in_file, out_file))

    return pairs


def _stable_split_bucket(key: str) -> int:
    """Map a split key to a stable bucket in [0, 9]."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def _pair_split_key(root: Path, in_file: Path, out_file: Path) -> str:
    """Build a stable per-pair split key from the pair paths."""
    try:
        in_rel = in_file.relative_to(root).as_posix()
    except ValueError:
        in_rel = in_file.as_posix()

    try:
        out_rel = out_file.relative_to(root).as_posix()
    except ValueError:
        out_rel = out_file.as_posix()

    return f"{in_rel}|{out_rel}"


def discover_dataset_splits(data_dirs: str) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Discover dataset splits from a comma-delimited list of roots.

    Supported layouts:
    - Explicit splits:
        root/train/<song>/in.wav
        root/train/<song>/out.wav
        root/val/<song>/in.wav
        root/val/<song>/out.wav
        root/test/<song>/...
    - Unsplit roots:
        root/<song>/in.wav
        root/<song>/out.wav

    Explicit `train/` and `val/` folders win for a root. Unsplit roots fall
    back to a stable per-pair hash split.
    """
    split_pairs: Dict[str, List[Tuple[Path, Path]]] = {name: [] for name in SPLIT_NAMES}
    hashed_candidates: List[Tuple[Path, Path, Path]] = []

    for data_dir in parse_data_dirs(data_dirs):
        present_splits = [name for name in SPLIT_NAMES if (data_dir / name).is_dir()]

        if present_splits:
            if not {"train", "val"}.issubset(present_splits):
                raise ValueError(
                    f"{data_dir} has partial split directories {present_splits}. "
                    "Use both train/ and val/, or no split folders at all."
                )

            split_pairs["train"].extend(_find_pairs_in_root(data_dir / "train"))
            split_pairs["val"].extend(_find_pairs_in_root(data_dir / "val"))
            if (data_dir / "test").is_dir():
                split_pairs["test"].extend(_find_pairs_in_root(data_dir / "test"))
            continue

        for in_file, out_file in _find_pairs_in_root(data_dir):
            hashed_candidates.append((data_dir, in_file, out_file))

    hashed_candidates.sort(key=lambda item: _pair_split_key(item[0], item[1], item[2]))

    for root, in_file, out_file in hashed_candidates:
        split_key = _pair_split_key(root, in_file, out_file)
        if _stable_split_bucket(split_key) == 0:
            split_pairs["val"].append((in_file, out_file))
        else:
            split_pairs["train"].append((in_file, out_file))

    if not split_pairs["val"] and split_pairs["train"]:
        split_pairs["val"].append(split_pairs["train"].pop())

    return split_pairs


def _sanitize_audio(audio: torch.Tensor, max_val: float = 10.0) -> torch.Tensor:
    """
    Sanitize audio tensor: replace NaN/Inf, clip extreme values.

    Args:
        audio: Audio tensor
        max_val: Maximum absolute value (default 10.0, well above normal audio range)

    Returns:
        Sanitized audio tensor
    """
    # Replace NaN with 0
    if torch.isnan(audio).any():
        audio = torch.nan_to_num(audio, nan=0.0)

    # Replace Inf with max_val
    if torch.isinf(audio).any():
        audio = torch.clamp(audio, min=-max_val, max=max_val)

    # Clip extreme values
    audio = torch.clamp(audio, min=-max_val, max=max_val)

    return audio


class VocalSeparatorDataset(Dataset):
    """
    Dataset that loads a concrete set of paired audio files for vocal separation.
    Unicode-safe for international filenames (Japanese, Korean, etc.).

    Direct root loading expects:
        data_dir/
            song1/
                in.wav
                out.wav
            song2/
                ...

    Training split discovery lives in `discover_dataset_splits()`.

    Supports multiple directories via comma-delimited string:
        data_dirs="./data1,./data2,./data3"
    """

    def __init__(
        self,
        data_dirs: Optional[str] = None,
        sr: int = 44100,
        chunk_seconds: float = 5.0,
        augment: bool = True,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
    ):
        super().__init__()
        self.data_dirs = parse_data_dirs(data_dirs) if data_dirs else []
        self.sr = sr
        self.chunk_samples = int(chunk_seconds * sr)
        self.augment = augment

        if pairs is not None:
            self.pairs = [(_normalize_path(in_file), _normalize_path(out_file)) for in_file, out_file in pairs]
        else:
            if not data_dirs:
                raise ValueError("data_dirs is required when pairs are not provided")
            self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            raise ValueError(f"No audio pairs found in {data_dirs or 'provided pairs'}")

        if self.data_dirs:
            logger.info(
                f"Found {len(self.pairs)} audio pairs from {len(self.data_dirs)} director{'y' if len(self.data_dirs) == 1 else 'ies'}"
            )
        else:
            logger.info(f"Found {len(self.pairs)} provided audio pairs")

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find all (in.wav, out.wav) pairs in subdirectories across all data directories."""
        pairs = []

        for data_dir in self.data_dirs:
            pairs.extend(_find_pairs_in_root(data_dir))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio file and ensure stereo @ target sample rate."""
        audio, sr = torchaudio.load(str(path))

        # Resample if needed
        if sr != self.sr:
            audio = torchaudio.transforms.Resample(sr, self.sr)(audio)

        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        return audio

    def _random_chunk(
        self,
        original: torch.Tensor,
        instrumental: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract random chunk of chunk_samples length."""
        length = min(original.shape[1], instrumental.shape[1])

        if length <= self.chunk_samples:
            # Pad if too short
            pad_amount = self.chunk_samples - length + 1
            original = torch.nn.functional.pad(original, (0, pad_amount))
            instrumental = torch.nn.functional.pad(instrumental, (0, pad_amount))
            length = self.chunk_samples + 1

        # Random start position
        max_start = length - self.chunk_samples
        start = random.randint(0, max_start)

        return (
            original[:, start:start + self.chunk_samples],
            instrumental[:, start:start + self.chunk_samples],
        )

    def _augment(
        self,
        original: torch.Tensor,
        instrumental: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation with safety checks."""
        # Random gain (±3dB) - mild variation
        if random.random() < 0.5:
            gain = 10 ** (random.uniform(-3, 3) / 20)
            original = original * gain
            instrumental = instrumental * gain

        # Aggressive gain boost (6-12dB) - simulate loud choruses
        # Applied 30% of the time to help model learn loud sections
        if random.random() < 0.3:
            boost_db = random.uniform(6, 12)
            boost = 10 ** (boost_db / 20)
            original = original * boost
            instrumental = instrumental * boost

        # Normalize to prevent clipping after any gain changes
        max_val = max(original.abs().max(), instrumental.abs().max())
        if max_val > 0.99:
            scale = 0.99 / max_val
            original = original * scale
            instrumental = instrumental * scale

        # Random channel swap
        if random.random() < 0.5:
            original = original.flip(0)
            instrumental = instrumental.flip(0)

        # Random polarity inversion
        if random.random() < 0.3:
            original = -original
            instrumental = -instrumental

        return original, instrumental

    def _get_item_with_fallback(self, idx: int, max_retries: int = 2) -> dict:
        """Get item with fallback on failure. Fast path - minimal validation."""
        original_path, instrumental_path = self.pairs[idx]

        try:
            # Load audio - no validation, just load
            original = self._load_audio(original_path)
            instrumental = self._load_audio(instrumental_path)

            # Sanitize audio to prevent NaN/Inf from corrupt files
            original = _sanitize_audio(original)
            instrumental = _sanitize_audio(instrumental)

        except Exception as e:
            if max_retries <= 0:
                # Return zeros as last resort
                zeros = torch.zeros(2, self.chunk_samples)
                return {
                    "original_L": zeros[0],
                    "original_R": zeros[1],
                    "inst_L": zeros[0].clone(),
                    "inst_R": zeros[1].clone(),
                }
            # Try a different sample
            fallback_idx = (idx + 1) % len(self.pairs)
            return self._get_item_with_fallback(fallback_idx, max_retries - 1)

        # Random chunk
        original, instrumental = self._random_chunk(original, instrumental)

        # Augment
        if self.augment:
            original, instrumental = self._augment(original, instrumental)

        return {
            "original_L": original[0],
            "original_R": original[1],
            "inst_L": instrumental[0],
            "inst_R": instrumental[1],
        }

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                - original_L: (chunk_samples,) left channel of original
                - original_R: (chunk_samples,) right channel of original
                - inst_L: (chunk_samples,) left channel of instrumental
                - inst_R: (chunk_samples,) right channel of instrumental
        """
        return self._get_item_with_fallback(idx, max_retries=3)


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader - fast path."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


if __name__ == "__main__":
    # Test dataset
    import sys
    from model.config import load_runtime_config

    if len(sys.argv) > 1:
        data_dirs = sys.argv[1]  # Can be comma-delimited
    else:
        data_dirs = load_runtime_config().data_dirs

    split_pairs = discover_dataset_splits(data_dirs)
    print(
        f"Split sizes: train={len(split_pairs['train'])}, "
        f"val={len(split_pairs['val'])}, test={len(split_pairs['test'])}"
    )

    dataset = VocalSeparatorDataset(data_dirs=data_dirs, pairs=split_pairs["train"], chunk_seconds=5.0)
    print(f"Train dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"original_L shape: {sample['original_L'].shape}")
        print(f"inst_L shape: {sample['inst_L'].shape}")
