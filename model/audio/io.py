"""Audio I/O utilities - loading, saving, and length normalization."""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, List

from .convert import normalize_sample_rate


def load_audio(path: str, sr: int = 44100, mono: bool = False) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        path: Path to audio file
        sr: Target sample rate (default 44100)
        mono: If True, convert to mono by averaging channels
        
    Returns:
        Tuple of (audio_data, sample_rate)
        audio_data shape: (samples,) for mono, (samples, channels) for stereo
    """
    audio, loaded_sr = sf.read(path, dtype="float32", always_2d=not mono)

    if mono and audio.ndim == 2:
        audio = audio.mean(axis=1)

    if loaded_sr != sr:
        audio = normalize_sample_rate(audio, loaded_sr, sr)

    return audio, sr


def save_audio(path: str, audio: np.ndarray, sr: int = 44100) -> None:
    """
    Save audio to WAV file.
    
    Args:
        path: Output path
        audio: Audio data, shape (samples,) or (samples, channels)
        sr: Sample rate
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure float32 and clipped to [-1, 1]
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    
    sf.write(str(path), audio, sr, subtype='PCM_16')


def ensure_same_length(*tracks: np.ndarray) -> List[np.ndarray]:
    """
    Ensure all tracks have the same length by trimming to shortest.
    
    Args:
        *tracks: Variable number of audio arrays
        
    Returns:
        List of arrays, all trimmed to the same length
    """
    if len(tracks) == 0:
        return []
    
    # Find minimum length
    min_length = min(len(t) for t in tracks)
    
    # Trim all to minimum
    return [t[:min_length] for t in tracks]


def get_audio_info(path: str) -> dict:
    """
    Get audio file metadata without loading full audio.
    
    Args:
        path: Path to audio file
        
    Returns:
        Dict with 'sample_rate', 'channels', 'duration', 'frames'
    """
    info = sf.info(path)
    return {
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'duration': info.duration,
        'frames': info.frames,
        'format': info.format,
    }
