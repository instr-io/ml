"""Audio format conversion and normalization utilities."""

from math import gcd
import numpy as np
from scipy import signal
import subprocess
import tempfile
from pathlib import Path


def to_wav(input_path: str, output_path: str, sample_rate: int = 44100) -> None:
    """
    Convert any audio format to WAV using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        sample_rate: Target sample rate
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-ar', str(sample_rate),
        '-ac', '2',  # stereo
        '-acodec', 'pcm_s16le',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")


def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak normalize audio to target amplitude.
    
    Args:
        audio: Audio data
        target_peak: Target peak amplitude (0.0 to 1.0)
        
    Returns:
        Normalized audio
    """
    if audio.size == 0:
        return audio
    
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def normalize_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    common = gcd(orig_sr, target_sr)
    up = target_sr // common
    down = orig_sr // common
    axis = 0 if audio.ndim == 2 else -1
    resampled = signal.resample_poly(audio, up, down, axis=axis)
    return np.asarray(resampled, dtype=np.float32)


def ensure_wav(path: str, sample_rate: int = 44100, in_place: bool = False) -> str:
    """
    Ensure file is a valid WAV at target sample rate.
    Converts if necessary, returns path to valid WAV.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        in_place: If True, overwrite original file with converted version
        
    Returns:
        Path to valid WAV file (same as input if in_place or already valid,
        otherwise a temp file)
    """
    import soundfile as sf
    import os
    
    path = Path(path)
    
    # Try to read as-is
    try:
        info = sf.info(str(path))
        if info.format == 'WAV' and info.samplerate == sample_rate:
            return str(path)
    except Exception:
        pass
    
    # Need to convert
    temp_path = tempfile.mktemp(suffix='.wav')
    to_wav(str(path), temp_path, sample_rate)
    
    if in_place:
        # Replace original with converted
        os.replace(temp_path, str(path))
        return str(path)
    
    return temp_path
