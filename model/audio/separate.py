"""Audio separation utilities - subtraction and center-sides extraction."""

import numpy as np
from typing import Dict


def invert_phase(audio: np.ndarray) -> np.ndarray:
    """
    Invert the phase of audio (multiply by -1).
    
    Args:
        audio: Audio data
        
    Returns:
        Phase-inverted audio
    """
    return -audio


def subtract(original: np.ndarray, instrumental: np.ndarray) -> np.ndarray:
    """
    Subtract instrumental from original to estimate vocals.
    
    Args:
        original: Original audio with vocals
        instrumental: Instrumental version (aligned, same length)
        
    Returns:
        Estimated vocals (residual)
    """
    min_len = min(len(original), len(instrumental))
    return original[:min_len] - instrumental[:min_len]


def add(track1: np.ndarray, track2: np.ndarray) -> np.ndarray:
    """
    Add two audio tracks together.
    
    Args:
        track1: First audio track
        track2: Second audio track
        
    Returns:
        Sum of tracks
    """
    min_len = min(len(track1), len(track2))
    return track1[:min_len] + track2[:min_len]


def extract_center_sides(
    original: np.ndarray, 
    instrumental: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Extract center (vocals) and unique components from original and instrumental.
    
    This is the core separation used for training data:
    - center: What's in original but not in instrumental (vocals)
    - left_unique: What's unique to original (should be ~vocals)
    - right_unique: What's unique to instrumental (should be minimal for good matches)
    
    Args:
        original: Original audio with vocals
        instrumental: Instrumental version (aligned, same length)
        
    Returns:
        Dict with 'center', 'left_unique', 'right_unique' arrays
    """
    min_len = min(len(original), len(instrumental))
    orig = original[:min_len].copy()
    inst = instrumental[:min_len].copy()
    
    # Normalize amplitudes for consistent subtraction
    orig_peak = np.max(np.abs(orig))
    inst_peak = np.max(np.abs(inst))
    
    if orig_peak > 0:
        orig = orig / orig_peak
    if inst_peak > 0:
        inst = inst / inst_peak
    
    # Center = what's in original but not instrumental (vocals estimate)
    center = orig - inst
    
    # Left unique = residual from original's perspective
    left_unique = center  # Same as center for this use case
    
    # Right unique = what's in instrumental but not original (should be minimal)
    right_unique = inst - orig
    
    return {
        'center': center,
        'left_unique': left_unique,
        'right_unique': right_unique,
    }


def mix_tracks(tracks: Dict[str, np.ndarray], gains: Dict[str, float] = None) -> np.ndarray:
    """
    Mix multiple tracks together with optional gain adjustments.
    
    Args:
        tracks: Dict of track_name -> audio array
        gains: Dict of track_name -> gain multiplier (default 1.0)
        
    Returns:
        Mixed audio
    """
    if not tracks:
        raise ValueError("No tracks to mix")
    
    gains = gains or {}
    
    # Find common length
    min_len = min(len(t) for t in tracks.values())
    
    # Mix
    result = np.zeros(min_len, dtype=np.float32)
    for name, track in tracks.items():
        gain = gains.get(name, 1.0)
        result += track[:min_len] * gain
    
    return result


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain in decibels to audio.
    
    Args:
        audio: Audio data
        gain_db: Gain in decibels
        
    Returns:
        Gained audio
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear
