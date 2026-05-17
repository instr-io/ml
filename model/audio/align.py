"""Audio alignment utilities using cross-correlation."""

import numpy as np
from scipy import signal
from typing import Tuple


def find_offset(reference: np.ndarray, target: np.ndarray, sr: int = 44100) -> int:
    """
    Find the sample offset to align target to reference using cross-correlation.
    
    Uses downsampling for efficiency on long audio files.
    
    Args:
        reference: Reference audio (1D mono array)
        target: Target audio to align (1D mono array)
        sr: Sample rate of the audio
        
    Returns:
        Offset in samples. Positive = target should be shifted right (starts later).
        Negative = target should be shifted left (starts earlier).
    """
    # Ensure mono
    if reference.ndim > 1:
        reference = np.mean(reference, axis=1)
    if target.ndim > 1:
        target = np.mean(target, axis=1)
    
    # Downsample for faster correlation (4kHz is enough for alignment)
    downsample_sr = 4000
    downsample_factor = sr // downsample_sr
    
    if downsample_factor > 1:
        ref_down = reference[::downsample_factor]
        tgt_down = target[::downsample_factor]
    else:
        ref_down = reference
        tgt_down = target
        downsample_factor = 1
    
    # Cross-correlate
    correlation = signal.correlate(ref_down, tgt_down, mode='full')
    
    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    
    # Convert to offset
    # In 'full' mode, output length is len(ref) + len(tgt) - 1
    # Zero lag is at index len(tgt) - 1
    zero_lag_idx = len(tgt_down) - 1
    offset_downsampled = peak_idx - zero_lag_idx
    
    # Scale back to original sample rate
    offset = offset_downsampled * downsample_factor
    
    return offset


def align_and_crop(
    track1: np.ndarray, 
    track2: np.ndarray, 
    sr: int = 44100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two tracks and crop to their overlapping region.
    
    Args:
        track1: First audio track (reference)
        track2: Second audio track (to be aligned)
        sr: Sample rate
        
    Returns:
        Tuple of (aligned_track1, aligned_track2) cropped to overlap
    """
    # Find offset
    offset = find_offset(track1, track2, sr)
    
    # Apply offset and find overlap
    if offset > 0:
        # track2 starts later than track1
        # Trim start of track1, keep track2 as-is
        t1_start = offset
        t2_start = 0
    else:
        # track2 starts earlier than track1
        # Trim start of track2, keep track1 as-is
        t1_start = 0
        t2_start = -offset
    
    # Determine end points
    t1_remaining = len(track1) - t1_start
    t2_remaining = len(track2) - t2_start
    overlap_length = min(t1_remaining, t2_remaining)
    
    # Extract overlapping regions
    aligned1 = track1[t1_start:t1_start + overlap_length]
    aligned2 = track2[t2_start:t2_start + overlap_length]
    
    return aligned1, aligned2


def compute_alignment_confidence(reference: np.ndarray, target: np.ndarray, sr: int = 44100) -> float:
    """
    Compute confidence score for alignment (0-1).
    Higher = clearer alignment peak.
    
    Args:
        reference: Reference audio
        target: Target audio
        sr: Sample rate
        
    Returns:
        Confidence score between 0 and 1
    """
    # Ensure mono
    if reference.ndim > 1:
        reference = np.mean(reference, axis=1)
    if target.ndim > 1:
        target = np.mean(target, axis=1)
    
    # Normalize signals
    ref_norm = reference / (np.max(np.abs(reference)) + 1e-10)
    tgt_norm = target / (np.max(np.abs(target)) + 1e-10)
    
    # Downsample
    downsample_factor = max(1, sr // 4000)
    ref_down = ref_norm[::downsample_factor]
    tgt_down = tgt_norm[::downsample_factor]
    
    # Normalized cross-correlation
    # This gives values between -1 and 1
    ref_centered = ref_down - np.mean(ref_down)
    tgt_centered = tgt_down - np.mean(tgt_down)
    
    ref_std = np.std(ref_centered)
    tgt_std = np.std(tgt_centered)
    
    if ref_std < 1e-10 or tgt_std < 1e-10:
        return 0.0
    
    # Compute normalized correlation
    correlation = signal.correlate(ref_centered, tgt_centered, mode='full')
    correlation = correlation / (len(ref_centered) * ref_std * tgt_std)
    
    # Peak of normalized correlation is the confidence
    peak = np.max(np.abs(correlation))
    
    return float(min(1.0, peak))
