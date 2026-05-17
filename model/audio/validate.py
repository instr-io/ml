"""Audio validation utilities for detecting bad matches."""

import numpy as np
from typing import Tuple, List
from enum import Enum


class FlagReason(Enum):
    """Reasons a pair might be flagged for manual review."""
    LOW_SIMILARITY = "similarity_below_threshold"
    DURATION_MISMATCH = "duration_differs_significantly"
    ALIGNMENT_FAILED = "no_clear_alignment_peak"
    RESIDUAL_NOT_QUIET = "loud_residual_in_instrumental_sections"


def similarity_score(original: np.ndarray, instrumental: np.ndarray) -> float:
    """
    Compute similarity score between original and instrumental.
    
    Uses normalized cross-correlation at zero lag (assuming pre-aligned).
    
    Args:
        original: Original audio with vocals
        instrumental: Instrumental version
        
    Returns:
        Similarity score between 0 and 1
    """
    # Ensure same length
    min_len = min(len(original), len(instrumental))
    orig = original[:min_len]
    inst = instrumental[:min_len]
    
    # Convert to mono if stereo
    if orig.ndim > 1:
        orig = np.mean(orig, axis=1)
    if inst.ndim > 1:
        inst = np.mean(inst, axis=1)
    
    # Normalized cross-correlation at zero lag
    orig_norm = orig - np.mean(orig)
    inst_norm = inst - np.mean(inst)
    
    orig_std = np.std(orig_norm)
    inst_std = np.std(inst_norm)
    
    if orig_std == 0 or inst_std == 0:
        return 0.0
    
    correlation = np.mean(orig_norm * inst_norm) / (orig_std * inst_std)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, correlation))


def find_quiet_regions(audio: np.ndarray, threshold_db: float = -40.0, min_duration: int = 4410) -> List[Tuple[int, int]]:
    """
    Find regions where audio is below threshold.
    
    Args:
        audio: Audio data (mono)
        threshold_db: Threshold in dB below peak
        min_duration: Minimum duration in samples to count as a region
        
    Returns:
        List of (start, end) sample indices for quiet regions
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Convert to amplitude
    amplitude = np.abs(audio)
    
    # Compute threshold
    peak = np.max(amplitude)
    if peak == 0:
        return [(0, len(audio))]
    
    threshold = peak * (10 ** (threshold_db / 20))
    
    # Find quiet samples
    is_quiet = amplitude < threshold
    
    # Find contiguous regions
    regions = []
    in_region = False
    start = 0
    
    for i, quiet in enumerate(is_quiet):
        if quiet and not in_region:
            start = i
            in_region = True
        elif not quiet and in_region:
            if i - start >= min_duration:
                regions.append((start, i))
            in_region = False
    
    # Handle region at end
    if in_region and len(audio) - start >= min_duration:
        regions.append((start, len(audio)))
    
    return regions


def compute_residual_score(original: np.ndarray, instrumental: np.ndarray) -> float:
    """
    Compute how clean the residual (original - instrumental) is.
    
    Good matches have residual that's quiet during instrumental sections.
    
    Args:
        original: Original audio
        instrumental: Instrumental audio (pre-aligned, same length)
        
    Returns:
        Score between 0 and 1 (higher = better match)
    """
    min_len = min(len(original), len(instrumental))
    orig = original[:min_len]
    inst = instrumental[:min_len]
    
    # Convert to mono
    if orig.ndim > 1:
        orig = np.mean(orig, axis=1)
    if inst.ndim > 1:
        inst = np.mean(inst, axis=1)
    
    # Normalize both
    orig_peak = np.max(np.abs(orig))
    inst_peak = np.max(np.abs(inst))
    
    if orig_peak > 0:
        orig = orig / orig_peak
    if inst_peak > 0:
        inst = inst / inst_peak
    
    # Compute residual
    residual = orig - inst
    
    # Find quiet regions in instrumental
    quiet_regions = find_quiet_regions(inst, threshold_db=-30.0)
    
    if not quiet_regions:
        # No quiet regions found, use overall residual energy
        residual_energy = np.mean(residual ** 2)
        return max(0.0, 1.0 - residual_energy * 10)
    
    # Compute residual energy in quiet regions
    quiet_residual_energies = []
    for start, end in quiet_regions:
        region_residual = residual[start:end]
        quiet_residual_energies.append(np.mean(region_residual ** 2))
    
    avg_quiet_residual = np.mean(quiet_residual_energies)
    
    # Low residual in quiet regions = good match
    # Score: 1.0 for avg_quiet_residual near 0, decreasing as it increases
    score = np.exp(-avg_quiet_residual * 50)
    
    return float(score)


def is_bad_match(
    original: np.ndarray, 
    instrumental: np.ndarray,
    similarity_threshold: float = 0.5,
    residual_threshold: float = 0.05,  # Very lenient - residual check is noisy for vocal tracks
) -> Tuple[bool, float, str]:
    """
    Determine if a pair is a bad match that should be flagged.
    
    Note: For vocal/instrumental pairs, the residual (vocals) WILL be loud
    in many places, so we rely primarily on similarity score.
    
    Args:
        original: Original audio
        instrumental: Instrumental audio (pre-aligned)
        similarity_threshold: Minimum similarity score (main check)
        residual_threshold: Minimum residual score (secondary, lenient)
        
    Returns:
        Tuple of (is_bad, score, reason)
    """
    sim_score = similarity_score(original, instrumental)
    
    # Primary check: similarity
    if sim_score < similarity_threshold:
        return True, sim_score, FlagReason.LOW_SIMILARITY.value
    
    # Secondary check: residual (very lenient)
    residual_score_val = compute_residual_score(original, instrumental)
    
    if residual_score_val < residual_threshold:
        return True, residual_score_val, FlagReason.RESIDUAL_NOT_QUIET.value
    
    # Use similarity as the main score (more reliable)
    return False, sim_score, "ok"
