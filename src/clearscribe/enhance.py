"""Voice enhancement pipeline.

Steps:
    1. Load audio and convert to mono float32.
    2. High-pass filter to remove rumble / handling noise (< 80 Hz).
    3. Spectral-gating noise reduction (noisereduce).
    4. Loudness normalisation to a broadcast-friendly target (pyloudnorm).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


@dataclass
class EnhanceSettings:
    """Tunable settings for the enhancement pipeline."""

    highpass_hz: float = 80.0
    noise_reduction_strength: float = 0.85  # 0 (off) .. 1 (aggressive)
    stationary_noise: bool = False  # True for constant hum/hiss, False for varying noise
    target_lufs: float = -16.0  # good default for voice/podcast content
    peak_ceiling: float = 0.98  # avoid clipping after gain


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32. Returns (samples, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return mono, sr


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """4th-order Butterworth high-pass filter to remove low-frequency rumble."""
    if cutoff_hz <= 0:
        return audio
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def reduce_noise(
    audio: np.ndarray,
    sr: int,
    strength: float = 0.85,
    stationary: bool = False,
) -> np.ndarray:
    """Spectral-gating noise reduction via the noisereduce library."""
    if strength <= 0:
        return audio
    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=float(np.clip(strength, 0.0, 1.0)),
        stationary=stationary,
    )
    return cleaned.astype(np.float32)


def normalize_loudness(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -16.0,
    peak_ceiling: float = 0.98,
) -> np.ndarray:
    """Normalise integrated loudness to target LUFS, with a hard peak ceiling."""
    # pyloudnorm needs at least ~0.4 s of audio to measure loudness
    if len(audio) < int(sr * 0.5):
        peak = np.max(np.abs(audio)) or 1.0
        return (audio / peak * peak_ceiling).astype(np.float32)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio.astype(np.float64))
    if not np.isfinite(loudness):
        return audio
    normalized = pyln.normalize.loudness(audio.astype(np.float64), loudness, target_lufs)

    peak = np.max(np.abs(normalized))
    if peak > peak_ceiling:
        normalized = normalized / peak * peak_ceiling
    return normalized.astype(np.float32)


def enhance_audio(
    input_path: str | Path,
    output_path: str | Path,
    settings: EnhanceSettings | None = None,
) -> Path:
    """Run the full enhancement pipeline on a file and write a 16-bit WAV.

    Returns the output path.
    """
    settings = settings or EnhanceSettings()
    input_path, output_path = Path(input_path), Path(output_path)

    logger.info("Loading %s", input_path)
    audio, sr = load_mono(input_path)

    logger.info("High-pass filter at %.0f Hz", settings.highpass_hz)
    audio = highpass(audio, sr, settings.highpass_hz)

    logger.info("Noise reduction (strength=%.2f)", settings.noise_reduction_strength)
    audio = reduce_noise(
        audio, sr,
        strength=settings.noise_reduction_strength,
        stationary=settings.stationary_noise,
    )

    logger.info("Loudness normalisation to %.1f LUFS", settings.target_lufs)
    audio = normalize_loudness(audio, sr, settings.target_lufs, settings.peak_ceiling)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    logger.info("Enhanced audio written to %s", output_path)
    return output_path
