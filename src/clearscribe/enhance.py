"""Voice enhancement pipeline — "regular recording in, podcast out".

Signal chain:
    1. High-pass filter        — remove rumble / handling noise
    2. De-hum (optional)       — notch mains hum 50/60 Hz + harmonics
    3. Noise reduction         — spectral gating (noisereduce)
    4. Noise gate              — push pauses into silence
    5. Compressor              — even out loud/quiet speech
    6. De-esser                — tame harsh sibilance
    7. Voice EQ                — warmth + presence + air
    8. Loudness normalisation  — broadcast target (default -16 LUFS)
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

from clearscribe import dsp

logger = logging.getLogger(__name__)


@dataclass
class EnhanceSettings:
    """Tunable settings for the enhancement pipeline.

    Defaults are the "podcast" preset: safe, natural-sounding values that
    work for most spoken-voice recordings.
    """

    # Filtering / noise
    highpass_hz: float = 80.0
    dehum_hz: float = 0.0            # 0 = off; set 50 (EU) or 60 (US)
    auto_notch: bool = True          # auto-remove tonal whines/beeps
    backend: str = "spectral"        # "spectral" (built-in) or "deep" (DeepFilterNet)
    noise_reduction_strength: float = 0.85
    stationary_noise: bool = False
    two_pass_noise_reduction: bool = True  # stationary pass then adaptive pass

    # Dynamics
    gate_threshold_db: float = -45.0  # 0 disables (use with gate_ratio)
    gate_ratio: float = 2.5           # <=1 disables
    comp_threshold_db: float = -22.0
    comp_ratio: float = 3.0           # <=1 disables
    deess_threshold_db: float = -30.0
    deess_ratio: float = 4.0          # <=1 disables

    # Tone
    warmth_db: float = 1.5
    presence_db: float = 2.5
    air_db: float = 1.5

    # Level
    target_lufs: float = -16.0
    peak_ceiling: float = 0.98


#: Named presets for the UI/CLI.
PRESETS: dict[str, EnhanceSettings] = {
    "podcast": EnhanceSettings(),
    "gentle": EnhanceSettings(
        noise_reduction_strength=0.6, comp_ratio=2.0, gate_ratio=1.5,
        presence_db=1.5, air_db=1.0, deess_ratio=2.0,
        two_pass_noise_reduction=False,
    ),
    "aggressive": EnhanceSettings(
        noise_reduction_strength=0.95, comp_ratio=4.0, comp_threshold_db=-24.0,
        gate_ratio=3.0, gate_threshold_db=-42.0,
    ),
    "raw": EnhanceSettings(  # denoise + normalise only
        gate_ratio=1.0, comp_ratio=1.0, deess_ratio=1.0,
        warmth_db=0.0, presence_db=0.0, air_db=0.0,
    ),
}


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32. Returns (samples, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """4th-order Butterworth high-pass filter to remove low-frequency rumble."""
    if cutoff_hz <= 0:
        return audio
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def reduce_noise(
    audio: np.ndarray, sr: int, strength: float = 0.85,
    stationary: bool = False, two_pass: bool = False,
) -> np.ndarray:
    """Spectral-gating noise reduction via the noisereduce library.

    With two_pass=True, a stationary pass first removes steady hiss and
    tonal residue (this is what constant high-frequency noise responds to),
    then an adaptive pass handles varying noise.
    """
    if strength <= 0:
        return audio
    strength = float(np.clip(strength, 0.0, 1.0))
    if two_pass:
        audio = nr.reduce_noise(y=audio, sr=sr, stationary=True,
                                prop_decrease=strength * 0.6).astype(np.float32)
    cleaned = nr.reduce_noise(
        y=audio, sr=sr, prop_decrease=strength, stationary=stationary,
    )
    return cleaned.astype(np.float32)


def normalize_loudness(
    audio: np.ndarray, sr: int,
    target_lufs: float = -16.0, peak_ceiling: float = 0.98,
) -> np.ndarray:
    """Normalise integrated loudness to target LUFS, with a hard peak ceiling."""
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


def enhance_array(
    audio: np.ndarray, sr: int, settings: EnhanceSettings | None = None
) -> np.ndarray:
    """Run the full enhancement chain on an in-memory signal."""
    s = settings or EnhanceSettings()

    logger.info("High-pass filter at %.0f Hz", s.highpass_hz)
    audio = highpass(audio, sr, s.highpass_hz)

    if s.dehum_hz > 0:
        logger.info("De-hum at %.0f Hz + harmonics", s.dehum_hz)
        audio = dsp.dehum(audio, sr, s.dehum_hz)

    if s.auto_notch:
        tones = dsp.find_tonal_noises(audio, sr)
        if tones:
            logger.info("Auto-notch removing tonal noise at: %s Hz",
                        ", ".join(f"{f:.0f}" for f in tones))
            audio = dsp.remove_tonal_noises(audio, sr, tones)

    if s.backend == "deep":
        from clearscribe.deep import deep_denoise
        logger.info("Deep denoise (DeepFilterNet)")
        audio = deep_denoise(audio, sr)
    else:
        logger.info("Noise reduction (strength=%.2f, two_pass=%s)",
                    s.noise_reduction_strength, s.two_pass_noise_reduction)
        audio = reduce_noise(audio, sr, s.noise_reduction_strength,
                             s.stationary_noise, s.two_pass_noise_reduction)

    if s.gate_ratio > 1.0:
        logger.info("Noise gate at %.0f dB", s.gate_threshold_db)
        audio = dsp.gate(audio, sr, s.gate_threshold_db, s.gate_ratio)

    if s.comp_ratio > 1.0:
        logger.info("Compressor %.1f:1 at %.0f dB", s.comp_ratio, s.comp_threshold_db)
        audio = dsp.compress(audio, sr, s.comp_threshold_db, s.comp_ratio)

    if s.deess_ratio > 1.0:
        logger.info("De-esser %.1f:1", s.deess_ratio)
        audio = dsp.deess(audio, sr,
                          threshold_db=s.deess_threshold_db, ratio=s.deess_ratio)

    if s.warmth_db or s.presence_db or s.air_db:
        logger.info("Voice EQ (warmth %+.1f, presence %+.1f, air %+.1f dB)",
                    s.warmth_db, s.presence_db, s.air_db)
        audio = dsp.voice_eq(audio, sr, s.warmth_db, s.presence_db, s.air_db)

    logger.info("Loudness normalisation to %.1f LUFS", s.target_lufs)
    return normalize_loudness(audio, sr, s.target_lufs, s.peak_ceiling)


def enhance_audio(
    input_path: str | Path,
    output_path: str | Path,
    settings: EnhanceSettings | None = None,
) -> Path:
    """Run the full enhancement pipeline on a file and write a 16-bit WAV."""
    input_path, output_path = Path(input_path), Path(output_path)
    logger.info("Loading %s", input_path)
    audio, sr = load_mono(input_path)

    audio = enhance_array(audio, sr, settings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    logger.info("Enhanced audio written to %s", output_path)
    return output_path
