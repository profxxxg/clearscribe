import numpy as np
import soundfile as sf

from clearscribe.enhance import (
    EnhanceSettings,
    enhance_audio,
    highpass,
    load_mono,
    normalize_loudness,
    reduce_noise,
)

SR = 16000


def test_load_mono_returns_mono(noisy_wav):
    audio, sr = load_mono(noisy_wav)
    assert audio.ndim == 1
    assert sr == SR
    assert audio.dtype == np.float32


def test_highpass_removes_rumble():
    t = np.linspace(0, 1, SR, endpoint=False)
    rumble = np.sin(2 * np.pi * 30 * t).astype(np.float32)  # 30 Hz
    filtered = highpass(rumble, SR, cutoff_hz=80.0)
    # 30 Hz energy should be strongly attenuated by an 80 Hz high-pass
    assert np.sqrt(np.mean(filtered[SR // 2:] ** 2)) < 0.1 * np.sqrt(
        np.mean(rumble ** 2)
    )


def test_reduce_noise_lowers_noise_floor(noisy_wav):
    audio, sr = load_mono(noisy_wav)
    cleaned = reduce_noise(audio, sr, strength=0.9)
    # First second is noise-only: its RMS should drop after reduction
    noise_before = np.sqrt(np.mean(audio[:sr] ** 2))
    noise_after = np.sqrt(np.mean(cleaned[:sr] ** 2))
    assert noise_after < noise_before * 0.6


def test_normalize_loudness_hits_peak_ceiling():
    rng = np.random.default_rng(0)
    quiet = (rng.normal(0, 0.01, 2 * SR)).astype(np.float32)
    out = normalize_loudness(quiet, SR, target_lufs=-16.0, peak_ceiling=0.98)
    assert np.max(np.abs(out)) <= 0.9801


def test_enhance_audio_end_to_end(noisy_wav, tmp_path):
    out = tmp_path / "out" / "clean.wav"
    result = enhance_audio(noisy_wav, out, EnhanceSettings())
    assert result.exists()
    enhanced, sr = sf.read(str(out))
    original, _ = sf.read(str(noisy_wav))
    assert sr == SR
    assert len(enhanced) == len(original)  # duration preserved
    assert np.max(np.abs(enhanced)) <= 1.0  # no clipping
