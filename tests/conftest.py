"""Shared fixtures: generate a synthetic 'noisy voice' WAV for testing."""

import numpy as np
import pytest
import soundfile as sf

SR = 16000


@pytest.fixture
def noisy_wav(tmp_path):
    """3 s file: 1 s noise-only lead-in, then a 220 Hz 'voice' tone + noise."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    # A tone with harmonics, crudely voice-like
    voice = (
        0.30 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 440 * t)
        + 0.08 * np.sin(2 * np.pi * 660 * t)
    )
    noise_full = rng.normal(0, 0.05, 3 * SR)  # broadband noise everywhere
    signal = np.concatenate([np.zeros(SR), voice]) + noise_full
    # Add low-frequency rumble the high-pass should remove
    t_full = np.linspace(0, 3.0, 3 * SR, endpoint=False)
    signal += 0.10 * np.sin(2 * np.pi * 30 * t_full)

    path = tmp_path / "noisy.wav"
    sf.write(str(path), signal.astype(np.float32), SR)
    return path
