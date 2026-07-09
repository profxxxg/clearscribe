"""Broadcast-style DSP building blocks: dynamics and EQ.

Everything here is pure NumPy/SciPy — no models, fully deterministic.
These are the stages that take "cleaned" audio to "podcast" audio:

    de-hum -> gate -> compressor -> de-esser -> EQ

Loudness normalisation (enhance.py) then sets the final level.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

_EPS = 1e-10


# ---------------------------------------------------------------- envelope

def envelope(
    audio: np.ndarray,
    sr: int,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
    hop: int = 64,
) -> np.ndarray:
    """Peak envelope follower with asymmetric attack/release smoothing.

    Computed on small hops for speed, then expanded back to sample rate.
    """
    n = len(audio)
    pad = (-n) % hop
    framed = np.abs(np.pad(audio, (0, pad))).reshape(-1, hop)
    peaks = framed.max(axis=1)

    hop_ms = 1000.0 * hop / sr
    a_att = np.exp(-hop_ms / max(attack_ms, 0.01))
    a_rel = np.exp(-hop_ms / max(release_ms, 0.01))

    env = np.empty_like(peaks)
    state = peaks[0]
    for i, p in enumerate(peaks):
        coeff = a_att if p > state else a_rel
        state = coeff * state + (1.0 - coeff) * p
        env[i] = state

    return np.repeat(env, hop)[:n]


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, _EPS))


def _lin(db: np.ndarray) -> np.ndarray:
    return 10.0 ** (db / 20.0)


# ---------------------------------------------------------------- dynamics

def compress(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -22.0,
    ratio: float = 3.0,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
    knee_db: float = 6.0,
) -> np.ndarray:
    """Soft-knee downward compressor. Evens out loud/quiet speech.

    No makeup gain — final loudness normalisation handles level.
    """
    if ratio <= 1.0:
        return audio
    env_db = _db(envelope(audio, sr, attack_ms, release_ms))
    over = env_db - threshold_db

    # Soft knee: quadratic transition around the threshold
    half_knee = knee_db / 2.0
    gain_red = np.where(
        over <= -half_knee,
        0.0,
        np.where(
            over >= half_knee,
            over * (1.0 - 1.0 / ratio),
            (1.0 - 1.0 / ratio) * (over + half_knee) ** 2 / (2.0 * knee_db),
        ),
    )
    return (audio * _lin(-gain_red)).astype(np.float32)


def gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -45.0,
    ratio: float = 2.5,
    attack_ms: float = 2.0,
    release_ms: float = 60.0,
    floor_db: float = -18.0,
) -> np.ndarray:
    """Downward expander ("soft gate"). Pushes pauses further into silence
    without the pumping of a hard gate. `floor_db` caps max attenuation.
    """
    if ratio <= 1.0:
        return audio
    env_db = _db(envelope(audio, sr, attack_ms, release_ms))
    under = threshold_db - env_db
    gain_red = np.clip(np.where(under > 0, under * (ratio - 1.0), 0.0),
                       0.0, -floor_db)
    return (audio * _lin(-gain_red)).astype(np.float32)


def deess(
    audio: np.ndarray,
    sr: int,
    low_hz: float = 5000.0,
    high_hz: float = 9000.0,
    threshold_db: float = -30.0,
    ratio: float = 4.0,
) -> np.ndarray:
    """De-esser: compress only the sibilance band, leave the rest untouched."""
    nyq = sr / 2.0
    high_hz = min(high_hz, nyq * 0.95)
    if low_hz >= high_hz:
        return audio
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    band = sosfilt(sos, audio).astype(np.float32)
    rest = audio - band

    env_db = _db(envelope(band, sr, attack_ms=1.0, release_ms=40.0))
    over = env_db - threshold_db
    gain_red = np.where(over > 0, over * (1.0 - 1.0 / ratio), 0.0)
    return (rest + band * _lin(-gain_red)).astype(np.float32)


# ---------------------------------------------------------------- EQ

def _rbj_shelf_peak(sr, f0, gain_db, q, kind):
    """RBJ audio-EQ-cookbook biquad: 'lowshelf', 'highshelf', or 'peak'."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)

    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + two_sqrtA_alpha),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - two_sqrtA_alpha)]
        a = [(A + 1) + (A - 1) * cw + two_sqrtA_alpha,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - two_sqrtA_alpha]
    elif kind == "highshelf":
        two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + two_sqrtA_alpha),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - two_sqrtA_alpha)]
        a = [(A + 1) - (A - 1) * cw + two_sqrtA_alpha,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - two_sqrtA_alpha]
    else:
        raise ValueError(kind)
    return tf2sos(np.array(b) / a[0], np.array(a) / a[0])


def voice_eq(
    audio: np.ndarray,
    sr: int,
    warmth_db: float = 1.5,     # low shelf @ 160 Hz
    presence_db: float = 2.5,   # peak @ 3 kHz — intelligibility
    air_db: float = 1.5,        # high shelf @ 10 kHz — "expensive mic" sheen
) -> np.ndarray:
    """Gentle broadcast voice EQ. All gains in dB; 0 disables a band."""
    out = audio
    nyq = sr / 2.0
    if warmth_db:
        out = sosfilt(_rbj_shelf_peak(sr, 160.0, warmth_db, 0.707, "lowshelf"), out)
    if presence_db and 3000.0 < nyq:
        out = sosfilt(_rbj_shelf_peak(sr, 3000.0, presence_db, 1.0, "peak"), out)
    if air_db and 10000.0 < nyq * 0.95:
        out = sosfilt(_rbj_shelf_peak(sr, 10000.0, air_db, 0.707, "highshelf"), out)
    return out.astype(np.float32)


def dehum(
    audio: np.ndarray,
    sr: int,
    hum_hz: float = 50.0,
    harmonics: int = 4,
    q: float = 35.0,
) -> np.ndarray:
    """Notch out mains hum and its harmonics (50 Hz Europe, 60 Hz US)."""
    out = audio
    for k in range(1, harmonics + 1):
        f = hum_hz * k
        if f >= sr / 2.0 * 0.95:
            break
        b, a = iirnotch(f, q, fs=sr)
        out = sosfilt(tf2sos(b, a), out)
    return out.astype(np.float32)
