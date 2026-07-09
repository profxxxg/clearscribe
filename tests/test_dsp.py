import numpy as np

from clearscribe import dsp

SR = 16000
RNG = np.random.default_rng(1)


def rms_db(x):
    return 20 * np.log10(max(np.sqrt(np.mean(np.asarray(x) ** 2)), 1e-10))


def tone(freq, seconds=1.0, amp=0.3):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_compressor_reduces_dynamic_range():
    quiet, loud = tone(300, amp=0.05), tone(300, amp=0.8)
    audio = np.concatenate([quiet, loud])
    out = dsp.compress(audio, SR, threshold_db=-22.0, ratio=4.0)
    range_before = rms_db(loud) - rms_db(quiet)
    range_after = rms_db(out[SR:]) - rms_db(out[:SR])
    assert range_after < range_before - 3  # at least 3 dB less dynamic range


def test_compressor_ratio_one_is_identity():
    audio = tone(300)
    assert np.allclose(dsp.compress(audio, SR, ratio=1.0), audio)


def test_gate_attenuates_quiet_parts_only():
    speech, pause = tone(300, amp=0.4), tone(300, amp=0.002)
    audio = np.concatenate([speech, pause])
    out = dsp.gate(audio, SR, threshold_db=-45.0, ratio=3.0)
    assert rms_db(out[SR + SR // 2:]) < rms_db(pause) - 6  # pause pushed down
    assert abs(rms_db(out[SR // 4: SR // 2]) - rms_db(speech)) < 1.5  # speech kept


def test_deesser_targets_sibilance_band():
    voice, sibilance = tone(300, amp=0.3), tone(7000, amp=0.4)
    audio = voice + sibilance
    out = dsp.deess(audio, SR, threshold_db=-30.0, ratio=6.0)
    from scipy.signal import butter, sosfilt
    band = butter(4, [5000, 7900], btype="bandpass", fs=SR, output="sos")
    low = butter(4, 1000, btype="lowpass", fs=SR, output="sos")
    assert rms_db(sosfilt(band, out)) < rms_db(sosfilt(band, audio)) - 3
    assert abs(rms_db(sosfilt(low, out)) - rms_db(sosfilt(low, audio))) < 1.0


def test_voice_eq_boosts_presence_band():
    audio = (RNG.normal(0, 0.1, 2 * SR)).astype(np.float32)  # white noise
    out = dsp.voice_eq(audio, SR, warmth_db=0, presence_db=4.0, air_db=0)
    from scipy.signal import butter, sosfilt
    band = butter(2, [2500, 3600], btype="bandpass", fs=SR, output="sos")
    assert rms_db(sosfilt(band, out)) > rms_db(sosfilt(band, audio)) + 2


def test_dehum_notches_50hz_and_harmonics():
    hum = tone(50, 5.0, 0.3) + tone(100, 5.0, 0.15) + tone(300, 5.0, 0.3)
    out = dsp.dehum(hum, SR, hum_hz=50.0, harmonics=2)[2 * SR:]  # steady state
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    def peak_at(f):
        return spectrum[np.abs(freqs - f) < 2].max()
    assert peak_at(50) < 0.1 * peak_at(300)   # hum crushed
    assert peak_at(100) < 0.2 * peak_at(300)  # harmonic crushed


def test_envelope_tracks_level_changes():
    audio = np.concatenate([tone(300, amp=0.05), tone(300, amp=0.5)])
    env = dsp.envelope(audio, SR)
    assert len(env) == len(audio)
    assert env[int(1.5 * SR)] > env[SR // 2] * 4
