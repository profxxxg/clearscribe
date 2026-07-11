"""SystemRecorder tests with a fake sounddevice (no real mic in CI)."""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

from clearscribe.recorder import SystemRecorder


@pytest.fixture
def fake_sounddevice(monkeypatch):
    fake = types.ModuleType("sounddevice")

    class FakeInputStream:
        def __init__(self, samplerate, channels, dtype, callback):
            self.callback = callback
            self.samplerate = samplerate
            self.channels = channels

        def start(self):
            # Simulate 1 second of captured audio in 10 chunks
            chunk = np.full((self.samplerate // 10, self.channels), 0.1,
                            dtype=np.float32)
            for _ in range(10):
                self.callback(chunk, len(chunk), None, None)

        def stop(self):
            pass

        def close(self):
            pass

    fake.InputStream = FakeInputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_record_start_stop_produces_wav(fake_sounddevice):
    rec = SystemRecorder(samplerate=16000)
    assert not rec.recording
    rec.start()
    assert rec.recording
    path = rec.stop()
    assert not rec.recording
    audio, sr = sf.read(str(path))
    assert sr == 16000
    assert abs(len(audio) - 16000) < 100  # ~1 s captured


def test_stop_without_start_errors(fake_sounddevice):
    with pytest.raises(RuntimeError, match="Not recording"):
        SystemRecorder().stop()


def test_double_start_errors(fake_sounddevice):
    rec = SystemRecorder(samplerate=16000)
    rec.start()
    with pytest.raises(RuntimeError, match="Already recording"):
        rec.start()
    rec.stop()


def test_missing_sounddevice_gives_hint(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def block(name, *a, **k):
        if name == "sounddevice":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(RuntimeError, match="pip install sounddevice"):
        SystemRecorder().start()
