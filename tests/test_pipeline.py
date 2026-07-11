"""Pipeline + CLI tests. Whisper is mocked so the suite runs fully offline."""

from unittest.mock import patch

import pytest

from clearscribe.cli import main
from clearscribe.pipeline import run_pipeline
from clearscribe.transcribe import Segment, Transcript

FAKE_TRANSCRIPT = Transcript(
    segments=[Segment(0.0, 2.0, "Testing one two.")],
    language="en",
    duration=3.0,
)


def test_enhance_only_pipeline(noisy_wav, tmp_path):
    outputs = run_pipeline(noisy_wav, tmp_path / "out", transcribe=False)
    assert set(outputs) == {"enhanced_audio"}
    assert outputs["enhanced_audio"].exists()


def test_full_pipeline_with_mocked_whisper(noisy_wav, tmp_path):
    with patch("clearscribe.transcribe.transcribe_audio",
               return_value=FAKE_TRANSCRIPT) as mock_t:
        outputs = run_pipeline(
            noisy_wav, tmp_path / "out", formats=["txt", "srt", "vtt", "json"]
        )
    # Transcription should run on the ENHANCED audio, not the original
    assert mock_t.call_args.args[0] == outputs["enhanced_audio"]
    for fmt in ("txt", "srt", "vtt", "json"):
        assert outputs[fmt].exists()
        assert "Testing one two." in outputs[fmt].read_text(encoding="utf-8")


def test_pipeline_rejects_unknown_format(noisy_wav, tmp_path):
    with pytest.raises(ValueError, match="Unknown transcript format"):
        run_pipeline(noisy_wav, tmp_path, formats=["docx"])


def test_cli_enhance_only(noisy_wav, tmp_path, capsys):
    rc = main([str(noisy_wav), "-o", str(tmp_path / "cli_out"), "--enhance-only"])
    assert rc == 0
    assert "enhanced_audio" in capsys.readouterr().out


def test_cli_conflicting_flags(noisy_wav):
    rc = main([str(noisy_wav), "--no-enhance", "--enhance-only"])
    assert rc == 2


def test_cli_missing_file(tmp_path):
    rc = main([str(tmp_path / "nope.wav"), "--enhance-only"])
    assert rc == 1


def test_deep_backend_is_dispatched(noisy_wav, tmp_path):
    """backend='deep' must route through clearscribe.deep.deep_denoise."""
    import numpy as np
    from clearscribe.enhance import EnhanceSettings

    with patch("clearscribe.deep.deep_denoise",
               side_effect=lambda a, sr, atten_lim_db=100.0: np.zeros_like(a)) as mock_deep:
        run_pipeline(noisy_wav, tmp_path / "out", transcribe=False,
                     enhance_settings=EnhanceSettings(backend="deep"))
    mock_deep.assert_called_once()


def test_deep_backend_missing_gives_helpful_error(noisy_wav, tmp_path):
    import builtins
    from clearscribe.enhance import EnhanceSettings
    real_import = builtins.__import__

    def block_df(name, *a, **k):
        if name.startswith("df") or name == "torch":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    with patch("builtins.__import__", side_effect=block_df):
        with pytest.raises(ImportError, match="deep"):
            run_pipeline(noisy_wav, tmp_path / "out", transcribe=False,
                         enhance_settings=EnhanceSettings(backend="deep"))


def test_torchaudio_compat_shim(monkeypatch):
    """Shim must create torchaudio.backend.common when modern torchaudio lacks it."""
    import sys
    import types

    from clearscribe.deep import _ensure_torchaudio_compat

    fake_ta = types.ModuleType("torchaudio")  # modern torchaudio: no .backend
    for name in list(sys.modules):
        if name == "torchaudio" or name.startswith("torchaudio."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    _ensure_torchaudio_compat()
    from torchaudio.backend.common import AudioMetaData  # must not raise
    assert AudioMetaData is not None
    assert sys.modules["torchaudio"].backend.common.AudioMetaData is AudioMetaData


def test_deep_denoise_padding_and_atten(monkeypatch):
    """deep_denoise must pass atten limit through and return exact length."""
    import sys
    import types

    import numpy as np

    calls = {}

    class FakeTensor:
        def __init__(self, arr): self.arr = np.asarray(arr)
        def unsqueeze(self, _): return self
        def squeeze(self, _): return self
        def cpu(self): return self
        def numpy(self): return self.arr

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = lambda a: FakeTensor(a)
    class _NoGrad:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    fake_torch.no_grad = _NoGrad

    fake_df = types.ModuleType("df")
    fake_enh = types.ModuleType("df.enhance")
    class FakeState:
        def sr(self): return 16000
    def fake_init_df(): return "model", FakeState(), None
    def fake_enhance(model, state, tensor, atten_lim_db=None):
        calls["atten"] = atten_lim_db
        calls["in_len"] = len(tensor.arr)
        return tensor  # identity
    fake_enh.init_df, fake_enh.enhance = fake_init_df, fake_enhance
    fake_df.enhance = fake_enh
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "df", fake_df)
    monkeypatch.setitem(sys.modules, "df.enhance", fake_enh)

    import clearscribe.deep as deep
    monkeypatch.setattr(deep, "_MODEL_CACHE", {})

    audio = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    out = deep.deep_denoise(audio, 16000, atten_lim_db=40.0)
    assert len(out) == len(audio)             # padding fully trimmed
    assert calls["atten"] == 40.0             # limit passed to the model
    assert calls["in_len"] == len(audio) + 2 * 8000  # 0.5 s reflect each side
    assert np.allclose(out, audio, atol=1e-6)  # identity model -> identity out

    out2 = deep.deep_denoise(audio, 16000, atten_lim_db=100.0)
    assert calls["atten"] is None             # 100 means unlimited
    assert len(out2) == len(audio)
