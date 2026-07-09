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
