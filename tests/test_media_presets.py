"""Tests for universal media input and user preset persistence."""

import subprocess

import numpy as np
import pytest

from clearscribe.enhance import PRESETS, EnhanceSettings, load_mono
from clearscribe.media import ffmpeg_exe, to_wav
from clearscribe.presets import (all_presets, delete_user_preset,
                                 load_user_presets, save_user_preset)

ffmpeg = ffmpeg_exe()
needs_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="no ffmpeg available")


@pytest.fixture
def video_file(noisy_wav, tmp_path):
    """Wrap the noisy wav into a real MP4 video (black frames + AAC audio)."""
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=3",
         "-i", str(noisy_wav), "-shortest", "-c:v", "libx264",
         "-c:a", "aac", str(mp4)],
        check=True, capture_output=True)
    return mp4


@needs_ffmpeg
def test_to_wav_extracts_audio_from_video(video_file):
    wav = to_wav(video_file)
    audio, sr = load_mono(wav)
    assert sr > 0 and len(audio) > sr * 2  # ~3 s of audio came out


@needs_ffmpeg
def test_load_mono_reads_video_directly(video_file):
    audio, sr = load_mono(video_file)
    assert audio.ndim == 1 and len(audio) > sr * 2


def test_to_wav_error_on_garbage(tmp_path):
    bad = tmp_path / "not_media.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(RuntimeError, match="could not decode"):
        to_wav(bad)


@pytest.fixture
def preset_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARSCRIBE_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path / "cfg"


def test_preset_roundtrip(preset_dir):
    s = EnhanceSettings(noise_reduction_strength=0.42, comp_ratio=5.5,
                        backend="deep", dehum_hz=50.0)
    save_user_preset("my-mic", s)
    loaded = load_user_presets()["my-mic"]
    assert loaded == s
    assert "my-mic" in all_presets() and "podcast" in all_presets()
    assert delete_user_preset("my-mic") and "my-mic" not in load_user_presets()


def test_preset_name_validation(preset_dir):
    with pytest.raises(ValueError, match="empty"):
        save_user_preset("   ", EnhanceSettings())
    with pytest.raises(ValueError, match="built-in"):
        save_user_preset("podcast", EnhanceSettings())


def test_user_preset_cannot_shadow_builtin(preset_dir):
    import json
    from clearscribe.presets import _presets_file
    _presets_file().parent.mkdir(parents=True, exist_ok=True)
    _presets_file().write_text(json.dumps(
        {"podcast": {"noise_reduction_strength": 0.0}}))
    assert all_presets()["podcast"] == PRESETS["podcast"]  # built-in wins
