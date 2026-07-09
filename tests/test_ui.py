"""UI smoke tests (skipped automatically if gradio isn't installed)."""

import pytest

gr = pytest.importorskip("gradio")

from clearscribe.ui import _apply_preset, _settings_from_ui, build_app, enhance_for_ui


def test_app_builds():
    app = build_app()
    assert app is not None


def test_apply_preset_returns_all_knobs():
    values = _apply_preset("gentle")
    assert len(values) == 9
    assert values[0] == 0.6  # gentle noise-reduction strength


def test_settings_from_ui_maps_dehum_choice():
    s = _settings_from_ui("podcast", "Spectral (built-in)", True,
                          0.85, False, "50 Hz (EU/most of world)",
                          -45, 3.0, -22, 4.0, 1.5, 2.5, 1.5, -16)
    assert s.dehum_hz == 50.0
    assert s.comp_ratio == 3.0
    assert s.backend == "spectral" and s.auto_notch


def test_settings_from_ui_deep_backend():
    s = _settings_from_ui("podcast", "Deep AI — DeepFilterNet", False,
                          0.85, False, "Off",
                          -45, 3.0, -22, 4.0, 1.5, 2.5, 1.5, -16)
    assert s.backend == "deep" and not s.auto_notch


def test_enhance_for_ui_end_to_end(noisy_wav):
    out_path, state_path, stats = enhance_for_ui(
        str(noisy_wav), "podcast", "Spectral (built-in)", True, 0.85, False,
        "Off", -45, 3.0, -22, 4.0, 1.5, 2.5, 1.5, -16,
    )
    assert out_path == state_path
    assert out_path.endswith(".enhanced.wav")
    assert "Noise floor" in stats


def test_enhance_for_ui_rejects_missing_file():
    with pytest.raises(gr.Error):
        enhance_for_ui(None, "podcast", "Spectral (built-in)", True, 0.85,
                       False, "Off", -45, 3.0, -22, 4.0, 1.5, 2.5, 1.5, -16)
