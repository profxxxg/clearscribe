"""Optional deep-learning denoise backend using DeepFilterNet.

DeepFilterNet is a neural speech-enhancement model trained on noisy speech —
including *babble* (background voices/chatter) — which classic spectral
methods fundamentally cannot remove, because background speech occupies the
same frequencies as the main voice.

Install (Python 3.8–3.11):
    pip install torch torchaudio deepfilternet

Python 3.12+ note: DeepFilterNet's compiled component (deepfilterlib) has no
prebuilt wheels beyond Python 3.11, so pip tries to compile it and demands the
Rust toolchain ("Cargo ... is not installed"). Easiest fix: run ClearScribe in
a Python 3.11 virtual environment. See INSTALL.md.

The model (~50 MB) downloads automatically on first use and is cached.
"""

from __future__ import annotations

import logging
import sys
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}


def _ensure_torchaudio_compat() -> None:
    """Make DeepFilterNet importable on modern torchaudio.

    DeepFilterNet 0.5.x does `from torchaudio.backend.common import
    AudioMetaData`, but torchaudio >= 2.6 removed the `backend` module.
    We never use torchaudio I/O (ClearScribe feeds DeepFilterNet raw
    tensors and resamples with scipy), so a minimal stand-in module is
    enough to satisfy the import.
    """
    import importlib
    import types

    try:
        import torchaudio
    except ImportError:
        return  # handled by the caller's install hint
    try:
        importlib.import_module("torchaudio.backend.common")
        return  # old torchaudio — nothing to do
    except Exception:
        pass

    AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
    if AudioMetaData is None:
        try:  # some versions keep it in the private module
            from torchaudio._backend.common import AudioMetaData  # type: ignore
        except Exception:
            class AudioMetaData:  # minimal stand-in (df uses it as a type)
                def __init__(self, sample_rate=0, num_frames=0,
                             num_channels=0, bits_per_sample=0, encoding=""):
                    self.sample_rate = sample_rate
                    self.num_frames = num_frames
                    self.num_channels = num_channels
                    self.bits_per_sample = bits_per_sample
                    self.encoding = encoding

    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend
    logger.info("Applied torchaudio>=2.6 compatibility shim for DeepFilterNet")

if sys.version_info >= (3, 12):
    INSTALL_HINT = (
        "The 'deep' backend needs DeepFilterNet, which has no prebuilt wheels "
        f"for your Python ({sys.version_info.major}.{sys.version_info.minor}) — "
        "pip would try to compile it and ask for the Rust toolchain.\n"
        "Easiest fix: create a Python 3.11 virtual environment and install "
        "there:\n"
        "  py -3.11 -m venv .venv311        (Windows; on macOS/Linux: python3.11 -m venv .venv311)\n"
        "  .venv311\\Scripts\\activate\n"
        '  pip install ".[ui]" torch torchaudio deepfilternet\n'
        "See INSTALL.md for the full walkthrough."
    )
else:
    INSTALL_HINT = (
        "The 'deep' backend needs DeepFilterNet and PyTorch.\n"
        "Install them with:\n"
        "  pip install torch torchaudio deepfilternet"
    )


def is_available() -> bool:
    """True if the deep backend's dependencies are importable."""
    try:
        _ensure_torchaudio_compat()
        import df.enhance  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def deep_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Denoise with DeepFilterNet3. Handles resampling to/from its 48 kHz."""
    try:
        _ensure_torchaudio_compat()
        import torch
        from df.enhance import enhance, init_df
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e

    from scipy.signal import resample_poly

    if "model" not in _MODEL_CACHE:
        logger.info("Loading DeepFilterNet model (downloads ~50 MB on first use)")
        model, df_state, _ = init_df()
        _MODEL_CACHE.update(model=model, state=df_state)
    model, df_state = _MODEL_CACHE["model"], _MODEL_CACHE["state"]

    target_sr = int(df_state.sr())
    x = audio.astype(np.float32)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)

    with torch.no_grad():
        out = enhance(model, df_state, torch.from_numpy(x).unsqueeze(0))
    y = out.squeeze(0).cpu().numpy()

    if sr != target_sr:
        g = gcd(sr, target_sr)
        y = resample_poly(y, sr // g, target_sr // g)
    return y[: len(audio)].astype(np.float32)
