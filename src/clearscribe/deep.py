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


def deep_denoise(
    audio: np.ndarray, sr: int, atten_lim_db: float = 100.0
) -> np.ndarray:
    """Denoise with DeepFilterNet3. Handles resampling to/from its 48 kHz.

    Two protections against the model clipping soft speech at boundaries:
    - the signal is reflect-padded by 0.5 s on both ends so the model's
      noise estimate has warmed up before the real audio starts, then the
      padding is trimmed off;
    - `atten_lim_db` caps the maximum suppression (DeepFilterNet's own
      parameter): 100 = unlimited; lower values mix the result toward the
      original, keeping soft word onsets/endings at the cost of a slightly
      higher noise floor.
    """
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

    pad = min(int(0.5 * target_sr), max(len(x) - 1, 0))
    if pad > 0:  # reflect-pad: realistic warm-up material for the model
        x_padded = np.concatenate([x[:pad][::-1], x, x[-pad:][::-1]])
    else:
        x_padded = x

    lim = None if atten_lim_db is None or atten_lim_db >= 100 else float(atten_lim_db)
    tensor = torch.from_numpy(x_padded).unsqueeze(0)
    with torch.no_grad():
        try:
            out = enhance(model, df_state, tensor, atten_lim_db=lim)
        except TypeError:  # very old df versions lack the kwarg
            out = enhance(model, df_state, tensor)
    y = out.squeeze(0).cpu().numpy()
    y = y[pad: pad + len(x)]

    if sr != target_sr:
        g = gcd(sr, target_sr)
        y = resample_poly(y, sr // g, target_sr // g)
    return y[: len(audio)].astype(np.float32)
