"""Optional deep-learning denoise backend using DeepFilterNet.

DeepFilterNet is a neural speech-enhancement model trained on noisy speech —
including *babble* (background voices/chatter) — which classic spectral
methods fundamentally cannot remove, because background speech occupies the
same frequencies as the main voice.

Install:
    pip install torch
    pip install "deepfilternet @ git+https://github.com/Rikorose/DeepFilterNet.git#subdirectory=DeepFilterNet"

The model (~50 MB) downloads automatically on first use and is cached.
"""

from __future__ import annotations

import logging
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}

INSTALL_HINT = (
    "The 'deep' backend needs DeepFilterNet and PyTorch.\n"
    "Install them with:\n"
    "  pip install torch\n"
    '  pip install "deepfilternet @ '
    'git+https://github.com/Rikorose/DeepFilterNet.git#subdirectory=DeepFilterNet"'
)


def is_available() -> bool:
    """True if the deep backend's dependencies are importable."""
    try:
        import df.enhance  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def deep_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Denoise with DeepFilterNet3. Handles resampling to/from its 48 kHz."""
    try:
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
