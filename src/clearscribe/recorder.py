"""Browser-independent microphone recording.

Gradio's built-in recorder runs inside the browser (MediaRecorder / Web
Audio), which some browsers and privacy settings break — the classic symptom
is a 0:00 recording and a frozen page. Since ClearScribe runs locally, we can
sidestep the browser entirely and capture audio from the system microphone
in Python via the `sounddevice` library (bundled PortAudio, pip-installable).
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INSTALL_HINT = ("System-mic recording needs the 'sounddevice' package:  "
                "pip install sounddevice")


class SystemRecorder:
    """Start/stop recording from the default input device. Thread-safe."""

    def __init__(self, samplerate: int = 48000):
        self.samplerate = samplerate
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError(INSTALL_HINT) from e

        with self._lock:
            if self._stream is not None:
                raise RuntimeError("Already recording — press Stop first.")
            self._chunks = []

            def _callback(indata, frames, time_info, status):
                if status:
                    logger.warning("Recording status: %s", status)
                self._chunks.append(indata.copy())

            try:
                stream = sd.InputStream(
                    samplerate=self.samplerate, channels=1,
                    dtype="float32", callback=_callback,
                )
                stream.start()
            except Exception as e:
                raise RuntimeError(
                    f"Could not open the microphone: {e}. Check that a mic "
                    "is connected, and that Windows Settings → Privacy → "
                    "Microphone allows desktop apps to use it."
                ) from e
            self._stream = stream
            logger.info("System recording started at %d Hz", self.samplerate)

    def stop(self) -> Path:
        """Stop and write the capture to a WAV file. Returns its path."""
        import soundfile as sf

        with self._lock:
            if self._stream is None:
                raise RuntimeError("Not recording — press Record first.")
            stream, self._stream = self._stream, None
        stream.stop()
        stream.close()

        if not self._chunks:
            raise RuntimeError("No audio was captured — is the mic muted?")
        data = np.concatenate(self._chunks)[:, 0]
        if len(data) < self.samplerate * 0.25:
            raise RuntimeError("Recording was shorter than 0.25 s — "
                               "try again and speak for a moment.")
        path = Path(tempfile.mkdtemp()) / "recording.wav"
        sf.write(str(path), data, self.samplerate, subtype="PCM_16")
        logger.info("System recording saved: %s (%.1f s)",
                    path, len(data) / self.samplerate)
        return path
