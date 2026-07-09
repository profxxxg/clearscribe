"""Universal media input: decode any audio or video format via ffmpeg.

ClearScribe bundles a static ffmpeg binary through the `imageio-ffmpeg`
package, so users don't need a system ffmpeg install. This enables:
  - reading formats libsndfile can't (M4A, AAC, OPUS, WMA, ...)
  - extracting the audio track from video files (MP4, MKV, MOV, WEBM, ...)
  - Gradio's microphone recording (browsers record WebM; Gradio needs an
    `ffmpeg` on PATH to convert it — see expose_ffmpeg_on_path()).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: Extensions libsndfile handles natively — no ffmpeg needed.
SNDFILE_EXTS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif", ".mp3"}

FFMPEG_HINT = (
    "Reading this file format needs ffmpeg. It ships with ClearScribe's "
    "dependencies — reinstall with:  pip install imageio-ffmpeg"
)


def ffmpeg_exe() -> str | None:
    """Path to an ffmpeg binary: system one if present, else the bundled one."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def expose_ffmpeg_on_path() -> None:
    """Make `ffmpeg` findable via PATH for libraries that call it by name.

    Gradio converts microphone recordings (WebM) by invoking `ffmpeg` — if
    it isn't on PATH the UI appears to freeze after recording. The bundled
    binary has a versioned filename, so we place an alias copy named
    `ffmpeg` in ~/.clearscribe/bin and prepend that to PATH.
    """
    if shutil.which("ffmpeg"):
        return
    exe = ffmpeg_exe()
    if exe is None:
        logger.warning("No ffmpeg found — microphone recording and "
                       "video/M4A input will not work. %s", FFMPEG_HINT)
        return
    alias_dir = Path.home() / ".clearscribe" / "bin"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    try:
        if not alias.exists() or alias.stat().st_size != Path(exe).stat().st_size:
            shutil.copy2(exe, alias)
            alias.chmod(0o755)
        os.environ["PATH"] = str(alias_dir) + os.pathsep + os.environ.get("PATH", "")
        logger.info("Exposed bundled ffmpeg on PATH: %s", alias)
    except OSError as e:  # pragma: no cover - permission edge cases
        logger.warning("Could not expose ffmpeg on PATH (%s). %s", e, FFMPEG_HINT)


def to_wav(src: str | Path, dst: str | Path | None = None) -> Path:
    """Extract/convert the audio of any media file to 16-bit WAV.

    Works for audio formats libsndfile can't read and for video files
    (the video stream is dropped, sample rate is preserved).
    """
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError(FFMPEG_HINT)
    src = Path(src)
    dst = Path(dst) if dst else Path(tempfile.mkdtemp()) / f"{src.stem}.wav"
    cmd = [exe, "-y", "-i", str(src), "-vn", "-acodec", "pcm_s16le", str(dst)]
    logger.info("Decoding %s with ffmpeg", src.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dst.exists():
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        raise RuntimeError(
            f"ffmpeg could not decode '{src.name}': {tail[0]}. "
            "Is there an audio track in this file?"
        )
    return dst
