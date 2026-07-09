"""Speech-to-text transcription using faster-whisper (open-source Whisper)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single transcribed segment with timing."""

    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """Full transcription result."""

    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "small",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "int8",
) -> Transcript:
    """Transcribe an audio file with faster-whisper.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.).
        model_size: Whisper model size — tiny, base, small, medium, large-v3.
            "small" is a good accuracy/speed balance on CPU.
        language: ISO language code (e.g. "en"). None = auto-detect.
        device: "cpu", "cuda", or "auto".
        compute_type: "int8" is fast on CPU; use "float16" on GPU.

    Note:
        The first run downloads the model from Hugging Face (~500 MB for
        "small"); it is cached locally afterwards.
    """
    # Lazy import so enhancement-only usage never requires the model stack.
    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model '%s' (device=%s)", model_size, device)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info("Transcribing %s", audio_path)
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,  # skip long silences
        beam_size=5,
    )

    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip()) for s in raw_segments
    ]
    logger.info(
        "Transcribed %d segments (language=%s, duration=%.1fs)",
        len(segments), info.language, info.duration,
    )
    return Transcript(
        segments=segments,
        language=info.language or "",
        duration=info.duration or 0.0,
    )
