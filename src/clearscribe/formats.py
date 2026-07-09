"""Transcript output formats: plain text, SRT, WebVTT, and JSON."""

from __future__ import annotations

import json

from clearscribe.transcribe import Transcript


def _timestamp(seconds: float, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep=',' for SRT, '.' for VTT)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(transcript: Transcript) -> str:
    """Plain-text transcript, one paragraph."""
    return transcript.text + "\n"


def to_srt(transcript: Transcript) -> str:
    """SubRip subtitle format."""
    blocks = []
    for i, seg in enumerate(transcript.segments, start=1):
        blocks.append(
            f"{i}\n"
            f"{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n"
            f"{seg.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    """WebVTT subtitle format."""
    blocks = ["WEBVTT\n"]
    for seg in transcript.segments:
        blocks.append(
            f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}\n"
            f"{seg.text}\n"
        )
    return "\n".join(blocks)


def to_json(transcript: Transcript) -> str:
    """Structured JSON with segments and metadata."""
    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "text": transcript.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcript.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


WRITERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}
