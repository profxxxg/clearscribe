"""End-to-end pipeline: enhance a recording, then write transcripts."""

from __future__ import annotations

import logging
from pathlib import Path

from clearscribe.enhance import EnhanceSettings, enhance_audio
from clearscribe.formats import WRITERS

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    enhance: bool = True,
    transcribe: bool = True,
    formats: list[str] | None = None,
    model_size: str = "small",
    language: str | None = None,
    enhance_settings: EnhanceSettings | None = None,
) -> dict[str, Path]:
    """Run enhancement and/or transcription. Returns a dict of written files."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    formats = formats or ["txt", "srt"]

    unknown = set(formats) - set(WRITERS)
    if unknown:
        raise ValueError(f"Unknown transcript format(s): {sorted(unknown)}. "
                         f"Choose from {sorted(WRITERS)}.")

    outputs: dict[str, Path] = {}

    audio_for_transcription = input_path
    if enhance:
        enhanced_path = output_dir / f"{stem}.enhanced.wav"
        enhance_audio(input_path, enhanced_path, enhance_settings)
        outputs["enhanced_audio"] = enhanced_path
        audio_for_transcription = enhanced_path

    if transcribe:
        from clearscribe.transcribe import transcribe_audio

        transcript = transcribe_audio(
            audio_for_transcription, model_size=model_size, language=language
        )
        for fmt in formats:
            out_path = output_dir / f"{stem}.{fmt}"
            out_path.write_text(WRITERS[fmt](transcript), encoding="utf-8")
            outputs[fmt] = out_path
            logger.info("Wrote %s", out_path)

    return outputs
