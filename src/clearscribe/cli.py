"""ClearScribe command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from clearscribe import __version__
from clearscribe.enhance import EnhanceSettings
from clearscribe.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clearscribe",
        description="AI voice enhancer + transcript writer (open source).",
    )
    p.add_argument("input", help="Input audio file (WAV, MP3, FLAC, M4A, ...)")
    p.add_argument("-o", "--output-dir", default="clearscribe_output",
                   help="Directory for outputs (default: clearscribe_output)")
    p.add_argument("--formats", default="txt,srt",
                   help="Comma-separated transcript formats: txt,srt,vtt,json")
    p.add_argument("--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="Whisper model size (default: small)")
    p.add_argument("--language", default=None,
                   help="Language code, e.g. 'en' or 'ar' (default: auto-detect)")
    p.add_argument("--no-enhance", action="store_true",
                   help="Skip enhancement, transcribe the original audio")
    p.add_argument("--enhance-only", action="store_true",
                   help="Only enhance the audio, skip transcription")
    p.add_argument("--strength", type=float, default=0.85,
                   help="Noise reduction strength 0-1 (default: 0.85)")
    p.add_argument("--stationary", action="store_true",
                   help="Use stationary noise reduction (constant hum/hiss)")
    p.add_argument("--target-lufs", type=float, default=-16.0,
                   help="Loudness target in LUFS (default: -16)")
    p.add_argument("--highpass", type=float, default=80.0,
                   help="High-pass cutoff in Hz, 0 to disable (default: 80)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--version", action="version", version=f"clearscribe {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.no_enhance and args.enhance_only:
        print("error: --no-enhance and --enhance-only are mutually exclusive",
              file=sys.stderr)
        return 2

    settings = EnhanceSettings(
        highpass_hz=args.highpass,
        noise_reduction_strength=args.strength,
        stationary_noise=args.stationary,
        target_lufs=args.target_lufs,
    )

    try:
        outputs = run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            enhance=not args.no_enhance,
            transcribe=not args.enhance_only,
            formats=[f.strip() for f in args.formats.split(",") if f.strip()],
            model_size=args.model,
            language=args.language,
            enhance_settings=settings,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print("Done. Files written:")
    for kind, path in outputs.items():
        print(f"  {kind:>15}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
