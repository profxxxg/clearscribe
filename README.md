# ClearScribe 🎙️✨

**AI voice enhancer + transcript writer — fully open source, runs 100% locally.**

Feed it a noisy voice recording. Get back a cleaned-up audio file plus transcripts in TXT, SRT, VTT, and JSON. No API keys, no cloud, no cost.

## What it does

1. **Enhances your audio**
   - High-pass filter removes low-frequency rumble and handling noise
   - Spectral-gating noise reduction ([noisereduce](https://github.com/timsainb/noisereduce)) cleans hiss, hum, and background noise
   - Loudness normalisation to −16 LUFS ([pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)) for consistent, podcast-ready volume
2. **Transcribes it** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), an optimised open-source implementation of OpenAI's Whisper — multilingual, accurate, and fast on CPU
3. **Writes transcripts** as plain text, SRT subtitles, WebVTT captions, and structured JSON

## Install

```bash
git clone https://github.com/Profxxxg/clearscribe.git
cd clearscribe
pip install .
```

Requires Python 3.10+. The first transcription run downloads the Whisper model (~500 MB for `small`) and caches it locally.

## Usage

```bash
# Enhance + transcribe (default: txt and srt outputs)
clearscribe interview.wav

# All transcript formats, medium model, English
clearscribe podcast.mp3 --formats txt,srt,vtt,json --model medium --language en

# Just clean the audio
clearscribe voiceover.wav --enhance-only --strength 0.9

# Just transcribe, skip enhancement
clearscribe lecture.m4a --no-enhance --model small
```

Outputs land in `clearscribe_output/` (change with `-o`):

```
clearscribe_output/
├── interview.enhanced.wav
├── interview.txt
└── interview.srt
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `small` | Whisper size: `tiny` `base` `small` `medium` `large-v3` |
| `--language` | auto | Language code, e.g. `en`, `ar`, `fr` |
| `--strength` | `0.85` | Noise reduction strength (0–1) |
| `--stationary` | off | Better for constant hum/hiss |
| `--target-lufs` | `-16` | Loudness target |
| `--highpass` | `80` | High-pass cutoff in Hz (0 disables) |

### Use it as a library

```python
from clearscribe.pipeline import run_pipeline

outputs = run_pipeline("interview.wav", "out/", formats=["srt", "json"])
print(outputs["srt"].read_text())
```

## Run the tests

```bash
pip install .[dev]
pytest
```

The test suite runs fully offline — transcription is mocked, and enhancement is verified against synthetic noisy audio (measured noise-floor reduction, rumble attenuation, no clipping, duration preserved).

## How the enhancement works

The pipeline is deliberately classic DSP + one modern trick:

- A 4th-order Butterworth high-pass strips energy below 80 Hz, where voice carries almost no information but rumble lives.
- `noisereduce` estimates the noise spectrum and applies spectral gating — attenuating time-frequency bins that look like noise.
- `pyloudnorm` measures integrated loudness (ITU-R BS.1770) and gains the signal to −16 LUFS with a peak ceiling to prevent clipping.

Transcription then runs on the **enhanced** audio, which measurably helps Whisper on noisy recordings.

## Contributing

Issues and PRs welcome! Ideas on the roadmap:

- [ ] Batch processing of folders
- [ ] Optional DeepFilterNet backend for stronger denoising
- [ ] Speaker diarisation
- [ ] Word-level timestamps

## License

[MIT](LICENSE) — free for anyone to use, modify, and distribute.

---

Built by [George Megally Girgis](https://github.com/Profxxxg) — Mechatronics Engineer (robotics · automation · embedded systems).
