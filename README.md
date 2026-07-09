# ClearScribe 🎙️✨

**AI voice enhancer + transcript writer — fully open source, runs 100% locally.**

Feed it a noisy voice recording. Get back a cleaned-up audio file plus transcripts in TXT, SRT, VTT, and JSON. No API keys, no cloud, no cost.

## Background voices? Use the Deep AI engine

Classic spectral denoising **cannot** separate one voice from other voices — background speech lives in the same frequencies as the main speaker, so there is nothing for a filter to grab. That's a mathematical limit, not a tuning problem. For recordings with people talking in the background, enable the **deep** backend (DeepFilterNet, a neural model trained on exactly this "babble" noise):

```bash
pip install torch
pip install "deepfilternet @ git+https://github.com/Rikorose/DeepFilterNet.git#subdirectory=DeepFilterNet"
```

Then pick **Deep AI** in the web app, or use `--backend deep` on the CLI. The model (~50 MB) downloads once and is cached.

## What it does

1. **Enhances your audio with a full broadcast chain**
   - High-pass filter removes low-frequency rumble and handling noise
   - Mains **de-hum** notches out 50/60 Hz hum and its harmonics
   - **Auto-notch** detects and removes constant tonal noises — electronics whines, beeps, monitor coil whine (typically 1–8 kHz)
   - Two-pass spectral **noise reduction** ([noisereduce](https://github.com/timsainb/noisereduce)) cleans hiss and background noise — or switch to the optional **Deep AI engine** ([DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)), which is dramatically better at background voices and chatter
   - **Noise gate** pushes pauses into clean silence
   - **Compressor** (soft-knee) evens out loud and quiet speech — the core of the "podcast sound"
   - **De-esser** tames harsh S/T sibilance
   - **Voice EQ** adds warmth (160 Hz), presence (3 kHz), and air (10 kHz)
   - **Loudness normalisation** to −16 LUFS ([pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)) — the podcast distribution standard
2. **Transcribes it** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), an optimised open-source implementation of OpenAI's Whisper — multilingual, accurate, and fast on CPU
3. **Writes transcripts** as plain text, SRT subtitles, WebVTT captions, and structured JSON

> 📖 **New here? Follow the [step-by-step Installation & Usage Guide](INSTALL.md)** — Windows, macOS, and Linux, with troubleshooting.

## Web app (easiest)

```bash
pip install "clearscribe[ui] @ git+https://github.com/Profxxxg/clearscribe.git"
clearscribe-ui
```

Then open the local URL it prints. Upload (or record) audio, pick a preset, hit **Enhance**, and **compare before/after side by side** — with measured stats (noise floor, loudness, peaks). Transcription is one more click, with downloadable TXT/SRT/VTT/JSON.

Presets: `podcast` (recommended) · `gentle` · `aggressive` · `raw`. Every stage is adjustable under *Advanced settings*.

## Install (CLI / library)

```bash
git clone https://github.com/Profxxxg/clearscribe.git
cd clearscribe
pip install .        # add .[ui] for the web app
```

Requires Python 3.10+. The first transcription run downloads the Whisper model (~500 MB for `small`) and caches it locally.

## Usage

```bash
# Enhance + transcribe (default: podcast preset, txt and srt outputs)
clearscribe interview.wav

# Presets and EU mains-hum removal
clearscribe interview.wav --preset aggressive --dehum 50

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
| `--preset` | `podcast` | `podcast` `gentle` `aggressive` `raw` |
| `--dehum` | off | Mains hum removal: `50` (EU) or `60` (US) |
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

- A 4th-order Butterworth high-pass strips energy below 80 Hz; optional notch filters remove mains hum and harmonics.
- `noisereduce` estimates the noise spectrum and applies spectral gating — attenuating time-frequency bins that look like noise.
- A downward expander (soft gate) deepens silences; a soft-knee compressor with an asymmetric attack/release envelope follower evens speech dynamics; a split-band de-esser compresses only the 5–9 kHz sibilance band.
- RBJ-cookbook biquad EQ adds broadcast tone: low shelf (warmth), presence peak, high shelf (air).
- `pyloudnorm` measures integrated loudness (ITU-R BS.1770) and gains the signal to −16 LUFS with a peak ceiling to prevent clipping.

On a synthetic noisy-speech benchmark the default chain lands within 0.1 dB of −16 LUFS, drops the noise floor by ~5 dB on top of spectral denoising, and attenuates 50 Hz hum by ~40 dB.

Transcription then runs on the **enhanced** audio, which measurably helps Whisper on noisy recordings.

## Contributing

Issues and PRs welcome! Ideas on the roadmap:

- [ ] Batch processing of folders
- [ ] Speaker diarisation
- [ ] Word-level timestamps
- [ ] Reference-track loudness matching

## License

[MIT](LICENSE) — free for anyone to use, modify, and distribute.

---

Built by [George Megally Girgis](https://github.com/Profxxxg) — Mechatronics Engineer (robotics · automation · embedded systems).
