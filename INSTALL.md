# 📦 Installation & Usage Guide

The **recommended setup** below gives you the full ClearScribe: the broadcast
enhancement chain **plus the Deep AI engine** (DeepFilterNet), which is what
removes background voices and heavy noise. It takes ~10 minutes, mostly
downloads. A lighter install without Deep AI is at the end.

> ℹ️ Why Python 3.11 specifically? The Deep AI engine's compiled component
> ships prebuilt only for Python 3.8–3.11. Python 3.11 alongside your other
> Pythons is completely fine — the setup below keeps everything in its own
> virtual environment.

---

## 1 · Install Python 3.11

Use **3.11.9** — the last 3.11 with a Windows installer (newer 3.11.x are
source-only security releases):

- [Direct installer download (Windows 64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
- [Release page](https://www.python.org/downloads/release/python-3119/) (macOS installer there too; Linux: `sudo apt install python3.11 python3.11-venv` or your distro's equivalent)

**Windows:** in the installer, keep the **py launcher** ticked and tick
✅ **"Add python.exe to PATH"**. Then **open a new terminal** (PATH changes
don't reach already-open ones).

## 2 · Get the code

```bash
git clone https://github.com/Profxxxg/clearscribe.git
cd clearscribe
```

(No Git? **Code → Download ZIP** on GitHub, extract, open a terminal in the folder.)

## 3 · Create the virtual environment

**Windows (PowerShell or CMD):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

> ⚠️ **Mind the space** in `venv .venv` — module `venv`, space, folder
> `.venv`. Typing `venv.venv` gives `No module named venv.venv`.
> If PowerShell blocks activation, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> or use CMD.

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`.

## 4 · Install ClearScribe with the Deep AI engine

```bash
pip install ".[ui]"
pip install torch torchaudio deepfilternet
```

The second line is ~1 GB (PyTorch) — it can look frozen for a few minutes;
it isn't. Verify:

```bash
clearscribe --version
```

## 5 · Run it

```bash
clearscribe-ui
```

Open the printed address (usually `http://127.0.0.1:7860`). Then:

1. **Upload** any audio or video file — WAV, MP3, M4A, MP4, MKV, ... — or
   record with the **🎙️ Record / ⏹ Stop & use** buttons (records from your
   system microphone directly, no browser involvement)
2. Pick the **Denoise engine**: *Deep AI* for background voices/chatter and
   heavy noise (first use downloads its ~50 MB model, one time), *Spectral*
   for light cleanup — it's faster
3. Click **✨ Enhance**, then compare the **Original** and **Enhanced**
   players side by side, with measured stats
4. Optionally **📝 Transcribe** and download TXT/SRT/VTT/JSON
5. Dialled-in settings? Name them and **💾 Save preset** — they persist in
   "My saved presets" (`~/.clearscribe/presets.json`)

> **Deep AI clipping soft word starts/endings?** Lower the **"Deep AI max
> suppression"** slider in Advanced settings (try 30–60 dB) — gentler
> suppression keeps quiet speech at the cost of slightly more residual noise.

## 6 · Everyday use — no reinstalling!

Everything above is **one-time**. From then on:

```powershell
cd path\to\clearscribe
.venv\Scripts\activate      # (source .venv/bin/activate on macOS/Linux)
clearscribe-ui
```

You only touch pip again to update (section 9).

---

## 7 · Command line & Python library

```bash
# Enhance + transcribe with the deep engine, all transcript formats
clearscribe interview.wav --backend deep --formats txt,srt,vtt,json

# Presets, hum removal, saving your settings
clearscribe voiceover.mp4 --enhance-only --preset aggressive --dehum 50 --save-preset street
clearscribe next_take.mp4 --enhance-only --preset street
```

`clearscribe --help` lists everything. As a library:

```python
from clearscribe.pipeline import run_pipeline
from clearscribe.enhance import PRESETS

outputs = run_pipeline("interview.wav", "out/", formats=["srt", "json"],
                       enhance_settings=PRESETS["podcast"])
```

## 8 · Run the tests (contributors)

```bash
pip install ".[dev]"
pytest -v
```

The suite runs fully offline — Whisper and DeepFilterNet are mocked, DSP
stages are verified against synthetic audio with measurable assertions.

## 9 · Updating

```bash
cd clearscribe
git pull
pip install ".[ui]" --upgrade
```

---

## Alternative: lite install (no Deep AI)

Works on any Python **3.10+**, no PyTorch download; you get the full
spectral chain (denoise, auto-notch, gate, compressor, de-esser, EQ,
loudness) and transcription — but not the neural engine that handles
background voices:

```bash
python -m venv .venv
.venv\Scripts\activate          # or: source .venv/bin/activate
pip install ".[ui]"
clearscribe-ui
```

You can add the Deep AI engine later by redoing steps 1, 3, 4 with
Python 3.11.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `pip` / `py` is not recognized (Windows) | Install Python from [python.org](https://www.python.org/downloads/) with **"Add python.exe to PATH"** ticked (the `py` launcher is included — Microsoft Store Pythons don't have it), then **open a new terminal**. Or call the interpreter directly: `"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m venv .venv` |
| `No module named venv.venv` | Missing space: it's `python -m venv .venv` |
| `clearscribe: command not found` | The venv isn't activated — redo the activate command from step 3 |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or use CMD |
| `Cargo, the Rust package manager, is not installed` installing deepfilternet | You're on Python 3.12+ (no prebuilt wheels) — use the Python 3.11 setup above |
| `No module named 'torchaudio'` using Deep AI | `pip install torchaudio` in the venv (version clash: `pip install torch torchaudio --upgrade`) |
| `No module named 'torchaudio.backend'` using Deep AI | Update ClearScribe (`git pull` + `pip install .`) — it ships a compatibility shim. Or pin: `pip install torch==2.5.1 torchaudio==2.5.1` |
| 🎙️ Record says "could not open the microphone" | Check a mic is connected and Windows Settings → Privacy & security → Microphone allows desktop apps; close other apps using the mic exclusively |
| First transcription / first Deep AI run hangs at the start | One-time model download (~500 MB Whisper `small`; ~50 MB DeepFilterNet) — watch the terminal |
| Deep AI cuts the start/end of sentences | Lower **"Deep AI max suppression"** in Advanced settings to 30–60 dB |
| Transcription slow | Smaller model: `--model base` or `tiny` |
| Wrong language detected | Set it: `--language en`, or type it in the UI |
| Port 7860 already in use | Close the other Gradio app, or `GRADIO_SERVER_PORT=7861 clearscribe-ui` |
| Sounds over-processed | `gentle` preset, or lower noise reduction / compressor |
| Background voices still audible on Spectral | That's expected — spectral filtering can't separate voice from voice. Use the Deep AI engine |
| A constant whine/beep survives (1–8 kHz) | Keep **Auto-remove tonal noises** ticked; set the Presence EQ slider to 0 so leftover noise there isn't boosted |

Still stuck? [Open an issue](https://github.com/Profxxxg/clearscribe/issues)
with your OS, Python version, and the full error message.
