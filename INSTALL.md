# 📦 Installation & Usage Guide

This guide takes you from nothing to a working ClearScribe in about 5 minutes.
It covers **Windows**, **macOS**, and **Linux**.

---

## 1 · Prerequisites

You need **Python 3.10 or newer**.

Check what you have:

```bash
python --version        # Windows
python3 --version       # macOS / Linux
```

If it prints `3.10.x` or higher, you're good. If not, install Python from
[python.org/downloads](https://www.python.org/downloads/).
**Windows users:** during installation, tick ✅ **"Add Python to PATH"** —
this is the #1 cause of `python is not recognized` errors.

Optional: [Git](https://git-scm.com/downloads), for cloning the repo.
No Git? Click **Code → Download ZIP** on the GitHub page and extract it instead.

---

## 2 · Get the code

```bash
git clone https://github.com/Profxxxg/clearscribe.git
cd clearscribe
```

(Or extract the downloaded ZIP and open a terminal inside the extracted folder.)

---

## 3 · Create a virtual environment (recommended)

This keeps ClearScribe's dependencies separate from the rest of your system.

**Windows (PowerShell or CMD):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

> ⚠️ **Mind the space** in `venv .venv` — it's the module `venv`, a space, then
> the folder name `.venv`. Typing `venv.venv` gives
> `No module named venv.venv`.

> If PowerShell complains about execution policy, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> then try activating again — or just use CMD instead.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Do this activation step again
whenever you come back in a new terminal.

---

## 4 · Install ClearScribe

**With the web app (recommended for most people):**

```bash
pip install ".[ui]"
```

**CLI / library only (lighter):**

```bash
pip install .
```

This pulls in the open-source stack: `noisereduce`, `pyloudnorm`, `scipy`,
`soundfile`, `faster-whisper` — and `gradio` if you chose `[ui]`.

Verify it worked:

```bash
clearscribe --version
```

---

## 5 · Use the web app

```bash
clearscribe-ui
```

Your terminal prints a local address, usually `http://127.0.0.1:7860`.
Open it in any browser. Then:

1. **Upload** an audio file (or click the microphone icon to record directly)
2. Pick a **preset** — `podcast` is the recommended full chain
3. Click **✨ Enhance**
4. **Compare before and after** — the original and enhanced players sit side
   by side, and a stats table shows the measured noise floor, loudness, and
   peak levels for both
5. (Optional) Click **📝 Transcribe enhanced audio** — pick a Whisper model
   size and formats (TXT, SRT, VTT, JSON), then download the transcript files

Tweak any stage (noise reduction, gate, compressor, de-esser, EQ, loudness)
under **Advanced settings**. To stop the app, press `Ctrl+C` in the terminal.

> **First transcription is slow — that's normal.** faster-whisper downloads
> the model on first use (~500 MB for `small`) and caches it. Every run after
> that starts instantly. You need an internet connection for that first run only.

---

## 5½ · Everyday use — no reinstalling!

All the `pip install` steps above are **one-time**. The packages live inside
the `.venv` folder. From then on, using ClearScribe is just:

```powershell
cd path\to\clearscribe
.venv\Scripts\activate     # (source .venv/bin/activate on macOS/Linux)
clearscribe-ui
```

You only run pip again when updating ClearScribe itself (section 9).

---

## 6 · Use the command line

```bash
# Enhance + transcribe with defaults (podcast preset; txt + srt out)
clearscribe interview.wav

# Choose formats, model, and language
clearscribe podcast.mp3 --formats txt,srt,vtt,json --model medium --language en

# Only clean up the audio
clearscribe voiceover.wav --enhance-only --preset aggressive --dehum 50

# Only transcribe, skip enhancement
clearscribe lecture.wav --no-enhance --model small
```

Results land in `clearscribe_output/` (change with `-o mydir`).
Run `clearscribe --help` for every option.

---

## 7 · Use it as a Python library

```python
from clearscribe.pipeline import run_pipeline
from clearscribe.enhance import PRESETS

outputs = run_pipeline(
    "interview.wav", "out/",
    formats=["srt", "json"],
    enhance_settings=PRESETS["podcast"],
)
print(outputs["srt"].read_text(encoding="utf-8"))
```

---

## 8 · Run the tests (for contributors)

```bash
pip install ".[dev]"
pytest -v
```

All 28 tests run fully offline — transcription is mocked and the DSP stages
are verified against synthetic audio with measurable assertions.

---

## 9 · Updating

```bash
cd clearscribe
git pull
pip install ".[ui]" --upgrade
```

---

## Optional: the Deep AI denoise engine

If your recordings have **background voices/chatter** or heavy noise the
built-in denoiser can't handle, install the DeepFilterNet backend.

> ⚠️ **Requires Python 3.8–3.11.** DeepFilterNet's compiled component has no
> prebuilt wheels for Python 3.12+ yet — on 3.12 pip tries to compile it and
> fails with *"Cargo, the Rust package manager, is not installed"*.

**If your Python is 3.8–3.11**, it's just:

```bash
pip install torch torchaudio deepfilternet
```

**If your Python is 3.12 or newer**, run ClearScribe in a Python 3.11
environment instead (both Pythons coexist happily):

1. Install Python 3.11. **Use 3.11.9** — it's the last 3.11 with a Windows
   installer (newer 3.11.x are source-only security releases):
   [direct installer download](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
   · [release page](https://www.python.org/downloads/release/python-3119/)
2. In the `clearscribe` folder, create and activate a 3.11 venv:

   **Windows:**
   ```powershell
   py -3.11 -m venv .venv311
   .venv311\Scripts\activate
   ```
   **macOS / Linux:**
   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   ```
3. Install everything in it:
   ```bash
   pip install ".[ui]"
   pip install torch torchaudio deepfilternet
   ```

(Alternative for those who want to stay on 3.12: install
[Rust](https://rustup.rs/) and, on Windows, the Microsoft C++ Build Tools,
then `pip install deepfilternet` compiles from source.)

Then choose **Deep AI — DeepFilterNet** in the web app's "Denoise engine"
selector, or add `--backend deep` on the command line. The model (~50 MB)
downloads automatically on first use. Note: PyTorch is a large download
(1 GB+), and processing is slower than the built-in engine — but the quality
difference on background voices is night and day.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `pip` is not recognized (Windows) | Reinstall Python with **"Add Python to PATH"** ticked, or use `py -m pip ...` |
| `clearscribe: command not found` | Your venv isn't activated — redo step 3's activate command |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or use CMD |
| Mic recording shows **0:00** and the page hangs | The browser's recorder produced no audio — this fails **in the browser**, before ClearScribe receives anything. In order: ① check the site's microphone permission (padlock icon in the address bar); ② check Windows Settings → Privacy → Microphone allows desktop apps/browsers; ③ try Chrome or Edge — Brave and strict-privacy browsers often block the recording APIs; ④ update Gradio: `pip install -U gradio`; ⑤ reliable fallback: record with your system voice recorder (Windows Sound Recorder saves .m4a) and upload the file — ClearScribe accepts it directly |
| First transcription hangs at the start | It's downloading the Whisper model (one-time, ~500 MB for `small`) — watch the terminal for progress |
| Transcription is slow | Use a smaller model: `--model base` or `--model tiny`; larger = more accurate but slower on CPU |
| Wrong language detected | Set it explicitly: `--language en` (or type it in the UI's language box) |
| Port 7860 already in use | Another Gradio app is running — close it, or `GRADIO_SERVER_PORT=7861 clearscribe-ui` |
| Enhanced audio sounds over-processed | Try the `gentle` preset, or lower the noise-reduction and compressor sliders |
| Background voices still audible | Spectral denoising can't separate voice from voice — install the Deep AI engine (section above) and select it |
| `Cargo, the Rust package manager, is not installed` when installing deepfilternet | You're on Python 3.12+ where no prebuilt wheels exist — use a Python 3.11 venv as shown in the Deep AI section above |
| `No module named 'torchaudio'` when using Deep AI | Install it in the same venv: `pip install torchaudio` (if versions clash: `pip install torch torchaudio --upgrade`) |
| `No module named 'torchaudio.backend'` when using Deep AI | Your ClearScribe predates the built-in compatibility shim — update ClearScribe (`git pull` + `pip install .` ), or pin the older pair: `pip install torch==2.5.1 torchaudio==2.5.1` |
| `'py' is not recognized` (Windows) | The `py` launcher is missing — common if your Python came from the Microsoft Store. Install Python from [python.org](https://www.python.org/downloads/) (the launcher is included), **open a new terminal**, and retry. Or call the interpreter directly: `"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m venv .venv311` |
| A constant whine/beep survives (often 1–8 kHz) | Make sure **Auto-remove tonal noises** is ticked (it's on by default); also try setting the Presence EQ slider to 0 so leftover noise there isn't boosted |

Still stuck? [Open an issue](https://github.com/Profxxxg/clearscribe/issues) —
include your OS, Python version, and the full error message.
