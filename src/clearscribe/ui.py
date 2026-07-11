"""ClearScribe web UI — drag-and-drop enhancement + transcription.

Launch with:  clearscribe-ui   (or: python -m clearscribe.ui)
Requires the [ui] extra:  pip install clearscribe[ui]
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from clearscribe import __version__
from clearscribe.enhance import PRESETS, EnhanceSettings, enhance_array, load_mono
from clearscribe.formats import WRITERS
from clearscribe.media import expose_ffmpeg_on_path, to_wav
from clearscribe.presets import load_user_presets, save_user_preset
from clearscribe.recorder import SystemRecorder

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The web UI needs gradio. Install it with:  pip install clearscribe[ui]"
    ) from e


def _settings_from_ui(
    preset, backend_choice, auto_notch, strength, stationary, dehum_choice,
    gate_thresh, comp_ratio, comp_thresh, deess_ratio, warmth, presence, air,
    target_lufs,
) -> EnhanceSettings:
    base = PRESETS.get(preset, PRESETS["podcast"])
    return EnhanceSettings(
        highpass_hz=base.highpass_hz,
        dehum_hz={"Off": 0.0, "50 Hz (EU/most of world)": 50.0,
                  "60 Hz (Americas)": 60.0}[dehum_choice],
        backend="deep" if backend_choice.startswith("Deep") else "spectral",
        auto_notch=auto_notch,
        noise_reduction_strength=strength,
        stationary_noise=stationary,
        two_pass_noise_reduction=base.two_pass_noise_reduction,
        gate_threshold_db=gate_thresh,
        gate_ratio=base.gate_ratio,
        comp_threshold_db=comp_thresh,
        comp_ratio=comp_ratio,
        deess_threshold_db=base.deess_threshold_db,
        deess_ratio=deess_ratio,
        warmth_db=warmth,
        presence_db=presence,
        air_db=air,
        target_lufs=target_lufs,
    )


KNOB_ORDER = "backend auto_notch strength stationary dehum gate comp_ratio " \
             "comp_thresh deess warmth presence air lufs".split()


def _settings_to_knobs(s: EnhanceSettings):
    """Map an EnhanceSettings onto the 13 UI controls (KNOB_ORDER)."""
    backend = "Deep AI — DeepFilterNet" if s.backend == "deep" else "Spectral (built-in)"
    dehum = {50.0: "50 Hz (EU/most of world)", 60.0: "60 Hz (Americas)"}.get(
        s.dehum_hz, "Off")
    return (backend, s.auto_notch, s.noise_reduction_strength,
            s.stationary_noise, dehum, s.gate_threshold_db, s.comp_ratio,
            s.comp_threshold_db, s.deess_ratio, s.warmth_db, s.presence_db,
            s.air_db, s.target_lufs)


def _apply_preset(preset: str):
    s = PRESETS[preset]
    return (s.noise_reduction_strength, s.gate_threshold_db, s.comp_ratio,
            s.comp_threshold_db, s.deess_ratio, s.warmth_db, s.presence_db,
            s.air_db, s.target_lufs)


def enhance_for_ui(file_path, mic_path, *ui_args, progress=gr.Progress()):
    audio_path = file_path or mic_path
    if not audio_path:
        raise gr.Error("Please upload a file or record something first.")
    progress(0.1, desc="Loading / decoding audio")
    try:
        audio, sr = load_mono(audio_path)
    except Exception as e:
        raise gr.Error(
            f"Couldn't read this file ({e}). If this was a microphone "
            "recording showing 0:00, the browser produced an empty file — "
            "see the recording tips in INSTALL.md, or record with your "
            "system's voice recorder and upload the file instead.")
    if len(audio) < sr * 0.25:
        raise gr.Error(
            "Audio is too short or empty (under 0.25 s). If this was a mic "
            "recording at 0:00, your browser's recorder failed — check the "
            "mic permission, try Chrome/Edge, or upload a file instead.")
    minutes = len(audio) / sr / 60.0
    if minutes > 40:
        gr.Warning(f"This is a long recording ({minutes:.0f} min) — processing "
                   "may take a while and use several GB of RAM. Consider "
                   "splitting files over ~1 hour.")

    progress(0.3, desc="Enhancing (denoise → gate → compress → de-ess → EQ)")
    settings = _settings_from_ui(*ui_args)
    enhanced = enhance_array(audio, sr, settings)

    progress(0.9, desc="Writing output")
    outdir = Path(tempfile.mkdtemp())
    out = outdir / (Path(audio_path).stem + ".enhanced.wav")
    sf.write(str(out), enhanced, sr, subtype="PCM_16")
    # Always-playable copy of exactly what was enhanced (video/M4A sources
    # may not play in the browser directly)
    orig = outdir / (Path(audio_path).stem + ".original.wav")
    sf.write(str(orig), audio, sr, subtype="PCM_16")

    stats = _stats_markdown(audio, enhanced, sr)
    return str(orig), str(out), str(out), stats


def _stats_markdown(before: np.ndarray, after: np.ndarray, sr: int) -> str:
    def rms_db(x):
        return 20 * np.log10(max(np.sqrt(np.mean(x**2)), 1e-10))

    # Estimate noise floor from the quietest 10% of 50 ms frames
    def floor_db(x):
        f = sr // 20
        frames = x[: len(x) // f * f].reshape(-1, f)
        r = np.sqrt(np.mean(frames**2, axis=1))
        quiet = np.sort(r)[: max(1, len(r) // 10)]
        return 20 * np.log10(max(quiet.mean(), 1e-10))

    return (
        f"| | Before | After |\n|---|---|---|\n"
        f"| Overall level (RMS) | {rms_db(before):.1f} dBFS | {rms_db(after):.1f} dBFS |\n"
        f"| Noise floor (quietest 10%) | {floor_db(before):.1f} dBFS | {floor_db(after):.1f} dBFS |\n"
        f"| Peak | {20*np.log10(max(np.abs(before).max(),1e-10)):.1f} dBFS "
        f"| {20*np.log10(max(np.abs(after).max(),1e-10)):.1f} dBFS |"
    )


def transcribe_for_ui(enhanced_path, model_size, language, formats,
                      progress=gr.Progress()):
    if not enhanced_path:
        raise gr.Error("Enhance an audio file first (or upload one and click Enhance).")
    if not formats:
        raise gr.Error("Pick at least one transcript format.")
    progress(0.05, desc=f"Loading Whisper '{model_size}' (first run downloads it)")
    from clearscribe.transcribe import transcribe_audio

    transcript = transcribe_audio(
        enhanced_path, model_size=model_size,
        language=language.strip() or None if language else None,
    )
    progress(0.9, desc="Writing transcript files")
    outdir = Path(tempfile.mkdtemp())
    stem = Path(enhanced_path).stem.replace(".enhanced", "")
    files = []
    for fmt in formats:
        p = outdir / f"{stem}.{fmt}"
        p.write_text(WRITERS[fmt](transcript), encoding="utf-8")
        files.append(str(p))
    return transcript.text or "(no speech detected)", files


def build_app() -> "gr.Blocks":
    with gr.Blocks(title="ClearScribe") as app:
        gr.Markdown(
            f"# 🎙️ ClearScribe v{__version__}\n"
            "**Open-source voice enhancer + transcript writer.** Upload a recording, "
            "get podcast-quality audio and transcripts — everything runs locally."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1 · Original")
                input_file = gr.File(
                    label="Upload audio or video (WAV, MP3, M4A, MP4, MKV, ...)",
                    type="filepath")
                original_audio = gr.Audio(label="Original — listen",
                                          type="filepath", interactive=False)
                mic_audio = gr.Audio(label="... or record", type="filepath",
                                     sources=["microphone"], format="wav")
                gr.Markdown("Browser recorder not working (0:00 / freeze)? "
                            "Use the **system mic** instead:")
                with gr.Row():
                    rec_start_btn = gr.Button("🎙️ Record (system mic)")
                    rec_stop_btn = gr.Button("⏹ Stop & use recording")
                rec_status = gr.Markdown("")
            with gr.Column():
                gr.Markdown("### 2 · Enhanced — compare side by side")
                output_audio = gr.Audio(label="Enhanced result", type="filepath",
                                        interactive=False)

        preset = gr.Radio(list(PRESETS), value="podcast", label="Preset",
                          info="'podcast' is the recommended full chain")

        with gr.Row():
            user_preset_dd = gr.Dropdown(
                choices=sorted(load_user_presets()), value=None,
                label="My saved presets",
                info="Pick one to load all its settings")
            preset_name_box = gr.Textbox(
                label="Save current settings as",
                placeholder="e.g. my-usb-mic")
            save_preset_btn = gr.Button("💾 Save preset")

        with gr.Row():
            backend_choice = gr.Radio(
                ["Spectral (built-in)", "Deep AI — DeepFilterNet"],
                value="Spectral (built-in)", label="Denoise engine",
                info="Deep AI is far better at background voices/chatter; "
                     "needs the optional [deep] install (see INSTALL.md)")
            auto_notch = gr.Checkbox(
                True, label="Auto-remove tonal noises",
                info="Finds constant whines/beeps (e.g. electronics around "
                     "3–4 kHz) in the noise floor and notches them out")

        with gr.Accordion("Advanced settings — what each control does", open=False):
            gr.Markdown(
                "The chain runs top to bottom: **clean → shape dynamics → shape tone → set loudness.** "
                "Hover any control for details; presets set sensible values for all of them."
            )
            with gr.Row():
                strength = gr.Slider(
                    0, 1, 0.85, step=0.05, label="Noise reduction",
                    info="How much background noise the spectral denoiser removes. "
                         "Too high can sound 'underwater' — back off if the voice gets hollow")
                stationary = gr.Checkbox(
                    False, label="Stationary noise (constant hiss/hum)",
                    info="Tick when the noise never changes: fan, hiss, air-con. "
                         "Leave off for variable noise like traffic or chatter")
                dehum_choice = gr.Dropdown(
                    ["Off", "50 Hz (EU/most of world)", "60 Hz (Americas)"],
                    value="Off", label="Mains hum removal",
                    info="Notches electrical hum and its harmonics. Pick your region's "
                         "mains frequency if you hear a low buzz")
            with gr.Row():
                gate_thresh = gr.Slider(
                    -70, -20, -45, step=1, label="Gate threshold (dB)",
                    info="Audio quieter than this is treated as a pause and pushed "
                         "toward silence. Raise it if background sounds leak between "
                         "words; lower it if word endings get clipped")
                comp_ratio = gr.Slider(
                    1, 8, 3, step=0.5, label="Compressor ratio",
                    info="Evens out loud vs quiet speech — the core 'podcast sound'. "
                         "2–3 = natural, 4+ = dense radio voice, 1 = off")
                comp_thresh = gr.Slider(
                    -40, -10, -22, step=1, label="Compressor threshold (dB)",
                    info="Level where compression starts. Lower = more of the "
                         "speech gets evened out")
                deess_ratio = gr.Slider(
                    1, 8, 4, step=0.5, label="De-esser ratio",
                    info="Tames harsh S/SH sounds (5–9 kHz sibilance) without "
                         "dulling the rest. 1 = off")
            with gr.Row():
                warmth = gr.Slider(
                    0, 6, 1.5, step=0.5, label="Warmth (dB @160 Hz)",
                    info="Low-shelf boost: adds body and fullness to the voice")
                presence = gr.Slider(
                    0, 6, 2.5, step=0.5, label="Presence (dB @3 kHz)",
                    info="Peak boost for clarity and intelligibility. NOTE: also "
                         "boosts any leftover noise near 3 kHz — set to 0 if noise "
                         "there survives denoising")
                air = gr.Slider(
                    0, 6, 1.5, step=0.5, label="Air (dB @10 kHz)",
                    info="High-shelf boost: adds sparkle / 'expensive mic' sheen")
                target_lufs = gr.Slider(
                    -24, -12, -16, step=1, label="Loudness target (LUFS)",
                    info="Final overall loudness. −16 is the podcast standard; "
                         "−14 for louder platforms like Spotify")

        enhance_btn = gr.Button("✨ Enhance", variant="primary")
        stats_md = gr.Markdown()
        enhanced_state = gr.State()

        gr.Markdown("### 3 · Transcribe (optional)")
        with gr.Row():
            model_size = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"],
                                     value="small", label="Whisper model")
            language = gr.Textbox(label="Language code (blank = auto)",
                                  placeholder="en, ar, fr ...")
            formats = gr.CheckboxGroup(list(WRITERS), value=["txt", "srt"],
                                       label="Transcript formats")
        transcribe_btn = gr.Button("📝 Transcribe enhanced audio")
        transcript_box = gr.Textbox(label="Transcript", lines=6)
        transcript_files = gr.Files(label="Download transcripts")

        _recorder = SystemRecorder()

        def _rec_start():
            try:
                _recorder.start()
            except RuntimeError as e:
                raise gr.Error(str(e))
            return ("🔴 **Recording from the system microphone…** "
                    "speak, then press ⏹ Stop.")

        def _rec_stop():
            try:
                path = _recorder.stop()
            except RuntimeError as e:
                raise gr.Error(str(e))
            return str(path), "✅ Recording captured — press ✨ Enhance."

        rec_start_btn.click(_rec_start, outputs=rec_status)
        rec_stop_btn.click(_rec_stop, outputs=[mic_audio, rec_status])

        _PLAYABLE = {".wav", ".mp3", ".ogg", ".oga", ".flac", ".m4a", ".opus"}

        def _preview_original(path):
            if not path:
                return None
            p = Path(path)
            if p.suffix.lower() in _PLAYABLE:
                return str(p)
            try:  # video or exotic audio: extract a playable wav
                return str(to_wav(p))
            except RuntimeError as e:
                raise gr.Error(str(e))

        input_file.change(_preview_original, inputs=input_file,
                          outputs=original_audio)

        def _check_recording(path):
            if not path:
                return
            try:
                a, rec_sr = load_mono(path)
                too_short = len(a) < rec_sr * 0.25
            except Exception:
                too_short = True
            if too_short:
                gr.Warning(
                    "That recording came back empty (0:00) — the browser's "
                    "recorder failed before ClearScribe received any audio. "
                    "Check the microphone permission for this site, try "
                    "Chrome or Edge (Brave/strict privacy modes often block "
                    "recording), or record with your system voice recorder "
                    "and upload the file.")

        mic_audio.change(_check_recording, inputs=mic_audio, outputs=[])

        knobs = [strength, gate_thresh, comp_ratio, comp_thresh, deess_ratio,
                 warmth, presence, air, target_lufs]
        preset.change(_apply_preset, inputs=preset, outputs=knobs)

        all_knobs = [backend_choice, auto_notch, strength, stationary,
                     dehum_choice, gate_thresh, comp_ratio, comp_thresh,
                     deess_ratio, warmth, presence, air, target_lufs]

        def _save_preset(name, preset_val, *knob_vals):
            settings = _settings_from_ui(preset_val, *knob_vals)
            try:
                save_user_preset(name, settings)
            except ValueError as e:
                raise gr.Error(str(e))
            gr.Info(f"Saved preset '{name.strip()}'")
            return gr.update(choices=sorted(load_user_presets()),
                             value=name.strip())

        def _load_preset(name):
            presets = load_user_presets()
            if not name or name not in presets:
                return [gr.update()] * len(all_knobs)
            return list(_settings_to_knobs(presets[name]))

        save_preset_btn.click(_save_preset,
                              inputs=[preset_name_box, preset] + all_knobs,
                              outputs=user_preset_dd)
        user_preset_dd.change(_load_preset, inputs=user_preset_dd,
                              outputs=all_knobs)

        enhance_btn.click(
            enhance_for_ui,
            inputs=[input_file, mic_audio, preset] + all_knobs,
            outputs=[original_audio, output_audio, enhanced_state, stats_md],
        )
        transcribe_btn.click(
            transcribe_for_ui,
            inputs=[enhanced_state, model_size, language, formats],
            outputs=[transcript_box, transcript_files],
        )

        gr.Markdown("---\n[MIT licensed · source on GitHub]"
                    "(https://github.com/Profxxxg/clearscribe)")
    return app


def main() -> None:  # entry point: clearscribe-ui
    expose_ffmpeg_on_path()  # Gradio needs `ffmpeg` to convert mic recordings
    build_app().launch()


if __name__ == "__main__":
    main()
