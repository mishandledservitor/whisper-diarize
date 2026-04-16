#!/usr/bin/env python3
"""
Whisper Diarize Local — Speech-to-Text with Speaker Diarization
WhisperX (Whisper + wav2vec2 alignment) + pyannote 3.1 diarization.
Built for long-form audio quality on Intel macOS (CPU only).

Usage:
    python whisper_diarize_local.py                                # Interactive mode
    python whisper_diarize_local.py interview.mp3                  # Quick diarized transcribe
    python whisper_diarize_local.py -m small interview.wav         # Specify model
    python whisper_diarize_local.py -o transcript.srt podcast.mp3  # Save as SRT
    python whisper_diarize_local.py --min-speakers 2 --max-speakers 2 call.mp3
    python whisper_diarize_local.py --inbox                        # Batch process inbox/
    python whisper_diarize_local.py --list-models                  # List models
"""

import argparse
import json
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Model catalog ───────────────────────────────────────────────────────────

MODELS = {
    "tiny":     {"params": "39M",   "size": "~75 MB",   "note": "fastest, low quality"},
    "base":     {"params": "74M",   "size": "~140 MB",  "note": "fast, basic quality"},
    "small":    {"params": "244M",  "size": "~460 MB",  "note": "decent quality, faster than medium"},
    "medium":   {"params": "769M",  "size": "~1.5 GB",  "note": "high quality (recommended)"},
    "large-v2": {"params": "1.55B", "size": "~2.9 GB",  "note": "best quality, very slow on CPU"},
    "large-v3": {"params": "1.55B", "size": "~2.9 GB",  "note": "best quality, very slow on CPU"},
}

ALL_MODELS = list(MODELS.keys())
DEFAULT_MODEL = "medium"

SUPPORTED_FORMATS = [
    "mp3", "mp4", "wav", "m4a", "ogg", "flac",
    "aac", "webm", "mkv", "mov", "avi", "opus",
]

DIARIZE_DIR = os.path.dirname(os.path.abspath(__file__))
INBOX_DIR = os.path.join(DIARIZE_DIR, "inbox")
OUTPUT_DIR = os.path.join(DIARIZE_DIR, "output")
PROCESSED_DIR = os.path.join(DIARIZE_DIR, "processed")
TOKEN_FILE = os.path.join(DIARIZE_DIR, ".hf_token")

DIARIZATION_PIPELINE = "pyannote/speaker-diarization-3.1"

# ── Utilities ───────────────────────────────────────────────────────────────

def resolve_path(path):
    return os.path.abspath(os.path.expanduser(path))


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def format_timestamp_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def load_hf_token():
    """Read HF token from env, then from .hf_token file."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok.strip()
    if os.path.isfile(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    return None


def save_hf_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token.strip() + "\n")
    os.chmod(TOKEN_FILE, 0o600)


# ── Output formatters ───────────────────────────────────────────────────────

def _segment_speaker(seg):
    return seg.get("speaker", "SPEAKER_??")


def format_text(segments):
    """Plain text grouped by speaker turn."""
    lines = []
    last_speaker = None
    buffer = []
    for seg in segments:
        spk = _segment_speaker(seg)
        text = seg["text"].strip()
        if not text:
            continue
        if spk != last_speaker:
            if buffer:
                lines.append(f"{last_speaker}: " + " ".join(buffer))
            buffer = [text]
            last_speaker = spk
        else:
            buffer.append(text)
    if buffer:
        lines.append(f"{last_speaker}: " + " ".join(buffer))
    return "\n\n".join(lines)


def format_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if not text:
            continue
        spk = _segment_speaker(seg)
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"[{spk}] {text}")
        lines.append("")
    return "\n".join(lines)


def format_vtt(segments):
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        spk = _segment_speaker(seg)
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(f"<v {spk}>{text}")
        lines.append("")
    return "\n".join(lines)


def format_json(segments, info):
    speakers = sorted({_segment_speaker(s) for s in segments if s.get("text", "").strip()})
    data = {
        "language": info.get("language"),
        "duration": round(info.get("duration", 0), 2),
        "model": info.get("model"),
        "diarization_pipeline": DIARIZATION_PIPELINE,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "segments": [],
    }
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        data["segments"].append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "speaker": _segment_speaker(seg),
            "text": text,
        })
    return json.dumps(data, indent=2, ensure_ascii=False)


def detect_format_from_path(path):
    _, ext = os.path.splitext(path)
    ext = ext.lower().lstrip(".")
    return {"srt": "srt", "vtt": "vtt", "json": "json", "txt": "text"}.get(ext, "text")

# ── Core ────────────────────────────────────────────────────────────────────

def load_models(model_name, hf_token, language=None):
    """Load WhisperX ASR model, alignment model, and diarization pipeline."""
    print(f"⏳ Loading WhisperX model ({model_name})...")
    start = time.time()
    import whisperx

    asr = whisperx.load_model(
        model_name,
        device="cpu",
        compute_type="int8",
        language=language,
    )
    print(f"   ✅ ASR loaded in {time.time() - start:.1f}s")

    print(f"⏳ Loading diarization pipeline ({DIARIZATION_PIPELINE})...")
    start = time.time()
    diarize_model = whisperx.DiarizationPipeline(
        model_name=DIARIZATION_PIPELINE,
        use_auth_token=hf_token,
        device="cpu",
    )
    print(f"   ✅ Diarization loaded in {time.time() - start:.1f}s")

    return asr, diarize_model


def load_align_model(language_code):
    import whisperx
    print(f"⏳ Loading alignment model for language: {language_code}...")
    start = time.time()
    align_model, align_meta = whisperx.load_align_model(
        language_code=language_code,
        device="cpu",
    )
    print(f"   ✅ Alignment loaded in {time.time() - start:.1f}s")
    return align_model, align_meta


def transcribe_and_diarize(asr, diarize_model, audio_path,
                           language=None, batch_size=4,
                           min_speakers=None, max_speakers=None):
    audio_path = resolve_path(audio_path)
    if not os.path.isfile(audio_path):
        print(f"⚠  File not found: {audio_path}")
        print(f"   (working dir: {os.getcwd()})")
        return None, None

    _, ext = os.path.splitext(audio_path)
    ext = ext.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        print(f"⚠  Unsupported format: .{ext}")
        print(f"   Supported: {', '.join(SUPPORTED_FORMATS)}")
        return None, None

    import whisperx

    file_size = os.path.getsize(audio_path)
    print(f"\n🎤 File: {os.path.basename(audio_path)} ({file_size / (1024*1024):.1f} MB)")
    if language:
        print(f"🌐 Language: {language}")
    else:
        print("🌐 Language: auto-detect")
    if min_speakers or max_speakers:
        print(f"👥 Speakers: min={min_speakers or 'auto'}, max={max_speakers or 'auto'}")

    overall_start = time.time()

    # ── 1. ASR (transcribe with WhisperX) ────────────────────────────────────
    print("\n  [1/3] 📝 Transcribing...")
    t0 = time.time()
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000.0
    print(f"        Audio duration: {format_time(duration)}")

    asr_result = asr.transcribe(audio, batch_size=batch_size, language=language)
    detected_language = asr_result.get("language", language)
    asr_segments = asr_result.get("segments", [])
    print(f"        ✅ {len(asr_segments)} segments in {format_time(time.time() - t0)}"
          f" (detected: {detected_language})")

    # ── 2. Forced alignment ──────────────────────────────────────────────────
    print(f"\n  [2/3] 🎯 Aligning word-level timestamps...")
    t0 = time.time()
    try:
        align_model, align_meta = load_align_model(detected_language)
        aligned = whisperx.align(
            asr_segments, align_model, align_meta, audio,
            device="cpu", return_char_alignments=False,
        )
        aligned_segments = aligned.get("segments", asr_segments)
        # Free alignment model (it's per-language and large)
        del align_model
        print(f"        ✅ Aligned in {format_time(time.time() - t0)}")
    except Exception as e:
        print(f"        ⚠  Alignment failed ({e}); falling back to ASR timestamps")
        aligned_segments = asr_segments

    # ── 3. Diarization ───────────────────────────────────────────────────────
    print(f"\n  [3/3] 👥 Diarizing speakers...")
    t0 = time.time()
    diarize_kwargs = {}
    if min_speakers:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_model(audio, **diarize_kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, {"segments": aligned_segments})
    final_segments = result.get("segments", aligned_segments)
    speakers = sorted({s.get("speaker", "SPEAKER_??") for s in final_segments
                       if s.get("text", "").strip()})
    print(f"        ✅ {len(speakers)} speaker(s) in {format_time(time.time() - t0)}")

    elapsed = time.time() - overall_start
    print(f"\n✅ Done in {format_time(elapsed)} "
          f"(real-time factor: {elapsed / duration:.2f}x)" if duration > 0 else "")
    print(f"   Speakers: {', '.join(speakers) if speakers else '(none detected)'}")

    info = {
        "language": detected_language,
        "duration": duration,
        "model": getattr(asr, "model_name", None) or "whisperx",
    }
    return final_segments, info


def print_models():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       🤖  WHISPER + DIARIZATION MODEL CATALOG  🤖        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print(f"  ┌─ {'Model':<10} {'Params':<8} {'Size':<11} {'Notes'} ─┐")
    print(f"  │{'─' * 65}│")
    for name, mi in MODELS.items():
        marker = " ★" if name == DEFAULT_MODEL else ""
        print(f"  │  {name:<10} {mi['params']:<8} {mi['size']:<11} {mi['note']}{marker}")
    print(f"  └{'─' * 65}┘\n")
    print(f"  ★ = default model ({DEFAULT_MODEL})")
    print(f"  Diarization pipeline: {DIARIZATION_PIPELINE}\n")


# ── Inbox / batch processing ───────────────────────────────────────────────

def scan_inbox():
    os.makedirs(INBOX_DIR, exist_ok=True)
    files = []
    for name in sorted(os.listdir(INBOX_DIR)):
        if name.startswith("."):
            continue
        ext = os.path.splitext(name)[1].lower().lstrip(".")
        if ext in SUPPORTED_FORMATS:
            files.append(os.path.join(INBOX_DIR, name))
    return files


def process_inbox(asr, diarize_model, language=None, out_format="text",
                  batch_size=4, min_speakers=None, max_speakers=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = scan_inbox()
    if not files:
        print("\n  📭 Inbox is empty — drop audio files into:")
        print(f"     {INBOX_DIR}/\n")
        return

    ext_map = {"text": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json"}
    out_ext = ext_map.get(out_format, ".txt")

    print(f"\n  📬 {len(files)} file(s) in inbox:\n")
    for f in files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"     • {os.path.basename(f)}  ({size:.1f} MB)")
    print()

    succeeded = 0
    failed = 0

    for i, audio_path in enumerate(files, 1):
        basename = os.path.basename(audio_path)
        name_no_ext = os.path.splitext(basename)[0]

        print(f"\n{'═' * 60}")
        print(f"  [{i}/{len(files)}]  {basename}")
        print(f"{'═' * 60}")

        segments, info = transcribe_and_diarize(
            asr, diarize_model, audio_path,
            language=language, batch_size=batch_size,
            min_speakers=min_speakers, max_speakers=max_speakers,
        )
        if segments is None:
            print(f"  ⚠  Skipping {basename}")
            failed += 1
            continue

        if out_format == "json":
            result = format_json(segments, info)
        else:
            formatter = {"text": format_text, "srt": format_srt,
                         "vtt": format_vtt}.get(out_format, format_text)
            result = formatter(segments)

        out_path = os.path.join(OUTPUT_DIR, name_no_ext + out_ext)
        with open(out_path, "w") as f:
            f.write(result)
        print(f"  💾 Saved: output/{name_no_ext}{out_ext}")

        dest = os.path.join(PROCESSED_DIR, basename)
        if os.path.exists(dest):
            base, ext = os.path.splitext(basename)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(PROCESSED_DIR, f"{base}_{counter}{ext}")
                counter += 1
        shutil.move(audio_path, dest)
        print(f"  📦 Moved:  processed/{os.path.basename(dest)}")
        succeeded += 1

    print(f"\n{'═' * 60}")
    print(f"  ✅ Done!  {succeeded} processed", end="")
    if failed:
        print(f", {failed} failed", end="")
    print(f"\n  📂 Transcripts in: {OUTPUT_DIR}/")
    print(f"{'═' * 60}\n")


# ── Interactive mode ────────────────────────────────────────────────────────

def interactive_mode(asr, diarize_model, model_name, hf_token):
    language = None
    output_format = "text"
    min_speakers = None
    max_speakers = None
    batch_size = 4

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   🎧  WHISPER DIARIZE — INTERACTIVE MODE  🎧             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Model: {model_name}  |  Language: auto  |  Format: {output_format}")
    inbox_files = scan_inbox()
    if inbox_files:
        print(f"\n  📬 {len(inbox_files)} file(s) waiting in inbox — type /inbox to process")

    print("\n  Drop a file path or use a command:")
    print("    /inbox                   — process all files in inbox/")
    print("    /lang <code>             — set language (e.g. /lang en), 'auto' to detect")
    print("    /format <fmt>            — set output: text, srt, vtt, json")
    print("    /speakers <min> <max>    — fix speaker count (e.g. /speakers 2 2)")
    print("    /speakers auto           — let pyannote decide")
    print("    /models                  — list all models")
    print("    /quit                    — exit\n")

    while True:
        try:
            text = input("  ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!\n")
            break

        if not text:
            continue

        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("/quit", "/exit", "/q"):
                print("\n  👋 Goodbye!\n")
                break
            elif cmd == "/lang":
                if not args:
                    print(f"  ℹ  Language: {'auto' if not language else language}")
                elif args[0].lower() == "auto":
                    language = None
                    print("  ✅ Language: auto-detect")
                else:
                    language = args[0].lower()
                    print(f"  ✅ Language: {language}")
            elif cmd == "/format":
                if not args:
                    print(f"  ℹ  Format: {output_format}")
                elif args[0] in ("text", "srt", "vtt", "json"):
                    output_format = args[0]
                    print(f"  ✅ Format: {output_format}")
                else:
                    print(f"  ⚠  Unknown format. Use: text, srt, vtt, json")
            elif cmd == "/speakers":
                if not args:
                    print(f"  ℹ  Speakers: min={min_speakers or 'auto'}, max={max_speakers or 'auto'}")
                elif args[0].lower() == "auto":
                    min_speakers = max_speakers = None
                    print("  ✅ Speakers: auto-detect")
                elif len(args) == 2 and args[0].isdigit() and args[1].isdigit():
                    min_speakers = int(args[0])
                    max_speakers = int(args[1])
                    print(f"  ✅ Speakers: min={min_speakers}, max={max_speakers}")
                else:
                    print("  ⚠  Use: /speakers <min> <max>  or  /speakers auto")
            elif cmd == "/inbox":
                process_inbox(asr, diarize_model, language, output_format,
                              batch_size, min_speakers, max_speakers)
            elif cmd == "/models":
                print_models()
            else:
                print(f"  ⚠  Unknown command: {cmd}")
            continue

        audio_path = resolve_path(text)
        if not os.path.isfile(audio_path):
            print(f"  ⚠  File not found: {audio_path}")
            continue

        segments, info = transcribe_and_diarize(
            asr, diarize_model, audio_path,
            language=language, batch_size=batch_size,
            min_speakers=min_speakers, max_speakers=max_speakers,
        )
        if segments is None:
            continue

        if output_format == "json":
            result = format_json(segments, info)
        else:
            formatter = {"text": format_text, "srt": format_srt,
                         "vtt": format_vtt}.get(output_format, format_text)
            result = formatter(segments)

        print(f"\n{'─' * 60}")
        print(result)
        print(f"{'─' * 60}")

        try:
            save = input("\n  💾 Save to file? (path or Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            continue
        if save:
            save_path = resolve_path(save)
            out_fmt = detect_format_from_path(save_path)
            if out_fmt == "json":
                out_text = format_json(segments, info)
            else:
                out_formatter = {"text": format_text, "srt": format_srt,
                                 "vtt": format_vtt}.get(out_fmt, format_text)
                out_text = out_formatter(segments)
            with open(save_path, "w") as f:
                f.write(out_text)
            print(f"  ✅ Saved to: {save_path}")


# ── Token bootstrap ─────────────────────────────────────────────────────────

def ensure_hf_token():
    """Get a HF token, prompting if necessary."""
    token = load_hf_token()
    if token:
        return token

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  🔑  Hugging Face token required for diarization         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n  pyannote/speaker-diarization-3.1 is gated. To use it:")
    print("    1. Create a free account at https://huggingface.co")
    print("    2. Accept terms at:")
    print("       https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("       https://huggingface.co/pyannote/segmentation-3.0")
    print("    3. Create a READ token at https://huggingface.co/settings/tokens")
    print()
    try:
        tok = input("  Paste your HF token (or Ctrl+C to abort): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  ⚠  No token provided. Cannot continue.")
        sys.exit(1)
    if not tok:
        print("  ⚠  Empty token. Cannot continue.")
        sys.exit(1)
    save_hf_token(tok)
    print(f"  ✅ Token saved to {TOKEN_FILE} (chmod 600)")
    return tok


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whisper Diarize Local — STT + speaker diarization (WhisperX + pyannote)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    Interactive mode
  %(prog)s interview.mp3                      Quick diarized transcribe
  %(prog)s -m small recording.wav             Use a smaller model (faster)
  %(prog)s -o transcript.srt podcast.mp3      Save as SRT subtitles
  %(prog)s --min-speakers 2 --max-speakers 2 call.mp3
  %(prog)s --inbox                            Batch-process all files in inbox/
  %(prog)s --list-models                      Show models
        """,
    )
    parser.add_argument("audio", nargs="?", help="Audio file to transcribe + diarize")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Whisper model size (default: {DEFAULT_MODEL})")
    parser.add_argument("-l", "--language", default=None,
                        help="Language code (auto-detect if omitted)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path — .txt, .srt, .vtt, or .json")
    parser.add_argument("-f", "--format", default=None,
                        choices=["text", "srt", "vtt", "json"],
                        help="Output format (default: text, or inferred from -o)")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimum number of speakers (improves diarization accuracy)")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximum number of speakers")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="ASR batch size (default 4 — lower for low-RAM machines)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models")
    parser.add_argument("--inbox", action="store_true",
                        help="Process all files in inbox/, save to output/")
    parser.add_argument("--no-print", action="store_true",
                        help="Don't print transcript to terminal")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token (otherwise read from .hf_token or HF_TOKEN env)")

    args = parser.parse_args()

    if args.list_models:
        print_models()
        sys.exit(0)

    if args.model not in ALL_MODELS:
        print(f"⚠  Unknown model: {args.model}")
        print_models()
        sys.exit(1)

    # Resolve HF token
    if args.hf_token:
        save_hf_token(args.hf_token)
        token = args.hf_token
    else:
        token = ensure_hf_token()

    asr, diarize_model = load_models(args.model, token, language=args.language)

    out_format = args.format
    if not out_format and args.output:
        out_format = detect_format_from_path(args.output)
    if not out_format:
        out_format = "text"

    if args.inbox:
        process_inbox(asr, diarize_model, args.language, out_format,
                      args.batch_size, args.min_speakers, args.max_speakers)
        return

    if args.audio:
        segments, info = transcribe_and_diarize(
            asr, diarize_model, args.audio,
            language=args.language, batch_size=args.batch_size,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        )
        if segments is None:
            sys.exit(1)
    else:
        interactive_mode(asr, diarize_model, args.model, token)
        return

    if out_format == "json":
        result = format_json(segments, info)
    else:
        formatter = {"text": format_text, "srt": format_srt,
                     "vtt": format_vtt}.get(out_format, format_text)
        result = formatter(segments)

    if not args.no_print:
        print(f"\n{'─' * 60}")
        print(result)
        print(f"{'─' * 60}")

    if args.output:
        out_path = resolve_path(args.output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(result)
        print(f"\n💾 Saved to: {out_path}")


if __name__ == "__main__":
    main()
