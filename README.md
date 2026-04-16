# Whisper Diarize Local

> Speech-to-text with speaker diarization, fully offline. WhisperX + pyannote 3.1.

**Version 1.0.0** | [Changelog](CHANGELOG.md)

```
whisper-diarize/
├── whisper_diarize_local.py    WhisperX + pyannote pipeline
├── setup_whisper_diarize.sh    Installer
├── uninstall_whisper_diarize.sh
├── inbox/                      Drop audio files here
├── output/                     Diarized transcripts saved here
└── processed/                  Originals moved here after processing
```

This is the **third voxbox tool**, built specifically for **long-form audio + speaker identification**.

Unlike the other voxbox submodules, this one **does require PyTorch** (pinned to 2.2.2 for Intel Mac compatibility). pyannote.audio has no PyTorch-free alternative at comparable quality. The other voxbox tools (`kokoro-tts`, `whisper-stt`) remain PyTorch-free.

---

## Quick Start

```bash
# from the voxbox root
chmod +x whisper-diarize/setup_whisper_diarize.sh
./whisper-diarize/setup_whisper_diarize.sh

# transcribe + diarize a file
./whisper-diarize/whisper-diarize interview.mp3

# batch-process inbox
./whisper-diarize/whisper-diarize --inbox
```

---

## Setup

### Prerequisites

- **macOS** (Intel or Apple Silicon)
- **Python 3.10 or 3.11** (installer will install python@3.10 via Homebrew if needed)
- **ffmpeg** (installed automatically)
- **~4 GB disk space** (PyTorch + Whisper medium + pyannote models)
- **Hugging Face account + token** (free) — pyannote 3.1 is a gated model

### Hugging Face token — required

Diarization uses [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), which requires accepting terms of use:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Generate a **read** token at https://huggingface.co/settings/tokens
4. Provide the token in any of these ways:
   - Paste it during setup (saved to `.hf_token`, chmod 600)
   - Export `HF_TOKEN` in your shell
   - Pass `--hf-token` on the CLI

### Installation

```bash
chmod +x setup_whisper_diarize.sh
./setup_whisper_diarize.sh
```

Steps:
1. Verifies Homebrew, Python 3.10, ffmpeg
2. Creates a venv
3. Installs PyTorch 2.2.2 (CPU), pyannote.audio 3.1.1, whisperx, soundfile
4. Prompts for HF token (or skip and enter on first run)
5. Pre-downloads the Whisper `medium` model (~1.5 GB)

---

## Usage

```bash
# Interactive mode
./whisper-diarize

# Quick diarize
./whisper-diarize interview.mp3

# Specify model
./whisper-diarize -m small recording.wav
./whisper-diarize -m large-v3 podcast.mp3

# Save as SRT (subtitles with [SPEAKER_00] tags)
./whisper-diarize -o transcript.srt podcast.mp3

# Force speaker count (huge accuracy + speed win when known)
./whisper-diarize --min-speakers 2 --max-speakers 2 interview.mp3

# Force language (skips auto-detect)
./whisper-diarize -l en talk.m4a

# Batch process all files in inbox/
./whisper-diarize --inbox

# List available models
./whisper-diarize --list-models
```

### Interactive commands

```
/inbox                  — process all files in inbox/
/lang <code>            — set language (e.g. /lang en), 'auto' to detect
/format <fmt>           — output format: text, srt, vtt, json
/speakers <min> <max>   — fix speaker count (e.g. /speakers 2 2)
/speakers auto          — let pyannote decide
/models                 — list models
/quit                   — exit
```

---

## Output formats

### `text` (default)

Speaker-grouped paragraphs:

```
SPEAKER_00: Hi, thanks for having me on the show today.

SPEAKER_01: Of course, glad to have you. Let's start with your background.

SPEAKER_00: Sure, I grew up in Brooklyn and...
```

### `srt` / `vtt`

Subtitle blocks with `[SPEAKER_XX]` prefix on each cue. Drop into video players or editors.

### `json`

Machine-readable, includes per-segment timestamps + speaker IDs + a top-level speaker list.

---

## Models

| Model    | Params | Size    | Notes                              |
|----------|--------|---------|------------------------------------|
| tiny     | 39M    | ~75 MB  | fastest, low quality               |
| base     | 74M    | ~140 MB | fast, basic quality                |
| small    | 244M   | ~460 MB | decent quality, faster than medium |
| medium ★ | 769M   | ~1.5 GB | high quality (default)             |
| large-v2 | 1.55B  | ~2.9 GB | best quality, very slow on CPU     |
| large-v3 | 1.55B  | ~2.9 GB | best quality, very slow on CPU     |

**For Intel CPU, `medium` is the practical sweet spot** for long-form audio. `large-v3` is noticeably better but ~2× slower again.

---

## Performance on Intel Mac (CPU only)

WhisperX runs three stages: ASR → wav2vec2 alignment → pyannote diarization. Realistic budget on a 2018 Intel MBP:

| Audio length | medium model |
|--------------|--------------|
| 15 min       | ~30–60 min   |
| 30 min       | ~1–2 hr      |
| 1 hour       | ~3–5 hr      |
| 3 hours      | overnight    |

Diarization itself adds ~0.3–0.5× audio duration on top of ASR. **Pinning the speaker count via `--min-speakers / --max-speakers` is the single biggest performance + accuracy win** when you know how many people are talking.

---

## How it works

1. **WhisperX ASR** — runs Whisper on VAD-filtered chunks (no hallucination drift on long audio)
2. **wav2vec2 alignment** — re-anchors each word to its actual audio position so speaker boundaries are crisp
3. **pyannote 3.1 diarization** — clusters voice activity into speaker turns
4. **Word-speaker assignment** — overlays speaker labels onto the aligned transcript

---

## Troubleshooting

### `Could not download 'pyannote/speaker-diarization-3.1'`
You haven't accepted the model terms or your token lacks read permission. Visit the model pages, accept the terms, and regenerate a token.

### `ImportError: No module named 'whisperx'`
Run `./setup_whisper_diarize.sh` again, or activate the venv manually:
```bash
source whisper-diarize/venv/bin/activate
pip install whisperx
```

### Setup fails on Python 3.12+
WhisperX/pyannote work most reliably on Python 3.10/3.11. The setup script auto-installs `python@3.10` via Homebrew when needed.

### `torch` install fails on Intel Mac
PyTorch 2.4+ dropped Intel Mac support. The setup pins `torch==2.2.2` deliberately. Don't upgrade.

### Diarization is slow / inaccurate
- Provide `--min-speakers N --max-speakers N` if you know the count
- Use a smaller Whisper model (`small` or `base`) — diarization quality is unaffected
- Audio quality matters more than model size for diarization — clean audio with separated mics gives the best results

### "OpenMP runtime is already initialized" warning
Harmless. Set `KMP_DUPLICATE_LIB_OK=TRUE` in your shell to silence it.

---

## License

MIT — same as voxbox.
Whisper, WhisperX, and pyannote each carry their own licenses; check their repos for details.
