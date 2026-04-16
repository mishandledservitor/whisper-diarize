# Changelog

All notable changes to Whisper Diarize Local are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-04-16

### Added
- `whisper_diarize_local.py` — WhisperX + pyannote 3.1 pipeline
  - Whisper ASR via WhisperX (medium model by default)
  - wav2vec2 word-level alignment for accurate timestamps
  - pyannote 3.1 speaker diarization
  - Output formats: speaker-grouped text, SRT, VTT, JSON (with speaker labels)
- 6 model sizes (tiny → large-v3); `medium` is the default for the long-form quality target
- `--min-speakers` / `--max-speakers` flags for known speaker counts (improves accuracy + speed)
- Inbox workflow (`inbox/`, `output/`, `processed/`) matching the `whisper-stt` pattern
- Interactive mode with `/inbox`, `/lang`, `/format`, `/speakers`, `/models`
- HF token bootstrap — read from env, `.hf_token` file (chmod 600), or interactive prompt
- `setup_whisper_diarize.sh` — installs Python 3.10, ffmpeg, PyTorch 2.2.2 (Intel-Mac compatible), pyannote.audio 3.1.1, whisperx
- `uninstall_whisper_diarize.sh` — interactive cleanup with confirmation prompts
