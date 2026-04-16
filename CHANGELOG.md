# Changelog

All notable changes to Whisper Diarize Local are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.0.2] — 2026-04-16

### Fixed
- **Threading deadlock / segfault on Intel macOS.** Whisperx + pyannote + torch 2.2.2 deadlocked at 0% CPU on multi-core Intel Macs because two OpenMP runtimes (MKL's `libiomp5` and torch's `libomp`) fight for the same thread pool, and forked workers from faster-whisper / pyannote inherit broken thread state. Fixed by setting `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false` at module import (before any torch / numpy / faster-whisper import).
- **CTranslate2 segfault during language detection** when `OMP_NUM_THREADS > 1`. The single-thread cap is now mandatory on this stack, not optional. Bumping it triggers an immediate segfault in the transcription step. Documented inline.
- `torch.set_num_threads()` is now pinned to match the OMP env var to avoid cache contention from oversubscription.

### Added
- LICENSE file (MIT) with upstream attribution for Whisper, WhisperX, pyannote.audio, PyTorch, transformers, and the gated diarization model.
- File manifest section in README.
- Suppression filter for the noisy "Lightning automatically upgraded your loaded checkpoint" log line. The upgrade is in-memory only and works fine; the message was pure visual noise. Filter targets the specific Lightning logger and is also attached to root handlers as a fallback.

### Changed
- `.gitignore` expanded to cover IDE/editor files (`.vscode/`, `.idea/`, swap files), local env files (`.env`, `.envrc`), and Python build artifacts. `.hf_token` was already ignored and stays so.

### Performance baseline (2018 Intel 15" MacBook Pro / macOS Sequoia)
- 2m 10s audio, `tiny` model, 4 speakers detected: **9m 05s total** (RTF 4.17×).
  - Transcribe: 35s
  - Align: 36s
  - Diarize: 7m 54s ← bottleneck (87% of total)
- Speed roughly scales with model size; `medium` is realistically 20–40 min for the same file.
- For faster turnaround, run on Apple Silicon or CUDA — the threading constraints above don't apply there.

## [1.0.1] — 2026-04-16

### Fixed
- **Dependency conflict on Intel Mac.** `numpy>=2` is dragged in transitively by `pyannote-core`, `asteroid-filterbanks`, and `speechbrain`, but `torch==2.2.2` (the last Intel-Mac-compatible release) was compiled against `numpy<2` and crashes at runtime with `RuntimeError: Numpy is not available` from the C extension bridge.
- Setup script now installs the full dependency tree first, then force-reinstalls `numpy==1.26.4` as the very last step. Pip prints a conflict warning, but everything works correctly at runtime.
- Pinned `transformers>=4.27,<4.44` (newer transformers declares `torch>=2.4` which fails on Intel Mac).
- Added `matplotlib` to the explicit pip install line — pyannote.audio uses it indirectly and it isn't always pulled in.
- Setup now uses Homebrew's `python@3.10` explicitly (via `/usr/local/bin/python3.10`). 3.11/3.12 have additional pyannote/whisperx compatibility issues.

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
