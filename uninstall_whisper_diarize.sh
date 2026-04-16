#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Whisper Diarize Local — Uninstall Script
# ══════════════════════════════════════════════════════════════════════════════

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       🗑  WHISPER DIARIZE LOCAL — UNINSTALL              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "   Directory: $INSTALL_DIR"
echo ""

# ── 1. Virtual environment ───────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/venv" ]; then
    SIZE=$(du -sh "$INSTALL_DIR/venv" 2>/dev/null | awk '{print $1}')
    echo "🐍 Virtual environment ($SIZE — includes PyTorch ~2 GB)"
    read -p "   Delete? [y/N] " c; [[ "$c" =~ ^[Yy]$ ]] && rm -rf "$INSTALL_DIR/venv" && echo "   ✅ Removed" || echo "   ⏭  Skipped"
fi

# ── 2. Cached models ────────────────────────────────────────────────────────
echo ""
CACHE_DIR="$HOME/.cache/huggingface/hub"
DIARIZE_CACHES=$(find "$CACHE_DIR" \
    -maxdepth 1 -type d \
    \( -name "models--Systran--faster-whisper-*" \
    -o -name "models--pyannote--*" \
    -o -name "models--openai--whisper-*" \
    -o -name "models--jonatasgrosman--*" \
    -o -name "models--facebook--wav2vec2-*" \) \
    2>/dev/null)
if [ -n "$DIARIZE_CACHES" ]; then
    TOTAL_SIZE=$(du -shc $DIARIZE_CACHES 2>/dev/null | tail -1 | awk '{print $1}')
    echo "📦 Cached models ($TOTAL_SIZE total)"
    echo "$DIARIZE_CACHES" | while read d; do
        S=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
        echo "      $(basename $d) ($S)"
    done
    read -p "   Delete all cached models? [y/N] " c
    if [[ "$c" =~ ^[Yy]$ ]]; then
        echo "$DIARIZE_CACHES" | xargs rm -rf
        echo "   ✅ Removed"
    else
        echo "   ⏭  Skipped"
    fi
else
    echo "📦 No cached models found"
fi

# ── 3. HF token ─────────────────────────────────────────────────────────────
echo ""
if [ -f "$INSTALL_DIR/.hf_token" ]; then
    echo "🔑 Hugging Face token (.hf_token)"
    read -p "   Delete? [y/N] " c; [[ "$c" =~ ^[Yy]$ ]] && rm -f "$INSTALL_DIR/.hf_token" && echo "   ✅ Removed" || echo "   ⏭  Skipped"
fi

# ── 4. Launcher ─────────────────────────────────────────────────────────────
echo ""
if [ -f "$INSTALL_DIR/whisper-diarize" ]; then
    echo "🚀 Launcher script"
    read -p "   Delete? [y/N] " c; [[ "$c" =~ ^[Yy]$ ]] && rm -f "$INSTALL_DIR/whisper-diarize" && echo "   ✅ Removed" || echo "   ⏭  Skipped"
fi

# ── 5. Scripts themselves ───────────────────────────────────────────────────
echo ""
echo "📄 Scripts: whisper_diarize_local.py, setup_whisper_diarize.sh, uninstall_whisper_diarize.sh, README.md"
read -p "   Delete all scripts? (full removal) [y/N] " c
if [[ "$c" =~ ^[Yy]$ ]]; then
    rm -f "$INSTALL_DIR/whisper_diarize_local.py" \
          "$INSTALL_DIR/setup_whisper_diarize.sh" \
          "$INSTALL_DIR/uninstall_whisper_diarize.sh" \
          "$INSTALL_DIR/README.md" \
          "$INSTALL_DIR/CHANGELOG.md" \
          "$INSTALL_DIR/VERSION"
    echo "   ✅ Removed"
    rmdir "$INSTALL_DIR" 2>/dev/null && echo "   ✅ Removed empty directory"
fi

echo ""
echo "✅ Uninstall complete."
echo ""
