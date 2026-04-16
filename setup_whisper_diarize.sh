#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Whisper Diarize Local — Setup Script for macOS
# Uses WhisperX (Whisper + wav2vec2) + pyannote 3.1 for diarization.
# Requires PyTorch — pinned to 2.2.2 for Intel Mac compatibility.
# ══════════════════════════════════════════════════════════════════════════════

set -e

# Ensure Homebrew is in PATH (macOS Intel + Apple Silicon)
for brewdir in /usr/local/bin /opt/homebrew/bin; do
    [[ -d "$brewdir" ]] && export PATH="$brewdir:$PATH"
done

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$INSTALL_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   🎤  WHISPER DIARIZE LOCAL — macOS Setup  🎤            ║"
echo "║   WhisperX + pyannote 3.1 (CPU, Intel Mac compatible)    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "   Install directory: $INSTALL_DIR"
echo ""
echo "   ⚠  Heads-up: this submodule pulls in PyTorch (~2 GB) — required by"
echo "      pyannote.audio. The other voxbox tools stay PyTorch-free."
echo ""

# ── 1. Check for Homebrew ────────────────────────────────────────────────────
echo "🔍 Step 1/6: Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "   ⚠  Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "   ✅ Homebrew found"
fi

# ── 2. Check Python ─────────────────────────────────────────────────────────
echo ""
echo "🔍 Step 2/6: Checking Python..."
PYTHON_CMD=""

if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "   ✅ Python 3.10 found (recommended for WhisperX)"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ] && [ "$MINOR" -le 11 ]; then
        PYTHON_CMD="python3"
        echo "   ✅ Python $PYTHON_VERSION found"
    elif [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 12 ]; then
        echo "   ⚠  Python $PYTHON_VERSION — pyannote/whisperx work best on 3.10/3.11"
        echo "      Installing python@3.10 via Homebrew for a cleaner venv..."
        brew install python@3.10
        PYTHON_CMD="python3.10"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "   ⚠  Python 3.10/3.11 required. Installing python@3.10 via Homebrew..."
    brew install python@3.10
    PYTHON_CMD="python3.10"
fi

# ── 3. Check ffmpeg ─────────────────────────────────────────────────────────
echo ""
echo "🔍 Step 3/6: Checking ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "   ⚠  ffmpeg not found. Installing..."
    brew install ffmpeg
else
    echo "   ✅ ffmpeg found"
fi

# ── 4. Create venv & install packages ───────────────────────────────────────
echo ""
echo "📦 Step 4/6: Setting up Python environment..."

if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "   🐍 Creating virtual environment ($PYTHON_CMD)..."
    $PYTHON_CMD -m venv "$INSTALL_DIR/venv"
else
    echo "   ✅ Virtual environment exists"
fi

source "$INSTALL_DIR/venv/bin/activate"

echo "   📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# Detect architecture for torch wheel selection
ARCH=$(uname -m)
echo "   📦 Installing PyTorch 2.2.2 (CPU, $ARCH)..."
# 2.2.2 is the last version with reliable Intel Mac support; works on Apple Silicon too.
pip install -q "torch==2.2.2" "torchaudio==2.2.2"

echo "   📦 Installing WhisperX + pyannote.audio..."
pip install -q "pyannote.audio==3.1.1"
pip install -q "whisperx"

echo "   📦 Installing audio I/O helpers..."
pip install -q soundfile numpy

# ── 5. HF token bootstrap ───────────────────────────────────────────────────
echo ""
echo "🔑 Step 5/6: Hugging Face token..."
TOKEN_FILE="$INSTALL_DIR/.hf_token"

if [ -f "$TOKEN_FILE" ]; then
    echo "   ✅ Existing token found at .hf_token (chmod 600)"
elif [ -n "$HF_TOKEN" ] || [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "   ✅ HF token found in environment"
else
    echo ""
    echo "   pyannote/speaker-diarization-3.1 is a gated model. Before first run:"
    echo "     1. Create a free account: https://huggingface.co"
    echo "     2. Accept terms at:"
    echo "        - https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "        - https://huggingface.co/pyannote/segmentation-3.0"
    echo "     3. Generate a READ token: https://huggingface.co/settings/tokens"
    echo ""
    read -p "   Paste your HF token now (or Enter to skip and provide later): " TOKEN_INPUT
    if [ -n "$TOKEN_INPUT" ]; then
        echo "$TOKEN_INPUT" > "$TOKEN_FILE"
        chmod 600 "$TOKEN_FILE"
        echo "   ✅ Token saved to .hf_token (chmod 600)"
    else
        echo "   ⏭  Skipped — you'll be prompted on first run."
    fi
fi

# ── 6. Pre-download default model ──────────────────────────────────────────
echo ""
echo "📥 Step 6/6: Pre-downloading default Whisper model (medium, ~1.5 GB)..."
echo "   This may take a while on first install."
echo ""

python -c "
import warnings; warnings.filterwarnings('ignore')
import whisperx
print('   ⏳ Downloading Whisper medium...')
whisperx.load_model('medium', device='cpu', compute_type='int8')
print('   ✅ Whisper medium ready')
" || echo "   ⚠  Whisper download failed — will retry on first use"

# ── Create launcher ─────────────────────────────────────────────────────────
cat > "$INSTALL_DIR/whisper-diarize" << LAUNCHER
#!/bin/bash
SCRIPT_DIR="$INSTALL_DIR"
source "\$SCRIPT_DIR/venv/bin/activate"
exec python "\$SCRIPT_DIR/whisper_diarize_local.py" "\$@"
LAUNCHER

chmod +x "$INSTALL_DIR/whisper-diarize"

# ── Done! ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                 ✅  SETUP COMPLETE!                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Quick start:                                            ║"
echo "║    cd $INSTALL_DIR"
echo "║    ./whisper-diarize interview.mp3                       ║"
echo "║                                                          ║"
echo "║  Interactive mode:                                       ║"
echo "║    ./whisper-diarize                                     ║"
echo "║                                                          ║"
echo "║  Batch process inbox:                                    ║"
echo "║    ./whisper-diarize --inbox                             ║"
echo "║                                                          ║"
echo "║  Save as SRT:                                            ║"
echo "║    ./whisper-diarize -o out.srt podcast.mp3              ║"
echo "║                                                          ║"
echo "║  Fix speaker count (faster, more stable):                ║"
echo "║    ./whisper-diarize --min-speakers 2 --max-speakers 2 ..║"
echo "║                                                          ║"
echo "║  Note: long files on Intel CPU are slow.                 ║"
echo "║  Budget ~3-6x audio duration with the medium model.      ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
