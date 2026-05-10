#!/bin/bash
#
# setup-deps.sh — Install all Python dependencies and pretrained model weights.
#
# Run this once on a fresh instance BEFORE touching datasets.
#
# Usage:
#   bash scripts/setup-deps.sh [--skip-pretrained]
#
set -e

SKIP_PRETRAINED=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-pretrained) SKIP_PRETRAINED=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TTS_SRC_DIR="$ROOT_DIR/models/tts/src"
TTS_REPO_DIR="$ROOT_DIR/models/tts/tts-arabic-pytorch"

echo "=============================================="
echo "  Voice Trainer — Dependency Setup"
echo "=============================================="
echo ""
echo "  Root:      $ROOT_DIR"
echo "  TTS Repo:  $TTS_REPO_DIR"
echo ""

# ─── Git submodules ────────────────────────────────────────────────
echo "📂 Initialising git submodules..."
cd "$ROOT_DIR"
git submodule update --init --recursive 2>/dev/null || true
echo ""

# ─── Virtual environment ───────────────────────────────────────────
echo "🐍 Setting up Python virtual environment..."
VENV_DIR="$ROOT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    PYTHON_BIN=""
    for candidate in python3.12 python3.11 python3.10; do
        if command -v "$candidate" &>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON_BIN" ]; then
        echo "❌ No compatible Python found (need 3.10–3.12)."
        echo "   Install with: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
    echo "   Using $PYTHON_BIN ($(${PYTHON_BIN} --version))"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "   Python: $(python --version) at $(which python)"
echo ""

# ─── Python packages ───────────────────────────────────────────────
echo "📦 Installing Python packages..."
pip install --upgrade pip --quiet

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    # Detect GPU compute capability
    SM=$(python -c "
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f'{major}{minor}')
else:
    print('0')
" 2>/dev/null)
    echo "   GPU detected — compute capability: sm_${SM}"
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
       python -c "
import torch
major, minor = torch.cuda.get_device_capability(0)
# Check if the installed torch actually supports this GPU
props = torch.cuda.get_device_properties(0)
import subprocess, sys
out = subprocess.run([sys.executable, '-c',
    'import torch; print(torch.version.cuda)'], capture_output=True, text=True)
" 2>/dev/null; then
        # Test if the current install produces warnings about unsupported capability
        WARN=$(python -c "
import warnings, io, torch
buf = io.StringIO()
import warnings as _w
with _w.catch_warnings(record=True) as w:
    _w.simplefilter('always')
    torch.cuda.get_device_capability(0)
    for warning in w:
        if 'not compatible' in str(warning.message):
            print('unsupported')
            break
" 2>/dev/null)
        if [ "$WARN" = "unsupported" ] || [ "${SM:-0}" -ge "120" ]; then
            echo "   ⚠️  Blackwell GPU (sm_${SM}) — PyTorch stable does not support it."
            echo "   Installing PyTorch nightly with CUDA 12.8..."
            pip install --no-cache-dir --pre \
                torch torchaudio \
                --index-url https://download.pytorch.org/whl/nightly/cu128
        else
            echo "   ✅ PyTorch CUDA already installed and compatible."
        fi
    fi
else
    echo "   ⚠️  PyTorch CUDA not detected. Detecting best install..."
    # Try to read GPU capability even without a working torch
    SM_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$SM_RAW" ] && [ "$SM_RAW" -ge "120" ] 2>/dev/null; then
        echo "   Blackwell GPU detected (sm_${SM_RAW}) — installing PyTorch nightly + CUDA 12.8..."
        pip install --no-cache-dir --pre \
            torch torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        echo "   Installing PyTorch 2.6.0 + CUDA 12.4 (stable)..."
        pip install --no-cache-dir \
            torch==2.6.0 torchaudio==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124
    fi
fi

pip install --no-cache-dir torbi
pip install --no-cache-dir -r "$ROOT_DIR/requirements.txt"
pip install --no-cache-dir datasets huggingface_hub
echo ""

# ─── Pretrained models ─────────────────────────────────────────────
if [ "$SKIP_PRETRAINED" = true ]; then
    echo "⏭️  Skipping pretrained model download (--skip-pretrained)"
else
    echo "🤖 Downloading pretrained FastPitch + HiFi-GAN weights..."
    cp "$TTS_SRC_DIR/download_files.py" "$TTS_REPO_DIR/"
    cd "$TTS_REPO_DIR"
    python download_files.py
    cd "$ROOT_DIR"
fi
echo ""

echo "=============================================="
echo "  ✅ Dependency setup complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "    1. Download + format dataset:"
echo "         bash scripts/download-dataset.sh"
echo "    2. Review datasets/doda-dataset/doda/data.csv"
echo "    3. Merge + prepare for training:"
echo "         bash scripts/prepare-dataset.sh"
echo "    4. Train:"
echo "         bash models/tts/src/finetune.sh"
echo ""
