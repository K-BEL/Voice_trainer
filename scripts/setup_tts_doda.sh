#!/bin/bash
#
# setup_tts_doda.sh — End-to-end setup for TTS training on the DODa dataset
#
# Usage:
#   bash scripts/setup_tts_doda.sh [--skip-download] [--skip-pretrained] [--epochs N]
#
# This script:
#   1. Downloads and formats the DODa dataset from HuggingFace
#   2. Merges datasets and converts audio to WAV
#   3. Downloads pretrained FastPitch + HiFi-GAN weights
#   4. Extracts F0 pitch features (requires CUDA GPU)
#   5. Generates a training config YAML
#   6. Launches finetuning
#
set -e

# ─── Parse arguments ───────────────────────────────────────────────
SKIP_DOWNLOAD=false
SKIP_PRETRAINED=false
EPOCHS=100
SAVE_INTERVAL=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download)    SKIP_DOWNLOAD=true; shift ;;
        --skip-pretrained)  SKIP_PRETRAINED=true; shift ;;
        --epochs)           EPOCHS="$2"; shift 2 ;;
        --save-interval)    SAVE_INTERVAL="$2"; shift 2 ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Resolve paths ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS_DIR="$ROOT_DIR/datasets/doda-dataset"
DODA_DIR="$DATASETS_DIR/doda"
ALL_DATASETS_DIR="$ROOT_DIR/datasets/tts-all-datasets"
TTS_SRC_DIR="$ROOT_DIR/models/tts/src"
TTS_REPO_DIR="$ROOT_DIR/models/tts/tts-arabic-pytorch"
TOOLS_DIR="$ROOT_DIR/tools/dataset"
CHECKPOINTS_DIR="$ROOT_DIR/models/tts/checkpoints"

echo "=============================================="
echo "  Voice Trainer — TTS Setup for DODa Dataset"
echo "=============================================="
echo ""
echo "  Root:           $ROOT_DIR"
echo "  Dataset:        $DODA_DIR"
echo "  TTS Repo:       $TTS_REPO_DIR"
echo "  Checkpoints:    $CHECKPOINTS_DIR"
echo "  Epochs:         $EPOCHS"
echo "  Save interval:  $SAVE_INTERVAL"
echo ""

# ─── Step 1: Install dependencies ─────────────────────────────────
echo "📦 Step 1/7: Installing Python dependencies..."

# Initialize git submodules (tts-arabic-pytorch)
cd "$ROOT_DIR"
git submodule update --init --recursive 2>/dev/null || true

# Create and activate virtual environment using a PyTorch-compatible Python
VENV_DIR="$ROOT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "   Creating virtual environment at $VENV_DIR..."
    
    # Find a Python version compatible with PyTorch (3.10, 3.11, 3.12)
    PYTHON_BIN=""
    for candidate in python3.12 python3.11 python3.10; do
        if command -v "$candidate" &>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    
    if [ -z "$PYTHON_BIN" ]; then
        echo "   ❌ No PyTorch-compatible Python found (need 3.10, 3.11, or 3.12)."
        echo "   Install one with: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
    
    echo "   Using $PYTHON_BIN ($(${PYTHON_BIN} --version))"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "   Using Python: $(python --version) at $(which python)"

# Upgrade pip
pip install --upgrade pip --quiet

# Install PyTorch with CUDA support first (skip if already installed)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "   ⚠️  PyTorch CUDA not detected. Installing PyTorch 2.6.0 with CUDA 12.4..."
    pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
else
    echo "   ✅ PyTorch with CUDA already installed."
fi

# Install project requirements
echo "   Installing dependencies..."
pip install --no-cache-dir torbi  # Uses precompiled wheel for PyTorch 2.6 + CUDA 12.4
pip install --no-cache-dir -r "$ROOT_DIR/requirements.txt"



# Install datasets library for HuggingFace downloads
pip install --no-cache-dir datasets huggingface_hub

echo ""

# ─── Step 2: Download DODa dataset ────────────────────────────────
if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "⏭️  Skipping DODa download (--skip-download)"
else
    echo "📥 Step 2/7: Downloading DODa dataset from HuggingFace..."
    
    # Check if logged into HuggingFace (prefer `hf`, fallback to legacy CLI)
    if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
        echo "   🔑 HuggingFace login required (DODa dataset needs authentication)."
        if command -v hf >/dev/null 2>&1; then
            hf auth login
        elif command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli login
        else
            echo "   ❌ Hugging Face CLI not found. Install with: pip install huggingface_hub"
            exit 1
        fi
    fi
    if [ -f "$DODA_DIR/data.csv" ] && [ -d "$DODA_DIR/audios" ]; then
        echo "   Dataset already exists at $DODA_DIR"
        read -p "   Re-download? [y/N]: " REDOWNLOAD
        if [[ "$REDOWNLOAD" =~ ^[Yy]$ ]]; then
            python "$ROOT_DIR/scripts/download_format_doda.py" --output-dir "$DODA_DIR"
        else
            echo "   Keeping existing dataset."
        fi
    else
        python "$ROOT_DIR/scripts/download_format_doda.py" --output-dir "$DODA_DIR"
    fi
fi
echo ""

# ─── Step 3: Merge datasets ───────────────────────────────────────
echo "🔀 Step 3/7: Merging datasets into $ALL_DATASETS_DIR..."

# Delete all-datasets directory if exists
if [ -d "$ALL_DATASETS_DIR" ]; then
    rm -rf "$ALL_DATASETS_DIR"
fi

# Get all dataset folders (doda + any future ones)
folders=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | tr '\n' ' ')

if [ -z "$folders" ]; then
    echo "❌ No dataset folders found in $DATASETS_DIR"
    exit 1
fi

echo "   Found datasets: $folders"
# shellcheck disable=SC2086
python "$TOOLS_DIR/merge-datasets.py" --datasets $folders --output "$ALL_DATASETS_DIR"
echo ""

# ─── Step 4: Convert MP3 to WAV ───────────────────────────────────
echo "🔊 Step 4/7: Converting audio files to WAV format..."
bash "$TOOLS_DIR/mp3-to-wav.sh" "$ALL_DATASETS_DIR"
echo ""

# ─── Step 5: Download pretrained models ───────────────────────────
if [ "$SKIP_PRETRAINED" = true ]; then
    echo "⏭️  Skipping pretrained model download (--skip-pretrained)"
else
    echo "📦 Step 5/7: Downloading pretrained FastPitch + HiFi-GAN models..."
    
    # Copy download script and run it
    cp "$TTS_SRC_DIR/download_files.py" "$TTS_REPO_DIR/"
    cd "$TTS_REPO_DIR"
    python download_files.py
    cd "$ROOT_DIR"
fi
echo ""

# ─── Step 6: Extract F0 pitch features ────────────────────────────
echo "🎵 Step 6/7: Extracting F0 pitch features (this may take a while)..."

cd "$TTS_REPO_DIR"
cp "$TTS_SRC_DIR/extract_f0_penn.py" .

# Run F0 extraction — show output in real-time and capture to file
F0_LOG="$ROOT_DIR/f0_extraction.log"
python extract_f0_penn.py --audios_dir "$ALL_DATASETS_DIR/audios" 2>&1 | tee "$F0_LOG"

# Extract mean and std from the log
mean=$(grep -oE "mean [0-9.]+" "$F0_LOG" | head -1 | awk '{print $2}')
std=$(grep -oE "std [0-9.]+" "$F0_LOG" | head -1 | awk '{print $2}')

if [ -z "$mean" ] || [ -z "$std" ]; then
    echo "❌ Failed to extract F0 mean/std. Check the output above for errors."
    exit 1
fi

echo "   F0 Mean: $mean"
echo "   F0 Std:  $std"
echo ""

# ─── Step 7: Generate training config ─────────────────────────────
echo "⚙️  Step 7/7: Generating training configuration..."

cd "$TTS_SRC_DIR"
python generate-config.py \
    --train_data_path "$ALL_DATASETS_DIR" \
    --output_path "$TTS_SRC_DIR/config.yaml" \
    --f0_mean "$mean" \
    --f0_std "$std" \
    --restore_model "$TTS_REPO_DIR/pretrained/fastpitch_raw_ms.pth" \
    --checkpoint_dir "$CHECKPOINTS_DIR" \
    --n_save_states_iter "$SAVE_INTERVAL" \
    --n_save_backup_iter "$SAVE_INTERVAL" \
    --epochs "$EPOCHS"

echo ""
echo "=============================================="
echo "  ✅ Setup complete!"
echo "=============================================="
echo ""
echo "  Config saved to: $TTS_SRC_DIR/config.yaml"
echo "  Checkpoints at:  $CHECKPOINTS_DIR"
echo ""
echo "  To start training, run:"
echo "    bash models/tts/src/finetune.sh"
echo ""
echo "  To test checkpoints after training:"
echo "    bash models/tts/src/test-ckpts.sh models/tts/checkpoints"
echo ""
