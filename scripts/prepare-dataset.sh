#!/bin/bash
#
# prepare-dataset.sh — Merge datasets, convert to WAV, extract F0, generate config.
#
# Run this AFTER manually reviewing the raw dataset produced by download-dataset.sh.
#
# Usage:
#   bash scripts/prepare-dataset.sh [--epochs N] [--save-interval N]
#
set -e

EPOCHS=100
SAVE_INTERVAL=2500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS_DIR="$ROOT_DIR/datasets/doda-dataset"
ALL_DATASETS_DIR="$ROOT_DIR/datasets/tts-all-datasets"
TTS_SRC_DIR="$ROOT_DIR/models/tts/src"
TTS_REPO_DIR="$ROOT_DIR/models/tts/tts-arabic-pytorch"
TOOLS_DIR="$ROOT_DIR/tools/dataset"
CHECKPOINTS_DIR="$ROOT_DIR/models/tts/checkpoints"
VENV_DIR="$ROOT_DIR/venv"

echo "=============================================="
echo "  Voice Trainer — Dataset Preparation"
echo "=============================================="
echo ""
echo "  Source datasets : $DATASETS_DIR"
echo "  Merged output   : $ALL_DATASETS_DIR"
echo "  Epochs          : $EPOCHS"
echo "  Save interval   : $SAVE_INTERVAL"
echo ""

# ─── Activate venv ─────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "❌ Virtual environment not found. Run setup-deps.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ─── Step 1: Merge datasets ────────────────────────────────────────
echo "🔀 Step 1/4: Merging datasets into $ALL_DATASETS_DIR..."

# Clear the previous merged output so we start fresh
if [ -d "$ALL_DATASETS_DIR" ]; then
    echo "   Removing previous merged dataset..."
    rm -rf "$ALL_DATASETS_DIR"
fi

# Discover all per-dataset sub-folders (doda + any extras you have added)
folders=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | tr '\n' ' ')

if [ -z "$folders" ]; then
    echo "❌ No dataset folders found in $DATASETS_DIR"
    echo "   Run download-dataset.sh first."
    exit 1
fi

echo "   Found datasets: $folders"
# shellcheck disable=SC2086
python "$TOOLS_DIR/merge-datasets.py" --datasets $folders --output "$ALL_DATASETS_DIR"
echo ""

# ─── Step 2: Convert MP3 → WAV ────────────────────────────────────
echo "🔊 Step 2/4: Converting audio files to WAV..."
bash "$TOOLS_DIR/mp3-to-wav.sh" "$ALL_DATASETS_DIR"
echo ""

# ─── Step 3: Extract F0 pitch features ────────────────────────────
echo "🎵 Step 3/4: Extracting F0 pitch features (may take a while)..."

cd "$TTS_REPO_DIR"
cp "$TTS_SRC_DIR/extract_f0_penn.py" .

F0_LOG="$ROOT_DIR/f0_extraction.log"
python extract_f0_penn.py --audios_dir "$ALL_DATASETS_DIR/audios" 2>&1 | tee "$F0_LOG"

mean=$(grep -oE "mean [0-9.]+" "$F0_LOG" | head -1 | awk '{print $2}')
std=$(grep -oE "std [0-9.]+"   "$F0_LOG" | head -1 | awk '{print $2}')

if [ -z "$mean" ] || [ -z "$std" ]; then
    echo "❌ Failed to extract F0 mean/std. Check the output above for errors."
    exit 1
fi

echo "   F0 Mean : $mean"
echo "   F0 Std  : $std"
echo ""

# ─── Step 4: Generate training config ─────────────────────────────
echo "⚙️  Step 4/4: Generating training configuration..."

cd "$TTS_SRC_DIR"
python generate-config.py \
    --train_data_path "$ALL_DATASETS_DIR" \
    --output_path     "$TTS_SRC_DIR/config.yaml" \
    --f0_mean         "$mean" \
    --f0_std          "$std" \
    --restore_model   "$TTS_REPO_DIR/pretrained/fastpitch_raw_ms.pth" \
    --checkpoint_dir  "$CHECKPOINTS_DIR" \
    --n_save_states_iter "$SAVE_INTERVAL" \
    --n_save_backup_iter "$SAVE_INTERVAL" \
    --epochs          "$EPOCHS"
echo ""

echo "=============================================="
echo "  ✅ Dataset preparation complete!"
echo "=============================================="
echo ""
echo "  Merged dataset : $ALL_DATASETS_DIR"
echo "  Config         : $TTS_SRC_DIR/config.yaml"
echo "  Checkpoints    : $CHECKPOINTS_DIR"
echo ""
echo "  To start training, run:"
echo "    bash models/tts/src/finetune.sh"
echo ""
