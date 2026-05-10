#!/bin/bash
#
# download-dataset.sh — Download and format the DODa dataset from HuggingFace.
#
# After this script finishes you should MANUALLY REVIEW:
#   datasets/doda-dataset/doda/data.csv
#   datasets/doda-dataset/doda/audios/
#
# Then run prepare-dataset.sh to merge, convert and extract pitch.
#
# Usage:
#   bash scripts/download-dataset.sh [--output-dir PATH] [--split SPLIT]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS_DIR="$ROOT_DIR/datasets/doda-dataset"
DODA_DIR="$DATASETS_DIR/doda"
VENV_DIR="$ROOT_DIR/venv"

OUTPUT_DIR="$DODA_DIR"
SPLIT="train"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --split)      SPLIT="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  Voice Trainer — Dataset Download"
echo "=============================================="
echo ""
echo "  Output dir : $OUTPUT_DIR"
echo "  Split      : $SPLIT"
echo ""

# ─── Activate venv ─────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "❌ Virtual environment not found. Run setup-deps.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ─── HuggingFace auth check ────────────────────────────────────────
echo "🔑 Checking HuggingFace authentication..."
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "   Login required (DODa dataset needs authentication)."
    if command -v hf >/dev/null 2>&1; then
        hf auth login
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli login
    else
        echo "❌ Hugging Face CLI not found. Install with: pip install huggingface_hub"
        exit 1
    fi
fi
echo ""

# ─── Download ──────────────────────────────────────────────────────
if [ -f "$OUTPUT_DIR/data.csv" ] && [ -d "$OUTPUT_DIR/audios" ]; then
    echo "⚠️  Dataset already exists at $OUTPUT_DIR"
    read -rp "   Re-download? [y/N]: " REDOWNLOAD
    if [[ "$REDOWNLOAD" =~ ^[Yy]$ ]]; then
        python "$ROOT_DIR/scripts/download_format_doda.py" \
            --output-dir "$OUTPUT_DIR" \
            --split "$SPLIT"
    else
        echo "   Keeping existing dataset."
    fi
else
    python "$ROOT_DIR/scripts/download_format_doda.py" \
        --output-dir "$OUTPUT_DIR" \
        --split "$SPLIT"
fi

echo ""
echo "=============================================="
echo "  ✅ Download complete!"
echo "=============================================="
echo ""
echo "  ⚑ REVIEW YOUR DATASET BEFORE CONTINUING:"
echo ""
echo "    Captions CSV : $OUTPUT_DIR/data.csv"
echo "    Audio files  : $OUTPUT_DIR/audios/"
echo ""
echo "  Run the interactive review tool (plays 20 random samples):"
echo "    python scripts/review_dataset.py --dataset-dir \"$OUTPUT_DIR\" --n 20"
echo ""
echo "  Controls: Enter/k = keep | r = remove | q = quit"
echo "  Rejected rows are removed from data.csv automatically."
echo ""
echo "  When satisfied, run:"
echo "    bash scripts/prepare-dataset.sh"
echo ""
