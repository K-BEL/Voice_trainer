#!/bin/bash
#
# setup_tts_doda.sh — One-shot setup that chains all three stages.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  For a REVIEWED, production-quality dataset we recommend running     │
# │  each stage separately so you can spot-check the CSV before merging: │
# │                                                                       │
# │   1. bash scripts/setup-deps.sh                                      │
# │   2. bash scripts/download-dataset.sh                                │
# │      → Review datasets/doda-dataset/doda/data.csv manually           │
# │   3. bash scripts/prepare-dataset.sh                                 │
# │   4. bash models/tts/src/finetune.sh                                 │
# └─────────────────────────────────────────────────────────────────────┘
#
# Usage (automated, no manual review):
#   bash scripts/setup_tts_doda.sh [--skip-download] [--skip-pretrained]
#                                   [--epochs N] [--save-interval N]
#
set -e

SKIP_DOWNLOAD=false
SKIP_PRETRAINED=false
EPOCHS=100
SAVE_INTERVAL=2500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download)    SKIP_DOWNLOAD=true;  shift ;;
        --skip-pretrained)  SKIP_PRETRAINED=true; shift ;;
        --epochs)           EPOCHS="$2";         shift 2 ;;
        --save-interval)    SAVE_INTERVAL="$2";  shift 2 ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "  Voice Trainer — Full Setup (all stages)"
echo "=============================================="
echo ""

# ─── Stage 1: Dependencies + pretrained models ─────────────────────
PRETRAINED_FLAG=""
[ "$SKIP_PRETRAINED" = true ] && PRETRAINED_FLAG="--skip-pretrained"
bash "$SCRIPT_DIR/setup-deps.sh" $PRETRAINED_FLAG

# ─── Stage 2: Download dataset ─────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "⏭️  Skipping DODa download (--skip-download)"
    echo ""
else
    bash "$SCRIPT_DIR/download-dataset.sh"
fi

# ─── Stage 3: Prepare dataset + generate config ────────────────────
bash "$SCRIPT_DIR/prepare-dataset.sh" \
    --epochs "$EPOCHS" \
    --save-interval "$SAVE_INTERVAL"

echo ""
echo "  To start training, run:"
echo "    bash models/tts/src/finetune.sh"
echo ""
