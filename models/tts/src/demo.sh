#!/bin/bash
# Launch the Voice Trainer Gradio demo for interactive checkpoint testing.
#
# Usage:
#   bash models/tts/src/demo.sh [checkpoints_dir]
#
# Default checkpoints dir: models/tts/checkpoints

set -e

# Go to the directory of this script
cd "$(dirname "$0")"

src_dir=$(pwd)
root_dir="$src_dir/../../.."

# Activate virtual environment
# shellcheck disable=SC1091
source "$root_dir/venv/bin/activate"

# Resolve checkpoints directory
ckpt_dir="${1:-$src_dir/../checkpoints}"
ckpt_dir=$(realpath "$ckpt_dir")

# copy demo script to tts-arabic-pytorch
cp "$src_dir/demo.py" ../tts-arabic-pytorch/

# Go to the directory of the TTS model
cd ../tts-arabic-pytorch/

echo "=============================================="
echo "  Voice Trainer — Gradio Demo"
echo "=============================================="
echo ""
echo "  Checkpoints: $ckpt_dir"
echo "  Open http://0.0.0.0:7860 in your browser"
echo ""

python demo.py --ckpt_dir "$ckpt_dir" "$@"
