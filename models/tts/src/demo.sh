#!/bin/bash
# Launch the Voice Trainer Gradio demo for interactive checkpoint testing.
#
# Usage:
#   bash models/tts/src/demo.sh [checkpoints_dir]
#
# Default checkpoints dir: models/tts/checkpoints

set -e

# Resolve checkpoints directory relative to CWD before we change directories
if [ -n "$1" ]; then
    ckpt_dir=$(realpath "$1")
fi

# Go to the directory of this script
cd "$(dirname "$0")"
src_dir=$(pwd)

if [ -z "$ckpt_dir" ]; then
    ckpt_dir=$(realpath "$src_dir/../checkpoints")
fi

root_dir="$src_dir/../../.."

# Activate virtual environment
# shellcheck disable=SC1091
source "$root_dir/venv/bin/activate"

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
