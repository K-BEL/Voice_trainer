#!/bin/bash
set -e

# Go to the directory of this script
cd "$(dirname "$0")"

src_dir=$(pwd)
root_dir="$src_dir/../../.."

# Activate virtual environment
# shellcheck disable=SC1091
source "$root_dir/venv/bin/activate"

# copy updated script file
cp "$src_dir/train_fp_adv.py" ../tts-arabic-pytorch/

# Go to the directory of the TTS model
cd ../tts-arabic-pytorch/

# Auto-resume logic
CHECKPOINT_FILE="$root_dir/models/tts/checkpoints/states.pth"
PRETRAINED_FILE="$root_dir/models/tts/tts-arabic-pytorch/pretrained/fastpitch_raw_ms.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "🔄 Found existing states.pth — resuming training!"
    sed -i "s|restore_model:.*|restore_model: $CHECKPOINT_FILE|g" "$src_dir/config.yaml"
    export VT_RESUME_OPTIMIZERS="1"
    export VT_RESUME_PROGRESS="1"
else
    echo "🚀 Starting fresh fine-tuning from pretrained model."
    sed -i "s|restore_model:.*|restore_model: $PRETRAINED_FILE|g" "$src_dir/config.yaml"
    export VT_RESUME_OPTIMIZERS="0"
    export VT_RESUME_PROGRESS="0"
fi

# Re-enable Automatic Mixed Precision (AMP) for RTX 5070 Ti stability
export VT_AMP="1"
export LD_LIBRARY_PATH=""
python train_fp_adv.py --config "$src_dir/config.yaml"
