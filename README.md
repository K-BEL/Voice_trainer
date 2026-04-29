# Voice Trainer

Moroccan Darija TTS training pipeline based on FastPitch + HiFi-GAN, using the [DODa dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset).

## What This Repo Does

- Downloads and formats DODa audio/text data
- Merges datasets and ensures WAV-ready training data
- Extracts F0 (pitch) features
- Generates a training config automatically
- Fine-tunes FastPitch + HiFi-GAN
- Runs checkpoint-based inference tests

## Requirements

- Linux environment recommended (local Linux, cloud VM, Vast, etc.)
- Python 3.10+ (3.12 supported in current workflow)
- NVIDIA GPU + CUDA
- `ffmpeg`
- Hugging Face account with access to DODa

Install `ffmpeg`:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

## Quickstart

### 1) Clone

```bash
git clone --recurse-submodules https://github.com/K-BEL/Voice_trainer.git
cd Voice_trainer
```

`--recurse-submodules` is required to pull `models/tts/tts-arabic-pytorch`.

### 2) Setup data + dependencies

```bash
bash scripts/setup_tts_doda.sh
```

This performs:

1. Environment/dependency installation
2. DODa download + formatting
3. Dataset merge and audio prep
4. Pretrained model download
5. Pitch extraction
6. Config generation at `models/tts/src/config.yaml`

### 3) Activate Environment

The setup script creates a virtual environment at `./venv`. To activate it manually for testing or running scripts:

```bash
source venv/bin/activate
```

Common setup options:

```bash
# Skip dataset download
bash scripts/setup_tts_doda.sh --skip-download

# Skip pretrained model download
bash scripts/setup_tts_doda.sh --skip-pretrained

# Customize training plan at config-generation time
bash scripts/setup_tts_doda.sh --epochs 200 --save-interval 1000
```

## Train

```bash
bash models/tts/src/finetune.sh
```

Outputs:

- Checkpoints: `models/tts/checkpoints/`
- TensorBoard logs: `models/tts/logs/`

## Test Checkpoints

Generate test audio for all `.pth` checkpoints in a directory:

```bash
bash models/tts/src/test-ckpts.sh models/tts/checkpoints
```

Generated audio is saved under:

- `models/tts/results/<checkpoint_name>/`

## Typical Workflow

1. Run setup once
2. Start training
3. Let it run for a while
4. Test checkpoints periodically (or use the Gradio demo)
5. Best checkpoints are saved automatically as `best_model.pth`

## Interactive Demo

Test any checkpoint interactively in your browser:

```bash
bash models/tts/src/demo.sh models/tts/checkpoints
```

Then open `http://0.0.0.0:7860` in your browser. Features:

- Text input with Darija/Arabic
- Checkpoint selector (auto-discovers all `.pth` files)
- Adjustable speaker ID and pace
- Real-time audio playback

## Training Features

### Mixed Precision (AMP)

Enabled by default. Gives ~2x speedup and uses ~40% less VRAM.

```bash
# Disable if you encounter issues
VT_AMP=0 bash models/tts/src/finetune.sh
```

### Learning Rate Scheduler

Cosine annealing with warm-up (1000 iters by default). Gradually decays the learning rate for better convergence.

### Validation & Early Stopping

Automatically splits your dataset 90/10 and tracks validation loss. Saves `best_model.pth` when validation improves. Stops training if no improvement for 10 epochs.

### Auto Checkpoint Cleanup

Automatically deletes old numbered checkpoints, keeping only the 3 most recent. Prevents disk-full crashes.

### Environment Variables

All training features are controllable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `VT_AMP` | `1` | Enable mixed precision training |
| `VT_KEEP_CKPTS` | `3` | Number of checkpoints to keep (0 = disable cleanup) |
| `VT_VAL_SPLIT` | `0.1` | Validation split ratio (0 = disable validation) |
| `VT_PATIENCE` | `10` | Early stopping patience (0 = disable) |
| `VT_WARMUP_ITERS` | `1000` | LR warmup iterations |
| `VT_LR_MIN_RATIO` | `0.01` | Minimum LR as ratio of initial LR |
| `VT_GAN_WARMUP_ITERS` | `1000` | GAN loss warmup iterations |
| `VT_RESUME_OPTIMIZERS` | `0` | Resume optimizer states from checkpoint |
| `VT_RESUME_PROGRESS` | `0` | Resume epoch/iter counters from checkpoint |
| `VT_DISABLE_TB` | `0` | Disable TensorBoard logging |

Example with custom settings:

```bash
VT_KEEP_CKPTS=5 VT_PATIENCE=20 VT_AMP=1 bash models/tts/src/finetune.sh
```

## Export Artifacts (Important for Cloud Instances)

Before terminating your VM/instance, archive and download outputs:

```bash
cd /workspace/Voice_trainer
tar -czf export_voice_trainer_$(date +%Y%m%d_%H%M).tar.gz \
  models/tts/checkpoints \
  models/tts/src/config.yaml \
  models/tts/results \
  models/tts/logs
```

Then copy locally with `scp`:

```bash
scp -P <PORT> root@<IP>:/workspace/Voice_trainer/export_voice_trainer_*.tar.gz .
```

## Troubleshooting

### Hugging Face auth issues

- Use `hf auth login` (new CLI) in the same environment running setup.
- Confirm access to DODa on Hugging Face.

### GPU detected but not used

- Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

### `invalid phonemes` messages

- Some rows may be skipped by tokenizer/phonemizer logic.
- A small amount is expected; too many may reduce final quality.

### NaN/unstable training

- AMP handles most NaN/Inf gradient issues automatically now.
- If problems persist: `VT_AMP=0` to disable mixed precision.
- Reduce learning rates in `models/tts/src/config.yaml` (`g_lr`, `d_lr`).

### Disk space issues

- Auto-cleanup is now enabled by default (keeps 3 checkpoints).
- Adjust with `VT_KEEP_CKPTS=5` for more safety margin.

## Project Layout

```text
Voice_trainer/
├── scripts/
│   ├── setup_tts_doda.sh
│   └── download_format_doda.py
├── models/
│   └── tts/
│       ├── src/
│       │   ├── finetune.sh
│       │   ├── test-ckpts.sh
│       │   ├── demo.sh           ← NEW: Launch Gradio demo
│       │   ├── demo.py           ← NEW: Interactive web UI
│       │   ├── train_fp_adv.py   ← UPGRADED: AMP, LR sched, validation
│       │   ├── generate-config.py
│       │   ├── extract_f0_penn.py
│       │   ├── download_files.py
│       │   └── test_raw_model.py
│       └── tts-arabic-pytorch/
├── tools/dataset/
│   ├── merge-datasets.py
│   └── mp3-to-wav.sh
├── requirements.txt
└── pyproject.toml
```

## License

AGPL-3.0. See `LICENSE`.

## Credits

- [nipponjo/tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch)
- [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)

