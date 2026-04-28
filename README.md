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
4. Test checkpoints periodically
5. Keep best checkpoints only

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

- Reduce learning rates in `models/tts/src/config.yaml` (`g_lr`, `d_lr`).
- Increase GAN warmup if your run uses adversarial warmup controls.
- Keep monitoring gradient norms and checkpoint quality.

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
│       │   ├── train_fp_adv.py
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
