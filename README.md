# Voice Trainer

A TTS (Text-to-Speech) training pipeline for Moroccan Darija, using the [DODa dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset).  
Built on [FastPitch + HiFi-GAN](https://github.com/nipponjo/tts-arabic-pytorch).

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (NVIDIA)
- `ffmpeg` — `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- A [HuggingFace account](https://huggingface.co/join) with access to the DODa dataset

---

## 1. Download the Project

```bash
git clone --recurse-submodules https://github.com/K-BEL/Voice_trainer.git
cd Voice_trainer
```

> The `--recurse-submodules` flag is important — it pulls the `tts-arabic-pytorch` dependency.

---

## 2. Setup (One Command)

The setup script installs everything and prepares the dataset:

```bash
bash scripts/setup_tts_doda.sh
```

This will:
1. Install Python dependencies (PyTorch, librosa, penn, etc.)
2. Download the DODa dataset from HuggingFace
3. Merge and convert audio files to WAV
4. Download pretrained FastPitch + HiFi-GAN weights
5. Extract F0 pitch features from the audio (GPU accelerated)
6. Generate the training config (`models/tts/src/config.yaml`)

### Setup Options

```bash
# Already have the dataset downloaded? Skip it:
bash scripts/setup_tts_doda.sh --skip-download

# Already have pretrained weights? Skip them:
bash scripts/setup_tts_doda.sh --skip-pretrained

# Change training parameters:
bash scripts/setup_tts_doda.sh --epochs 200 --save-interval 1000
```

---

## 3. Train

Once setup is done, start training:

```bash
bash models/tts/src/finetune.sh
```

- Checkpoints are saved to `models/tts/checkpoints/`
- Logs are written for TensorBoard monitoring
- Training resumes automatically if a checkpoint exists

---

## 4. Test

Generate test audio from your trained checkpoints:

```bash
bash models/tts/src/test-ckpts.sh models/tts/checkpoints
```

This runs inference on a set of Darija test phrases and saves the output `.wav` files to `models/tts/results/<checkpoint_name>/`.

---

## Project Structure

```
Voice_trainer/
├── scripts/
│   ├── setup_tts_doda.sh        # One-command setup
│   └── download_format_doda.py  # DODa dataset downloader
├── models/
│   ├── tts/
│   │   ├── src/
│   │   │   ├── finetune.sh          # Start training
│   │   │   ├── test-ckpts.sh        # Test checkpoints
│   │   │   ├── train_fp_adv.py      # Training loop
│   │   │   ├── generate-config.py   # Config generator
│   │   │   ├── extract_f0_penn.py   # Pitch extraction
│   │   │   ├── download_files.py    # Pretrained weight downloader
│   │   │   └── test_raw_model.py    # Inference script
│   │   └── tts-arabic-pytorch/      # FastPitch+HiFiGAN (submodule)
│   └── scripts/
│       └── setup-tts.sh
├── tools/dataset/
│   ├── merge-datasets.py        # Merge multiple datasets
│   └── mp3-to-wav.sh            # Audio conversion
├── requirements.txt
└── pyproject.toml
```

---

## License

AGPL-3.0 — see [LICENSE](./LICENSE).

## Credits

- [nipponjo/tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch)
- [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)
