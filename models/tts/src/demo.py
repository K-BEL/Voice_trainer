"""Interactive Gradio demo for testing Voice Trainer checkpoints.

Launch with:
    bash models/tts/src/demo.sh [checkpoints_dir]
"""

import argparse
import os
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from models.fastpitch import FastPitch2Wave


def load_bigvgan(model_name="nvidia/bigvgan_v2_22khz_80band_256x", use_cuda=True):
    """Load a pretrained BigVGAN model from Hugging Face."""
    try:
        import bigvgan
    except ImportError:
        print("Installing bigvgan...")  # noqa: T201
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bigvgan"])
        import bigvgan

    from huggingface_hub import hf_hub_download
    import json

    # Download config and weights manually
    config_file = hf_hub_download(repo_id=model_name, filename="config.json")
    model_file = hf_hub_download(repo_id=model_name, filename="bigvgan_generator.pth")
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Instantiate class manually
    model = bigvgan.BigVGAN(**config)
    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict["model"])
    
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    
    model.remove_weight_norm()
    model.eval()
    return model


def find_checkpoints(ckpt_dir: str) -> list[str]:
    """Discover all .pth checkpoint files in a directory."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return []
    files = sorted(ckpt_path.glob("*.pth"), key=os.path.getmtime, reverse=True)
    return [f.name for f in files]


def load_model(ckpt_dir: str, ckpt_name: str, use_cuda: bool = True):
    """Load a FastPitch2Wave model from a checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model = FastPitch2Wave(ckpt_path)
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    return model


# Global model cache to avoid reloading on every request
_cached_model = None
_cached_ckpt_name = None
_cached_vocoder = None
_cached_vocoder_type = None


import re


def synthesize(text: str, ckpt_name: str, speaker_id: int, pace: float, ckpt_dir: str, vocoder_type: str):
    """Generate speech from text using the selected checkpoint and vocoder."""
    global _cached_model, _cached_ckpt_name, _cached_vocoder, _cached_vocoder_type  # noqa: PLW0603

    if not text.strip():
        return None

    # Load or reuse FastPitch model
    if _cached_ckpt_name != ckpt_name or _cached_model is None:
        _cached_model = load_model(ckpt_dir, ckpt_name)
        _cached_ckpt_name = ckpt_name

    use_cuda = next(_cached_model.parameters()).is_cuda

    # Load or reuse Vocoder
    if _cached_vocoder_type != vocoder_type or _cached_vocoder is None:
        if vocoder_type == "BigVGAN (v2 Universal)":
            _cached_vocoder = load_bigvgan(use_cuda=use_cuda)
        else:
            _cached_vocoder = None  # Use bundled HiFi-GAN
        _cached_vocoder_type = vocoder_type

    # Strip punctuation to prevent KeyError in the phonemizer
    clean_text = re.sub(r'[^\w\s]', '', text)

    # Generate waveform
    if _cached_vocoder_type == "BigVGAN (v2 Universal)":
        # Get mel from FastPitch, then pass to BigVGAN
        mel = _cached_model.model.ttmel(clean_text, speed=pace, speaker_id=speaker_id)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        with torch.inference_mode():
            wave = _cached_vocoder(mel)
        wave = wave[0].cpu()
    else:
        # Use default FastPitch2Wave path (HiFi-GAN)
        try:
            wave = _cached_model.tts(clean_text, speaker_id=speaker_id, speed=pace, phonemize=False)
        except TypeError:
            wave = _cached_model.tts(clean_text, speaker_id=speaker_id, speed=pace)
        wave = wave.unsqueeze(0).cpu()

    # Save to temp file
    out_path = "/tmp/vt_demo_output.wav"  # noqa: S108
    torchaudio.save(out_path, wave, 22050)
    return out_path


def build_ui(ckpt_dir: str):
    """Build and return the Gradio interface."""
    checkpoints = find_checkpoints(ckpt_dir)

    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")  # noqa: T201
        return None

    print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}")  # noqa: T201

    with gr.Blocks(
        title="Voice Trainer — Darija TTS Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎙️ Voice Trainer — Darija TTS Demo")
        gr.Markdown("Test your fine-tuned checkpoints interactively.")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text (Darija / Arabic)",
                    placeholder="اكتب شي حاجة هنا...",
                    lines=3,
                    value="السلام عليكم صاحبي",
                )
                ckpt_dropdown = gr.Dropdown(
                    label="Checkpoint",
                    choices=checkpoints,
                    value=checkpoints[0],
                )
                with gr.Row():
                    speaker_id = gr.Slider(
                        label="Speaker ID",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                    )
                    pace = gr.Slider(
                        label="Pace",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                    )
                vocoder_dropdown = gr.Dropdown(
                    label="Vocoder (Audio Engine)",
                    choices=["HiFi-GAN (Original)", "BigVGAN (v2 Universal)"],
                    value="HiFi-GAN (Original)",
                )
                generate_btn = gr.Button("🔊 Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Audio", type="filepath")

        # Refresh button for new checkpoints
        refresh_btn = gr.Button("🔄 Refresh Checkpoints", size="sm")

        def refresh_ckpts():
            new_ckpts = find_checkpoints(ckpt_dir)
            return gr.update(choices=new_ckpts, value=new_ckpts[0] if new_ckpts else None)

        refresh_btn.click(fn=refresh_ckpts, outputs=[ckpt_dropdown])

        generate_btn.click(
            fn=lambda text, ckpt, sid, p, voc: synthesize(text, ckpt, int(sid), p, ckpt_dir, voc),
            inputs=[text_input, ckpt_dropdown, speaker_id, pace, vocoder_dropdown],
            outputs=[audio_output],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Trainer Gradio Demo")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints/",
        help="Path to checkpoints directory",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = build_ui(args.ckpt_dir)
    if demo is not None:
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
    else:
        print("No checkpoints found. Train a model first!")  # noqa: T201
