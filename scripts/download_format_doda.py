import argparse
import csv
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download and format DODa dataset for TTS training")
    parser.add_argument("--token", type=str, default=True, help="Hugging Face access token (optional if logged in)")
    parser.add_argument("--output-dir", type=str, default="datasets/doda-dataset/doda", help="Output directory for the formatted dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audios_dir = output_dir / "audios"
    audios_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = output_dir / "data.csv"

    print(f"Loading DODa dataset (split: {args.split})...")
    # Load dataset with authentication
    ds = load_dataset("atlasia/DODa-audio-dataset", split=args.split, token=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Formatting and saving to {output_dir}...")

    kept = 0
    dropped_no_caption = 0
    dropped_no_audio = 0

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["audio", "caption"])

        for idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
            # ── Caption guard ──────────────────────────────────────────────
            # ONLY use the Arabic script column. If it is missing or empty,
            # drop the sample entirely. Do NOT fall back to Latin (darija_Ltn)
            # because the Arabic phonemiser will mangle it into garbage phonemes.
            caption = (sample.get("darija_Arab_new") or "").strip()
            if not caption:
                dropped_no_caption += 1
                continue

            # ── Audio guard ────────────────────────────────────────────────
            audio_dict = sample.get("audio") or {}
            audio_bytes = audio_dict.get("bytes")
            if not audio_bytes:
                dropped_no_audio += 1
                continue

            audio_path_orig = audio_dict.get("path", "")
            ext = Path(audio_path_orig).suffix if audio_path_orig else ".wav"
            if not ext:
                ext = ".wav"

            audio_filename = f"doda_sample_{idx:06d}{ext}"
            audio_path = audios_dir / audio_filename

            # Save raw audio file directly
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Write to CSV
            writer.writerow([audio_filename, caption])
            kept += 1

    total = kept + dropped_no_caption + dropped_no_audio
    print(f"\n✅ DODa dataset formatting complete!")
    print(f"   Kept    : {kept:,} / {total:,} samples")
    print(f"   Dropped : {dropped_no_caption:,} missing Arabic caption  |  {dropped_no_audio:,} missing audio")
    print(f"   Dataset ready at: {output_dir}")
    print("   Review data.csv, then run: bash scripts/prepare-dataset.sh")

if __name__ == "__main__":
    main()
