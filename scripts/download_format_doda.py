import argparse
import csv
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm

# ── DODa speaker index map ────────────────────────────────────────────────────
# Source: https://huggingface.co/datasets/atlasia/DODa-audio-dataset
# 7 contributors (4 females, 3 males). Ranges are by original sample index.
SPEAKER_RANGES = {
    "F1": [(0, 999), (8000, 8999)],           # 2,000 samples
    "M3": [(1000, 1999)],                      # 1,000 samples
    "F2": [(2000, 2730)],                      #   731 samples
    "M1": [(2731, 2800), (4000, 4999),         # 4,462 samples  ← most data
           (6000, 6999), (10000, 10999),
           (11000, 11999), (12351, 12742)],
    "M2": [(2801, 2999), (3000, 3999),         # 2,550 samples
           (9000, 9999), (12000, 12350)],
    "F3": [(5000, 5999)],                      # 1,000 samples
    "F4": [(7000, 7999)],                      # 1,000 samples
}


def _index_matches_speaker(idx: int, speaker: str) -> bool:
    """Return True if sample index belongs to the given speaker."""
    for lo, hi in SPEAKER_RANGES[speaker]:
        if lo <= idx <= hi:
            return True
    return False


def main():
    valid_speakers = list(SPEAKER_RANGES.keys())
    parser = argparse.ArgumentParser(description="Download and format DODa dataset for TTS training")
    parser.add_argument("--token", type=str, default=True, help="Hugging Face access token (optional if logged in)")
    parser.add_argument("--output-dir", type=str, default="datasets/doda-dataset/doda", help="Output directory for the formatted dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")
    parser.add_argument("--speaker", type=str, default="M1",
                        choices=valid_speakers,
                        help=f"Keep only samples from this speaker. Choices: {valid_speakers}. Default: M1 (most data)")
    parser.add_argument("--all-speakers", action="store_true", default=False,
                        help="Disable speaker filtering; keep all speakers (multi-speaker)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audios_dir = output_dir / "audios"
    audios_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = output_dir / "data.csv"

    filter_speaker = None if args.all_speakers else args.speaker
    if filter_speaker:
        total_for_speaker = sum(hi - lo + 1 for lo, hi in SPEAKER_RANGES[filter_speaker])
        print(f"🎤 Speaker filter: {filter_speaker} (~{total_for_speaker:,} samples)")
    else:
        print("🎤 Speaker filter: DISABLED (all speakers)")

    print(f"Loading DODa dataset (split: {args.split})...")
    # Load dataset with authentication
    ds = load_dataset("atlasia/DODa-audio-dataset", split=args.split, token=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Formatting and saving to {output_dir}...")

    kept = 0
    dropped_no_caption = 0
    dropped_no_audio = 0
    dropped_wrong_speaker = 0

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["audio", "caption"])

        for idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
            # ── Speaker filter ─────────────────────────────────────────────
            if filter_speaker and not _index_matches_speaker(idx, filter_speaker):
                dropped_wrong_speaker += 1
                continue

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

    total = kept + dropped_no_caption + dropped_no_audio + dropped_wrong_speaker
    print(f"\n✅ DODa dataset formatting complete!")
    print(f"   Kept    : {kept:,} / {total:,} samples")
    if filter_speaker:
        print(f"   Speaker : {filter_speaker} only")
        print(f"   Dropped : {dropped_wrong_speaker:,} wrong speaker  |  {dropped_no_caption:,} missing caption  |  {dropped_no_audio:,} missing audio")
    else:
        print(f"   Dropped : {dropped_no_caption:,} missing Arabic caption  |  {dropped_no_audio:,} missing audio")
    print(f"   Dataset ready at: {output_dir}")
    print("   Review data.csv, then run: bash scripts/prepare-dataset.sh")

if __name__ == "__main__":
    main()
