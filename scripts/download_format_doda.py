import argparse
import csv
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download and format DODa dataset for darija-chatbot")
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
    
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["audio", "caption"])
        
        for idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
            audio_dict = sample["audio"]
            audio_bytes = audio_dict["bytes"]
            audio_path_orig = audio_dict.get("path", "")
            ext = Path(audio_path_orig).suffix if audio_path_orig else ".wav"
            if not ext:
                ext = ".wav"
            
            # Using darija_Arab_new as the main transcription target
            caption = sample.get("darija_Arab_new", "")
            if not caption:
                caption = sample.get("darija_Ltn", "")
            
            audio_filename = f"doda_sample_{idx:06d}{ext}"
            audio_path = audios_dir / audio_filename
            
            # Save raw audio file directly
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Write to CSV
            writer.writerow([audio_filename, caption])

    print("✅ DODa dataset formatting complete!")
    print(f"Dataset ready at: {output_dir}")
    print("You can now run the appropriate 'prepare-data.sh' script.")

if __name__ == "__main__":
    main()
