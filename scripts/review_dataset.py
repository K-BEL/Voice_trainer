#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_dataset.py — Interactive terminal-based dataset quality checker.

Picks N random audio/caption pairs, plays each one, and asks you to
keep or remove it. At the end it removes all rejected rows from data.csv
(with an automatic backup first).

Usage:
    python scripts/review_dataset.py \
        --dataset-dir  datasets/doda-dataset/doda \
        --n            20

Controls (one key + Enter):
    Enter / k   → keep
    r           → remove
    q           → quit early (remaining rows kept as-is)

Requirements:
    - ffplay  (from ffmpeg) for audio playback
    OR
    - aplay   (from alsa-utils, Linux only)
    Install ffmpeg with: sudo apt install ffmpeg
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# RTL / Arabic display fix
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_pkg(pkg: str, import_name: str | None = None) -> bool:
    """Try to import a package; install it quietly if missing. Return success."""
    import_name = import_name or pkg
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass

    # Try standard install first, then --user (for macOS system Python)
    for extra in ([], ["--user"]):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg] + extra,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            __import__(import_name)
            return True
        except Exception:
            continue
    return False


_RTL_AVAILABLE: bool | None = None


def fix_rtl(text: str) -> str:
    """Reshape + reorder Arabic text so it renders correctly in dumb terminals."""
    global _RTL_AVAILABLE
    if _RTL_AVAILABLE is None:
        _RTL_AVAILABLE = (
            _ensure_pkg("arabic-reshaper", "arabic_reshaper")
            and _ensure_pkg("python-bidi", "bidi")
        )
    if not _RTL_AVAILABLE:
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


# ──────────────────────────────────────────────────────────────────────────────
# Audio playback
# ──────────────────────────────────────────────────────────────────────────────

def _find_player() -> str | None:
    """Return the name of an available command-line audio player, or None."""
    for cmd in ("ffplay", "aplay", "mpv", "play"):
        if shutil.which(cmd):
            return cmd
    return None


def play_audio(path: str, player: str) -> bool:
    """Play an audio file. Returns True on success, False if no audio hardware."""
    kwargs = dict(check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if player == "ffplay":
        r = subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], **kwargs
        )
    elif player == "aplay":
        r = subprocess.run(["aplay", "--quiet", path], **kwargs)
    elif player == "mpv":
        r = subprocess.run(["mpv", "--no-video", "--really-quiet", path], **kwargs)
    elif player == "play":  # sox
        r = subprocess.run(["play", "-q", path], **kwargs)
    else:
        return False
    # Return code 1 usually means no audio device on headless servers
    return r.returncode == 0


# ──────────────────────────────────────────────────────────────────────────────
# Core review loop
# ──────────────────────────────────────────────────────────────────────────────

def review_dataset(dataset_dir: str, n: int = 20, seed: int | None = None) -> None:
    dataset_dir = Path(dataset_dir).resolve()
    csv_path = dataset_dir / "data.csv"
    audios_dir = dataset_dir / "audios"

    if not csv_path.exists():
        print(f"❌  CSV not found: {csv_path}")
        sys.exit(1)
    if not audios_dir.is_dir():
        print(f"❌  Audios folder not found: {audios_dir}")
        sys.exit(1)

    player = _find_player()
    audio_available = player is not None
    if not audio_available:
        print("⚠️  No audio player found (ffplay / aplay / mpv / play).")
        print("   Install with: sudo apt install ffmpeg")
        print("   Continuing in caption-only mode.\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"\n📋  Dataset: {csv_path}")
    print(f"    Total samples : {total:,}")

    # ── Pick random sample ────────────────────────────────────────────────────
    rng = random.Random(seed)
    sample_size = min(n, total)
    sample_indices = rng.sample(range(total), sample_size)

    print(f"    Reviewing     : {sample_size} random samples")
    print("─" * 60)
    print("  Controls → Enter/k = keep | r = remove | q = quit")
    print("─" * 60)

    to_remove: set[int] = set()

    for review_num, row_idx in enumerate(sample_indices, start=1):
        row = rows[row_idx]
        audio_filename = row["audio"]
        caption = row["caption"]
        audio_path = audios_dir / audio_filename

        print(f"\n[{review_num}/{sample_size}]  {audio_filename}")
        print(f"  Caption : {fix_rtl(caption)}")

        # Play audio
        if audio_available and audio_path.exists():
            print(f"  ▶ Playing ({player})…", end="", flush=True)
            ok = play_audio(str(audio_path), player)
            if ok:
                print(" done")
            else:
                print(" ⚠️  no audio hardware (headless server) — caption-only mode")
                audio_available = False  # don't try again for remaining samples
        elif not audio_path.exists():
            print("  ⚠️  Audio file not found, skipping playback.")

        # Decision prompt
        while True:
            try:
                answer = input("  Keep or Remove? [Enter/k/r/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "q"

            if answer in ("", "k"):
                print("  ✅ Kept")
                break
            elif answer == "r":
                to_remove.add(row_idx)
                print("  🗑️  Marked for removal")
                break
            elif answer == "q":
                print("\n⏹  Review stopped early. Remaining samples kept.")
                break
            else:
                print("  Please enter k (keep), r (remove), or q (quit).")
        else:
            continue
        if answer == "q":
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Reviewed : {review_num} / {sample_size}")
    print(f"  To keep  : {review_num - len(to_remove)}")
    print(f"  To remove: {len(to_remove)}")

    if not to_remove:
        print("\n✅  Nothing to remove. data.csv is unchanged.")
        return

    # ── Confirm ───────────────────────────────────────────────────────────────
    try:
        confirm = input(f"\nPermanently remove {len(to_remove)} row(s) from data.csv? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "n"

    if confirm != "y":
        print("Aborted — data.csv unchanged.")
        return

    # ── Backup ────────────────────────────────────────────────────────────────
    backup_path = csv_path.with_suffix(".csv.bak")
    shutil.copy(csv_path, backup_path)
    print(f"📂  Backup saved → {backup_path}")

    # ── Write cleaned CSV ─────────────────────────────────────────────────────
    cleaned_rows = [row for i, row in enumerate(rows) if i not in to_remove]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "caption"])
        writer.writeheader()
        writer.writerows(cleaned_rows)

    removed_count = total - len(cleaned_rows)
    print(f"✅  Removed {removed_count} row(s). "
          f"Dataset now has {len(cleaned_rows):,} samples.")
    print(f"   data.csv → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively review N random audio/caption pairs from a dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/doda-dataset/doda",
        help="Directory containing data.csv and audios/",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of random samples to review (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()
    review_dataset(args.dataset_dir, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
