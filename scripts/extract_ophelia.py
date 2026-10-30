#!/usr/bin/env python3
"""
extract_ophelia.py

Extracts and profiles Ophelia's dialogue from Hamlet.
Computes act-level, scene-level, and conversational metrics of her voice and silence:
  - Total words and percentage of dialogue
  - Turn counts and average/median turn length per act
  - Scene presence vs. absence across all 20 scenes of Hamlet
  - Conversational partners (preceding and succeeding speakers)
"""

from pathlib import Path
import pandas as pd

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_PROCESSED_DIR / "hamlet_main_characters.csv"
OUTPUT_CSV = DATA_PROCESSED_DIR / "hamlet_ophelia_lines.csv"

# Fallback to legacy path if processed directory not yet populated
if not INPUT_CSV.exists() and (BASE_DIR / "data" / "hamlet_main_characters.csv").exists():
    INPUT_CSV = BASE_DIR / "data" / "hamlet_main_characters.csv"


def extract_ophelia():
    if not INPUT_CSV.exists():
        from parse_hamlet_main_chars import main as parse_main
        parse_main()

    df_main = pd.read_csv(INPUT_CSV)
    df_ophelia = df_main[df_main["character"] == "OPHELIA"].copy()
    df_ophelia.to_csv(OUTPUT_CSV, index=False)

    total_main_words = df_main["word_count"].sum()
    ophelia_words = df_ophelia["word_count"].sum()
    ophelia_turns = len(df_ophelia)

    print("=" * 60)
    print("OPHELIA: COMPUTATIONAL VOICE & SILENCE PROFILE")
    print("=" * 60)
    print(f"Total speech turns: {ophelia_turns}")
    print(f"Total words spoken: {ophelia_words:,}")
    print(f"Share of main character dialogue: {(ophelia_words / total_main_words) * 100:.2f}%")
    print(f"Overall average words per speech turn: {ophelia_words / ophelia_turns:.1f} words")

    print("\n--- Act-by-Act Breakdown ---")
    act_breakdown = df_ophelia.groupby("act")["word_count"].agg(
        turns="count",
        total_words="sum",
        mean_words="mean",
        median_words="median"
    ).reset_index()

    act_breakdown["share_of_ophelia_total"] = (act_breakdown["total_words"] / ophelia_words) * 100
    print(act_breakdown.to_string(index=False, formatters={
        "total_words": "{:,}".format,
        "mean_words": "{:.1f}".format,
        "median_words": "{:.1f}".format,
        "share_of_ophelia_total": "{:.1f}%".format
    }))

    print("\n--- Scene Presence Summary (Canonical 20 Scenes) ---")
    scenes_present = df_ophelia[["act", "scene"]].drop_duplicates().sort_values(by=["act", "scene"])
    print(f"Ophelia speaks in {len(scenes_present)} of 20 scenes:")
    for _, row in scenes_present.iterrows():
        subset = df_ophelia[(df_ophelia["act"] == row["act"]) & (df_ophelia["scene"] == row["scene"])]
        print(f"  - Act {row['act']}, Scene {row['scene']}: {len(subset)} turns, {subset['word_count'].sum()} words")

    print("\n--- Conversational Interaction Partners ---")
    print("Immediate preceding speakers (who Ophelia responds to):")
    print(df_ophelia["prev_character"].value_counts().to_string())

    print("\nImmediate succeeding speakers (who responds to Ophelia):")
    print(df_ophelia["next_character"].value_counts().to_string())

    print(f"\n[OK] Ophelia profile dataset updated: {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_ophelia()
