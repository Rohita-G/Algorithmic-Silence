#!/usr/bin/env python3
"""
visualize_ophelia.py

Generates publication-grade figures for speech volume, dialogue share,
and scene-level vocal presence across characters in Hamlet:
  - results/figures/speech_volume_comparison.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MAIN_CSV = DATA_PROCESSED_DIR / "hamlet_main_characters.csv"
OPHELIA_CSV = DATA_PROCESSED_DIR / "hamlet_ophelia_lines.csv"

if not MAIN_CSV.exists() and (BASE_DIR / "data" / "hamlet_main_characters.csv").exists():
    MAIN_CSV = BASE_DIR / "data" / "hamlet_main_characters.csv"
if not OPHELIA_CSV.exists() and (BASE_DIR / "data" / "hamlet_ophelia_lines.csv").exists():
    OPHELIA_CSV = BASE_DIR / "data" / "hamlet_ophelia_lines.csv"

# Stylistic settings
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]


def visualize_ophelia():
    print("=" * 65)
    print("GENERATING OPHELIA SPEECH VOLUME & SILENCE VISUALIZATIONS")
    print("=" * 65)

    if not OPHELIA_CSV.exists() or not MAIN_CSV.exists():
        from parse_hamlet_main_chars import main as parse_main
        parse_main()

    df_ophelia = pd.read_csv(OPHELIA_CSV)
    df_main = pd.read_csv(MAIN_CSV)

    # -------------------------------------------------------------
    # Figure: Spoken Dialogue Volume & Percentage Share
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    char_summary = df_main.groupby("character")["word_count"].sum().reset_index()
    total_words = char_summary["word_count"].sum()
    char_summary["pct"] = (char_summary["word_count"] / total_words) * 100
    char_summary = char_summary.sort_values(by="word_count", ascending=True)

    colors = ["#D95F02" if c == "OPHELIA" else "#7570B3" if c == "HAMLET" else "#888888" for c in char_summary["character"]]

    bars = plt.barh(char_summary["character"], char_summary["word_count"], color=colors, edgecolor="#222222", linewidth=0.8)

    for bar, pct, count in zip(bars, char_summary["pct"], char_summary["word_count"]):
        plt.text(
            bar.get_width() + 150, bar.get_y() + bar.get_height() / 2,
            f"{count:,} w  ({pct:.1f}%)",
            va="center", fontsize=9.5, fontweight="semibold", color="#222222"
        )

    plt.title("Comparative Spoken Dialogue Volume in Hamlet (7 Principal Roles)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Spoken Dialogue Words (Clean Count)", fontsize=11, fontweight="bold")
    plt.ylabel("Character", fontsize=11, fontweight="bold")
    plt.xlim(0, max(char_summary["word_count"]) * 1.22)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "speech_volume_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure: {fig_path}")


if __name__ == "__main__":
    visualize_ophelia()
