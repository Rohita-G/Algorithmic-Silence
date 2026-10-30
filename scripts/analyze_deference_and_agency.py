#!/usr/bin/env python3
"""
analyze_deference_and_agency.py

Analyzes conversational turn length distributions, brevity compression,
and honorific address formulas ("my lord") across principal characters in Hamlet.
Generates publication-ready figures:
  - results/figures/deference_markers_comparison.png
  - results/figures/turn_length_distribution.png
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_PROCESSED_DIR / "hamlet_main_characters.csv"
if not INPUT_CSV.exists() and (BASE_DIR / "data" / "hamlet_main_characters.csv").exists():
    INPUT_CSV = BASE_DIR / "data" / "hamlet_main_characters.csv"

MAIN_CHARACTERS = [
    "HAMLET",
    "KING CLAUDIUS",
    "QUEEN GERTRUDE",
    "POLONIUS",
    "OPHELIA",
    "LAERTES",
    "HORATIO"
]

CHARACTER_PALETTE = {
    "OPHELIA": "#D95F02",
    "HAMLET": "#7570B3",
    "KING CLAUDIUS": "#E7298A",
    "QUEEN GERTRUDE": "#E6AB02",
    "POLONIUS": "#66A61E",
    "LAERTES": "#A6761D",
    "HORATIO": "#1F78B4"
}


def analyze_deference():
    print("=" * 65)
    print("ANALYZING DEFERENCE & CONVERSATIONAL TURN DYNAMICS")
    print("=" * 65)

    if not INPUT_CSV.exists():
        from parse_hamlet_main_chars import main as parse_main
        parse_main()

    df = pd.read_csv(INPUT_CSV)
    df = df[df["character"].isin(MAIN_CHARACTERS)].copy()

    # Exact 'my lord' occurrences
    df["exact_my_lord"] = df["line"].apply(
        lambda x: len(re.findall(r"\bmy\s+lord\b", str(x), re.IGNORECASE))
    )

    deference_summary = df.groupby("character").agg(
        total_words=("word_count", "sum"),
        total_turns=("turn_id", "count"),
        exact_my_lord=("exact_my_lord", "sum"),
        turns_with_deference=("exact_my_lord", lambda s: (s > 0).sum())
    ).reset_index()

    deference_summary["rate_per_1k_words"] = (
        deference_summary["exact_my_lord"] / deference_summary["total_words"]
    ) * 1000

    deference_summary["turn_prevalence_pct"] = (
        deference_summary["turns_with_deference"] / deference_summary["total_turns"]
    ) * 100

    deference_summary = deference_summary.sort_values(by="rate_per_1k_words", ascending=False)

    print("\n--- Honorific Address Rates ---")
    print(deference_summary.to_string(index=False, formatters={
        "total_words": "{:,}".format,
        "rate_per_1k_words": "{:.2f}".format,
        "turn_prevalence_pct": "{:.1f}%".format
    }))

    # -------------------------------------------------------------
    # Figure 1: Deference Markers Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    colors = [CHARACTER_PALETTE.get(c, "#888888") for c in deference_summary["character"]]
    bars = plt.bar(deference_summary["character"], deference_summary["rate_per_1k_words"], color=colors, edgecolor="#222222", linewidth=0.8)

    for bar, rate in zip(bars, deference_summary["rate_per_1k_words"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{rate:.1f}", ha="center", fontsize=9.5, fontweight="bold"
        )

    plt.title("Frequency of 'My Lord' Address Formula (per 1,000 Words)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Occurrences per 1,000 Spoken Words", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=9.5, fontweight="semibold")
    plt.ylim(0, max(deference_summary["rate_per_1k_words"]) * 1.18)
    plt.tight_layout()

    fig1_path = FIGURES_DIR / "deference_markers_comparison.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Turn Length Distribution Across Characters
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    sns.boxplot(
        data=df,
        x="character",
        y="word_count",
        hue="character",
        order=MAIN_CHARACTERS,
        palette=CHARACTER_PALETTE,
        legend=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "6"}
    )
    plt.yscale("log")
    plt.title("Conversational Turn Length Distribution in Hamlet (Log Scale)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Character", fontsize=11, fontweight="bold")
    plt.ylabel("Words per Speech Turn (Log Scale)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=9.5, fontweight="semibold")
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "turn_length_distribution.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure: {fig2_path}")


if __name__ == "__main__":
    analyze_deference()
