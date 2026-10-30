#!/usr/bin/env python3
"""
run_pipeline.py

Master execution script for the "Algorithmic Silence" research pipeline.
Runs all core parsing, network modeling, statistical deference analysis,
and publication figure generation in sequence with full progress logging.

Usage:
    python run_pipeline.py
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Core pipeline execution stages
PIPELINE_STEPS = [
    ("1. Text Parsing & Corpus Ingestion", SCRIPTS_DIR / "parse_hamlet_main_chars.py"),
    ("2. Ophelia Voice & Silence Extraction", SCRIPTS_DIR / "extract_ophelia.py"),
    ("3. Character Network Science & Centrality", SCRIPTS_DIR / "character_network.py"),
    ("4. Speech Volume & Dialogue Share Plotting", SCRIPTS_DIR / "visualize_ophelia.py"),
    ("5. Conversational Deference & Turn Dynamics", SCRIPTS_DIR / "analyze_deference_and_agency.py")
]

EXPECTED_OUTPUTS = [
    DATA_PROCESSED_DIR / "hamlet_all_dialogue.csv",
    DATA_PROCESSED_DIR / "hamlet_main_characters.csv",
    DATA_PROCESSED_DIR / "hamlet_ophelia_lines.csv",
    TABLES_DIR / "hamlet_network_centrality.csv",
    FIGURES_DIR / "speech_volume_comparison.png",
    FIGURES_DIR / "turn_length_distribution.png",
    FIGURES_DIR / "deference_markers_comparison.png",
    FIGURES_DIR / "hamlet_character_network_clean_labels.png",
    FIGURES_DIR / "ophelia_ego_network.png"
]


def run_pipeline():
    print("=" * 70)
    print("ALGORITHMIC SILENCE: END-TO-END RESEARCH PIPELINE")
    print("Computational Analysis of Ophelia in Shakespeare's Hamlet")
    print("=" * 70)

    for title, script_path in PIPELINE_STEPS:
        print(f"\n[*] Executing: {title}")
        print(f"  Script: {script_path.relative_to(BASE_DIR)}")
        result = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR)
        if result.returncode != 0:
            print(f"[ERROR] Failed while executing {script_path.name}")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("VERIFYING PIPELINE ARTIFACTS...")
    print("=" * 70)

    missing = []
    for out in EXPECTED_OUTPUTS:
        rel_str = str(out.relative_to(BASE_DIR))
        if out.exists():
            size = out.stat().st_size
            print(f"  [OK] {rel_str:<50} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {rel_str}")
            missing.append(rel_str)

    if missing:
        print(f"\n[FAIL] Pipeline verification failed: {len(missing)} outputs missing.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All core research pipeline outputs successfully generated and verified.")


if __name__ == "__main__":
    run_pipeline()
