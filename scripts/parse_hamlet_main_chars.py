#!/usr/bin/env python3
"""
parse_hamlet_main_chars.py

Scholarly parser for William Shakespeare's Hamlet (Folger Shakespeare Library text).
Correctly parses dramatic structure:
  - Eliminates preamble / editorial credits / dramatis personae before ACT 1
  - Accurately tracks Acts (1-5) and Scenes (1-20)
  - Identifies speech turns for ALL speakers without swallowing secondary characters
  - Maps character aliases to canonical names
  - Tracks conversational turn-taking adjacency (preceding and succeeding speaker)
  - Computes word counts, character counts, question flags, and deference markers
  - Generates:
      1) data/processed/hamlet_all_dialogue.csv (complete play dialogue, all characters)
      2) data/processed/hamlet_main_characters.csv (dialogue for 7 main characters)
      3) data/processed/hamlet_ophelia_lines.csv (Ophelia-specific dialogue with context)
"""

import re
import sys
from pathlib import Path
import pandas as pd

# Determine root repository directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Primary raw corpus path (with fallback to data/hamlet.txt if needed)
HAMLET_PATH = DATA_RAW_DIR / "hamlet.txt"
if not HAMLET_PATH.exists() and (BASE_DIR / "data" / "hamlet.txt").exists():
    HAMLET_PATH = BASE_DIR / "data" / "hamlet.txt"

ALL_DIALOGUE_CSV = DATA_PROCESSED_DIR / "hamlet_all_dialogue.csv"
MAIN_CHARS_CSV = DATA_PROCESSED_DIR / "hamlet_main_characters.csv"
OPHELIA_CSV = DATA_PROCESSED_DIR / "hamlet_ophelia_lines.csv"

# Canonical character mapping
SPEAKER_MAP = {
    "HAMLET": "HAMLET", "PRINCE": "HAMLET",
    "KING": "KING CLAUDIUS", "KING CLAUDIUS": "KING CLAUDIUS", "CLAUDIUS": "KING CLAUDIUS",
    "QUEEN": "QUEEN GERTRUDE", "QUEEN GERTRUDE": "QUEEN GERTRUDE", "GERTRUDE": "QUEEN GERTRUDE",
    "POLONIUS": "POLONIUS", "OPHELIA": "OPHELIA", "OPH.": "OPHELIA",
    "LAERTES": "LAERTES", "HORATIO": "HORATIO",
    "ROSENCRANTZ": "ROSENCRANTZ", "GUILDENSTERN": "GUILDENSTERN",
    "MARCELLUS": "MARCELLUS", "BARNARDO": "BARNARDO", "FRANCISCO": "FRANCISCO",
    "GHOST": "GHOST", "GRAVEDIGGER": "GRAVEDIGGER", "FIRST CLOWN": "GRAVEDIGGER", "CLOWN": "GRAVEDIGGER",
    "OTHER": "GRAVEDIGGER COMPANION", "SECOND CLOWN": "GRAVEDIGGER COMPANION",
    "OSRIC": "OSRIC", "REYNALDO": "REYNALDO", "FORTINBRAS": "FORTINBRAS", "CAPTAIN": "CAPTAIN",
    "FIRST PLAYER": "FIRST PLAYER", "PLAYER": "FIRST PLAYER",
    "PLAYER KING": "PLAYER KING", "PLAYER QUEEN": "PLAYER QUEEN",
    "LUCIANUS": "LUCIANUS", "PROLOGUE": "PROLOGUE",
    "GENTLEMAN": "GENTLEMAN", "GENTLEMEN": "GENTLEMEN",
    "MESSENGER": "MESSENGER", "SAILOR": "SAILOR", "DOCTOR": "DOCTOR",
    "VOLTEMAND": "VOLTEMAND", "CORNELIUS": "CORNELIUS",
    "ALL": "ALL", "LORD": "LORD", "AMBASSADOR": "AMBASSADOR"
}

MAIN_CHARACTERS = [
    "HAMLET",
    "KING CLAUDIUS",
    "QUEEN GERTRUDE",
    "POLONIUS",
    "OPHELIA",
    "LAERTES",
    "HORATIO"
]


def parse_play(filepath: Path = HAMLET_PATH) -> pd.DataFrame:
    """Parses Folger Shakespeare Hamlet into verified dialogue turns."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing corpus file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    start_pos = full_text.find("ACT 1")
    if start_pos == -1:
        start_pos = 0

    play_text = full_text[start_pos:]
    clean_play_text = re.sub(r"\[[\s\S]*?\]", "", play_text)
    play_lines = clean_play_text.splitlines()

    spk_re = re.compile(
        r"^([A-Z][A-Z0-9 \'\.\-]+?)"             # Speaker name in uppercase
        r"(?:,)?"                                # Optional comma
        r"(?:\s{2,}(.*)|\s*)$"                   # Either 2+ spaces followed by dialogue, OR end of line
    )

    records = []
    turn_id = 0
    current_act = 1
    current_scene = 1
    current_speaker = None
    current_dialogue = []

    def flush():
        nonlocal turn_id, current_speaker, current_dialogue
        if current_speaker and current_dialogue:
            text = " ".join(current_dialogue).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                turn_id += 1
                words = text.split()
                canonical = SPEAKER_MAP.get(current_speaker, current_speaker)
                records.append({
                    "turn_id": turn_id,
                    "act": current_act,
                    "scene": current_scene,
                    "character_raw": current_speaker,
                    "character": canonical,
                    "line": text,
                    "word_count": len(words),
                    "char_count": len(text)
                })
        current_speaker = None
        current_dialogue = []

    for raw_line in play_lines:
        line = raw_line.strip()
        if not line:
            continue
        m_act = re.match(r"^ACT\s+(\d+)", line, re.IGNORECASE)
        if m_act:
            flush()
            current_act = int(m_act.group(1))
            current_scene = 1
            continue
        m_scene = re.match(r"^Scene\s+(\d+)", line, re.IGNORECASE)
        if m_scene:
            flush()
            current_scene = int(m_scene.group(1))
            continue
        if line.startswith("="):
            continue

        m_spk = spk_re.match(raw_line)
        if m_spk:
            cand = m_spk.group(1).strip()
            alpha = re.sub(r"[^A-Z]", "", cand)
            if alpha.isupper() and cand not in ["ACT", "SCENE", "SCENE 1"]:
                flush()
                current_speaker = cand
                d_part = m_spk.group(2) if m_spk.group(2) else ""
                d_clean = d_part.strip()
                if d_clean:
                    current_dialogue.append(d_clean)
                continue

        if line and current_speaker:
            current_dialogue.append(line)

    flush()
    df = pd.DataFrame(records)

    # Sequential speaker adjacency
    df["prev_character"] = df["character"].shift(1)
    df["next_character"] = df["character"].shift(-1)

    # Reset adjacency boundaries across scenes (must be None/NaN across scene boundaries)
    scene_change_prev = (df["act"] != df["act"].shift(1)) | (df["scene"] != df["scene"].shift(1))
    df.loc[scene_change_prev, "prev_character"] = None

    scene_change_next = (df["act"] != df["act"].shift(-1)) | (df["scene"] != df["scene"].shift(-1))
    df.loc[scene_change_next, "next_character"] = None

    # Conversational markers
    df["has_question"] = df["line"].apply(lambda x: 1 if "?" in str(x) else 0)
    df["has_exclamation"] = df["line"].apply(lambda x: 1 if "!" in str(x) else 0)
    df["deference_count"] = df["line"].apply(
        lambda x: len(re.findall(r"\b(my\s+lord|your\s+lordship|lordship)\b", str(x), re.IGNORECASE))
    )

    # Ensure required column order
    cols = [
        "turn_id", "act", "scene", "character_raw", "character", "line",
        "word_count", "char_count", "prev_character", "next_character",
        "has_question", "has_exclamation", "deference_count"
    ]
    df = df[cols]
    return df


def main():
    print("=" * 65)
    print("PARSING SHAKESPEARE'S HAMLET (FOLGER DIGITAL TEXTS)")
    print("=" * 65)
    print(f"Reading from: {HAMLET_PATH}")

    df_all = parse_play(HAMLET_PATH)
    print(f"[OK] Total discrete dialogue turns: {len(df_all):,}")
    print(f"[OK] Total spoken words across play: {df_all['word_count'].sum():,}")
    print(f"[OK] Distinct speaking roles: {df_all['character'].nunique():,}")

    # Export complete dialogue dataset to data/processed/
    df_all.to_csv(ALL_DIALOGUE_CSV, index=False)
    print(f"[OK] Saved complete play dialogue to: {ALL_DIALOGUE_CSV}")

    # Export main 7 characters subset
    df_main = df_all[df_all["character"].isin(MAIN_CHARACTERS)].copy()
    df_main.to_csv(MAIN_CHARS_CSV, index=False)
    print(f"[OK] Saved main 7 characters dialogue ({len(df_main)} turns, {df_main['word_count'].sum():,} words) to: {MAIN_CHARS_CSV}")

    # Export Ophelia subset
    df_ophelia = df_all[df_all["character"] == "OPHELIA"].copy()
    df_ophelia.to_csv(OPHELIA_CSV, index=False)
    print(f"[OK] Saved Ophelia dialogue ({len(df_ophelia)} turns, {df_ophelia['word_count'].sum():,} words) to: {OPHELIA_CSV}")

    print("\n--- Speech Share Among 7 Principal Characters ---")
    summary = df_main.groupby("character")["word_count"].agg(
        turns="count",
        total_words="sum",
        mean_turn_length="mean",
        median_turn_length="median"
    ).reset_index()

    total_words_main = df_main["word_count"].sum()
    summary["percentage_share"] = (summary["total_words"] / total_words_main) * 100
    summary = summary.sort_values(by="total_words", ascending=False)

    print(summary.to_string(index=False, formatters={
        "total_words": "{:,}".format,
        "mean_turn_length": "{:.1f}".format,
        "median_turn_length": "{:.1f}".format,
        "percentage_share": "{:.2f}%".format
    }))


if __name__ == "__main__":
    main()
