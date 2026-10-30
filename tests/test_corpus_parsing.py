#!/usr/bin/env python3
"""
test_corpus_parsing.py

Unit and regression tests verifying the integrity of text parsing,
speaker attribution, turn-taking adjacency, and scene segmentation
for the Folger Shakespeare Library edition of Hamlet.
"""

from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

HAMLET_TXT = DATA_DIR / "raw" / "hamlet.txt"
if not HAMLET_TXT.exists():
    HAMLET_TXT = DATA_DIR / "hamlet.txt"

ALL_DIALOGUE_CSV = DATA_DIR / "processed" / "hamlet_all_dialogue.csv"
if not ALL_DIALOGUE_CSV.exists():
    ALL_DIALOGUE_CSV = DATA_DIR / "hamlet_all_dialogue.csv"

MAIN_CHARS_CSV = DATA_DIR / "processed" / "hamlet_main_characters.csv"
if not MAIN_CHARS_CSV.exists():
    MAIN_CHARS_CSV = DATA_DIR / "hamlet_main_characters.csv"

OPHELIA_CSV = DATA_DIR / "processed" / "hamlet_ophelia_lines.csv"
if not OPHELIA_CSV.exists():
    OPHELIA_CSV = DATA_DIR / "hamlet_ophelia_lines.csv"

CANONICAL_MAIN_CHARACTERS = {
    "HAMLET",
    "KING CLAUDIUS",
    "POLONIUS",
    "HORATIO",
    "LAERTES",
    "OPHELIA",
    "QUEEN GERTRUDE"
}

OPHELIA_CANONICAL_SCENES = {
    (1, 3),
    (2, 1),
    (3, 1),
    (3, 2),
    (4, 5)
}


def test_source_text_exists_and_verified():
    """Verify that raw Folger source text exists and has the expected editorial header."""
    assert HAMLET_TXT.exists(), f"Source corpus missing: {HAMLET_TXT}"
    with open(HAMLET_TXT, "r", encoding="utf-8") as f:
        header = "".join([f.readline() for _ in range(10)])
    assert "Folger Shakespeare Library" in header, "Source text does not match Folger Shakespeare Library edition."
    assert "FDT version 0.9.2" in header, "Expected Folger Digital Texts version 0.9.2."


def test_all_dialogue_structure():
    """Verify parsed dialogue across the complete play."""
    assert ALL_DIALOGUE_CSV.exists(), f"Parsed dialogue dataset missing: {ALL_DIALOGUE_CSV}"
    df = pd.read_csv(ALL_DIALOGUE_CSV)

    # 1. Total turn count in verified play
    assert len(df) == 1132, f"Expected 1,132 dialogue turns, got {len(df)}"

    # 2. Total words across play (clean spoken dialogue excluding stage directions)
    assert df["word_count"].sum() == 29562, f"Expected 29,562 clean words, got {df['word_count'].sum()}"

    # 3. Required columns
    expected_cols = [
        "turn_id", "act", "scene", "character_raw", "character", "line",
        "word_count", "char_count", "prev_character", "next_character",
        "has_question", "has_exclamation", "deference_count"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    # 4. Valid act ranges
    assert set(df["act"].unique()).issubset({1, 2, 3, 4, 5})

    # 5. Distinct speakers
    assert df["character"].nunique() == 33, f"Expected 33 distinct speakers, got {df['character'].nunique()}"


def test_main_characters_subset():
    """Verify main character dialogue filtering and counts."""
    assert MAIN_CHARS_CSV.exists(), f"Main character dataset missing: {MAIN_CHARS_CSV}"
    df = pd.read_csv(MAIN_CHARS_CSV)

    assert len(df) == 846, f"Expected 846 turns for main characters, got {len(df)}"
    assert df["word_count"].sum() == 24031, f"Expected 24,031 clean words for main characters, got {df['word_count'].sum()}"
    assert set(df["character"].unique()) == CANONICAL_MAIN_CHARACTERS


def test_ophelia_speech_profile():
    """Verify Ophelia's dialogue volume, turn count, and dramatic scene presence."""
    assert OPHELIA_CSV.exists(), f"Ophelia dataset missing: {OPHELIA_CSV}"
    df = pd.read_csv(OPHELIA_CSV)

    # 1. Speech turns
    assert len(df) == 58, f"Expected exactly 58 speech turns for Ophelia, got {len(df)}"

    # 2. Character identifier
    assert (df["character"] == "OPHELIA").all()

    # 3. Canonical scene presence
    scenes = set(zip(df["act"], df["scene"]))
    assert scenes == OPHELIA_CANONICAL_SCENES, f"Ophelia scenes mismatch: expected {OPHELIA_CANONICAL_SCENES}, got {scenes}"

    # 4. Absence from Act 5
    assert (df["act"] != 5).all(), "Ophelia should have zero speech turns in Act 5."

    # 5. Clean spoken dialogue word count (verified 1,180 words)
    assert df["word_count"].sum() == 1180, f"Expected 1,180 clean spoken words, got {df['word_count'].sum()}"


def test_clean_spoken_dialogue_verification():
    """Verify that independent verified parsing produces 1,180 spoken dialogue words
    and does not swallow King Claudius's 108-word speech into Ophelia's Turn 28."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vp", BASE_DIR / "tests" / "helpers" / "verification_pipeline.py")
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)

    df_clean = vp.parse_raw_corpus(HAMLET_TXT)
    oph_clean = df_clean[df_clean["character"] == "OPHELIA"].reset_index(drop=True)

    # 1. Verified turn count remains 58
    assert len(oph_clean) == 58, f"Expected 58 turns, got {len(oph_clean)}"

    # 2. Verified clean spoken dialogue is exactly 1,180 words
    assert oph_clean["word_count"].sum() == 1180, (
        f"Expected exactly 1,180 clean spoken dialogue words for Ophelia, got {oph_clean['word_count'].sum()}"
    )

    # 3. Verify Turn 28 (Act 3 Scene 1 soliloquy) has exactly 99 words, NOT 207 words
    turn_28 = oph_clean.iloc[27]  # 0-indexed
    assert turn_28["act"] == 3 and turn_28["scene"] == 1
    assert turn_28["word_count"] == 99, f"Turn 28 should contain 99 words (clean soliloquy), got {turn_28['word_count']}"
    assert "Love? His affections do not that way tend" not in turn_28["line"], (
        "Turn 28 must not contain King Claudius's subsequent speech."
    )


def test_turn_adjacency_continuity():
    """Verify that turn-taking adjacency records match contiguous dialogue rows."""
    df = pd.read_csv(ALL_DIALOGUE_CSV)
    for i in range(len(df) - 1):
        curr_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        # Only verify if in the same scene
        if (curr_row["act"] == next_row["act"]) and (curr_row["scene"] == next_row["scene"]):
            assert curr_row["next_character"] == next_row["character"], (
                f"Adjacency mismatch at turn {curr_row['turn_id']}: "
                f"next_character={curr_row['next_character']}, actual={next_row['character']}"
            )
            assert next_row["prev_character"] == curr_row["character"], (
                f"Preceding mismatch at turn {next_row['turn_id']}: "
                f"prev_character={next_row['prev_character']}, actual={curr_row['character']}"
            )
        else:
            # Must be None/NaN across scene or act boundaries
            assert pd.isna(curr_row["next_character"]), (
                f"Turn {curr_row['turn_id']} at scene boundary must have None for next_character"
            )
            assert pd.isna(next_row["prev_character"]), (
                f"Turn {next_row['turn_id']} at scene start must have None for prev_character"
            )


def test_dumb_show_and_multiline_stage_directions():
    """Verify that the 119-word dumb show and multi-line stage directions do not leak into dialogue."""
    df = pd.read_csv(ALL_DIALOGUE_CSV)
    
    # 1. Dumb show text in Act 3 Scene 2 must not appear in any dialogue turn
    dumb_show_snippets = [
        "Enter a King and a Queen, very lovingly",
        "She kneels and makes show of protestation",
        "pours poison in the sleeper's ears",
        "The dead body is carried away"
    ]
    for snippet in dumb_show_snippets:
        matches = df[df["line"].str.contains(snippet, case=False, na=False)]
        assert len(matches) == 0, f"Dumb show leaked into dialogue: {snippet} found in turn {matches['turn_id'].tolist()}"

    # 2. End of play stage direction [They exit, marching...] must not leak
    end_sd = "peal of ordnance are shot off"
    assert len(df[df["line"].str.contains(end_sd, case=False, na=False)]) == 0


def test_character_and_scene_totals_sum_to_play_totals():
    """Verify that character sums and scene sums equal the play totals exactly."""
    df = pd.read_csv(ALL_DIALOGUE_CSV)
    
    # 1. Character turn and word sums
    char_grouped = df.groupby("character").agg(turns=("turn_id", "count"), words=("word_count", "sum"))
    assert char_grouped["turns"].sum() == 1132, "Sum of character turns does not equal 1,132"
    assert char_grouped["words"].sum() == 29562, "Sum of character words does not equal 29,562"
    
    # 2. Scene turn and word sums
    scene_grouped = df.groupby(["act", "scene"]).agg(turns=("turn_id", "count"), words=("word_count", "sum"))
    assert len(scene_grouped) == 20, f"Expected 20 canonical scenes, got {len(scene_grouped)}"
    assert scene_grouped["turns"].sum() == 1132, "Sum of scene turns does not equal 1,132"
    assert scene_grouped["words"].sum() == 29562, "Sum of scene words does not equal 29,562"


def test_honorific_matching_and_case_normalization():
    """Verify that 'my lord' regex handles punctuation, case, and spacing properly."""
    import re
    df = pd.read_csv(OPHELIA_CSV)
    pattern = re.compile(r"\bmy\s+lord\b", re.IGNORECASE)
    
    total_matches = df["line"].apply(lambda text: len(pattern.findall(str(text)))).sum()
    assert total_matches == 27, f"Expected exactly 27 exact matches of 'my lord' in Ophelia, got {total_matches}"
    
    turns_with_honorific = df["line"].apply(lambda text: bool(pattern.search(str(text)))).sum()
    assert turns_with_honorific == 26, f"Expected exactly 26 turns containing 'my lord', got {turns_with_honorific}"

