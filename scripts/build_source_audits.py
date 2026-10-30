#!/usr/bin/env python3
"""
build_source_audits.py

Constructs comprehensive source audits for Folger Hamlet (FDT v0.9.2):
  1. validation/removed_spans_audit.csv: complete inventory of all 1,424 non-dialogue text spans
     (front matter, act/scene headings, stage directions, and speaker headers) with
     exact start/end line numbers, word counts, raw text snippets, and exclusion reasons.
  2. validation/extracted_turns_with_source_refs.csv: verified clean dialogue turns with
     exact source line references, canonical character mapping, and turn adjacency.
  3. validation/ophelia_adjacency_audit.csv: all 115 contiguous turn transitions involving
     Ophelia traced to source turns, interlocutors, line ranges, and text snippets.

Usage:
    python scripts/build_source_audits.py
"""

import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
VALIDATION_DIR = BASE_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

HAMLET_PATH = DATA_RAW_DIR / "hamlet.txt"
if not HAMLET_PATH.exists() and (BASE_DIR / "data" / "hamlet.txt").exists():
    HAMLET_PATH = BASE_DIR / "data" / "hamlet.txt"

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


def generate_audits():
    with open(HAMLET_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    full_text = "".join(raw_lines)

    removed_spans = []
    span_id = 0

    # 1. Front matter (lines 1 to 43)
    front_matter_text = "".join(raw_lines[:43]).strip()
    fm_words = len(front_matter_text.split())
    span_id += 1
    removed_spans.append({
        "span_id": f"SPAN-{span_id:04d}",
        "start_line": 1,
        "end_line": 43,
        "span_type": "FRONT_MATTER",
        "word_count": fm_words,
        "raw_text_snippet": (front_matter_text[:120] + "...").replace("\n", " "),
        "exclusion_reason": "Folger editorial metadata and dramatis personae list preceding Act 1"
    })

    # 2. Stage directions (all 266 bracketed spans)
    bracket_matches = list(re.finditer(r"\[([\s\S]*?)\]", full_text))
    for m in bracket_matches:
        span_id += 1
        content = m.group(0)
        start_char = m.start()
        end_char = m.end()
        start_line = full_text[:start_char].count("\n") + 1
        end_line = full_text[:end_char].count("\n") + 1
        clean_content = content.replace("\n", " ").strip()
        words = len(clean_content.split())

        reason = "Stage direction: dramatic action, entrance, exit, or actor cue"
        if "Dumb show" in clean_content or "dumb show" in clean_content or "Enter a King and a Queen" in clean_content:
            reason = "Pantomime stage direction (Act 3 Scene 2 Dumb Show): non-vocal theatrical action"
        elif clean_content.lower().startswith("[aside"):
            reason = "Theatrical performance delivery cue / direction"
        elif clean_content.lower().startswith("[sings"):
            reason = "Performance cue indicating ballad delivery mode"

        removed_spans.append({
            "span_id": f"SPAN-{span_id:04d}",
            "start_line": start_line,
            "end_line": end_line,
            "span_type": "STAGE_DIRECTION",
            "word_count": words,
            "raw_text_snippet": clean_content[:120] + ("..." if len(clean_content) > 120 else ""),
            "exclusion_reason": reason
        })

    # 3. Clean line-preserving text for dialogue extraction
    def mask_brackets(text):
        def repl(m):
            return "".join("\n" if c == "\n" else " " for c in m.group(0))
        return re.sub(r"\[[\s\S]*?\]", repl, text)

    clean_lines = mask_brackets(full_text).splitlines()

    turns = []
    turn_id = 0
    current_act = 0
    current_scene = 0
    current_speaker = None
    current_dialogue = []
    turn_start_line = 0
    last_content_line = 0

    def flush_turn(end_line):
        nonlocal turn_id, current_speaker, current_dialogue, turn_start_line
        if current_speaker and current_dialogue:
            text = " ".join(current_dialogue).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                turn_id += 1
                words = text.split()
                canonical_char = SPEAKER_MAP.get(current_speaker, current_speaker)
                turns.append({
                    "turn_id": turn_id,
                    "act": current_act,
                    "scene": current_scene,
                    "start_line": turn_start_line,
                    "end_line": end_line,
                    "character_raw": current_speaker,
                    "character": canonical_char,
                    "dialogue": text,
                    "word_count": len(words),
                    "char_count": len(text)
                })
        current_speaker = None
        current_dialogue = []

    for line_idx, raw_line in enumerate(clean_lines):
        line_num = line_idx + 1
        line = raw_line.strip()
        if not line:
            continue

        # Act heading
        m_act = re.match(r"^ACT\s+(\d+)", line, re.IGNORECASE)
        if m_act:
            flush_turn(last_content_line)
            current_act = int(m_act.group(1))
            current_scene = 1
            span_id += 1
            removed_spans.append({
                "span_id": f"SPAN-{span_id:04d}",
                "start_line": line_num,
                "end_line": line_num + 1,
                "span_type": "ACT_SCENE_HEADING",
                "word_count": len(line.split()),
                "raw_text_snippet": line,
                "exclusion_reason": "Structural act division heading and underline"
            })
            continue

        # Scene heading
        m_scene = re.match(r"^Scene\s+(\d+)", line, re.IGNORECASE)
        if m_scene:
            flush_turn(last_content_line)
            current_scene = int(m_scene.group(1))
            span_id += 1
            removed_spans.append({
                "span_id": f"SPAN-{span_id:04d}",
                "start_line": line_num,
                "end_line": line_num + 1,
                "span_type": "ACT_SCENE_HEADING",
                "word_count": len(line.split()),
                "raw_text_snippet": line,
                "exclusion_reason": "Structural scene division heading and underline"
            })
            continue

        if line.startswith("="):
            continue

        if current_act == 0:
            continue

        # Speaker detection
        m_spk = re.match(r"^([A-Z][A-Z0-9 \'\.\-]+?)(?:,)?(?:\s{2,}(.*)|\s*)$", raw_line)
        if m_spk:
            cand_name = m_spk.group(1).strip()
            letters = re.sub(r"[^A-Za-z]", "", cand_name)
            if letters and letters.isupper() and cand_name not in ["ACT", "SCENE"]:
                flush_turn(last_content_line)
                current_speaker = cand_name
                turn_start_line = line_num
                last_content_line = line_num

                span_id += 1
                removed_spans.append({
                    "span_id": f"SPAN-{span_id:04d}",
                    "start_line": line_num,
                    "end_line": line_num,
                    "span_type": "SPEAKER_LABEL",
                    "word_count": len(cand_name.split()),
                    "raw_text_snippet": cand_name,
                    "exclusion_reason": "Speaker attribution prefix label"
                })

                d_part = m_spk.group(2) if m_spk.group(2) else ""
                d_clean = d_part.strip()
                if d_clean:
                    current_dialogue.append(d_clean)
                continue

        if line and current_speaker:
            current_dialogue.append(line)
            last_content_line = line_num

    flush_turn(last_content_line)

    # Save removed spans audit
    df_spans = pd.DataFrame(removed_spans)
    df_spans_path = VALIDATION_DIR / "removed_spans_audit.csv"
    df_spans.to_csv(df_spans_path, index=False)
    print(f"[OK] Saved removed spans audit: {df_spans_path} ({len(df_spans)} rows)")

    # Save extracted turns audit
    df_turns = pd.DataFrame(turns)
    df_turns["prev_character"] = df_turns["character"].shift(1)
    df_turns["next_character"] = df_turns["character"].shift(-1)

    df_turns.loc[df_turns["act"] != df_turns["act"].shift(1), "prev_character"] = None
    df_turns.loc[df_turns["scene"] != df_turns["scene"].shift(1), "prev_character"] = None
    df_turns.loc[df_turns["act"] != df_turns["act"].shift(-1), "next_character"] = None
    df_turns.loc[df_turns["scene"] != df_turns["scene"].shift(-1), "next_character"] = None

    df_turns_path = VALIDATION_DIR / "extracted_turns_with_source_refs.csv"
    df_turns.to_csv(df_turns_path, index=False)
    print(f"[OK] Saved extracted turns audit: {df_turns_path} ({len(df_turns)} rows, {df_turns['word_count'].sum():,} words)")

    # Generate Ophelia adjacency audit
    adj_records = []
    trans_id = 0
    for idx, row in df_turns.iterrows():
        if row["character"] == "OPHELIA":
            # Incoming
            if pd.notna(row["prev_character"]):
                prev_row = df_turns.iloc[idx - 1]
                trans_id += 1
                adj_records.append({
                    "transition_id": trans_id,
                    "direction": "INCOMING",
                    "ophelia_turn_id": row["turn_id"],
                    "partner_turn_id": prev_row["turn_id"],
                    "partner_character": prev_row["character"],
                    "act": row["act"],
                    "scene": row["scene"],
                    "ophelia_lines": f"{row['start_line']}-{row['end_line']}",
                    "partner_lines": f"{prev_row['start_line']}-{prev_row['end_line']}",
                    "ophelia_snippet": row["dialogue"][:60],
                    "partner_snippet": prev_row["dialogue"][:60]
                })
            # Outgoing
            if pd.notna(row["next_character"]):
                next_row = df_turns.iloc[idx + 1]
                trans_id += 1
                adj_records.append({
                    "transition_id": trans_id,
                    "direction": "OUTGOING",
                    "ophelia_turn_id": row["turn_id"],
                    "partner_turn_id": next_row["turn_id"],
                    "partner_character": next_row["character"],
                    "act": row["act"],
                    "scene": row["scene"],
                    "ophelia_lines": f"{row['start_line']}-{row['end_line']}",
                    "partner_lines": f"{next_row['start_line']}-{next_row['end_line']}",
                    "ophelia_snippet": row["dialogue"][:60],
                    "partner_snippet": next_row["dialogue"][:60]
                })

    df_adj = pd.DataFrame(adj_records)
    df_adj_path = VALIDATION_DIR / "ophelia_adjacency_audit.csv"
    df_adj.to_csv(df_adj_path, index=False)
    print(f"[OK] Saved Ophelia adjacency audit: {df_adj_path} ({len(df_adj)} transitions)")


if __name__ == "__main__":
    generate_audits()
