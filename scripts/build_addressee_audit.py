#!/usr/bin/env python3
"""
build_addressee_audit.py

Generates validation/presence_addressee_annotations.csv:
Applies rule-based heuristic mapping to identify interlocutors, royal rank relations,
and contextual evidence for all 175 occurrences of honorific formulas ("my lord")
across the play dialogue.

Note on Provenance:
This is an automated heuristic baseline and single-researcher structural audit.
Formal double-blind inter-rater reliability (Cohen's kappa) with an independent
second reader remains an open requirement for full ground-truth validation.

Usage:
    python scripts/build_addressee_audit.py
"""

import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

TURNS_PATH = VALIDATION_DIR / "extracted_turns_with_source_refs.csv"
if not TURNS_PATH.exists():
    TURNS_PATH = BASE_DIR / "data" / "processed" / "hamlet_all_dialogue.csv"


def build_annotations():
    df = pd.read_csv(TURNS_PATH)
    text_col = "dialogue" if "dialogue" in df.columns else "line"

    pattern = re.compile(r"\bmy\s+lord\b", re.IGNORECASE)

    records = []
    ann_id = 0

    for idx, row in df.iterrows():
        text = str(row[text_col])
        matches = list(pattern.finditer(text))
        if not matches:
            continue

        char = row["character"]
        turn_id = row["turn_id"]
        act = row["act"]
        scene = row["scene"]
        start_line = row.get("start_line", "")
        end_line = row.get("end_line", "")
        line_ref = f"{start_line}-{end_line}" if start_line else ""
        prev_char = row["prev_character"]
        next_char = row["next_character"]

        # Heuristic rules grounded in dramatic scene context
        addressee = "UNKNOWN"
        evidence = "Inferred from surrounding dialogue"
        ambiguity = "CLEAR"
        rank_rel = "EQUAL"

        if char == "OPHELIA":
            if act in [1, 2]:
                addressee = "POLONIUS"
                evidence = f"Conversation in Act {act} Scene {scene} between daughter and father (Polonius)."
                rank_rel = "SUPERIOR_PATRIARCH"
            elif act == 3:
                addressee = "HAMLET"
                evidence = f"Dialogue in Act 3 Scene {scene} addressing Prince Hamlet (courtly suitor / Crown Prince)."
                rank_rel = "SUPERIOR_ROYAL"
        elif char == "HORATIO":
            if pd.notna(prev_char) and prev_char == "HAMLET":
                addressee = "HAMLET"
                evidence = "Direct conversational response to Hamlet."
                rank_rel = "SUPERIOR_ROYAL"
            elif act == 1 and scene in [1, 2]:
                addressee = "HAMLET" if "Hamlet" in text or prev_char == "HAMLET" else "GHOST_OR_ROYAL"
                evidence = "Dialogue on the watch / court addressing Prince or sovereign."
                rank_rel = "SUPERIOR_ROYAL"
            elif act == 5 and scene == 2:
                addressee = "HAMLET"
                evidence = "Attending Hamlet during the fencing match and dying moments."
                rank_rel = "SUPERIOR_ROYAL"
            else:
                addressee = "HAMLET"
                evidence = "Horatio's primary dramatic interlocutor."
                rank_rel = "SUPERIOR_ROYAL"
        elif char == "POLONIUS":
            if pd.notna(prev_char) and prev_char in ["KING CLAUDIUS", "QUEEN GERTRUDE"]:
                addressee = prev_char
                evidence = f"Addressing sovereign monarch ({prev_char})."
                rank_rel = "SUPERIOR_MONARCH"
            elif "King" in text or "Claudius" in text or act in [1, 2, 3]:
                addressee = "KING CLAUDIUS"
                evidence = "Addressing sovereign King Claudius in council / court."
                rank_rel = "SUPERIOR_MONARCH"
        elif char == "LAERTES":
            if act in [1, 4, 5]:
                addressee = "KING CLAUDIUS"
                evidence = "Addressing King Claudius for departure leave or revenge compact."
                rank_rel = "SUPERIOR_MONARCH"
        elif char == "QUEEN GERTRUDE":
            addressee = "KING CLAUDIUS" if act in [1, 2, 4] else "HAMLET"
            evidence = "Addressing royal husband (Claudius) or son (Hamlet)."
            rank_rel = "MONARCH_CONSORT"
        elif char == "HAMLET":
            if "Ghost" in text or (pd.notna(prev_char) and prev_char == "GHOST"):
                addressee = "GHOST"
                evidence = "Addressing the Ghost of his father, the late King."
                rank_rel = "SUPERIOR_PATRIARCH"
            elif pd.notna(prev_char) and prev_char == "POLONIUS":
                addressee = "POLONIUS"
                evidence = "Ironic / mocked deference to Lord Chamberlain Polonius."
                rank_rel = "INFERIOR_COUNCILLOR"
            else:
                addressee = "MULTIPLE_OR_AMBIGUOUS"
                evidence = "Courtly address or dramatic soliloquy reflection."
                ambiguity = "AMBIGUOUS"
                rank_rel = "VARIABLE"

        ann_id += 1
        records.append({
            "annotation_id": f"ANN-{ann_id:04d}",
            "turn_id": turn_id,
            "speaker": char,
            "act": act,
            "scene": scene,
            "source_lines": line_ref,
            "honorific_count": len(matches),
            "matched_text": "; ".join(m.group(0) for m in matches),
            "dialogue_snippet": text[:80] + ("..." if len(text) > 80 else ""),
            "addressee": addressee,
            "addressee_rank_relation": rank_rel,
            "contextual_evidence": evidence,
            "ambiguity_status": ambiguity
        })

    df_ann = pd.DataFrame(records)
    out_path = VALIDATION_DIR / "presence_addressee_annotations.csv"
    df_ann.to_csv(out_path, index=False)
    print(f"[OK] Saved addressee annotations: {out_path} ({len(df_ann)} rows)")


if __name__ == "__main__":
    build_annotations()
