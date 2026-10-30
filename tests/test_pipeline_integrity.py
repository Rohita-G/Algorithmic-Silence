#!/usr/bin/env python3
"""
test_pipeline_integrity.py

Integration and quality assurance tests verifying:
  - Required package imports
  - Generated output files and figures in results/
  - Centrality metrics consistency in results/tables/
  - Alignment with verified statistics from tests/fixtures/
"""

from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"


def test_dependency_imports():
    """Verify that all core analytical dependencies can be imported successfully."""
    try:
        import pandas as pd
        import numpy as np
        import scipy
        import matplotlib
        import seaborn as sns
        import networkx as nx
        import nltk
    except ImportError as e:
        pytest.fail(f"Failed to import core dependency: {e}")


def test_pipeline_outputs_exist_and_nonempty():
    """Verify that all core pipeline artifacts are generated and have non-zero size."""
    expected_files = [
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

    missing = []
    empty = []
    for f in expected_files:
        if not f.exists():
            missing.append(str(f.relative_to(BASE_DIR)))
        elif f.stat().st_size == 0:
            empty.append(str(f.relative_to(BASE_DIR)))

    assert not missing, f"Missing expected output files: {missing}"
    assert not empty, f"Output files are empty (0 bytes): {empty}"


def test_centrality_metrics_structure():
    """Verify network centrality CSV output format and mathematical validity."""
    centrality_csv = TABLES_DIR / "hamlet_network_centrality.csv"
    if not centrality_csv.exists() and (BASE_DIR / "output" / "hamlet_network_centrality.csv").exists():
        centrality_csv = BASE_DIR / "output" / "hamlet_network_centrality.csv"

    assert centrality_csv.exists(), f"Centrality CSV missing: {centrality_csv}"

    df = pd.read_csv(centrality_csv)
    assert len(df) == 7, f"Expected 7 main characters in centrality table, got {len(df)}"

    required_cols = [
        "Character", "Total_Words", "Turn_Exchanges",
        "Betweenness_Centrality", "Closeness_Centrality", "PageRank"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in centrality CSV"

    # Probability sum for PageRank must equal 1.0 (within numerical tolerance)
    assert abs(df["PageRank"].sum() - 1.0) < 1e-4, "PageRank values do not sum to 1.0"


def test_verified_statistics_alignment():
    """Verify that current processed dialogue and centrality tables match frozen benchmarks."""
    fixtures_csv = FIXTURES_DIR / "final_paper_statistics.csv"
    assert fixtures_csv.exists(), f"Benchmark fixtures missing: {fixtures_csv}"

    df_fix = pd.read_csv(fixtures_csv)
    fix_dict = dict(zip(df_fix["Metric_Name"], df_fix["Verified_Value"]))

    # 1. Verify frozen benchmark fixture entries
    assert int(fix_dict["Ophelia Spoken Words"]) == 1180
    assert fix_dict["Vocal Absence Ratio across Canonical Scenes"] == "75.0%"
    assert int(fix_dict["Ophelia Exact Phrase Count 'my lord'"]) == 27
    assert float(fix_dict["Ophelia Normalized Rate of 'my lord'"]) == 22.88
    assert float(fix_dict["Ophelia Betweenness Centrality (Inverted Distance)"]) == 0.000

    # 2. Verify production Ophelia dataset against benchmarks
    ophelia_csv = DATA_PROCESSED_DIR / "hamlet_ophelia_lines.csv"
    if not ophelia_csv.exists() and (BASE_DIR / "data" / "hamlet_ophelia_lines.csv").exists():
        ophelia_csv = BASE_DIR / "data" / "hamlet_ophelia_lines.csv"

    assert ophelia_csv.exists(), "Ophelia lines CSV missing"
    df_ophelia = pd.read_csv(ophelia_csv)

    assert df_ophelia["word_count"].sum() == 1180, "Ophelia clean words diverged from 1,180"
    assert len(df_ophelia) == 58, "Ophelia turns diverged from 58"

    # 3. Verify production deference count
    import re
    pattern = re.compile(r"\bmy\s+lord\b", re.IGNORECASE)
    ophelia_my_lord = df_ophelia["line"].apply(lambda t: len(pattern.findall(str(t)))).sum()
    assert ophelia_my_lord == 27, f"Ophelia 'my lord' count diverged: {ophelia_my_lord} != 27"

    # 4. Verify production centrality table
    centrality_csv = TABLES_DIR / "hamlet_network_centrality.csv"
    if centrality_csv.exists():
        df_cent = pd.read_csv(centrality_csv)
        oph_row = df_cent[df_cent["Character"] == "OPHELIA"]
        if not oph_row.empty:
            assert oph_row["Total_Words"].iloc[0] == 1180
            # 114 turn exchanges in the 7-character court subgraph (115 in full play including 1 with Prologue)
            assert oph_row["Turn_Exchanges"].iloc[0] == 114


def test_network_construction_synthetic_sequence_and_induced_subgraph():
    """Verify that graph construction does not create false adjacency across excluded characters.
    
    Regression test requested in STEP 1:
      Synthetic sequence: OPHELIA -> PROLOGUE -> HAMLET
      Expected:
        - Full graph contains OPHELIA-PROLOGUE and PROLOGUE-HAMLET
        - Induced subgraph on {OPHELIA, HAMLET} must NOT create OPHELIA-HAMLET
      Also tests:
        - Scene boundaries prevent edge creation
        - Consecutive turns by same character (self-loops) are excluded
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from character_network import construct_turn_adjacency_graph

    synthetic_df = pd.DataFrame([
        {"turn_id": 1, "act": 3, "scene": 2, "character": "OPHELIA"},
        {"turn_id": 2, "act": 3, "scene": 2, "character": "PROLOGUE"},
        {"turn_id": 3, "act": 3, "scene": 2, "character": "HAMLET"},
        # Self-loop test: consecutive speech turns in same scene
        {"turn_id": 4, "act": 3, "scene": 2, "character": "HAMLET"},
        # Scene boundary test: transitions across scene breaks must reset
        {"turn_id": 5, "act": 3, "scene": 3, "character": "KING CLAUDIUS"},
    ])

    # 1. Full graph from synthetic sequence
    G_full = construct_turn_adjacency_graph(synthetic_df)
    assert G_full.has_edge("OPHELIA", "PROLOGUE"), "Full graph must contain OPHELIA-PROLOGUE"
    assert G_full.has_edge("PROLOGUE", "HAMLET"), "Full graph must contain PROLOGUE-HAMLET"
    assert not G_full.has_edge("OPHELIA", "HAMLET"), "Full graph must not join OPHELIA-HAMLET across PROLOGUE"
    assert not G_full.has_edge("HAMLET", "HAMLET"), "Self-loops must be excluded"
    assert not G_full.has_edge("HAMLET", "KING CLAUDIUS"), "Edges must not cross scene boundaries"

    # 2. Induced subgraph selecting only OPHELIA and HAMLET
    G_sub = construct_turn_adjacency_graph(synthetic_df, characters=["OPHELIA", "HAMLET"])
    assert not G_sub.has_edge("OPHELIA", "HAMLET"), (
        "Induced subgraph on {OPHELIA, HAMLET} must NOT contain false edge OPHELIA-HAMLET"
    )
    assert set(G_sub.nodes()) == {"OPHELIA", "HAMLET"}
    assert G_sub.number_of_edges() == 0


def test_network_edge_weights_and_boundary_audit_agreement():
    """Verify that network edge weights strictly agree with the underlying turn boundary records
    and directional transition audit (validation/ophelia_adjacency_audit.csv)."""
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from character_network import construct_turn_adjacency_graph

    dialogue_csv = DATA_PROCESSED_DIR / "hamlet_all_dialogue.csv"
    if not dialogue_csv.exists() and (BASE_DIR / "data" / "hamlet_all_dialogue.csv").exists():
        dialogue_csv = BASE_DIR / "data" / "hamlet_all_dialogue.csv"
    assert dialogue_csv.exists(), "Complete dialogue CSV missing"

    df_dialogue = pd.read_csv(dialogue_csv)
    audit_csv = BASE_DIR / "validation" / "ophelia_adjacency_audit.csv"
    assert audit_csv.exists(), "Ophelia adjacency audit missing"
    df_audit = pd.read_csv(audit_csv)

    # 1. Verify audit transition counts
    assert len(df_audit) == 115, f"Expected 115 audit transitions, got {len(df_audit)}"
    inc = df_audit[df_audit["direction"] == "INCOMING"]
    out = df_audit[df_audit["direction"] == "OUTGOING"]
    assert len(inc) == 58, f"Expected 58 incoming transitions, got {len(inc)}"
    assert len(out) == 57, f"Expected 57 outgoing transitions (Turn 158 closes scene 1.3), got {len(out)}"

    # 2. Verify G_full edge weights match audit partner totals
    G_full = construct_turn_adjacency_graph(df_dialogue)
    expected_partners = {
        "HAMLET": 54,
        "POLONIUS": 22,
        "LAERTES": 18,
        "QUEEN GERTRUDE": 10,
        "KING CLAUDIUS": 10,
        "PROLOGUE": 1
    }
    for partner, expected_w in expected_partners.items():
        assert G_full.has_edge("OPHELIA", partner), f"G_full missing edge OPHELIA-{partner}"
        actual_w = G_full["OPHELIA"][partner]["count"]
        assert actual_w == expected_w, f"Edge weight mismatch OPHELIA-{partner}: {actual_w} != {expected_w}"

    # 3. Verify total transitions across full graph
    total_boundary_transitions = 0
    for i in range(len(df_dialogue) - 1):
        r1, r2 = df_dialogue.iloc[i], df_dialogue.iloc[i + 1]
        if (r1["act"] == r2["act"]) and (r1["scene"] == r2["scene"]) and (r1["character"] != r2["character"]):
            total_boundary_transitions += 1
    graph_total_weight = sum(d["count"] for u, v, d in G_full.edges(data=True))
    assert graph_total_weight == total_boundary_transitions == 1107, (
        f"Graph weight sum ({graph_total_weight}) does not match boundary transitions ({total_boundary_transitions})"
    )

