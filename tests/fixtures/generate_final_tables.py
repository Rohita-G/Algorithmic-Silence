#!/usr/bin/env python3
"""
generate_final_tables.py

Generates machine-readable final paper statistics and discrepancy logs for Task 3.
"""

from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent

stats_records = [
    {
        "Metric_ID": "STAT-01",
        "Category": "Dialogue Volume",
        "Metric_Name": "Ophelia Spoken Words",
        "Verified_Value": "1180",
        "Numerator": "1180",
        "Denominator": "N/A",
        "Scope": "Entire Play (Acts 1-4; Act 5 has 0)",
        "Classification": "VERIFIED_CORPUS_FACT",
        "Recommended_Reporting": "1,180 words across 58 dialogue turns"
    },
    {
        "Metric_ID": "STAT-02",
        "Category": "Dialogue Volume",
        "Metric_Name": "Total Spoken Words (All Characters)",
        "Verified_Value": "29562",
        "Numerator": "29562",
        "Denominator": "N/A",
        "Scope": "Entire Play (33 speaking roles [31 individual + 2 collective ALL and GENTLEMEN], 1,132 turns)",
        "Classification": "VERIFIED_CORPUS_FACT",
        "Recommended_Reporting": "29,562 clean spoken dialogue words across 1,132 speech turns"
    },
    {
        "Metric_ID": "STAT-03",
        "Category": "Dialogue Volume",
        "Metric_Name": "Total Spoken Words (7 Main Characters)",
        "Verified_Value": "24031",
        "Numerator": "24031",
        "Denominator": "N/A",
        "Scope": "Hamlet, Claudius, Polonius, Horatio, Laertes, Ophelia, Gertrude (846 turns)",
        "Classification": "VERIFIED_CORPUS_FACT",
        "Recommended_Reporting": "24,031 clean spoken words (81.29% of spoken dialogue)"
    },
    {
        "Metric_ID": "STAT-04",
        "Category": "Dialogue Share",
        "Metric_Name": "Ophelia Share of Total Spoken Dialogue",
        "Verified_Value": "3.99%",
        "Numerator": "1180",
        "Denominator": "29562",
        "Scope": "Play-wide clean spoken dialogue",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "3.99% of total play spoken dialogue"
    },
    {
        "Metric_ID": "STAT-05",
        "Category": "Dialogue Share",
        "Metric_Name": "Ophelia Share of Main Character Dialogue",
        "Verified_Value": "4.91%",
        "Numerator": "1180",
        "Denominator": "24031",
        "Scope": "7 major court characters",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "4.91% of main-character dialogue"
    },
    {
        "Metric_ID": "STAT-06",
        "Category": "Dialogue Volume",
        "Metric_Name": "Hamlet-to-Ophelia Word Ratio",
        "Verified_Value": "9.81:1",
        "Numerator": "11572",
        "Denominator": "1180",
        "Scope": "Play-wide clean spoken dialogue",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "Outspoken by Hamlet alone (11,572 words) by a factor of 9.81 to 1"
    },
    {
        "Metric_ID": "STAT-07",
        "Category": "Dramatic Presence",
        "Metric_Name": "Vocal Absence Ratio across Canonical Scenes",
        "Verified_Value": "75.0%",
        "Numerator": "15",
        "Denominator": "20",
        "Scope": "20 canonical scenes of Hamlet",
        "Classification": "VERIFIED_CORPUS_FACT",
        "Recommended_Reporting": "Zero spoken dialogue in 15 of 20 canonical scenes (75.0%); physically absent in 14, mute corpse in Act 5 Scene 1"
    },
    {
        "Metric_ID": "STAT-08",
        "Category": "Turn Length",
        "Metric_Name": "Ophelia Act 3 Turn Length Median & IQR",
        "Verified_Value": "Median=5.0, IQR=5.0",
        "Numerator": "n=29 turns",
        "Denominator": "Q1=4.0, Q3=9.0",
        "Scope": "Act 3 (Scene 1 and Scene 2)",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "Median of 5.0 words (IQR = 5.0 words, Mean = 10.9 words)"
    },
    {
        "Metric_ID": "STAT-09",
        "Category": "Turn Length",
        "Metric_Name": "Hamlet Act 3 Turn Length Median & IQR",
        "Verified_Value": "Median=14.0, IQR=30.0",
        "Numerator": "n=105 turns",
        "Denominator": "Q1=6.0, Q3=36.0",
        "Scope": "Act 3 (Scenes 1-4)",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "Median of 14.0 words (IQR = 30.0 words, Mean = 39.1 words)"
    },
    {
        "Metric_ID": "STAT-10",
        "Category": "Turn Length",
        "Metric_Name": "Ophelia Act 3 Scene 2 Stichomythic Mean & Median",
        "Verified_Value": "Mean=5.6, Median=5.0",
        "Numerator": "90 words",
        "Denominator": "16 turns",
        "Scope": "Act 3 Scene 2 (Play Scene)",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "16 turns totaling 90 words; mean length of 5.6 words, median of 5.0 words; 10 of 16 turns <= 5 words"
    },
    {
        "Metric_ID": "STAT-11",
        "Category": "Deference Markers",
        "Metric_Name": "Ophelia Exact Phrase Count 'my lord'",
        "Verified_Value": "27",
        "Numerator": "27",
        "Denominator": "58 turns",
        "Scope": "Ophelia spoken dialogue",
        "Classification": "VERIFIED_CORPUS_FACT",
        "Recommended_Reporting": "27 occurrences of 'my lord' (28 including 'lordship')"
    },
    {
        "Metric_ID": "STAT-12",
        "Category": "Deference Markers",
        "Metric_Name": "Hamlet Exact Phrase Count 'my lord'",
        "Verified_Value": "10",
        "Numerator": "10",
        "Denominator": "358 turns",
        "Scope": "Hamlet spoken dialogue",
        "Classification": "VERIFIED_CORPUS_FACT",
        "Recommended_Reporting": "10 occurrences of 'my lord'"
    },
    {
        "Metric_ID": "STAT-13",
        "Category": "Deference Markers",
        "Metric_Name": "Ophelia Normalized Rate of 'my lord'",
        "Verified_Value": "22.88",
        "Numerator": "27",
        "Denominator": "1180 words",
        "Scope": "Per 1,000 words spoken",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "22.88 occurrences per 1,000 words"
    },
    {
        "Metric_ID": "STAT-14",
        "Category": "Deference Markers",
        "Metric_Name": "Hamlet Normalized Rate of 'my lord'",
        "Verified_Value": "0.86",
        "Numerator": "10",
        "Denominator": "11572 words",
        "Scope": "Per 1,000 words spoken",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "0.86 occurrences per 1,000 words (rate ratio: 26.5:1)"
    },
    {
        "Metric_ID": "STAT-15",
        "Category": "Deference Markers",
        "Metric_Name": "Ophelia Turn-Level Prevalence of 'my lord'",
        "Verified_Value": "44.83%",
        "Numerator": "26 turns",
        "Denominator": "58 turns",
        "Scope": "Percentage of speech turns containing formula",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "44.8% of speech turns (26 of 58 turns)"
    },
    {
        "Metric_ID": "STAT-16",
        "Category": "Deference Markers",
        "Metric_Name": "Hamlet Turn-Level Prevalence of 'my lord'",
        "Verified_Value": "2.51%",
        "Numerator": "9 turns",
        "Denominator": "358 turns",
        "Scope": "Percentage of speech turns containing formula",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "2.5% of speech turns (9 of 358 turns)"
    },
    {
        "Metric_ID": "STAT-17",
        "Category": "Deference Markers",
        "Metric_Name": "Turn-Level Odds Ratio ('my lord' prevalence)",
        "Verified_Value": "31.51",
        "Numerator": "Contingency [26, 32] vs [9, 349]",
        "Denominator": "Woolf 95% CI: [13.60, 72.98]",
        "Scope": "Exploratory Turn-Level Fisher Exact Test",
        "Classification": "EXPLORATORY_INFERENTIAL_STAT",
        "Recommended_Reporting": "Odds Ratio = 31.51 (95% CI: [13.6, 73.0], Fisher exact p = 5.31e-18); framed as exploratory sensitivity check"
    },
    {
        "Metric_ID": "STAT-18",
        "Category": "Network Interaction",
        "Metric_Name": "Ophelia Conversational Partners & Degree",
        "Verified_Value": "Degree=6 (Full Play), Degree=5 (Court Subgraph)",
        "Numerator": "6 interlocutors in full play",
        "Denominator": "115 turn exchanges total (114 in court subgraph)",
        "Scope": "Turn-adjacency network",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "Direct dialogue with 6 roles: Hamlet (54), Polonius (22), Laertes (18), Gertrude (10), Claudius (10), Prologue (1)"
    },
    {
        "Metric_ID": "STAT-19",
        "Category": "Network Interaction",
        "Metric_Name": "Ophelia Betweenness Centrality (Inverted Distance)",
        "Verified_Value": "0.000",
        "Numerator": "0 shortest paths",
        "Denominator": "Distance = 1/weight",
        "Scope": "Full 33-character network & 7-character subgraph",
        "Classification": "VERIFIED_DESCRIPTIVE_STAT",
        "Recommended_Reporting": "Betweenness = 0.000; descriptively indicates peripheral, non-brokering position without shortest-path mediation (bridges 0 shortest paths under standard inverted-distance convention)"
    }
]

df_stats = pd.DataFrame(stats_records)
stats_csv_path = OUTPUT_DIR / "final_paper_statistics.csv"
df_stats.to_csv(stats_csv_path, index=False)
print(f"[OK] Saved: {stats_csv_path} ({len(df_stats)} rows)")

discrepancy_records = [
    {
        "Issue_ID": "DISC-01",
        "Domain": "Ophelia Word Count",
        "Thesis_Value": "1,245 words",
        "First_Audit_Value": "1,288 words",
        "Final_Verified_Value": "1,180 words",
        "Discrepancy_Analysis": "Original thesis included 49 words of Polonius ('Ophelia, walk you here...') and 16 words of Prologue (1,180 + 49 + 16 = 1,245). First audit removed those, but speaker regex failed on line 2744 'KING, [advancing with Polonius]' due to comma, swallowing 108 words of King Claudius into Ophelia (1,180 + 108 = 1,288). Final check verified 1,180 words across 58 turns.",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-02",
        "Domain": "Dialogue Share Denominators",
        "Thesis_Value": "4.15% (unclear denom)",
        "First_Audit_Value": "4.31% (play); 5.36% (main)",
        "Final_Verified_Value": "3.99% (play clean spoken); 4.91% (main spoken)",
        "Discrepancy_Analysis": "First audit calculated 1,288 / 29,888 = 4.31% and 1,288 / 24,021 = 5.36%. Verified numerator is 1,180. Play clean spoken dialogue is 29,562 words (excluding 290 multi-line stage direction words). Comparison subset is 24,031 words. Resulting shares: 1,180/29,562 = 3.99%; 1,180/24,031 = 4.91%.",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-03",
        "Domain": "Scene Absence Metric",
        "Thesis_Value": "Unstated",
        "First_Audit_Value": "75.0% (15 of 20 scenes)",
        "Final_Verified_Value": "75.0% (15 of 20 scenes zero dialogue; 14 physically absent, 1 mute corpse in 5.1)",
        "Discrepancy_Analysis": "Verified textually against all 20 scenes. In Act 5 Scene 1, Ophelia enters as a mute corpse in a coffin ('[Enter King, Queen, Laertes, Lords attendant, and the corpse of Ophelia, with a Doctor of Divinity]'). In the other 14 non-speaking scenes, she is completely absent from the stage.",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-04",
        "Domain": "Turn Length Statistics",
        "Thesis_Value": "Unstated",
        "First_Audit_Value": "Act 3 Mann-Whitney U = 825.5, p = 1.98e-4",
        "Final_Verified_Value": "Act 3: Ophelia median 5.0 (IQR 5.0) vs Hamlet median 14.0 (IQR 30.0), U = 853.0, p = 2.95e-4; All Acts: U = 9240.0, p = 0.179 (not significant)",
        "Discrepancy_Analysis": "Act 3 difference reproduces cleanly, but U-test violates i.i.d. turn independence due to conversational stichomythia in Act 3 Scene 2. Across all 5 acts, turn length difference is non-significant. Recommend descriptive reporting (medians and IQRs) and omit confirmatory inferential claims.",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-05",
        "Domain": "Deference Significance Test",
        "Thesis_Value": "Unstated",
        "First_Audit_Value": "Word-level Chi-square = 154.13, p = 2.17e-35",
        "Final_Verified_Value": "Turn-level Fisher Exact Test: Odds Ratio = 31.51 (95% CI: [13.6, 73.0]), p = 5.31e-18; Normalized rate: 22.88 vs 0.86 per 1k words",
        "Discrepancy_Analysis": "Word-level chi-square table subtracted 2-word phrase counts from single-word totals, violating multinomial sampling and mutual exclusivity. Turn-level formulation is discrete and defensible. In a single play, phrase clustering across scenes makes inference exploratory; recommend reporting descriptive rates and turn prevalence as primary evidence.",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-06",
        "Domain": "Network Centrality & Adjacency Construction",
        "Thesis_Value": "Unstated",
        "First_Audit_Value": "Degree 5 (court), Betweenness: Hamlet=0.000, Horatio=0.533, Ophelia=0.267",
        "Final_Verified_Value": "17 true edges in court subgraph (false Horatio--Laertes edge eliminated). Court exchanges: Hamlet 416, Claudius 166, Horatio 147, Polonius 138, Gertrude 124, Ophelia 114, Laertes 111. Inverted Distance Betweenness (d = 1/w): Hamlet=0.8667, all other 6 court roles=0.0000.",
        "Discrepancy_Analysis": "Filtering dialogue to main characters prior to turn adjacency created false edges (Horatio--Laertes) and inflated edge weights. Constructing adjacency from complete play removes false adjacency. NetworkX default betweenness treats weight as distance, causing inversion; specifying distance = 1/w correctly demonstrates Hamlet mediates 86.7% of shortest paths, while Ophelia and other characters mediate 0.0% (non-brokering peripheral position).",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-07",
        "Domain": "Stage Direction Leaks in Parser",
        "Thesis_Value": "Uninspected",
        "First_Audit_Value": "Uninspected (single-line strip)",
        "Final_Verified_Value": "290 words across 16 multi-line stage directions (e.g. Dumb Show, funeral entrance) leaked into male dialogue turns in verification_pipeline.py",
        "Discrepancy_Analysis": "Ophelia dialogue is 100% free of leaked stage directions (0 words leaked). The 290 leaked words affect Hamlet (196 words, including 145-word Dumb Show) and other males. Clean spoken dialogue is 29,562 words.",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-08",
        "Domain": "Act 5 Scene 1 Funeral Dialogue Word Count",
        "Thesis_Value": "1,034 words spoken over Ophelia's corpse",
        "First_Audit_Value": "1,034 words (fictitious speaker breakdown summing to 1,034)",
        "Final_Verified_Value": "589 clean spoken words (Turns 960-985, lines 5274-5401) or 635 words including Hamlet's entrance observation (lines 5267-5272)",
        "Discrepancy_Analysis": "The 1,034 words previously cited was Prince Hamlet's total word count across all of Act 5 Scene 1. Recomputation over exact Folger source boundaries establishes the ceremony dialogue from Laertes' 'What ceremony else?' to the King's exit contains exactly 589 clean spoken words (Hamlet 245, Laertes 112, Doctor 90, Gertrude 74, Claudius 62, Horatio 5, All 1).",
        "Status": "RESOLVED_FROZEN"
    },
    {
        "Issue_ID": "DISC-09",
        "Domain": "Role-Label Classification",
        "Thesis_Value": "Unstated",
        "First_Audit_Value": "32 speaking roles",
        "Final_Verified_Value": "33 speaking roles: 31 individual canonical characters + 2 collective speaker headings ('ALL' and 'GENTLEMEN')",
        "Status": "RESOLVED_FROZEN"
    }
]

df_disc = pd.DataFrame(discrepancy_records)
disc_csv_path = OUTPUT_DIR / "discrepancy_log.csv"
df_disc.to_csv(disc_csv_path, index=False)
print(f"[OK] Saved: {disc_csv_path} ({len(df_disc)} rows)")
