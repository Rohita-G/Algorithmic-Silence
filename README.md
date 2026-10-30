# Algorithmic Silence: Computational Analysis of Ophelia in Shakespeare’s Hamlet

A computational literary case study of Ophelia’s dramatic participation
in a specified Folger edition of Hamlet.

The project measures speech allocation, turn lengths, honorific address,
and within-scene speaker adjacency. These descriptive results are
interpreted alongside feminist literary criticism to examine patterns
of constrained participation without treating numerical differences
as automatic proof of gendered suppression or authorial intent.

## Research Questions

1. How are spoken words and speech turns distributed across characters,
   and what share belongs to Ophelia?

2. How do Ophelia’s turn lengths vary across scenes, and how do they
   compare with other speakers in comparable interaction segments?

3. How frequently does Ophelia use “my lord,” and how should that pattern
   be interpreted in relation to rank, addressee, and dramatic context?

4. Which speakers immediately precede or follow Ophelia within scenes,
   and how does her speaker-adjacency position differ between the
   principal-character subgraph and the full-play graph?

## Scope

This is a descriptive, single-play study based on one source edition.
It does not measure theatrical delivery, infer psychological states,
isolate gender from social rank, or establish authorial intent.

Speaker adjacency represents sequential speech, not necessarily direct
address. Automated tests support reproducibility and regression checking;
independent annotation remains a separate validation task.

---

## Repository Layout

```text
Algorithmic_Silence/
├── README.md                      # Project overview and reproduction guide
├── requirements.txt               # Pinned Python dependencies
├── run_pipeline.py                # Master execution script for analytical pipeline
├── CITATION.cff                   # Citation metadata (CFF format)
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
│
├── data/
│   ├── raw/
│   │   └── hamlet.txt             # Immutable source corpus (Folger Digital Texts v0.9.2)
│   └── processed/
│       ├── hamlet_all_dialogue.csv    # Canonical dialogue (1,132 turns, 29,562 words)
│       ├── hamlet_main_characters.csv # 7 principal characters (846 turns, 24,031 words)
│       └── hamlet_ophelia_lines.csv   # Ophelia dialogue subset (58 turns, 1,180 words)
│
├── scripts/                       # Core analytical pipeline code
│   ├── parse_hamlet_main_chars.py     # Stage 1: Folger ingestion and dialogue parsing
│   ├── extract_ophelia.py             # Stage 2: Ophelia dialogue and absence extraction
│   ├── character_network.py           # Stage 3: Turn-adjacency graph and centrality
│   ├── visualize_ophelia.py           # Stage 4: Dialogue volume visualizations
│   ├── analyze_deference_and_agency.py# Stage 5: Honorifics and turn length distributions
│   ├── build_source_audits.py         # Audit generator: removed spans, extracted turns, adjacency
│   └── build_addressee_audit.py       # Audit generator: heuristic addressee annotations
│
├── tests/                         # Quality assurance suite and verification benchmarks
│   ├── test_corpus_parsing.py         # Parser, token count, and attribution tests
│   ├── test_pipeline_integrity.py     # Pipeline output and benchmark alignment tests
│   ├── helpers/
│   │   └── verification_pipeline.py   # Ground-truth verification calculation engine
│   └── fixtures/
│       ├── final_paper_statistics.csv # Authoritative frozen benchmark metrics
│       ├── discrepancy_log.csv        # Reconciliation audit log
│       └── generate_final_tables.py   # Fixture table generator
│
├── results/                       # Current paper outputs
│   ├── tables/
│   │   ├── hamlet_network_centrality.csv # Network centrality metrics (degree, betweenness, PageRank)
│   │   ├── hamlet_full_play_network_centrality.csv # Full play network centrality metrics
│   │   └── corrected_results.csv         # Reconciled statistics and audit references
│   └── figures/
│       ├── speech_volume_comparison.png       # Fig 1: Dialogue volume across characters
│       ├── turn_length_distribution.png       # Fig 2: Turn length box plots
│       ├── deference_markers_comparison.png   # Fig 3: Honorific rates per 1,000 words
│       ├── hamlet_character_network_clean_labels.png # Fig 4: Turn-adjacency network
│       └── ophelia_ego_network.png            # Fig 5: Ophelia conversational ego network
│
└── validation/                    # Traceability, annotations, and validation evidence
    ├── extracted_turns_with_source_refs.csv # 1,132 turns with line-level source ranges
    ├── removed_spans_audit.csv    # 1,424 removed spans (SDs, headers) with line numbers
    ├── ophelia_adjacency_audit.csv# 115 Ophelia transitions mapped to source turns
    ├── presence_addressee_annotations.csv   # Heuristic & audited addressees for all 175 honorifics
    ├── validation_results.md      # Methodological checks, audit report, and protocol
    └── history/
        └── corrected_results.csv  # Historical audit correction ledger
```

---

## Environment Setup

### Requirements
- Python 3.10 or higher

### Setup
```bash
# Clone the repository
git clone https://github.com/Rohita-G/Algorithmic-Silence.git
cd Algorithmic-Silence

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

---

## Reproduction Command

Execute the complete end-to-end analytical pipeline from the project root:

```bash
python run_pipeline.py
```

This runs the five core pipeline stages:
1. `scripts/parse_hamlet_main_chars.py` -> Ingests `data/raw/hamlet.txt` and generates `data/processed/hamlet_all_dialogue.csv` and `data/processed/hamlet_main_characters.csv`.
2. `scripts/extract_ophelia.py` -> Extracts `data/processed/hamlet_ophelia_lines.csv` and computes speech totals.
3. `scripts/character_network.py` -> Builds turn-adjacency graphs, saves `results/tables/hamlet_network_centrality.csv`, and renders `results/figures/hamlet_character_network_clean_labels.png` and `results/figures/ophelia_ego_network.png`.
4. `scripts/visualize_ophelia.py` -> Generates `results/figures/speech_volume_comparison.png`.
5. `scripts/analyze_deference_and_agency.py` -> Generates `results/figures/turn_length_distribution.png` and `results/figures/deference_markers_comparison.png`.

---

## Automated Tests

Execute the automated test suite with verbose output:

```bash
pytest tests/ -v
```

The test suite executes 13 rigorous validation checks:
- Raw corpus existence and SHA-256 byte-for-byte immutability.
- Parsing accuracy: exactly 1,132 turns and 29,562 clean spoken words.
- Ophelia spoken dialogue accuracy: exactly 58 turns and 1,180 spoken words.
- Scene boundary discontinuity: conversational adjacency resets to `None` at scene boundaries.
- Benchmark alignment: verified alignment against frozen gold standards in `tests/fixtures/final_paper_statistics.csv`.
- Artifact completeness: validation that all generated CSVs and PNG figures exist and are non-empty.

---

## Research Reproduction and Artifact Verification

This repository provides the complete computational reproduction package and verification test suite:
- **Deterministic Pipeline:** Run `python run_pipeline.py` to regenerate all processed tables and publication figures from raw source text.
- **Auditable Validation Suite:** Run `pytest tests/ -v` to verify parsing accuracy, edge weights, and consistency against frozen benchmark statistics.

---

## Corpus Provenance and Usage Conditions

- **Corpus Text:** William Shakespeare, *Hamlet*, edited by Barbara A. Mowat, Paul Werstine, Michael Poston, and Rebecca Niles.
- **Corpus Version:** Folger Digital Texts (FDT), Version 0.9.2 (Released July 31, 2015).
- **Collation Base:** Editorial collation of the Second Quarto (Q2, 1604–1605) and First Folio (F1, 1623).
- **Source URL:** [https://shakespeare.folger.edu/shakespeares-works/hamlet/](https://shakespeare.folger.edu/shakespeares-works/hamlet/)
- **Source Checksum (SHA-256):** `3d9b03e4051a202ae263f65cd4d24371af5ce8655629a73bddf87aad253db5f3`
- **Usage Conditions:** The Folger source text is provided for non-commercial educational and scholarly research under terms established by the Folger Shakespeare Library.

---

## Generated Outputs

The pipeline deterministically generates the following outputs:

### Data Tables
- `data/processed/hamlet_all_dialogue.csv`: 1,132 turns, 29,562 spoken words across 33 dramatic roles.
- `data/processed/hamlet_main_characters.csv`: 846 turns, 24,031 spoken words for 7 court figures.
- `data/processed/hamlet_ophelia_lines.csv`: 58 turns, 1,180 spoken words for Ophelia.
- `results/tables/hamlet_network_centrality.csv`: Weighted degree, betweenness centrality, closeness, and PageRank for all characters.
- `results/tables/corrected_results.csv`: Audit comparison ledger mapping old vs. verified metrics.

### Manuscript Figures
- `results/figures/speech_volume_comparison.png`: Word count allocation across principal characters (Ophelia = 1,180 words, 4.91% court share).
- `results/figures/turn_length_distribution.png`: Turn length box plots highlighting stichomythic compression (Ophelia median = 6.0 words overall, 5.0 words in 3.2).
- `results/figures/deference_markers_comparison.png`: Frequency of *"my lord"* per 1,000 spoken words (Ophelia = 22.88 vs. Hamlet = 0.86; 26.6:1 ratio).
- `results/figures/hamlet_character_network_clean_labels.png`: Full turn-adjacency conversational network.
- `results/figures/ophelia_ego_network.png`: Ophelia's direct interlocutor subgraph (114 transitions across 5 court figures, plus 1 to Prologue).

---

## Known Validation Limitations

1. **Editorial Dependency:** All counts reflect the specific editorial decisions of the Folger edition (Q2/F1 collation). Alternative historical texts (e.g., First Quarto Q1 1603) differ in scene structure and dialogue length.
2. **Text vs. Performance:** This pipeline analyzes written dramatic verse and prose, not theatrical performance. Actor pauses, delivery speed, and staging choices dramatically alter audience perception of presence and silence.
3. **Turn-Adjacency vs. Co-Presence:** Network edges represent immediate dialogue transitions within a scene. Non-speaking physical presence on stage (e.g., Act 5 Scene 1, where Ophelia's corpse is present while 589 clean dialogue words are spoken over her funeral rites) is not captured by adjacency graphs, but is tracked via contextual scene audit (`validation/presence_addressee_annotations.csv`).
4. **Affective Lexicon Domain Mismatch:** Off-the-shelf modern sentiment tools (VADER, TextBlob) fail systematically on early modern dramatic verse due to syntactic inversions, courtly deference formulas (*"good my lord"*), and dramatic irony. Sentiment scores are excluded from core findings and treated as exploratory baseline demonstrations.
5. **Human Annotation Ground Truth:** While computational extraction and parsing are 100% automated and deterministic, multi-annotator inter-rater reliability (Cohen's $\kappa$) has not yet been collected with an independent second reader. The annotation schema and guidelines are documented in `validation/validation_results.md`.

---

## Citation

```bibtex
@software{g2026algorithmic,
  author    = {Rohita G.},
  title     = {Algorithmic Silence: Computational Analysis of Ophelia in Shakespeare's Hamlet},
  year      = {2026},
  url       = {https://github.com/Rohita-G/Algorithmic-Silence},
  version   = {1.0.0}
}
```
