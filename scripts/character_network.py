#!/usr/bin/env python3
"""
character_network.py

Network science analysis of Shakespeare's Hamlet with focus on Ophelia's structural position.
Constructs two network models:
  1. Turn-Adjacency Network: Direct conversational reply (Character A speaks -> Character B replies).
  2. Ophelia Ego Network: Direct interlocutors connected to Ophelia.

Computes academic centrality metrics (Degree, Weighted Degree, Betweenness, Closeness, PageRank),
exports metrics to CSV (results/tables/hamlet_network_centrality.csv), and generates publication-grade visualizations:
  - results/figures/hamlet_character_network_clean_labels.png
  - results/figures/ophelia_ego_network.png
"""

from pathlib import Path
from itertools import combinations
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_ALL_CSV = DATA_PROCESSED_DIR / "hamlet_all_dialogue.csv"
if not INPUT_ALL_CSV.exists() and (BASE_DIR / "data" / "hamlet_all_dialogue.csv").exists():
    INPUT_ALL_CSV = BASE_DIR / "data" / "hamlet_all_dialogue.csv"

INPUT_MAIN_CSV = DATA_PROCESSED_DIR / "hamlet_main_characters.csv"
if not INPUT_MAIN_CSV.exists() and (BASE_DIR / "data" / "hamlet_main_characters.csv").exists():
    INPUT_MAIN_CSV = BASE_DIR / "data" / "hamlet_main_characters.csv"

CENTRALITY_CSV = TABLES_DIR / "hamlet_network_centrality.csv"

MAIN_CHARACTERS = [
    "HAMLET",
    "KING CLAUDIUS",
    "QUEEN GERTRUDE",
    "POLONIUS",
    "OPHELIA",
    "LAERTES",
    "HORATIO"
]

NODE_COLORS = {
    "OPHELIA": "#D95F02",        # Highlight: Vibrant terracotta
    "HAMLET": "#7570B3",         # Melancholic purple / slate
    "KING CLAUDIUS": "#E7298A",  # Royal magenta
    "QUEEN GERTRUDE": "#E6AB02", # Tarnished gold
    "POLONIUS": "#66A61E",       # Pedantic green
    "LAERTES": "#A6761D",        # Earthy bronze
    "HORATIO": "#1F78B4"         # Loyal deep blue
}


def construct_turn_adjacency_graph(df_dialogue, characters=None):
    """
    Constructs an undirected, weighted turn-adjacency network from the complete
    chronological dialogue sequence.
    
    METHODOLOGICAL SAFEGUARD:
    Adjacency is computed across the complete speech sequence to prevent false
    adjacency artifacts from skipping intermediate characters.
    
    EDGE ATTRIBUTES:
      - count: observed contiguous turn adjacency frequency (integer >= 1)
      - weight: identical to count (for NetworkX tie-strength algorithms)
      - distance: 1.0 / count (modeling choice for shortest-path analyses)
    """
    all_edge_counts = Counter()
    for i in range(len(df_dialogue) - 1):
        row_curr = df_dialogue.iloc[i]
        row_next = df_dialogue.iloc[i + 1]
        if (row_curr["act"] == row_next["act"]) and (row_curr["scene"] == row_next["scene"]):
            c1 = row_curr["character"]
            c2 = row_next["character"]
            if c1 != c2:  # Exclude self-loops
                pair = tuple(sorted([c1, c2]))
                all_edge_counts[pair] += 1

    G = nx.Graph()
    if characters is not None:
        char_set = set(characters)
        for char in characters:
            G.add_node(char)
        for (c1, c2), count in all_edge_counts.items():
            if c1 in char_set and c2 in char_set:
                distance = 1.0 / count
                G.add_edge(c1, c2, count=count, weight=count, distance=distance)
    else:
        for char in df_dialogue["character"].unique():
            G.add_node(char)
        for (c1, c2), count in all_edge_counts.items():
            distance = 1.0 / count
            G.add_edge(c1, c2, count=count, weight=count, distance=distance)

    return G


def compute_centrality_dataframe(G, word_counts):
    """
    Computes centrality metrics with explicit definitions:
      - Unweighted_Degree: Number of distinct interlocutor neighbors
      - Turn_Exchanges: Total contiguous transitions (strength / weighted degree)
      - Betweenness_Unweighted: Primary betweenness using unweighted topological shortest paths
      - Betweenness_Distance: Sensitivity betweenness using inverse-frequency distance (d = 1 / count)
      - Betweenness_Centrality: Set to Betweenness_Unweighted as primary
      - Closeness_Centrality: Unweighted topological closeness
      - Eigenvector_Centrality: Weighted by interaction count
      - PageRank: Random-walk stationary distribution weighted by count
    """
    degree_unweighted = dict(G.degree())
    degree_weighted = dict(G.degree(weight="count"))
    bw_unweighted = nx.betweenness_centrality(G, weight=None, normalized=True)
    bw_distance = nx.betweenness_centrality(G, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(G)
    
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="count", max_iter=1000)
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}
        
    pagerank = nx.pagerank(G, weight="count")

    nodes = list(G.nodes())
    df = pd.DataFrame({
        "Character": nodes,
        "Total_Words": [word_counts.get(c, 0) for c in nodes],
        "Turn_Exchanges": [degree_weighted.get(c, 0) for c in nodes],
        "Unweighted_Degree": [degree_unweighted.get(c, 0) for c in nodes],
        "Betweenness_Unweighted": [bw_unweighted.get(c, 0.0) for c in nodes],
        "Betweenness_Distance": [bw_distance.get(c, 0.0) for c in nodes],
        "Betweenness_Centrality": [bw_unweighted.get(c, 0.0) for c in nodes],  # Primary metric
        "Closeness_Centrality": [closeness.get(c, 0.0) for c in nodes],
        "Eigenvector_Centrality": [eigenvector.get(c, 0.0) for c in nodes],
        "PageRank": [pagerank.get(c, 0.0) for c in nodes]
    }).sort_values(by="Turn_Exchanges", ascending=False)
    
    return df


def build_and_analyze_networks():
    print("=" * 65)
    print("HAMLET: CHARACTER NETWORK SCIENCE & STRUCTURAL POSITION")
    print("=" * 65)

    if not INPUT_ALL_CSV.exists():
        from parse_hamlet_main_chars import main as parse_main
        parse_main()

    df_all = pd.read_csv(INPUT_ALL_CSV)

    # Calculate total word counts per character for node sizing
    all_word_counts = df_all.groupby("character")["word_count"].sum().to_dict()

    # 1. Full Play Graph
    G_full = construct_turn_adjacency_graph(df_all)
    full_centrality_df = compute_centrality_dataframe(G_full, all_word_counts)
    full_centrality_path = TABLES_DIR / "hamlet_full_play_network_centrality.csv"
    full_centrality_df.to_csv(full_centrality_path, index=False)
    print(f"[OK] Saved Full Play Centrality to: {full_centrality_path}")

    # 2. Seven Principal Court Characters Subgraph
    G_turn = construct_turn_adjacency_graph(df_all, characters=MAIN_CHARACTERS)
    for char in MAIN_CHARACTERS:
        G_turn.nodes[char]["word_count"] = all_word_counts.get(char, 0)

    centrality_df = compute_centrality_dataframe(G_turn, all_word_counts)
    centrality_df.to_csv(CENTRALITY_CSV, index=False)
    print(f"[OK] Saved Principal Centrality Metrics to: {CENTRALITY_CSV}")

    print("\n--- Network Centrality Summary (7 Principal Roles) ---")
    print(centrality_df[["Character", "Total_Words", "Turn_Exchanges", "Unweighted_Degree", "Betweenness_Unweighted", "Betweenness_Distance", "PageRank"]].to_string(index=False))

    # -------------------------------------------------------------
    # Visualization 1: Publication-Grade Main Character Network
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 8.5), dpi=300)
    pos = nx.kamada_kawai_layout(G_turn, weight=None)

    # Scale node sizes based on square root of word count (visually balanced)
    node_sizes = [np.sqrt(G_turn.nodes[n]["word_count"]) * 65 + 400 for n in G_turn.nodes()]
    node_colors = [NODE_COLORS.get(n, "#888888") for n in G_turn.nodes()]

    # Scale edge widths based on interaction weight
    edges = G_turn.edges()
    weights = [G_turn[u][v]["weight"] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [1.0 + (w / max_weight) * 9.0 for w in weights]
    edge_alphas = [0.35 + (w / max_weight) * 0.5 for w in weights]

    # Draw edges with variable opacity
    for (u, v), w, width in zip(edges, weights, edge_widths):
        is_ophelia_edge = (u == "OPHELIA" or v == "OPHELIA")
        edge_color = "#D95F02" if is_ophelia_edge else "#7F7F7F"
        alpha = 0.85 if is_ophelia_edge else 0.4
        nx.draw_networkx_edges(
            G_turn, pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color=edge_color
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G_turn, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#222222",
        linewidths=1.8
    )

    # Draw labels with clear white halo for maximum legibility
    labels = {n: f"{n}\n({all_word_counts[n]:,} w)" for n in G_turn.nodes()}
    for node, (x, y) in pos.items():
        weight_font = "bold" if node == "OPHELIA" else "semibold"
        color_font = "#111111"
        plt.text(
            x, y, labels[node],
            fontsize=9.5,
            fontweight=weight_font,
            ha="center", va="center",
            color=color_font,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92, linewidth=0.8)
        )

    plt.title("Shakespeare's Hamlet: Principal Character Turn-Adjacency Network", fontsize=14, fontweight="bold", pad=18)
    plt.axis("off")
    plt.tight_layout()

    fig1_path = FIGURES_DIR / "hamlet_character_network_clean_labels.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure: {fig1_path}")

    # -------------------------------------------------------------
    # Visualization 2: Ophelia's Ego Network (Direct Ties Only)
    # -------------------------------------------------------------
    plt.figure(figsize=(8.5, 7.5), dpi=300)
    ophelia_neighbors = list(G_turn.neighbors("OPHELIA"))
    ego_nodes = ["OPHELIA"] + ophelia_neighbors
    G_ego = G_turn.subgraph(ego_nodes)

    ego_pos = nx.spring_layout(G_ego, seed=42, k=0.8)
    ego_pos["OPHELIA"] = np.array([0.0, 0.0])  # Pin Ophelia to center

    # Distribute neighbors in a circle around Ophelia
    num_neighbors = len(ophelia_neighbors)
    angles = np.linspace(0, 2 * np.pi, num_neighbors, endpoint=False)
    radius = 1.0
    for idx, neighbor in enumerate(ophelia_neighbors):
        ego_pos[neighbor] = np.array([radius * np.cos(angles[idx]), radius * np.sin(angles[idx])])

    # Node sizes and colors
    ego_node_sizes = [np.sqrt(G_ego.nodes[n]["word_count"]) * 60 + 400 for n in G_ego.nodes()]
    ego_node_colors = [NODE_COLORS.get(n, "#888888") for n in G_ego.nodes()]

    # Draw ego edges with weights labeled
    for u, v in G_ego.edges():
        w = G_ego[u][v]["weight"]
        width = 1.5 + (w / max_weight) * 10.0
        is_direct = (u == "OPHELIA" or v == "OPHELIA")
        color = "#D95F02" if is_direct else "#BDBDBD"
        alpha = 0.9 if is_direct else 0.4
        nx.draw_networkx_edges(G_ego, ego_pos, edgelist=[(u, v)], width=width, alpha=alpha, edge_color=color)

        if is_direct:
            mid_x = (ego_pos[u][0] + ego_pos[v][0]) / 2.0
            mid_y = (ego_pos[u][1] + ego_pos[v][1]) / 2.0
            plt.text(
                mid_x, mid_y, f"{w} turns",
                fontsize=8.5, fontweight="bold",
                color="#8C2D04",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF7BC", edgecolor="#FEC44F", alpha=0.95)
            )

    nx.draw_networkx_nodes(
        G_ego, ego_pos,
        node_size=ego_node_sizes,
        node_color=ego_node_colors,
        edgecolors="#222222",
        linewidths=2.0
    )

    for node, (x, y) in ego_pos.items():
        is_focal = (node == "OPHELIA")
        plt.text(
            x, y, f"{node}\n({all_word_counts[node]:,} w)",
            fontsize=9.5 if is_focal else 8.5,
            fontweight="bold" if is_focal else "medium",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaaaaa", alpha=0.92)
        )

    plt.title("Ophelia's Conversational Ego Network in the Court Subgraph", fontsize=13, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "ophelia_ego_network.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure: {fig2_path}")


if __name__ == "__main__":
    build_and_analyze_networks()
