"""
Majority Dynamics in Online Networks
=====================================
Implementazione completa del progetto COCOA'15.

Struttura:
  - majority_cascade(G, S)         : simula il processo di attivazione
  - cost_seeds_greedy(G, k, c, fi) : Algorithm 1 (Cost-Seeds-Greedy)
  - wtss_maximal(G, k, c)          : Algorithm 2 (WTSS adattato)
  - my_seeds(G, k, c)              : Algorithm 3 (custom: PageRank-weighted)
  - Funzioni costo: random, d(u)/2, 1/d(u) (custom)
  - Esperimenti STEP 1/2/3 con grafici
"""

import math
import random
import copy
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
#  GRAFO: Barabási-Albert (simula una rete sociale reale)
# ─────────────────────────────────────────────
def build_graph():
    G = nx.read_edgelist("Dataset/facebook_combined.txt")
    return G

# ─────────────────────────────────────────────
#  MAJORITY CASCADE
# ─────────────────────────────────────────────
def majority_cascade(G, S):
    """
    Simula il processo di attivazione con regola Majority.
    Un nodo v si attiva se almeno metà dei suoi vicini sono già attivi.
    Restituisce l'insieme Inf[S] dei nodi attivati.
    """
    infected = set(S)
    while True:
        new_infected = set()
        for v in G.nodes():
            if v not in infected:
                dv = G.degree(v)
                if dv == 0:
                    continue
                neighbors_infected = sum(1 for u in G.neighbors(v) if u in infected)
                if neighbors_infected >= math.ceil(dv / 2):
                    new_infected.add(v)
        if not new_infected:
            break
        infected |= new_infected
    return infected

# ─────────────────────────────────────────────
#  FUNZIONI f1, f2, f3  (per Algorithm 1)
# ─────────────────────────────────────────────
def f1(G, S):
    """f1(S) = sum_{v in V} min(|N(v)∩S|, ceil(d(v)/2))"""
    val = 0
    for v in G.nodes():
        dv = G.degree(v)
        overlap = sum(1 for u in G.neighbors(v) if u in S)
        val += min(overlap, math.ceil(dv / 2))
    return val

def f2(G, S):
    """f2(S) = sum_{v in V} sum_{i=1}^{|N(v)∩S|} max(ceil(d(v)/2) - i + 1, 0)"""
    val = 0
    for v in G.nodes():
        dv = G.degree(v)
        overlap = sum(1 for u in G.neighbors(v) if u in S)
        threshold = math.ceil(dv / 2)
        for i in range(1, overlap + 1):
            val += max(threshold - i + 1, 0)
    return val

def f3(G, S):
    """f3(S) = sum_{v in V} sum_{i=1}^{|N(v)∩S|} max((ceil(d(v)/2)-i+1)/(d(v)-i+1), 0)"""
    val = 0
    for v in G.nodes():
        dv = G.degree(v)
        if dv == 0:
            continue
        overlap = sum(1 for u in G.neighbors(v) if u in S)
        threshold = math.ceil(dv / 2)
        for i in range(1, overlap + 1):
            denom = dv - i + 1
            if denom > 0:
                val += max((threshold - i + 1) / denom, 0)
    return val

def marginal_gain_fi(G, v, S_current, fi_func):
    """Guadagno marginale Δ_v f_i(S) = f_i(S ∪ {v}) - f_i(S)"""
    S_new = S_current | {v}
    return fi_func(G, S_new) - fi_func(G, S_current)

# ─────────────────────────────────────────────
#  ALGORITHM 1: Cost-Seeds-Greedy
# ─────────────────────────────────────────────
def cost_seeds_greedy(G, k, c, fi_func):
    """
    Greedy che seleziona iterativamente il nodo u con
    massimo rapporto Δ_u f_i(S_d) / c(u), finché c(S_d) <= k.
    Restituisce S_p (l'ultimo seed set con costo ≤ k).
    """
    S_p = set()
    S_d = set()
    remaining = set(G.nodes())

    while True:
        best_u = None
        best_ratio = -1
        for v in remaining - S_d:
            cu = c[v]
            if cu == 0:
                continue
            ratio = marginal_gain_fi(G, v, S_d, fi_func) / cu
            if ratio > best_ratio:
                best_ratio = ratio
                best_u = v

        if best_u is None:
            break

        S_p = set(S_d)
        S_d = S_d | {best_u}

        cost_Sd = sum(c[u] for u in S_d)
        if cost_Sd > k:
            break

    return S_p

# ─────────────────────────────────────────────
#  ALGORITHM 2: WTSS adattato (seed set massimale)
# ─────────────────────────────────────────────
def wtss_maximal(G, k, c):
    """
    Versione adattata di WTSS che si ferma al seed set massimale
    con costo ≤ k. Threshold t(v) = ceil(d(v)/2).
    """
    # Inizializzazione
    delta = {v: G.degree(v) for v in G.nodes()}
    k_thresh = {v: math.ceil(G.degree(v) / 2) for v in G.nodes()}
    neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    U = set(G.nodes())
    S = set()
    current_cost = 0

    while U:
        # Case 1: esiste v in U con k(v) == 0
        case1 = [v for v in U if k_thresh[v] == 0]
        if case1:
            v = case1[0]
            for u in list(neighbors[v]):
                if u in U:
                    k_thresh[u] = max(0, k_thresh[u] - 1)
            # rimuovi v da U
            for u in list(neighbors[v]):
                if u in U:
                    delta[u] -= 1
                    neighbors[u].discard(v)
            neighbors[v] = set()
            U.remove(v)
            continue

        # Case 2: esiste v in U con delta(v) < k(v)
        case2 = [v for v in U if delta[v] < k_thresh[v]]
        if case2:
            v = case2[0]
            add_cost = c[v]
            if current_cost + add_cost > k:
                # Budget esaurito: restituiamo S corrente
                return S
            S.add(v)
            current_cost += add_cost
            for u in list(neighbors[v]):
                if u in U:
                    k_thresh[u] = max(0, k_thresh[u] - 1)
            for u in list(neighbors[v]):
                if u in U:
                    delta[u] -= 1
                    neighbors[u].discard(v)
            neighbors[v] = set()
            U.remove(v)
            continue

        # Case 3: scegli v = argmax c(u)*k(u) / (delta(u)*(delta(u)+1))
        best_v = None
        best_score = -1
        for u in U:
            d = delta[u]
            if d == 0:
                score = float('inf')
            else:
                score = (c[u] * k_thresh[u]) / (d * (d + 1))
            if score > best_score:
                best_score = score
                best_v = u

        v = best_v
        # Rimuovi v dal grafo
        for u in list(neighbors[v]):
            if u in U:
                delta[u] -= 1
                neighbors[u].discard(v)
        neighbors[v] = set()
        U.remove(v)

    return S

# ─────────────────────────────────────────────
#  ALGORITHM 3: My-Seeds (PageRank-Cost Greedy)
# ─────────────────────────────────────────────
def my_seeds(G, k, c):
    """
    Algorithm 3 (custom): ordina i nodi per PageRank / c(u),
    poi aggiunge nodi in ordine finché il budget non è esaurito.
    Intuizione: nodi con alto PageRank sono "centrali" nella rete
    e quindi buoni candidati come seed.
    """
    pr = nx.pagerank(G, alpha=0.85)
    # Ratio PageRank / costo
    ranked = sorted(G.nodes(), key=lambda v: pr[v] / max(c[v], 1e-9), reverse=True)
    S = set()
    current_cost = 0
    for v in ranked:
        if current_cost + c[v] <= k:
            S.add(v)
            current_cost += c[v]
    return S

# ─────────────────────────────────────────────
#  FUNZIONI COSTO
# ─────────────────────────────────────────────
def cost_random(G, low=1, high=5, seed=42):
    rng = random.Random(seed)
    return {v: rng.randint(low, high) for v in G.nodes()}

def cost_half_degree(G):
    return {v: max(1, math.ceil(G.degree(v) / 2)) for v in G.nodes()}

def cost_inverse_degree(G):
    """Costo custom: nodi con alto grado costano poco (hub-friendly)."""
    return {v: max(1, math.ceil(len(G.nodes()) / max(G.degree(v), 1))) for v in G.nodes()}

# ─────────────────────────────────────────────
#  MODIFICA ONLINE DEL GRAFO (STEP 2)
# ─────────────────────────────────────────────
def remove_edges(G, x, seed=42):
    G2 = G.copy()
    edges = list(G2.edges())
    rng = random.Random(seed)
    to_remove = rng.sample(edges, min(x, len(edges)))
    G2.remove_edges_from(to_remove)
    return G2

def remove_vertices(G, y, seed=42):
    G2 = G.copy()
    nodes = list(G2.nodes())
    rng = random.Random(seed)
    to_remove = rng.sample(nodes, min(y, len(nodes)))
    G2.remove_nodes_from(to_remove)
    return G2

# ─────────────────────────────────────────────
#  ESPERIMENTI
# ─────────────────────────────────────────────
def run_experiment(G, cost_funcs, budget_range, algorithms, label=""):
    """
    STEP 1: Per ogni funzione costo e ogni algoritmo,
    calcola |Inf[G,S]| al variare del budget k.
    """
    results = {}  # results[cost_name][algo_name] = list of |Inf[G,S]|
    seeds_map = {}  # seeds_map[cost_name][algo_name][k] = S

    for cost_name, c in cost_funcs.items():
        results[cost_name] = {}
        seeds_map[cost_name] = {}
        for algo_name, algo_fn in algorithms.items():
            inf_sizes = []
            seeds_map[cost_name][algo_name] = {}
            for k in budget_range:
                S = algo_fn(G, k, c)
                inf = majority_cascade(G, S)
                inf_sizes.append(len(inf))
                seeds_map[cost_name][algo_name][k] = S
            results[cost_name][algo_name] = inf_sizes
            print(f"  [{label}] cost={cost_name}, algo={algo_name}: done")

    return results, seeds_map

def run_edge_removal(G, S_fixed, x_range, seed=99):
    """STEP 2: Rimuove x archi e calcola |Inf[G',S]|."""
    base_inf = len(majority_cascade(G, S_fixed))
    inf_sizes = []
    for x in x_range:
        G2 = remove_edges(G, x, seed=seed)
        # Mantieni solo nodi di S presenti in G2
        S2 = S_fixed & set(G2.nodes())
        inf = majority_cascade(G2, S2)
        inf_sizes.append(len(inf))
    return base_inf, inf_sizes

def run_vertex_removal(G, S_fixed, y_range, seed=99):
    """STEP 3: Rimuove y vertici e calcola |Inf[G',S]|."""
    base_inf = len(majority_cascade(G, S_fixed))
    inf_sizes = []
    for y in y_range:
        G2 = remove_vertices(G, y, seed=seed)
        S2 = S_fixed & set(G2.nodes())
        inf = majority_cascade(G2, S2)
        inf_sizes.append(len(inf))
    return base_inf, inf_sizes

# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────
COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "s", "^", "D", "v"]

def plot_step1(results, budget_range, n_nodes, out_path):
    cost_names = list(results.keys())
    n_costs = len(cost_names)
    fig, axes = plt.subplots(1, n_costs, figsize=(7 * n_costs, 5), sharey=False)
    if n_costs == 1:
        axes = [axes]

    for ax, cost_name in zip(axes, cost_names):
        algo_results = results[cost_name]
        for i, (algo_name, inf_sizes) in enumerate(algo_results.items()):
            ax.plot(budget_range, inf_sizes, color=COLORS[i],
                    marker=MARKERS[i], label=algo_name, linewidth=2, markersize=5)
        ax.axhline(y=n_nodes, color="gray", linestyle="--", alpha=0.5, label=f"|V|={n_nodes}")
        ax.set_title(f"Funzione costo: {cost_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Budget k", fontsize=11)
        ax.set_ylabel("|Inf[G, S]|", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, n_nodes * 1.1)

    fig.suptitle("STEP 1 — |Inf[G,S]| al variare del budget k", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {out_path}")

def plot_step2(base_inf, inf_sizes_dict, x_range, n_nodes, out_path):
    """Confronto |Inf[G,S]| vs |Inf[G',S]| al variare di x (archi rimossi)."""
    n_algos = len(inf_sizes_dict)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5), sharey=False)
    if n_algos == 1:
        axes = [axes]

    for ax, (algo_name, inf_sizes) in zip(axes, inf_sizes_dict.items()):
        ax.axhline(y=base_inf, color="gray", linestyle="--", linewidth=2,
                   label=f"|Inf[G,S]| = {base_inf}")
        ax.plot(x_range, inf_sizes, color="#E91E63", marker="s",
                linewidth=2, markersize=5, label="|Inf[G',S]|")
        ax.fill_between(x_range, inf_sizes, base_inf,
                        alpha=0.15, color="#E91E63", label="Degrado")
        ax.set_title(f"Algoritmo: {algo_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Numero archi rimossi (x)", fontsize=11)
        ax.set_ylabel("|Inf|", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, n_nodes * 1.1)

    fig.suptitle("STEP 2 — Effetto rimozione archi su |Inf[G',S]|",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {out_path}")

def plot_step3(base_inf, inf_sizes_dict, y_range, n_nodes, out_path):
    """Confronto |Inf[G,S]| vs |Inf[G',S]| al variare di y (vertici rimossi)."""
    n_algos = len(inf_sizes_dict)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5), sharey=False)
    if n_algos == 1:
        axes = [axes]

    for ax, (algo_name, inf_sizes) in zip(axes, inf_sizes_dict.items()):
        ax.axhline(y=base_inf, color="gray", linestyle="--", linewidth=2,
                   label=f"|Inf[G,S]| = {base_inf}")
        ax.plot(y_range, inf_sizes, color="#4CAF50", marker="^",
                linewidth=2, markersize=5, label="|Inf[G',S]|")
        ax.fill_between(y_range, inf_sizes, base_inf,
                        alpha=0.15, color="#4CAF50", label="Degrado")
        ax.set_title(f"Algoritmo: {algo_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Numero vertici rimossi (y)", fontsize=11)
        ax.set_ylabel("|Inf|", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("STEP 3 — Effetto rimozione vertici su |Inf[G',S]|",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {out_path}")

def plot_network_stats(G, out_path):
    """Visualizza proprietà del grafo."""
    degrees = [d for _, d in G.degree()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(degrees, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribuzione dei gradi", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Grado", fontsize=11)
    axes[0].set_ylabel("Frequenza", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Draw small version of the graph
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx(G, pos=pos, ax=axes[1], node_size=20, node_color="#2196F3",
                     edge_color="#BBBBBB", with_labels=False, alpha=0.8, width=0.5)
    axes[1].set_title("Visualizzazione della rete (Barabási-Albert)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(f"Rete: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
                 f"<k>={2*G.number_of_edges()/G.number_of_nodes():.1f}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {out_path}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    out_dir = "/Users/giuseppepiosorrentino/RetiSocialiProj/Results"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Majority Dynamics in Online Networks")
    print("=" * 60)

    # ── Costruzione grafo ──────────────────────
    print("\n[1/6] Costruzione grafo...")
    G = build_graph()
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"      Nodi: {n_nodes}, Archi: {n_edges}")
    print(f"      Grado medio: {2*n_edges/n_nodes:.2f}")
    print(f"      Clustering medio: {nx.average_clustering(G):.4f}")

    plot_network_stats(G, f"{out_dir}/00_network_stats.png")

    # ── Funzioni costo ─────────────────────────
    print("\n[2/6] Calcolo funzioni costo...")
    c_random = cost_random(G, low=1, high=5)
    c_half   = cost_half_degree(G)
    c_inv    = cost_inverse_degree(G)

    cost_funcs = {
        "Random [1,5]":    c_random,
        "d(u)/2":          c_half,
        "n/d(u) (custom)": c_inv,
    }

    # ── Algoritmi ─────────────────────────────
    algorithms = {
        "Greedy-f1":  lambda G, k, c: cost_seeds_greedy(G, k, c, f1),
        "WTSS":       lambda G, k, c: wtss_maximal(G, k, c),
        "My-Seeds\n(PageRank)": lambda G, k, c: my_seeds(G, k, c),
    }

    # ── STEP 1: variare budget k ───────────────
    print("\n[3/6] STEP 1 — Esperimenti al variare del budget k...")
    budget_range = list(range(5, 101, 5))
    results, seeds_map = run_experiment(G, cost_funcs, budget_range, algorithms, label="G")

    # Usa come cost_name di riferimento "Random [1,5]" per Step 2/3
    REF_COST = "Random [1,5]"
    REF_K    = 40   # budget di riferimento per Step 2 e 3

    print("\n[4/6] Generazione grafici STEP 1...")
    plot_step1(results, budget_range, n_nodes, f"{out_dir}/01_step1_inf_vs_budget.png")

    # ── STEP 2: rimozione archi ────────────────
    print("\n[5/6] STEP 2 — Rimozione archi...")
    x_range = list(range(0, min(151, n_edges), 10))
    step2_inf = {}
    step2_base = {}
    for algo_name, algo_fn in algorithms.items():
        S_fixed = seeds_map[REF_COST][algo_name][REF_K]
        base_inf, inf_sizes = run_edge_removal(G, S_fixed, x_range)
        step2_inf[algo_name.replace("\n", " ")] = inf_sizes
        step2_base[algo_name.replace("\n", " ")] = base_inf
        print(f"  algo={algo_name.replace(chr(10),' ')}: base={base_inf}, "
              f"dopo {x_range[-1]} archi rimossi: {inf_sizes[-1]}")

    # Usa base comune (media) per il plot
    avg_base = int(np.mean(list(step2_base.values())))
    plot_step2(avg_base, step2_inf, x_range, n_nodes,
               f"{out_dir}/02_step2_edge_removal.png")

    # ── STEP 3: rimozione vertici ──────────────
    print("\n[6/6] STEP 3 — Rimozione vertici...")
    y_range = list(range(0, min(51, n_nodes), 5))
    step3_inf = {}
    step3_base = {}
    for algo_name, algo_fn in algorithms.items():
        S_fixed = seeds_map[REF_COST][algo_name][REF_K]
        base_inf, inf_sizes = run_vertex_removal(G, S_fixed, y_range)
        step3_inf[algo_name.replace("\n", " ")] = inf_sizes
        step3_base[algo_name.replace("\n", " ")] = base_inf
        print(f"  algo={algo_name.replace(chr(10),' ')}: base={base_inf}, "
              f"dopo {y_range[-1]} vertici rimossi: {inf_sizes[-1]}")

    avg_base3 = int(np.mean(list(step3_base.values())))
    plot_step3(avg_base3, step3_inf, y_range, n_nodes,
               f"{out_dir}/03_step3_vertex_removal.png")

    # ── Riepilogo ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Riepilogo file generati:")
    for f in ["00_network_stats.png",
              "01_step1_inf_vs_budget.png",
              "02_step2_edge_removal.png",
              "03_step3_vertex_removal.png"]:
        print(f"    {out_dir}/{f}")
    print("=" * 60)
    print("  COMPLETATO!")
