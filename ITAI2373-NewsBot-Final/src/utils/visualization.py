"""
Plotting functions (matplotlib for static report figures, Plotly for
interactive notebook/web views). Every function returns the figure and can
save it when ``path`` is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PALETTE = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#ca8a04", "#db2777"]
CATEGORY_COLORS = {"business": "#2563eb", "entertainment": "#db2777", "politics": "#dc2626",
                   "sport": "#16a34a", "tech": "#9333ea"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})


def _save(fig, path: Optional[Path]):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig


def bar(series: pd.Series, title: str, xlabel: str = "", path: Optional[Path] = None, color=None, horizontal=False):
    """Bar chart of a Series (category colours where possible)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = color or [CATEGORY_COLORS.get(str(i), PALETTE[0]) for i in series.index]
    (series.plot.barh if horizontal else series.plot.bar)(ax=ax, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if not horizontal:
        plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return _save(fig, path)


def confusion_matrix_plot(cm: List[List[int]], labels: List[str], title: str, path: Optional[Path] = None):
    """Row-normalised confusion matrix with counts."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def model_comparison(df: pd.DataFrame, title: str, metric: str = "macro_f1", path: Optional[Path] = None):
    """Horizontal bar chart comparing models on one metric."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    d = df.sort_values(metric)
    ax.barh(d["model"], d[metric], color=[PALETTE[1] if "Ensemble" in m or "2.0" in m else PALETTE[0] for m in d["model"]])
    lo = max(0, d[metric].min() - 0.05)
    ax.set_xlim(lo, 1.0)
    for y, v in enumerate(d[metric]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center")
    ax.set_title(title)
    ax.set_xlabel(metric)
    fig.tight_layout()
    return _save(fig, path)


def timeline(df: pd.DataFrame, x: str, y: str, group: Optional[str], title: str, ylabel: str = "",
             path: Optional[Path] = None, anomalies: Optional[pd.DataFrame] = None):
    """Line chart over periods, one line per group, anomalies circled."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    groups = df.groupby(group) if group else [("all", df)]
    for i, (name, g) in enumerate(groups):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", label=str(name),
                color=CATEGORY_COLORS.get(str(name), PALETTE[i % len(PALETTE)]))
    if anomalies is not None and not anomalies.empty:
        a = anomalies[anomalies["anomaly"]]
        ax.scatter(a[x], a[y], s=160, facecolors="none", edgecolors="black", linewidths=1.5, label="anomaly", zorder=5)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def heatmap(table: pd.DataFrame, title: str, path: Optional[Path] = None, fmt: str = "{:.2f}", cmap="RdYlGn",
            vmin=None, vmax=None):
    """Annotated heatmap of a DataFrame."""
    fig, ax = plt.subplots(figsize=(1.2 * table.shape[1] + 2, 0.55 * table.shape[0] + 1.6))
    im = ax.imshow(table.values.astype(float), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if pd.notna(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def topic_trends(trends: pd.DataFrame, title: str, top_n: int = 6, path: Optional[Path] = None):
    """Share of the ``top_n`` topics per period."""
    top = trends.groupby("topic")["share"].mean().nlargest(top_n).index
    d = trends[trends["topic"].isin(top)].pivot_table(index="period", columns="topic", values="share")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    d.plot(ax=ax, marker="o", color=PALETTE[: len(d.columns)])
    ax.set_title(title)
    ax.set_ylabel("average topic share")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def network(graph: nx.Graph, title: str, top_n: int = 40, path: Optional[Path] = None, seed: int = 7):
    """Draw the most-mentioned entities and the strongest links between them."""
    ranked = sorted(graph.nodes, key=lambda n: -graph.nodes[n].get("mentions", 0))
    places = [n for n in ranked if graph.nodes[n].get("type") in ("GPE", "LOC")][: top_n // 3]
    actors = [n for n in ranked if graph.nodes[n].get("type") not in ("GPE", "LOC")][: top_n - len(places)]
    nodes = actors + places  # people/organisations first, places capped at one third
    g = graph.subgraph(nodes)
    pos = nx.spring_layout(g, k=0.9, seed=seed, weight="weight")
    types = nx.get_node_attributes(g, "type")
    type_colors = {"PERSON": "#f97316", "ORG": "#2563eb", "GPE": "#16a34a", "LOC": "#16a34a", "PRODUCT": "#9333ea"}
    sizes = [60 + 6 * g.nodes[n].get("mentions", 1) ** 0.9 for n in g]
    weights = [0.3 + 0.15 * g.edges[e]["weight"] ** 0.7 for e in g.edges]
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(g, pos, width=weights, alpha=0.25, ax=ax)
    nx.draw_networkx_nodes(g, pos, node_size=sizes, node_color=[type_colors.get(types.get(n), "#64748b") for n in g],
                           alpha=0.85, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    for t, c in [("PERSON", "#f97316"), ("ORG", "#2563eb"), ("Place", "#16a34a")]:
        ax.scatter([], [], c=c, label=t)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def stacked_share(table: pd.DataFrame, title: str, path: Optional[Path] = None):
    """Stacked horizontal bars of shares (rows sum to 1)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    table.plot.barh(stacked=True, ax=ax, color=[CATEGORY_COLORS.get(c, PALETTE[i % 8]) for i, c in enumerate(table.columns)])
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    fig.tight_layout()
    return _save(fig, path)


def system_architecture(path: Optional[Path] = None):
    """Static architecture diagram for documentation and slides."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=9, bold=False):
        """Draw one labelled box."""
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", lw=1.2, alpha=0.95))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        """Draw one arrow."""
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3))

    box(0.2, 4.3, 2.3, 1.2, "Data sources\nBBC News (2,092, EN)\nWebz.io (13,213, 6 langs,\n2023–2026)", "#e2e8f0")
    box(0.2, 2.5, 2.3, 1.3, "data_processing\nloader · validator\npreprocessor (6 langs)\nfeatures (TF-IDF, LSA)", "#cbd5e1")
    arrow(1.35, 4.3, 1.35, 3.8)
    mods = [
        (3.1, "A · Content Analysis\nensemble classifier\nsentiment + emotions\nNER + knowledge graph\nLDA / NMF topics", "#dbeafe"),
        (5.4, "B · Language Models\nTextRank / DistilBART\nsemantic search\n(multilingual MiniLM)\ncontent enhancement", "#dcfce7"),
        (7.7, "C · Multilingual\nown char n-gram langID\nMarianMT translation\ncross-lingual analysis\ncultural context", "#fef9c3"),
    ]
    ax.plot([2.5, 2.8, 2.8, 8.75], [3.15, 3.15, 4.85, 4.85], color="#334155", lw=1.3)
    for x, t, c in mods:
        box(x, 2.3, 2.1, 2.2, t, c)
        arrow(x + 1.05, 4.85, x + 1.05, 4.5)
    box(3.1, 0.3, 6.7, 1.3, "NewsBot2 orchestrator  ·  analyze_article · batch_analysis · process_query · insights report · export",
        "#e9d5ff", 9.5, True)
    for x, _, _ in mods:
        arrow(x + 1.05, 2.3, x + 1.05, 1.6)
    box(10.3, 2.3, 1.6, 2.2, "D · Conversation\nintent classifier\nslot extraction\ncontext memory\nresponses\n(+ optional Ollama)", "#fee2e2")
    arrow(10.3, 3.4, 9.8, 1.2)
    box(10.3, 4.8, 1.6, 0.9, "User\n(chat / notebook)", "#f1f5f9")
    arrow(11.1, 4.8, 11.1, 4.5)
    box(10.3, 0.3, 1.6, 1.3, "Outputs\nreports · CSV/JSON\ncharts", "#f1f5f9")
    arrow(9.8, 0.95, 10.3, 0.95)
    ax.set_xlim(0, 12.1)
    ax.set_ylim(0, 6)
    ax.set_title("NewsBot 2.0 – system architecture", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
