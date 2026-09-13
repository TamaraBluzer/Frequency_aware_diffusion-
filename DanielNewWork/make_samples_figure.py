"""Draw the sample-comparison figure: real Planar graph beside two generated arms.

    python DanielNewWork/make_samples_figure.py

Writes DanielNewWork/figures/samples.pdf. Standalone on purpose — it does not
import scripts/make_paper_figures.py, so it cannot collide with that file.

Reads only artifacts that the leakage correction did not touch: the seed-0
sample tensors and reports for the k=8 pilot arms, plus the dataset split.

Two anti-cherry-picking rules, both stated in the figure caption: the sample
drawn is index 0 of every arm, and all panels share one Kamada-Kawai layout, so
the real graph is not handed the 2D coordinates its triangulation came from.

READ DanielNewWork/README.md BEFORE REUSING THE OUTPUT. The counts this script
prints are measurements and stand, but the reading of them in main_daniel.tex
predates the leakage correction and is wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fald.data.spectre import load_splits

RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "figures"

# `high` and `random` are measured and printed but not drawn: at column width
# there is room for three panels, and both look like `none`.
DRAWN = ["none", "low"]
MEASURED_ONLY = ["high", "random"]

BAND_COLOUR = {"none": "#666666", "low": "#1b7837", "high": "#d95f02", "random": "#c51b7d"}

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def arm_graph(band: str) -> nx.Graph:
    path = RESULTS / f"adjacency_diffusion_planar_{band}_k8_samples.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — cannot build the sample panel.")
    adj = torch.load(path, weights_only=False)[0]
    return nx.from_numpy_array((np.asarray(adj) > 0).astype(np.uint8))


def arm_ratio(band: str) -> float:
    path = RESULTS / f"adjacency_diffusion_planar_{band}_k8.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — cannot label the sample panel.")
    return json.loads(path.read_text())["evaluation"]["ratio"]


def triangles(graph: nx.Graph) -> int:
    return sum(nx.triangles(graph).values()) // 3


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    real = load_splits("planar")["val"][0]
    panels = [("real (val. split)", real, "#222222", None)]
    for band in DRAWN:
        panels.append((band, arm_graph(band), BAND_COLOUR[band], arm_ratio(band)))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.05, 1.30))
    for ax, (label, graph, colour, ratio) in zip(axes, panels):
        pos = nx.kamada_kawai_layout(graph)
        nx.draw_networkx_edges(graph, pos, ax=ax, width=0.35, edge_color=colour, alpha=0.75)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=3.5, node_color=colour, linewidths=0)
        ax.set_title(label if ratio is None else f"{label} (Ratio {ratio:.0f})", fontsize=7.0, pad=2.0)
        ax.set_axis_off()
        ax.set_aspect("equal")
        # Identical limits across panels, so the titles and counts line up
        # instead of floating with each drawing's convex hull.
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.text(
            0.5,
            -0.03,
            f"{graph.number_of_edges()}e, {triangles(graph)}$\\triangle$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            color="#555555",
        )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "samples.pdf", bbox_inches="tight")
    plt.close(fig)

    counts = {"real": triangles(real), **{b: triangles(arm_graph(b)) for b in DRAWN}}
    counts.update({b: triangles(arm_graph(b)) for b in MEASURED_ONLY})
    print(f"wrote {OUT / 'samples.pdf'}")
    print(f"triangle counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
