"""Draw the copying result: a conditioning graph beside its own generated copy.

    python DanielNewWork/make_copying_figure.py

Writes DanielNewWork/figures/copying.pdf and prints every number in the caption.

This is the per-graph, visual instance of the retrieval result in main.tex S4.3.
All three panels share one layout, computed on the conditioning graph, so an
edge the sample shares with it lands in the same place in both panels: copied
structure overlays, invented structure does not. That layout choice is what
makes the comparison legible and is stated in the caption -- it is not a
neutral drawing of each graph on its own terms.

The `none` arm is the control that carries the argument. It is drawn against
the same conditioning graph despite never having seen its spectrum, so its
panel shows what this figure looks like with no copying: overlap at the
mismatched rate, and no alignment to recover.

Standalone on purpose -- imports nothing from scripts/, so it cannot collide
with concurrent edits there.
"""

from __future__ import annotations

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

N = 64
IDX = 0  # index 0 of every arm, fixed in advance; see README
SHARED, INVENTED, BASE = "#1b7837", "#bbbbbb", "#222222"

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def upper(graph: nx.Graph) -> np.ndarray:
    return (nx.to_numpy_array(graph, nodelist=range(N)) > 0)[np.triu_indices(N, 1)]


def f1(target: np.ndarray, pred: np.ndarray) -> float:
    tp = int((target & pred).sum())
    precision = tp / max(pred.sum(), 1)
    recall = tp / max(target.sum(), 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def arm_samples(band: str) -> list[nx.Graph]:
    path = RESULTS / f"adjacency_diffusion_planar_{band}_k8_samples.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — cannot build the figure.")
    return [nx.from_numpy_array((np.asarray(a) > 0).astype(np.uint8)) for a in torch.load(path, weights_only=False)]


def arm_stats(real: list[np.ndarray], gen: list[np.ndarray]) -> dict[str, float]:
    """Matched vs mismatched F1 and top-1 retrieval, over all 32 graphs."""
    matched = [f1(real[i], gen[i]) for i in range(len(gen))]
    mismatched = [f1(real[j], gen[i]) for i in range(len(gen)) for j in range(len(real)) if i != j]
    top1 = sum(1 for i in range(len(gen)) if int(np.argmax([f1(real[j], gen[i]) for j in range(len(real))])) == i)
    return {
        "matched": float(np.mean(matched)),
        "mismatched": float(np.mean(mismatched)),
        "top1": top1,
        "n": len(gen),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    val = load_splits("planar")["val"]
    real_vecs = [upper(g) for g in val]
    cond = val[IDX]
    cond_vec = real_vecs[IDX]

    # One layout for every panel, taken from the conditioning graph.
    pos = nx.kamada_kawai_layout(cond)

    arms = {}
    for band in ("low", "none"):
        samples = arm_samples(band)
        arms[band] = {
            "graph": samples[IDX],
            "stats": arm_stats(real_vecs, [upper(g) for g in samples]),
        }

    panels = [
        ("conditioning graph", cond, None),
        ("low $k{=}8$", arms["low"]["graph"], "low"),
        ("none", arms["none"]["graph"], "none"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(3.10, 1.44))
    printed = {}
    for ax, (label, graph, band) in zip(axes, panels):
        if band is None:
            nx.draw_networkx_edges(graph, pos, ax=ax, width=0.4, edge_color=BASE, alpha=0.85)
            title = label
            sub = f"val. #{IDX}, {graph.number_of_edges()} edges"
        else:
            shared = [e for e in graph.edges() if cond.has_edge(*e)]
            invented = [e for e in graph.edges() if not cond.has_edge(*e)]
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=invented, width=0.3, edge_color=INVENTED, alpha=0.85)
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=shared, width=0.5, edge_color=SHARED, alpha=0.95)
            score = f1(cond_vec, upper(graph))
            title = f"{label}, $F_1$ {score:.2f}"
            sub = f"{len(shared)}/{graph.number_of_edges()} shared"
            printed[band] = (len(shared), graph.number_of_edges(), score)

        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=3.0, node_color="#444444", linewidths=0)
        ax.set_title(title, fontsize=6.8, pad=2.0)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#555555")

    fig.tight_layout(pad=0.22)
    fig.savefig(OUT / "copying.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT / 'copying.pdf'}\n")
    print("numbers to check against the caption:")
    print(f"  conditioning graph val #{IDX}: {cond.number_of_edges()} edges")
    for band, (shared, total, score) in printed.items():
        print(f"  {band:5s} sample #{IDX}: {shared}/{total} shared with it, F1 {score:.3f}")
    for band in ("low", "none"):
        s = arms[band]["stats"]
        print(
            f"  {band:5s} over all {s['n']}: matched F1 {s['matched']:.3f}, "
            f"mismatched {s['mismatched']:.3f}, top-1 {s['top1']}/{s['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
