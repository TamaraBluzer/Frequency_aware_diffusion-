"""Regenerate the ACL paper's figures from results/*.json — no hand-typed numbers.

    python scripts/make_paper_figures.py

Writes vector PDFs into paper/figures/, sized for the ACL two-column layout
(\\textwidth = 6.3in for a figure*, \\columnwidth = 3.0in for a single column).

Same contract as scripts/make_figures.py: every figure reads its own committed
JSON and raises if a source file is missing, rather than silently emitting a
figure with a gap in it.
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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.paths import results_dir

PAPER_FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"

# One colour per conditioning arm, shared across both panels of Figure 1 so a
# band keeps its identity between the frequency view and the information view.
BAND_COLOUR = {
    "low": "#008060",
    "high": "#d55e00",
    "random": "#a64d79",
    "gaussian": "#0072b2",
    "shuffled": "#7161a8",
}
BAND_MARKER = {
    "low": "o",
    "high": "s",
    "random": "^",
    "gaussian": "v",
    "shuffled": "D",
}
BAND_STYLE = {"low": "-", "high": "-", "random": "--", "gaussian": ":", "shuffled": "-."}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    }
)


def _load(name: str) -> dict:
    path = results_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the paper figure that needs it cannot be built. "
            f"See docs/REPORT.md for which results are committed."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def fig_sweep(out: Path) -> None:
    """Figure 1: the k-sweep in two views — by frequency, and by probe recovery.

    Left panel is the experiment the proposal asked for (quality against cutoff
    k, one line per band). Right panel re-plots the identical points against how
    many of the target's edges the rank-and-threshold probe recovers from the
    condition alone, which is what separates the two minima.

    Two things the panels have to keep straight, both at a reviewer's request:

    * Evidence weight. Twenty-five of the cells are a single run at seed 0 and
      carry no spread at all; only three cells (high/k=32, low/k=8, and the
      unconditioned baseline) were repeated over three seeds. Single runs are
      drawn as hollow markers and never carry an error bar; the repeated cells
      are filled stars with std bars, and the baseline a dashed line with a
      shaded +/-sd band.
    * What the x-axis of (b) is. It is what one specific probe recovers, not an
      information content that any decoder could reach, so both the label and
      the floor are attributed to the estimator that produced them.
    """
    sweep = _load("k_sweep_planar.json")
    seeds = _load("k_sweep_planar_seedcheck.json")
    leak = _load("condition_leakage_planar.json")

    ratio = {(r["band"], r["k"]): r["ratio"] for r in sweep["rows"]}
    recovery = {(r["band"], r["k"]): r["edge_recovery"] for r in leak["rows"]}
    agg = {}
    for name, band, k in (("none", "none", 2), ("low_k8", "low", 8), ("high_k32", "high", 32)):
        values = [r["ratio"] for r in seeds["rows"] if r["band"] == band and r["k"] == k]
        if len(values) != 3:
            raise ValueError(f"Expected three seeds for {name}, got {len(values)}")
        agg[name] = {"ratio_mean": np.mean(values), "ratio_std": np.std(values, ddof=1)}
    base_mean = agg["none"]["ratio_mean"]
    base_std = agg["none"]["ratio_std"]

    # The probe resolves a sign against the true edge set, so the analytic rate
    # is not its floor: a condition carrying nothing about the target still
    # scores above it, by an amount that depends on the orientation rule the
    # results file was written under. So measure the floor instead of asserting
    # it. The gaussian arm carries nothing by construction, so whatever it
    # scores under the active rule is what "nothing" looks like on this axis.
    gaussian_rows = [r["edge_recovery"] for r in leak["rows"] if r["band"] == "gaussian"]
    if not gaussian_rows:
        raise ValueError(
            "condition_leakage_planar.json has no gaussian rows — the empirical "
            "floor for Figure 1(b) cannot be measured."
        )
    floor = 100.0 * sum(gaussian_rows) / len(gaussian_rows)
    analytic = 100.0 * leak["rows"][0]["chance_recovery"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 2.85), sharey=True)
    band_handles: list = []
    evidence_handles: list = []

    # --- left: the frequency view -------------------------------------------
    ks = sweep["k_values"]
    for band in BAND_COLOUR:
        ys = [ratio[(band, k)] for k in ks]
        line, = left.plot(
            ks, ys, marker=BAND_MARKER[band], markersize=4.8,
            markerfacecolor="white", markeredgecolor=BAND_COLOUR[band],
            markeredgewidth=1.0, linewidth=1.25, linestyle=BAND_STYLE[band],
            color=BAND_COLOUR[band], label=band,
        )
        band_handles.append(line)

    left.axhspan(base_mean - base_std, base_mean + base_std, color="#666666", alpha=0.18, zorder=0)
    left.axhline(base_mean, color="#666666", linewidth=1.0, linestyle="--", zorder=1)

    # Overlay the three-seed means for the two cells that were re-run, with std
    # error bars: seed 0 alone was low/k=8's most favourable draw.
    repeated = [("high_k32", "high", 32), ("low_k8", "low", 8)]
    for name, band, k in repeated:
        left.errorbar(
            [k], [agg[name]["ratio_mean"]], yerr=[agg[name]["ratio_std"]], fmt="*",
            markersize=10, color=BAND_COLOUR[band], markeredgecolor="black",
            markeredgewidth=0.5, elinewidth=1.2, capsize=3, zorder=5,
        )
    left.annotate("low, $k=8$", (8, agg["low_k8"]["ratio_mean"]), xytext=(-12, 26),
                  textcoords="offset points", fontsize=8, color=BAND_COLOUR["low"],
                  arrowprops={"arrowstyle": "-", "color": BAND_COLOUR["low"]})
    left.annotate("high, $k=32$", (32, agg["high_k32"]["ratio_mean"]), xytext=(-74, 18),
                  textcoords="offset points", fontsize=8, color=BAND_COLOUR["high"],
                  arrowprops={"arrowstyle": "-", "color": BAND_COLOUR["high"]})
    left.set_xscale("log", base=2)
    left.set_xticks(ks, [str(k) for k in ks])
    left.set_xlabel("Cutoff $k$ (eigenpairs)")
    left.set_ylabel("MMD Ratio (lower is better)")
    left.set_title("(a) Quality versus cutoff", loc="left", pad=8)
    left.set_ylim(-15, 385)
    left.set_yticks([0, 100, 200, 300])

    # --- right: the probe view ----------------------------------------------
    for band in BAND_COLOUR:
        xs = [recovery[(band, k)] * 100 for k in ks]
        ys = [ratio[(band, k)] for k in ks]
        right.scatter(xs, ys, marker=BAND_MARKER[band], s=28, facecolors="none",
                      edgecolors=BAND_COLOUR[band], linewidths=1.0, zorder=3)
    for name, band, k in repeated:
        right.errorbar(
            [recovery[(band, k)] * 100], [agg[name]["ratio_mean"]],
            yerr=[agg[name]["ratio_std"]], fmt="*", markersize=10,
            color=BAND_COLOUR[band], markeredgecolor="black", markeredgewidth=0.5,
            elinewidth=1.2, capsize=3, zorder=5,
        )
    for band, k, label_xy in (("low", 8, (37, 127)), ("high", 32, (53, 67)),
                              ("low", 16, (43, 237)), ("high", 16, (48, 364))):
        key = f"{band}_k{k}"
        y = agg[key]["ratio_mean"] if key in agg else ratio[(band, k)]
        right.annotate(f"{band}, $k={k}$", (100 * recovery[(band, k)], y),
                       xytext=label_xy, textcoords="data", fontsize=8,
                       color=BAND_COLOUR[band],
                       arrowprops={"arrowstyle": "-", "color": BAND_COLOUR[band], "lw": 0.7})

    right.axhspan(base_mean - base_std, base_mean + base_std, color="#666666", alpha=0.18, zorder=0)
    right.axhline(base_mean, color="#666666", linewidth=1.0, linestyle="--", zorder=1)
    right.axvline(floor, color="#999999", linewidth=0.9, linestyle=":", zorder=1)
    right.annotate(
        f"Gaussian probe: {floor:.1f}%\nRandom edges: {analytic:.1f}%",
        xy=(floor + 2, 200), fontsize=7.5, color="#555555", va="center", ha="left",
        linespacing=1.35,
    )
    right.set_xlabel("Edges recovered by probe (%)")
    right.set_title("(b) Quality versus recovery", loc="left", pad=8)
    right.set_xlim(-4, 104)
    right.set_xticks([0, 25, 50, 75, 100])

    # --- legend proxies for how much evidence a marker stands on -------------
    # Drawn in neutral grey so they read as a marker convention rather than as
    # another band, and parked far outside the right panel's (already fixed)
    # limits so they exist for the legend only and never render inside a panel.
    off = (-1000.0, -1000.0)
    single_proxy, = right.plot(
        [off[0]], [off[1]], marker="o", markersize=4.8, markerfacecolor="white",
        markeredgecolor="#444444", markeredgewidth=1.0, linewidth=1.0, color="#444444",
    )
    evidence_handles.append((single_proxy, "Seed 0 (single run)"))
    seed_proxy = right.errorbar(
        [off[0]], [off[1]], yerr=[60.0], fmt="*", markersize=9, color="#444444",
        markeredgecolor="black", markeredgewidth=0.4, elinewidth=1.0, capsize=2.5,
    )
    evidence_handles.append((seed_proxy, "3 seeds: mean $\\pm$ SD"))
    evidence_handles.append(
        ((Patch(facecolor="#666666", alpha=0.18, edgecolor="none"),
          Line2D([], [], color="#666666", linewidth=1.0, dashes=(2.6, 1.6))),
         "None: mean $\\pm$ SD")
    )

    # --- shared legend, below both panels so it cannot sit on the data -------
    fig.subplots_adjust(left=0.10, right=0.985, top=0.87, bottom=0.32, wspace=0.16)
    fig.legend(band_handles, [h.get_label() for h in band_handles], ncol=5,
               loc="lower center", bbox_to_anchor=(0.53, 0.105), frameon=False,
               handlelength=2.0, columnspacing=1.7, handletextpad=0.5)
    fig.legend([h for h, _ in evidence_handles], [t for _, t in evidence_handles], ncol=3,
               loc="lower center", bbox_to_anchor=(0.53, 0.025), frameon=False,
               fontsize=7.8, handlelength=1.8, columnspacing=1.2, handletextpad=0.5,
               handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)})
    _save(fig, out, "sweep")


def fig_copying(out: Path) -> None:
    from paper_cpu_checks import unpack_graphs

    snapshot = _load("paper_sample_snapshot.json")
    reference = unpack_graphs(snapshot["reference"])[0]
    panels = [("(a) Reference graph", reference, None),
              ("(b) Low-frequency, $k=8$", unpack_graphs(snapshot["arms"]["low_seed0"]["graphs"])[0], "low"),
              ("(c) Unconditioned", unpack_graphs(snapshot["arms"]["none_seed0"]["graphs"])[0], "none")]
    pos = nx.kamada_kawai_layout(reference)
    coordinates = np.array(list(pos.values()))
    lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
    center = (lo + hi) / 2
    half = max(hi - lo) * 0.55
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.25))
    for ax, (title, graph, band) in zip(axes, panels):
        if band is None:
            nx.draw_networkx_edges(graph, pos, ax=ax, width=0.7, edge_color="#444444")
            subtitle = f"Validation graph 0: {graph.number_of_edges()} edges"
        else:
            shared = [edge for edge in graph.edges() if reference.has_edge(*edge)]
            extra = [edge for edge in graph.edges() if not reference.has_edge(*edge)]
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=extra, width=0.55,
                                   edge_color="#a8adb4", style="dashed", alpha=0.75)
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=shared, width=0.95,
                                   edge_color=BAND_COLOUR["low"])
            f1 = 2 * len(shared) / (graph.number_of_edges() + reference.number_of_edges())
            subtitle = f"{len(shared)}/{graph.number_of_edges()} shared; $F_1={f1:.2f}$"
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=7, node_color="#222222", linewidths=0)
        ax.set_title(title, fontsize=9, pad=7)
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.24, wspace=0.17)
    fig.legend([Line2D([], [], color=BAND_COLOUR["low"], lw=1.4),
                Line2D([], [], color="#969ca5", lw=1.1, linestyle="--")],
               ["Shared with reference", "Not in reference"], ncol=2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.015), fontsize=8.5)
    _save(fig, out, "copying")


def fig_downstream(out: Path) -> None:
    """Figure 2: SBM community-count accuracy, real-only vs three augmentations."""
    data = _load("stage9_downstream.json")
    sizes = data["train_sizes"]
    arms = [
        ("real_only", "Real only", "#222222", "o", "-"),
        ("real_plus_random", "+ True-generator SBM", "#0072b2", "^", "-"),
        ("real_plus_unconditioned", "+ Unconditioned diffusion", "#d55e00", "s", "--"),
        ("real_plus_conditioned", "+ High-frequency diffusion (k=32)", "#008060", "D", "--"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for key, label, colour, marker, style in arms:
        values = [np.array(data["results"][f"n{n}"][key]["accuracies"]) * 100 for n in sizes]
        ax.errorbar(sizes, [v.mean() for v in values], yerr=[v.std(ddof=1) for v in values],
                    marker=marker, markersize=5, linewidth=1.25, linestyle=style,
                    capsize=3, elinewidth=1.0, color=colour, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_ylim(80, 103)
    ax.set_xlabel("Real training graphs")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Exploratory downstream comparison: unverified synthetic labels", loc="left", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.34)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.075), fontsize=8)
    fig.text(0.5, 0.025, "Mean +/- sample SD across 5 seeds; not evidence of a low-frequency augmentation benefit.",
             ha="center", fontsize=7.5, color="#555555")
    _save(fig, out, "downstream")


FIGURES = [fig_sweep, fig_copying, fig_downstream]


def main() -> int:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(PAPER_FIGURES)
        print(f"wrote {fn.__name__}")
    print(f"\n{len(FIGURES)} figures written to {PAPER_FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
