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
    "low": "#1b7837",
    "high": "#d95f02",
    "random": "#c51b7d",
    "gaussian": "#3182bd",
    "shuffled": "#7570b3",
}
BAND_MARKER = {
    "low": "o",
    "high": "s",
    "random": "^",
    "gaussian": "v",
    "shuffled": "D",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
    }
)


def _load(name: str) -> dict:
    path = results_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the paper figure that needs it cannot be built. "
            f"See docs/REPORT.md for which results are committed."
        )
    return json.loads(path.read_text())


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
    agg = seeds["aggregates"]
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

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 1.85))

    band_handles: list = []
    evidence_handles: list = []

    # --- left: the frequency view -------------------------------------------
    ks = sweep["k_values"]
    for band in ["low", "high", "random", "gaussian", "shuffled"]:
        ys = [ratio[(band, k)] for k in ks]
        line, = left.plot(
            ks,
            ys,
            marker=BAND_MARKER[band],
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=BAND_COLOUR[band],
            markeredgewidth=0.9,
            linewidth=1.0,
            color=BAND_COLOUR[band],
            label=band,
        )
        band_handles.append(line)

    left.axhspan(
        base_mean - base_std,
        base_mean + base_std,
        color="#666666",
        alpha=0.22,
        zorder=0,
    )
    left.axhline(base_mean, color="#666666", linewidth=1.0, linestyle="--", zorder=1)

    # Overlay the three-seed means for the two cells that were re-run, with std
    # error bars: seed 0 alone was low/k=8's most favourable draw.
    repeated = [("high_k32", "high", 32), ("low_k8", "low", 8)]
    for name, band, k in repeated:
        left.errorbar(
            [k],
            [agg[name]["ratio_mean"]],
            yerr=[agg[name]["ratio_std"]],
            fmt="*",
            markersize=9,
            color=BAND_COLOUR[band],
            markeredgecolor="black",
            markeredgewidth=0.4,
            elinewidth=1.0,
            capsize=2.5,
            zorder=5,
        )

    left.set_xscale("log", base=2)
    left.set_xticks(ks)
    left.set_xticklabels([str(k) for k in ks])
    left.set_xlabel("$k$ (eigenpairs supplied)")
    left.set_ylabel("Ratio (lower is better)")
    left.set_title("(a) Frequency view")
    left.set_ylim(-15, 385)

    # --- right: the probe view ----------------------------------------------
    for band in ["low", "high", "random", "gaussian", "shuffled"]:
        xs = [recovery[(band, k)] * 100 for k in ks]
        ys = [ratio[(band, k)] for k in ks]
        right.scatter(
            xs,
            ys,
            marker=BAND_MARKER[band],
            s=24,
            facecolors="none",
            edgecolors=BAND_COLOUR[band],
            linewidths=0.9,
            zorder=3,
        )
    for name, band, k in repeated:
        right.errorbar(
            [recovery[(band, k)] * 100],
            [agg[name]["ratio_mean"]],
            yerr=[agg[name]["ratio_std"]],
            fmt="*",
            markersize=9,
            color=BAND_COLOUR[band],
            markeredgecolor="black",
            markeredgewidth=0.4,
            elinewidth=1.0,
            capsize=2.5,
            zorder=5,
        )

    right.axhspan(
        base_mean - base_std,
        base_mean + base_std,
        color="#666666",
        alpha=0.22,
        zorder=0,
    )
    right.axhline(base_mean, color="#666666", linewidth=1.0, linestyle="--", zorder=1)

    right.axvline(floor, color="#999999", linewidth=0.9, linestyle=":", zorder=1)
    right.annotate(
        f"probe floor {floor:.1f}%\n(gaussian arm; analytic {analytic:.1f}%)",
        xy=(floor + 2.5, 150),
        fontsize=6.0,
        color="#555555",
        va="center",
        ha="left",
        linespacing=1.35,
    )

    right.set_xlabel("edges recovered by the rank-threshold probe (%)")
    right.set_ylabel("Ratio (lower is better)")
    right.set_title("(b) Probe view")
    right.set_ylim(-15, 385)
    right.set_xlim(-4, 104)

    # --- legend proxies for how much evidence a marker stands on -------------
    # Drawn in neutral grey so they read as a marker convention rather than as
    # another band, and parked far outside the right panel's (already fixed)
    # limits so they exist for the legend only and never render inside a panel.
    off = (-1000.0, -1000.0)
    single_proxy, = right.plot(
        [off[0]],
        [off[1]],
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor="#444444",
        markeredgewidth=0.9,
        linewidth=1.0,
        color="#444444",
    )
    evidence_handles.append((single_proxy, "hollow: 1 run (seed 0), no spread"))
    seed_proxy = right.errorbar(
        [off[0]],
        [off[1]],
        yerr=[60.0],
        fmt="*",
        markersize=7.0,
        color="#444444",
        markeredgecolor="black",
        markeredgewidth=0.4,
        elinewidth=1.0,
        capsize=2.5,
    )
    evidence_handles.append((seed_proxy, "star: 3 seeds, mean $\\pm$ sd"))
    evidence_handles.append(
        (
            (
                Patch(facecolor="#666666", alpha=0.22, edgecolor="none"),
                Line2D([], [], color="#666666", linewidth=1.0, dashes=(2.6, 1.6)),
            ),
            "unconditioned: 3 seeds, mean $\\pm$ sd",
        )
    )

    # --- shared legend, below both panels so it cannot sit on the data -------
    fig.tight_layout(pad=0.4, rect=(0.0, 0.17, 1.0, 1.0))
    fig.legend(
        band_handles,
        [h.get_label() for h in band_handles],
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.078),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.legend(
        [h for h, _ in evidence_handles],
        [t for _, t in evidence_handles],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=6.5,
        handlelength=1.8,
        columnspacing=1.8,
        handletextpad=0.5,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
    )
    fig.savefig(out / "sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_downstream(out: Path) -> None:
    """Figure 2: SBM community-count accuracy, real-only vs three augmentations."""
    data = _load("stage9_downstream.json")
    sizes = data["train_sizes"]
    arms = [
        ("real_only", "real only", "#222222", "o", "-"),
        ("real_plus_random", "+ true-generator SBM", "#3182bd", "^", "-"),
        ("real_plus_unconditioned", "+ uncond. diffusion", "#d95f02", "s", "--"),
        ("real_plus_conditioned", "+ cond. diffusion", "#1b7837", "D", "--"),
    ]

    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    for key, label, colour, marker, style in arms:
        means = [data["results"][f"n{n}"][key]["mean"] * 100 for n in sizes]
        errs = [data["results"][f"n{n}"][key]["stderr"] * 100 for n in sizes]
        ax.errorbar(
            sizes,
            means,
            yerr=errs,
            marker=marker,
            markersize=3.5,
            linewidth=1.1,
            linestyle=style,
            capsize=2.5,
            elinewidth=0.9,
            color=colour,
            label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_ylim(87.5, 102.0)
    ax.set_xlabel("real training graphs")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout(pad=0.3)
    fig.savefig(out / "downstream.pdf", bbox_inches="tight")
    plt.close(fig)


FIGURES = [fig_sweep, fig_downstream]


def main() -> int:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(PAPER_FIGURES)
        print(f"wrote {fn.__name__}")
    print(f"\n{len(FIGURES)} figures written to {PAPER_FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
