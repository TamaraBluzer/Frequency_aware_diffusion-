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
    """Figure 1: the k-sweep in two views — by frequency, and by measured leakage.

    Left panel is the experiment the proposal asked for (quality against cutoff
    k, one line per band). Right panel re-plots the identical points against how
    much of the target graph each condition demonstrably carries, which is what
    separates the two minima.
    """
    sweep = _load("k_sweep_planar.json")
    seeds = _load("k_sweep_planar_seedcheck.json")
    leak = _load("condition_leakage_planar.json")

    ratio = {(r["band"], r["k"]): r["ratio"] for r in sweep["rows"]}
    recovery = {(r["band"], r["k"]): r["edge_recovery"] for r in leak["rows"]}
    agg = seeds["aggregates"]
    base_mean = agg["none"]["ratio_mean"]
    base_std = agg["none"]["ratio_std"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 1.55))

    # --- left: the frequency view -------------------------------------------
    ks = sweep["k_values"]
    for band in ["low", "high", "random", "gaussian", "shuffled"]:
        ys = [ratio[(band, k)] for k in ks]
        left.plot(
            ks,
            ys,
            marker=BAND_MARKER[band],
            markersize=3.5,
            linewidth=1.2,
            color=BAND_COLOUR[band],
            label=band,
        )
    left.axhspan(
        base_mean - base_std,
        base_mean + base_std,
        color="#666666",
        alpha=0.22,
        zorder=0,
    )
    left.axhline(
        base_mean,
        color="#666666",
        linewidth=1.0,
        linestyle="--",
        zorder=1,
        label="none ($\\pm$sd)",
    )

    # Overlay the three-seed means for the two cells that were re-run, with std
    # error bars: seed 0 alone was low/k=8's most favourable draw.
    for name, band, k in [("high_k32", "high", 32), ("low_k8", "low", 8)]:
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
    left.plot(
        [],
        [],
        "*",
        color="#444444",
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.4,
        label="3-seed mean",
    )

    left.set_xscale("log", base=2)
    left.set_xticks(ks)
    left.set_xticklabels([str(k) for k in ks])
    left.set_xlabel("$k$ (eigenpairs supplied)")
    left.set_ylabel("Ratio (lower is better)")
    left.set_title("(a) Frequency view")
    left.set_ylim(-15, 385)
    shared_handles, shared_labels = left.get_legend_handles_labels()

    # --- right: the information view ----------------------------------------
    for band in ["low", "high", "random", "gaussian", "shuffled"]:
        xs = [recovery[(band, k)] * 100 for k in ks]
        ys = [ratio[(band, k)] for k in ks]
        right.scatter(
            xs,
            ys,
            marker=BAND_MARKER[band],
            s=22,
            color=BAND_COLOUR[band],
            zorder=3,
        )
    chance = leak["rows"][0]["chance_recovery"] * 100
    right.axvline(chance, color="#999999", linewidth=0.9, linestyle=":", zorder=1)
    right.annotate(
        "chance",
        xy=(chance, 232),
        xytext=(chance + 1.5, 228),
        fontsize=6.5,
        color="#666666",
    )
    right.axhline(base_mean, color="#666666", linewidth=1.0, linestyle="--", zorder=1)

    right.legend(
        shared_handles,
        shared_labels,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(0.06, 0.40),
        frameon=True,
        framealpha=0.92,
        handlelength=1.6,
    )
    right.set_xlabel("edge recovery from the condition alone (%)")
    right.set_ylabel("Ratio (lower is better)")
    right.set_title("(b) Information view")
    right.set_ylim(-15, 385)
    right.set_xlim(-4, 104)

    fig.tight_layout(pad=0.4)
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
