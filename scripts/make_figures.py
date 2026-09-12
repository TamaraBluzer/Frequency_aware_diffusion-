"""Regenerate every report figure from results/*.json — no hand-typed numbers.

Run from a clean clone with the committed results/ directory present:

    python scripts/make_figures.py

Each figure function reads its own JSON file(s) and fails loudly (raises) if the
expected file is missing, rather than silently skipping — a missing figure should
break this script, not produce a report with a gap nobody notices.

Coverage is limited to what is actually committed under results/. Stage 7's full
48-config frequency sweep and Stage 6.5's scaled-up discrete D3PM runs were only
ever run on ephemeral Colab sessions and their raw JSON was never saved back to
this repo — see docs/REPORT.md for the numbers that are available (the k=8 pilot
subset of Stage 7, which IS committed) and a note on the gap.
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

FIGURES_DIR = results_dir() / "figures"


def _load(name: str) -> dict:
    path = results_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — this figure needs it. "
            f"See docs/REPORT.md for which results are committed."
        )
    return json.loads(path.read_text())


def fig_stage7_pilot_ratio(out: Path) -> None:
    """Stage 7 k=8 pilot: Ratio by band, with std error bars."""
    data = _load("frequency_pilot_planar_k8_summary.json")
    arms = data["arms"]
    bands = ["none", "low", "high", "random"]
    means = [arms[b]["ratio_mean"] for b in bands]
    stds = [arms[b]["ratio_std"] for b in bands]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#888888", "#2b8cbe", "#de2d26", "#a6a6a6"]
    ax.bar(bands, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel("Ratio (lower is better)")
    ax.set_title(f"Stage 7 pilot: frequency band vs Ratio (k={data['k']}, planar)")
    ax.set_xlabel("band")
    fig.tight_layout()
    fig.savefig(out / "stage7_pilot_ratio_by_band.png", dpi=150)
    plt.close(fig)


def fig_stage8_learned_prior(out: Path) -> None:
    """Stage 8: oracle vs end-to-end Ratio gap, high k=32 vs low k=8."""
    data = _load("stage8_results.json")
    diffusion = data["adjacency_diffusion_training"]
    e2e = data["end_to_end_sampling"]

    configs = ["high_k32", "low_k8"]
    oracle_ratio = [diffusion[c]["ratio"] for c in configs]
    e2e_ratio = [e2e[f"{c}_gs1.0"]["ratio"] for c in configs]
    er_ratio = diffusion["high_k32"]["er_ratio"]

    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, oracle_ratio, width, label="oracle spectra", color="#2b8cbe")
    ax.bar(x + width / 2, e2e_ratio, width, label="learned prior (e2e, gs=1.0)", color="#de2d26")
    ax.axhline(er_ratio, color="#888888", linestyle="--", label=f"ER baseline ({er_ratio:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["high (k=32)", "low (k=8)"])
    ax.set_ylabel("Ratio (lower is better)")
    ax.set_title("Stage 8: oracle vs learned-prior end-to-end sampling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stage8_oracle_vs_e2e.png", dpi=150)
    plt.close(fig)


def fig_stage8_guidance_sweep(out: Path) -> None:
    """Stage 8: guidance scale sweep, Ratio vs scale for high k=32."""
    data = _load("stage8_results.json")
    sweep = data["guidance_scale_sweep"]["results_by_scale"]
    scales = sorted(float(s) for s in sweep)
    ratios = [sweep[f"{s:.1f}"]["ratio"] for s in scales]
    er_ratio = data["adjacency_diffusion_training"]["high_k32"]["er_ratio"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scales, ratios, marker="o", color="#de2d26")
    ax.axhline(er_ratio, color="#888888", linestyle="--", label=f"ER baseline ({er_ratio:.1f})")
    best_scale = data["guidance_scale_sweep"]["best_scale"]
    best_ratio = data["guidance_scale_sweep"]["best_ratio"]
    ax.scatter([best_scale], [best_ratio], color="#2b8cbe", zorder=5, label=f"best (gs={best_scale})")
    ax.set_xlabel("guidance scale")
    ax.set_ylabel("Ratio (lower is better)")
    ax.set_title("Stage 8: guidance scale sweep (high k=32)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stage8_guidance_sweep.png", dpi=150)
    plt.close(fig)


def fig_stage9_downstream(out: Path) -> None:
    """Stage 9: downstream classifier accuracy by training-set size and arm."""
    data = _load("stage9_downstream.json")
    results = data["results"]
    train_sizes = data["train_sizes"]
    arms = ["real_only", "real_plus_random", "real_plus_unconditioned", "real_plus_conditioned"]
    labels = ["real only", "real + random", "real + uncond.", "real + cond."]
    colors = ["#888888", "#a6a6a6", "#2b8cbe", "#de2d26"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(train_sizes))
    width = 0.2
    for i, (arm, label, color) in enumerate(zip(arms, labels, colors)):
        means = [results[f"n{n}"][arm]["mean"] for n in train_sizes]
        stderrs = [results[f"n{n}"][arm]["stderr"] for n in train_sizes]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stderrs, capsize=3, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in train_sizes])
    ax.set_ylabel("classifier accuracy")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Stage 9: SBM community-count classification (5 seeds, ±stderr)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stage9_downstream_accuracy.png", dpi=150)
    plt.close(fig)


def fig_discrete_d3pm_ratio(out: Path) -> None:
    """Stage 6.5 pilot: discrete D3PM Ratio, none vs low, cycles-augmented arms."""
    files = {
        "none (adjacency)": "discrete_adjacency_diffusion_planar_none_k8.json",
        "none (+cycles)": "discrete_adjacency_diffusion_cycles_planar_none_k8.json",
        "low (+cycles)": "discrete_adjacency_diffusion_cycles_planar_low_k8.json",
    }
    labels, ratios, valid = [], [], []
    for label, fname in files.items():
        d = _load(fname)
        labels.append(label)
        ratios.append(d["evaluation"]["ratio"])
        valid.append(d["evaluation"]["vun/valid"])
    er_ratio = _load(files["none (adjacency)"])["er_ratio"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#888888", "#a6a6a6", "#2b8cbe"]
    bars = ax.bar(labels, ratios, color=colors)
    ax.axhline(er_ratio, color="#de2d26", linestyle="--", label=f"ER baseline ({er_ratio:.1f})")
    for bar, v in zip(bars, valid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"valid={v:.0%}",
                 ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Ratio (lower is better)")
    ax.set_title("Stage 6.5 pilot: discrete D3PM Ratio (k=8, planar)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stage6_5_discrete_d3pm_ratio.png", dpi=150)
    plt.close(fig)


def fig_eval_calibration(out: Path) -> None:
    """Eval harness calibration: self-similarity floor vs ER baseline."""
    data = _load("eval_calibration_planar.json")
    rows = data["rows"]
    names = ["train vs test", "train[:h] vs train[h:]", "ER(matched) vs test"]
    ratios = []
    for name in names:
        r = rows[name]["ratio"]
        ratios.append(r if r is not None else 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, ratios, color=["#2b8cbe", "#2b8cbe", "#de2d26"])
    ax.set_ylabel("Ratio")
    ax.set_yscale("log")
    ax.set_title("Eval harness calibration (planar)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out / "eval_calibration.png", dpi=150)
    plt.close(fig)


FIGURES = [
    fig_stage7_pilot_ratio,
    fig_stage8_learned_prior,
    fig_stage8_guidance_sweep,
    fig_stage9_downstream,
    fig_discrete_d3pm_ratio,
    fig_eval_calibration,
]


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(FIGURES_DIR)
        print(f"wrote {fn.__name__}")
    print(f"\n{len(FIGURES)} figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
