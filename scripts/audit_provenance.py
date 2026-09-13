"""Trace every numeric claim in paper/main.tex back to a committed result file.

An external reviewer asked us to "map each table and figure to its result files,
configuration, seed, and code version where available" and to "report component metrics:
extract existing degree, clustering, spectrum, wavelet, and orbit results."  This script is
that map, computed rather than transcribed.

What it does
------------
1. **Claim audit.**  Every numeric claim in the abstract, the three tables, the figure and the
   body text is declared below with the file and key it is supposed to come from and a
   derivation that recomputes it.  Nothing in the CLAIMS table is a hand-copied result value:
   the `paper` field is what `main.tex` prints, the `derivation` recomputes the backing value
   from `results/`, and the script reports whether they agree.  A claim with no backing file
   is declared with `source=None` and lands in `untraceable`.

2. **Component metrics.**  The twelve k=8 arms
   (`adjacency_diffusion_planar_{low,high,none,random}_k8{,_seed1,_seed2}.json`) are the only
   runs in the project that kept a full `evaluation` block, so degree / clustering / spectral
   / wavelet / orbit and valid / unique / novel / vun are extracted per arm and per seed and
   aggregated.  Both sample (ddof=1) and population (ddof=0) standard deviations are reported,
   because the paper's two tables do not use the same one.

3. **Lost breakdowns.**  The sweep files carry only `ratio` and `validity`.  Their per-metric
   components are *not* reconstructed here -- they are unrecoverable, and the JSON's own
   `provenance.caveat` says why.  They are recorded as a loss instead.

4. **Table 1 vs Table 2.**  The same nominal configuration (band=low, k=8, planar, 200 epochs)
   appears in both sets with different values.  The reconciliation section tests the "low is
   simply the noisiest arm" hypothesis against the data and records what the git history does
   and does not explain.

Reading the committed state, not the working tree
-------------------------------------------------
By default every result file is read from `git show HEAD:<path>`, so the audit describes what
is *committed* -- which is what a reviewer can reproduce.  `--source worktree` audits the
working tree instead, and `--source both` reports a per-file diff between them.  That flag
matters: at the time this script was written the leakage probe was being corrected in the
working tree and its committed numbers were about to change.

Usage
-----
    python scripts/audit_provenance.py                   # writes results/provenance_audit.json
    python scripts/audit_provenance.py --source both     # adds a HEAD-vs-worktree diff
    python scripts/audit_provenance.py --print           # human-readable summary to stdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER = REPO / "paper" / "main.tex"
DEFAULT_OUT = RESULTS / "provenance_audit.json"

# --------------------------------------------------------------------------------------
# File access
# --------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def _git_ok(*args: str) -> str | None:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def _tracked(path: str) -> bool:
    """True when `path` exists in HEAD.  `cat-file -e` prints nothing, so test the exit code."""
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{path}"], cwd=REPO, capture_output=True
        ).returncode
        == 0
    )


class Store:
    """Loads `results/*.json` from a chosen ref, caching and recording what it read."""

    def __init__(self, source: str) -> None:
        if source not in {"head", "worktree"}:
            raise ValueError(f"unknown source {source!r}")
        self.source = source
        self._cache: dict[str, Any] = {}
        self.missing: list[str] = []

    def load(self, name: str) -> dict | None:
        if name in self._cache:
            return self._cache[name]
        payload: dict | None = None
        if self.source == "head":
            raw = _git_ok("show", f"HEAD:results/{name}")
            if raw is not None:
                payload = json.loads(raw)
        else:
            path = RESULTS / name
            if path.is_file():
                payload = json.loads(path.read_text())
        if payload is None:
            self.missing.append(name)
        self._cache[name] = payload
        return payload


# --------------------------------------------------------------------------------------
# Small helpers used by the claim derivations
# --------------------------------------------------------------------------------------


def path_get(payload: Any, dotted: str) -> Any:
    """`a.b[2].c` -> nested lookup.  Raises KeyError/IndexError if absent."""
    node = payload
    for part in dotted.split("."):
        while "[" in part:
            head, _, rest = part.partition("[")
            if head:
                node = node[head]
            index, _, part = rest.partition("]")
            node = node[int(index)]
        if part:
            node = node[part]
    return node


def row(payload: dict, list_key: str = "rows", **predicate: Any) -> dict:
    """The single row of `payload[list_key]` matching every key=value in `predicate`."""
    hits = [
        item
        for item in payload[list_key]
        if all(item.get(key) == value for key, value in predicate.items())
    ]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} rows match {predicate} in {list_key}")
    return hits[0]


def mean(values) -> float:
    return statistics.fmean(values)


def sd(values, ddof: int = 1) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) if ddof == 1 else statistics.pstdev(values)


# The three-seed k=8 arms, in the order the paper's Table 1 rows appear.
K8_BANDS = ("none", "low", "high", "random")
K8_SEEDS = (0, 1, 2)
COMPONENT_METRICS = ("degree", "clustering", "spectral", "wavelet", "orbit")
VUN_FIELDS = ("valid", "unique", "novel", "vun")


def k8_file(band: str, seed: int) -> str:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return f"adjacency_diffusion_planar_{band}_k8{suffix}.json"


def k8_ratios(store: Store, band: str) -> list[float]:
    return [
        path_get(store.load(k8_file(band, seed)), "evaluation.ratio") for seed in K8_SEEDS
    ]


def k8_losses(store: Store, band: str) -> list[float]:
    return [
        path_get(store.load(k8_file(band, seed)), "best.validation_loss") for seed in K8_SEEDS
    ]


# --------------------------------------------------------------------------------------
# The claim table
# --------------------------------------------------------------------------------------
#
# Each entry:
#   id          stable identifier
#   location    where in the paper the number is printed
#   quantity    what it is
#   paper       the value as typeset in main.tex
#   tex         the literal substring of main.tex that carries it (checked to exist)
#   files       backing result file(s), or [] when nothing backs it
#   key         the JSON key / derivation, in words
#   fn          recomputes the backing value from `store`; None means untraceable
#   tol         absolute tolerance for the comparison
#   seeds       seeds behind the number
#   breakdown   "full"  -> the run has a per-metric evaluation block
#               "ratio-only" -> only an aggregate Ratio survived
#               "n/a"   -> not a Ratio-bearing claim


def build_claims() -> list[dict]:
    C: list[dict] = []

    def add(**kwargs: Any) -> None:
        kwargs.setdefault("tol", 0.05)
        kwargs.setdefault("seeds", None)
        kwargs.setdefault("breakdown", "n/a")
        kwargs.setdefault("files", [])
        kwargs.setdefault("fn", None)
        C.append(kwargs)

    # ---------------- Abstract ----------------
    add(
        id="abs.arms_x_cutoffs",
        location="abstract",
        quantity="sweep size (conditioned arms x cutoffs)",
        paper=25.0,
        tex="a sweep of five conditioning arms over five cutoffs",
        files=["k_sweep_planar.json"],
        key="len([r for r in rows if r.band != 'none'])",
        fn=lambda s: float(len([r for r in s.load("k_sweep_planar.json")["rows"] if r["band"] != "none"])),
        tol=0.0,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="abs.flat_cells",
        location="abstract",
        quantity="conditioned cells indistinguishable from baseline",
        paper=23.0,
        tex="23 of 25 cells sit",
        files=["k_sweep_planar.json"],
        key="25 conditioned cells minus the two minima",
        fn=lambda s: float(
            len([r for r in s.load("k_sweep_planar.json")["rows"] if r["band"] != "none"]) - 2
        ),
        tol=0.0,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="abs.low_k8_speedup",
        location="abstract / Table 2 / S4.3 / S5",
        quantity="low k=8 Ratio improvement over unconditioned baseline",
        paper=2.3,
        tex="improves the MMD Ratio $2.3\\times$",
        files=["k_sweep_planar_seedcheck.json"],
        key="aggregates.none.ratio_mean / aggregates.low_k8.ratio_mean",
        fn=lambda s: path_get(s.load("k_sweep_planar_seedcheck.json"), "aggregates.none.ratio_mean")
        / path_get(s.load("k_sweep_planar_seedcheck.json"), "aggregates.low_k8.ratio_mean"),
        tol=0.05,
        seeds=[0, 1, 2],
        breakdown="ratio-only",
    )
    add(
        id="abs.low_k8_leakage",
        location="abstract / Table 3 / S4.3",
        quantity="low k=8 edge recovery from the condition alone",
        paper=0.0,
        tex="recovering $0.0\\%$ of the target's edges",
        files=["condition_leakage_planar.json"],
        key="rows[band=low,k=8].edge_recovery x100",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_planar.json"), band="low", k=8)["edge_recovery"],
        tol=0.05,
    )
    add(
        id="abs.high_k32_speedup",
        location="abstract / Table 2 / S4.3",
        quantity="high k=32 Ratio improvement over unconditioned baseline",
        paper=29.7,
        tex="improves it $29.7\\times$",
        files=["k_sweep_planar_seedcheck.json"],
        key="aggregates.none.ratio_mean / aggregates.high_k32.ratio_mean",
        fn=lambda s: path_get(s.load("k_sweep_planar_seedcheck.json"), "aggregates.none.ratio_mean")
        / path_get(s.load("k_sweep_planar_seedcheck.json"), "aggregates.high_k32.ratio_mean"),
        tol=0.05,
        seeds=[0, 1, 2],
        breakdown="ratio-only",
    )
    add(
        id="abs.high_k32_leakage",
        location="abstract / Table 3 / S4.3 / S4.4 / S5",
        quantity="high k=32 edge recovery",
        paper=98.5,
        tex="recovers $98.5\\%$",
        files=["condition_leakage_planar.json"],
        key="rows[band=high,k=32].edge_recovery x100",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_planar.json"), band="high", k=32)["edge_recovery"],
        tol=0.05,
    )

    # ---------------- S3 Method ----------------
    add(
        id="method.n_generated",
        location="S3 Data",
        quantity="graphs generated per arm",
        paper=32.0,
        tex="as many graphs as the evaluation\nset holds (32)",
        files=[k8_file("low", 0)],
        key="evaluation.n_generated",
        fn=lambda s: float(path_get(s.load(k8_file("low", 0)), "evaluation.n_generated")),
        tol=0.0,
    )
    add(
        id="method.split",
        location="S3 Data",
        quantity="train/val/test split sizes",
        paper=float("nan"),
        tex="split 128/32/40",
        files=[],
        key="fald/data/spectre.py load_splits(): 200 graphs, 20% test, 80% of remainder train",
        fn=None,
        tol=0.0,
    )
    add(
        id="method.n_parameters",
        location="S3 Model",
        quantity="denoiser parameter count (M)",
        paper=1.83,
        tex="1.83M parameters",
        files=[k8_file("low", 0)],
        key="n_parameters / 1e6",
        fn=lambda s: path_get(s.load(k8_file("low", 0)), "n_parameters") / 1e6,
        tol=0.005,
    )
    add(
        id="method.timesteps",
        location="S3 Model",
        quantity="diffusion steps T",
        paper=100.0,
        tex="$T{=}100$ steps",
        files=[k8_file("low", 0)],
        key="config.timesteps",
        fn=lambda s: float(path_get(s.load(k8_file("low", 0)), "config.timesteps")),
        tol=0.0,
    )
    add(
        id="method.n_layers",
        location="S3 Model",
        quantity="denoiser layers",
        paper=4.0,
        tex="4 layers",
        files=[k8_file("low", 0)],
        key="config.n_layers",
        fn=lambda s: float(path_get(s.load(k8_file("low", 0)), "config.n_layers")),
        tol=0.0,
    )
    add(
        id="method.cond_drop",
        location="S3 Model",
        quantity="condition dropout probability",
        paper=0.1,
        tex="probability $0.1$",
        files=[k8_file("low", 0)],
        key="config.condition_drop_probability",
        fn=lambda s: float(path_get(s.load(k8_file("low", 0)), "config.condition_drop_probability")),
        tol=0.0,
    )
    add(
        id="method.training_budget",
        location="S3 Model",
        quantity="epochs / batch size / learning rate",
        paper=float("nan"),
        tex="trains 200 epochs at batch 16, learning rate $3\\!\\times\\!10^{-4}$",
        files=[],
        key="NOT recorded in any run report; only argparse defaults in "
        "scripts/train_adjacency_diffusion.py (--epochs 200, --batch-size 16, --lr 3e-4)",
        fn=None,
        tol=0.0,
    )
    add(
        id="method.autoencoder_latent",
        location="S3 Model",
        quantity="autoencoder per-pair latent floats",
        paper=32768.0,
        tex="32{,}768 floats for 2{,}016 binary decisions",
        files=["autoencoder_planar.json"],
        key="config.d_e x 64 x 64",
        fn=lambda s: float(path_get(s.load("autoencoder_planar.json"), "config.d_e") * 64 * 64),
        tol=0.0,
    )
    add(
        id="method.autoencoder_f1",
        location="S3 Model",
        quantity="autoencoder reconstruction F1",
        paper=1.0,
        tex="F1\n$=1.000$",
        files=["autoencoder_planar.json"],
        key="best.f1",
        fn=lambda s: float(path_get(s.load("autoencoder_planar.json"), "best.f1")),
        tol=0.0005,
    )
    add(
        id="method.wavelet_scales",
        location="S3 Evaluation",
        quantity="wavelet scales",
        paper=12.0,
        tex="12 ab-spline wavelet scales",
        files=[],
        key="fald/eval/descriptors.py docstring + evaluator.py table; not in any result JSON",
        fn=None,
        tol=0.0,
    )
    add(
        id="method.floor",
        location="S3 Evaluation",
        quantity="train-vs-test Ratio floor",
        paper=1.0,
        tex="the floor is $1.00$ by construction",
        files=["eval_calibration_planar.json"],
        key="rows['train vs test'].ratio",
        fn=lambda s: float(path_get(s.load("eval_calibration_planar.json"), "rows.train vs test.ratio")),
        tol=0.0,
    )
    add(
        id="method.er_calibration",
        location="S3 Evaluation",
        quantity="density-matched ER control, Ratio",
        paper=3035.0,
        tex="$3035\\times$",
        files=["eval_calibration_planar.json"],
        key="rows['ER(matched) vs test'].ratio",
        fn=lambda s: float(path_get(s.load("eval_calibration_planar.json"), "rows.ER(matched) vs test.ratio")),
        tol=0.5,
    )
    add(
        id="method.bootstraps",
        location="S3 Protocol",
        quantity="bootstrap resamples",
        paper=1000.0,
        tex="1{,}000 resamples",
        files=["frequency_pilot_planar_k8_summary.json"],
        key="bootstraps",
        fn=lambda s: float(path_get(s.load("frequency_pilot_planar_k8_summary.json"), "bootstraps")),
        tol=0.0,
    )
    add(
        id="method.full_spectrum_leak",
        location="S3 Protocol",
        quantity="edge recovery from the full spectrum",
        paper=100.0,
        tex="the full spectrum recovers $100.0\\%$",
        files=[],
        key="NOT in results/condition_leakage_planar.json (no full-spectrum row); "
        "stated only in docs/LEAKAGE.md prose",
        fn=None,
        tol=0.05,
    )

    # ---------------- Table 1 (k=8, three seeds) ----------------
    for band, ratio_paper, loss_paper, ratio_sd_paper, loss_sd_paper in (
        ("none", 327.2, 0.132, 8.8, 0.012),
        ("low", 158.0, 0.053, 16.2, 0.004),
        ("high", 339.1, 0.094, 9.2, 0.009),
        ("random", 326.5, 0.128, 11.8, 0.012),
    ):
        add(
            id=f"tab1.{band}.ratio_mean",
            location="Table 1",
            quantity=f"{band} k=8 Ratio, 3-seed mean",
            paper=ratio_paper,
            tex=f"\\texttt{{{band}}}",
            files=[k8_file(band, seed) for seed in K8_SEEDS],
            key="mean(evaluation.ratio) over seeds 0,1,2",
            fn=(lambda b: lambda s: mean(k8_ratios(s, b)))(band),
            tol=0.05,
            seeds=[0, 1, 2],
            breakdown="full",
        )
        add(
            id=f"tab1.{band}.ratio_sd",
            location="Table 1",
            quantity=f"{band} k=8 Ratio, 3-seed std (sample, ddof=1)",
            paper=ratio_sd_paper,
            tex=f"\\texttt{{{band}}}",
            files=[k8_file(band, seed) for seed in K8_SEEDS],
            key="stdev(evaluation.ratio), ddof=1",
            fn=(lambda b: lambda s: sd(k8_ratios(s, b), ddof=1))(band),
            tol=0.05,
            seeds=[0, 1, 2],
            breakdown="full",
        )
        add(
            id=f"tab1.{band}.loss_mean",
            location="Table 1",
            quantity=f"{band} k=8 validation loss, 3-seed mean",
            paper=loss_paper,
            tex=f"\\texttt{{{band}}}",
            files=[k8_file(band, seed) for seed in K8_SEEDS],
            key="mean(best.validation_loss)",
            fn=(lambda b: lambda s: mean(k8_losses(s, b)))(band),
            tol=0.0005,
            seeds=[0, 1, 2],
            breakdown="full",
        )
        add(
            id=f"tab1.{band}.loss_sd",
            location="Table 1",
            quantity=f"{band} k=8 validation loss, 3-seed std (sample, ddof=1)",
            paper=loss_sd_paper,
            tex=f"\\texttt{{{band}}}",
            files=[k8_file(band, seed) for seed in K8_SEEDS],
            key="stdev(best.validation_loss), ddof=1",
            fn=(lambda b: lambda s: sd(k8_losses(s, b), ddof=1))(band),
            tol=0.0005,
            seeds=[0, 1, 2],
            breakdown="full",
        )
        add(
            id=f"tab1.{band}.validity",
            location="Table 1",
            quantity=f"{band} k=8 validity",
            paper=0.0,
            tex="0\\%",
            files=[k8_file(band, seed) for seed in K8_SEEDS],
            key="mean(evaluation.vun/valid) x100",
            fn=(
                lambda b: lambda s: 100.0
                * mean(
                    path_get(s.load(k8_file(b, seed)), "evaluation.vun/valid") for seed in K8_SEEDS
                )
            )(band),
            tol=0.0,
            seeds=[0, 1, 2],
            breakdown="full",
        )

    # ---------------- S4.1 body ----------------
    add(
        id="s41.ratio_halved",
        location="S4.1 body",
        quantity="low vs none Ratio, relative reduction (%)",
        paper=50.0,
        tex="halves the Ratio",
        files=[k8_file(b, s_) for b in ("low", "none") for s_ in K8_SEEDS],
        key="100 * (1 - mean(low) / mean(none))",
        fn=lambda s: 100.0 * (1.0 - mean(k8_ratios(s, "low")) / mean(k8_ratios(s, "none"))),
        tol=5.0,
        seeds=[0, 1, 2],
        breakdown="full",
    )
    add(
        id="s41.loss_drop",
        location="S4.1 body / S5",
        quantity="low vs none validation-loss reduction (%)",
        paper=60.0,
        tex="cuts validation loss by 60\\%",
        files=[k8_file(b, s_) for b in ("low", "none") for s_ in K8_SEEDS],
        key="100 * (1 - mean(low loss) / mean(none loss))",
        fn=lambda s: 100.0 * (1.0 - mean(k8_losses(s, "low")) / mean(k8_losses(s, "none"))),
        tol=0.6,
        seeds=[0, 1, 2],
        breakdown="full",
    )
    for other, lo, hi in (("none", -189.4, -152.1), ("high", -206.7, -152.4), ("random", -195.1, -142.2)):
        for bound, value in (("lo", lo), ("hi", hi)):
            add(
                id=f"s41.ci_low_minus_{other}.{bound}",
                location="S4.1 body",
                quantity=f"paired bootstrap 95% CI for low-minus-{other} ({bound})",
                paper=value,
                tex="Paired hierarchical bootstrap 95\\% intervals",
                files=["frequency_pilot_planar_k8_summary.json"],
                key=f"paired_hierarchical_bootstrap.low_minus_{other}.ci_95[{0 if bound == 'lo' else 1}]",
                fn=(
                    lambda o, i: lambda s: path_get(
                        s.load("frequency_pilot_planar_k8_summary.json"),
                        f"paired_hierarchical_bootstrap.low_minus_{o}.ci_95[{i}]",
                    )
                )(other, 0 if bound == "lo" else 1),
                tol=0.05,
                seeds=[0, 1, 2],
                breakdown="full",
            )
    add(
        id="s41.condition_sensitivity",
        location="S4.1 body",
        quantity="edge decisions changed by adding the condition (%)",
        paper=12.95,
        tex="changes $12.95\\%$ of\nedge decisions",
        files=["adjacency_diffusion_planar_low_k8_condition_sensitivity.json"],
        key="same_noise_edge_edit_fraction x100",
        fn=lambda s: 100.0
        * path_get(
            s.load("adjacency_diffusion_planar_low_k8_condition_sensitivity.json"),
            "same_noise_edge_edit_fraction",
        ),
        tol=0.005,
        seeds=[0],
    )

    # ---------------- S4.2 / Figure 1(a) ----------------
    def flat_cells(s: Store) -> list[float]:
        rows = s.load("k_sweep_planar.json")["rows"]
        return [
            r["ratio"]
            for r in rows
            if r["band"] != "none"
            and not (r["band"] == "low" and r["k"] == 8)
            and not (r["band"] == "high" and r["k"] == 32)
        ]

    add(
        id="s42.flat_mean",
        location="S4.2 body / Figure 1(a)",
        quantity="mean Ratio over the 23 non-minimum conditioned cells",
        paper=327.2,
        tex="with mean $327.2$ and a spread of $18.5$",
        files=["k_sweep_planar.json"],
        key="mean(rows.ratio) over conditioned cells minus the two minima",
        fn=lambda s: mean(flat_cells(s)),
        tol=0.05,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s42.flat_sd",
        location="S4.2 body / Figure 1(a)",
        quantity="std of the 23 non-minimum conditioned cells (sample, ddof=1)",
        paper=18.5,
        tex="with mean $327.2$ and a spread of $18.5$",
        files=["k_sweep_planar.json"],
        key="stdev(same population), ddof=1",
        fn=lambda s: sd(flat_cells(s), ddof=1),
        tol=0.05,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s42.baseline",
        location="S4.2 body / Figure 1(a)",
        quantity="unconditioned baseline Ratio, seed 0",
        paper=335.7,
        tex="unconditioned baseline of $335.7$",
        files=["k_sweep_planar.json"],
        key="rows[band=none,k=2].ratio",
        fn=lambda s: row(s.load("k_sweep_planar.json"), band="none", k=2)["ratio"],
        tol=0.05,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s42.flat_min",
        location="S4.2 body",
        quantity="lowest Ratio among the 23 non-minimum conditioned cells",
        paper=270.5,
        tex="span $270.5$ to $347.9$",
        files=["k_sweep_planar.json"],
        key="min(rows.ratio) over the 23-cell population",
        fn=lambda s: min(flat_cells(s)),
        tol=0.05,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s42.flat_max",
        location="S4.2 body",
        quantity="highest Ratio among the 23 non-minimum conditioned cells",
        paper=347.9,
        tex="span $270.5$ to $347.9$",
        files=["k_sweep_planar.json"],
        key="max(rows.ratio) over the 23-cell population",
        fn=lambda s: max(flat_cells(s)),
        tol=0.05,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s42.high_k32_seed0",
        location="S4.2 body / Figure 1(a)",
        quantity="high k=32 Ratio, seed 0",
        paper=10.5,
        tex="$10.5$ for \\texttt{high}",
        files=["k_sweep_planar.json"],
        key="rows[band=high,k=32].ratio",
        fn=lambda s: row(s.load("k_sweep_planar.json"), band="high", k=32)["ratio"],
        tol=0.05,
        seeds=[0],
        breakdown="ratio-only",
    )
    for k_value, paper_value, tex in ((4, 316.4, "$316.4$ at $k{=}4$"), (16, 270.5, "$270.5$ at $k{=}16$"), (8, 116.3, "$116.3$ for \\texttt{low}, $k{=}8$")):
        add(
            id=f"s42.low_k{k_value}",
            location="S4.2 body / Figure 1(a)",
            quantity=f"low k={k_value} Ratio, seed 0",
            paper=paper_value,
            tex=tex,
            files=["k_sweep_planar.json"],
            key=f"rows[band=low,k={k_value}].ratio",
            fn=(lambda kk: lambda s: row(s.load("k_sweep_planar.json"), band="low", k=kk)["ratio"])(k_value),
            tol=0.05,
            seeds=[0],
            breakdown="ratio-only",
        )
    for band, bound, paper_value in (
        ("gaussian", "min", 322.0),
        ("gaussian", "max", 333.0),
        ("shuffled", "min", 329.0),
        ("shuffled", "max", 342.0),
    ):
        add(
            id=f"s42.{band}_{bound}",
            location="S4.2 body",
            quantity=f"{band} control Ratio range, {bound}",
            paper=paper_value,
            tex=f"\\texttt{{{band}}}",
            files=["k_sweep_planar.json"],
            key=f"{bound}(rows[band={band}].ratio), rounded to the paper's integer span",
            fn=(
                lambda b, fun: lambda s: float(
                    fun(r["ratio"] for r in s.load("k_sweep_planar.json")["rows"] if r["band"] == b)
                )
            )(band, min if bound == "min" else max),
            tol=1.0,
            seeds=[0],
            breakdown="ratio-only",
        )

    # ---------------- Table 2 (three-seed confirmation) ----------------
    for arm, key, ratio_paper, sd_paper in (
        ("high k=32", "high_k32", 11.1, 0.6),
        ("low k=8", "low_k8", 141.9, 19.3),
        ("none", "none", 330.1, 6.2),
    ):
        add(
            id=f"tab2.{key}.ratio_mean",
            location="Table 2",
            quantity=f"{arm} Ratio, 3-seed mean",
            paper=ratio_paper,
            tex="Three-seed confirmation",
            files=["k_sweep_planar_seedcheck.json"],
            key=f"aggregates.{key}.ratio_mean (equals mean of the three rows)",
            fn=(lambda kk: lambda s: path_get(s.load("k_sweep_planar_seedcheck.json"), f"aggregates.{kk}.ratio_mean"))(key),
            tol=0.05,
            seeds=[0, 1, 2],
            breakdown="ratio-only",
        )
        add(
            id=f"tab2.{key}.ratio_sd",
            location="Table 2",
            quantity=f"{arm} Ratio, 3-seed std (POPULATION, ddof=0)",
            paper=sd_paper,
            tex="Three-seed confirmation",
            files=["k_sweep_planar_seedcheck.json"],
            key=f"aggregates.{key}.ratio_std -- note ddof=0, unlike Table 1's ddof=1",
            fn=(lambda kk: lambda s: path_get(s.load("k_sweep_planar_seedcheck.json"), f"aggregates.{kk}.ratio_std"))(key),
            tol=0.05,
            seeds=[0, 1, 2],
            breakdown="ratio-only",
        )
    add(
        id="s42.high_k32_sd_body",
        location="S4.2 body",
        quantity="high k=32 std quoted in prose",
        paper=0.58,
        tex="std $0.58$ on a mean of $11.11$",
        files=["k_sweep_planar_seedcheck.json"],
        key="aggregates.high_k32.ratio_std (ddof=0)",
        fn=lambda s: path_get(s.load("k_sweep_planar_seedcheck.json"), "aggregates.high_k32.ratio_std"),
        tol=0.005,
        seeds=[0, 1, 2],
        breakdown="ratio-only",
    )
    add(
        id="s42.low_k8_seed0_draw",
        location="S4.2 body",
        quantity="low k=8 seed-0 draw",
        paper=116.27,
        tex="draw of $116.27$",
        files=["k_sweep_planar_seedcheck.json"],
        key="rows[band=low,k=8,seed=0].ratio",
        fn=lambda s: row(s.load("k_sweep_planar_seedcheck.json"), band="low", k=8, seed=0)["ratio"],
        tol=0.005,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s42.low_k8_quotable",
        location="S4.2 body",
        quantity="low k=8 number the paper says to quote",
        paper=141.91,
        tex="so $141.91$ is the number to quote",
        files=["k_sweep_planar_seedcheck.json"],
        key="aggregates.low_k8.ratio_mean",
        fn=lambda s: path_get(s.load("k_sweep_planar_seedcheck.json"), "aggregates.low_k8.ratio_mean"),
        tol=0.005,
        seeds=[0, 1, 2],
        breakdown="ratio-only",
    )
    add(
        id="s42.high_k32_only_valid",
        location="S4.2 body",
        quantity="arms in the sweep producing a valid planar graph",
        paper=1.0,
        tex="the only arm in the sweep that produced a valid\nplanar graph",
        files=["k_sweep_planar.json"],
        key="count of rows with validity > 0",
        fn=lambda s: float(len([r for r in s.load("k_sweep_planar.json")["rows"] if r["validity"] > 0])),
        tol=0.0,
        seeds=[0],
        breakdown="ratio-only",
    )

    # ---------------- Table 3 (leakage) ----------------
    leak_table = {
        "low": [0.0, 0.0, 0.0, 0.0, 0.6],
        "high": [21.8, 31.4, 49.3, 76.7, 98.5],
        "random": [14.8, 15.6, 19.2, 26.3, 43.6],
        "gaussian": [7.9, 8.5, 8.2, 8.7, 9.4],
        "shuffled": [8.7, 8.7, 8.8, 8.7, 8.3],
    }
    for band, values in leak_table.items():
        for k_value, paper_value in zip((2, 4, 8, 16, 32), values):
            add(
                id=f"tab3.{band}.k{k_value}",
                location="Table 3 / Figure 1(b)",
                quantity=f"{band} k={k_value} edge recovery (%)",
                paper=paper_value,
                tex=f"\\texttt{{{band}}}",
                files=["condition_leakage_planar.json"],
                key=f"rows[band={band},k={k_value}].edge_recovery x100",
                fn=(
                    lambda b, kk: lambda s: 100.0
                    * row(s.load("condition_leakage_planar.json"), band=b, k=kk)["edge_recovery"]
                )(band, k_value),
                tol=0.05,
                seeds=[0],
            )
    add(
        id="tab3.chance",
        location="Table 3 caption / S4.3",
        quantity="chance edge recovery on Planar (%)",
        paper=8.8,
        tex="chance is 8.8\\%",
        files=["condition_leakage_planar.json"],
        key="rows[*].chance_recovery x100 (constant across rows)",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_planar.json"), band="low", k=8)["chance_recovery"],
        tol=0.05,
        seeds=[0],
    )
    add(
        id="s43.low_minus_high_points",
        location="S4.3 body",
        quantity="low-minus-high Ratio gap at k=8 (points)",
        paper=170.0,
        tex="wins by ${\\sim}170$\nRatio points",
        files=[k8_file(b, s_) for b in ("low", "high") for s_ in K8_SEEDS],
        key="mean(high) - mean(low) over the three k=8 seeds",
        fn=lambda s: mean(k8_ratios(s, "high")) - mean(k8_ratios(s, "low")),
        tol=15.0,
        seeds=[0, 1, 2],
        breakdown="full",
    )
    add(
        id="s43.sbm_low_max",
        location="S4.3 body",
        quantity="SBM low-band edge recovery, max over k (%)",
        paper=3.5,
        tex="\\texttt{low} at or below $3.5\\%$",
        files=["condition_leakage_sbm.json"],
        key="max(rows[band=low].edge_recovery) x100",
        fn=lambda s: 100.0
        * max(r["edge_recovery"] for r in s.load("condition_leakage_sbm.json")["rows"] if r["band"] == "low"),
        tol=0.05,
        seeds=[0],
    )
    add(
        id="s43.sbm_chance",
        location="S4.3 body",
        quantity="SBM chance edge recovery (%)",
        paper=9.4,
        tex="against $9.4\\%$ chance",
        files=["condition_leakage_sbm.json"],
        key="rows[*].chance_recovery x100",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_sbm.json"), band="low", k=8)["chance_recovery"],
        tol=0.05,
        seeds=[0],
    )
    add(
        id="s43.sbm_high_k32",
        location="S4.3 body",
        quantity="SBM high k=32 edge recovery (%)",
        paper=87.7,
        tex="\\texttt{high} $87.7\\%$ at $k{=}32$",
        files=["condition_leakage_sbm.json"],
        key="rows[band=high,k=32].edge_recovery x100",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_sbm.json"), band="high", k=32)["edge_recovery"],
        tol=0.05,
        seeds=[0],
    )

    # ---------------- S4.4 validity ----------------
    add(
        id="s44.low_k16_connected",
        location="S4.4 body",
        quantity="low k=16 reconstructions connected (%)",
        paper=71.9,
        tex="already $71.9\\%$ connected",
        files=["condition_leakage_planar.json"],
        key="rows[band=low,k=16].connected_frac x100",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_planar.json"), band="low", k=16)["connected_frac"],
        tol=0.05,
        seeds=[0],
    )
    add(
        id="s44.high_k32_planar",
        location="S4.4 body",
        quantity="high k=32 reconstructions planar (%)",
        paper=3.1,
        tex="only \\texttt{high}, $k{=}32$ reaches it, at $3.1\\%$",
        files=["condition_leakage_planar.json"],
        key="rows[band=high,k=32].planar_frac x100",
        fn=lambda s: 100.0 * row(s.load("condition_leakage_planar.json"), band="high", k=32)["planar_frac"],
        tol=0.05,
        seeds=[0],
    )
    add(
        id="s44.k32_connected_both",
        location="S4.4 body",
        quantity="connected fraction at k=32 in both bands",
        paper=1.0,
        tex="by $k{=}32$ every reconstruction is connected in both bands",
        files=["condition_leakage_planar.json"],
        key="min(connected_frac) over rows[band in {low,high}, k=32]",
        fn=lambda s: min(
            row(s.load("condition_leakage_planar.json"), band=b, k=32)["connected_frac"]
            for b in ("low", "high")
        ),
        tol=0.0,
        seeds=[0],
    )
    for cid, quantity, paper_value, tex in (
        ("s44.misplaced_mean", "mean misplaced edges at 98.5% recovery", 2.7, "misplaced edges is $2.7$"),
        ("s44.misplaced_max", "max misplaced edges", 6.0, "(max 6)"),
        ("s44.zero_misplaced", "reconstructions with zero misplaced edges (%)", 3.1, "\\emph{zero} misplaced edges is $3.1\\%$"),
        ("s44.edges_per_graph", "mean edges per validation graph", 177.0, "with 177\nedges"),
        ("s44.median_distance", "high k=32 median distance-to-planarity", 4.0, "median distance 4"),
        ("s44.distance_over_12", "high k=16 / low k=32 edge removals needed", 12.0, "over 12 edge removals"),
    ):
        add(
            id=cid,
            location="S4.4 body",
            quantity=quantity,
            paper=paper_value,
            tex=tex,
            files=[],
            key="NOT in any results/*.json; stated only in docs/ZERO_VALIDITY.md prose. "
            "scripts/report_breakdown.py:distance_to_planar can recompute the distance figures, "
            "but no run report carries a validity_diagnosis block and the samples are gone.",
            fn=None,
        )
    add(
        id="s44.discrete_cycles_zero_validity",
        location="S4.4 body",
        quantity="discrete cycle-feature arm validity",
        paper=0.0,
        tex="reports $0\\%$ validity at every scale",
        files=[
            "discrete_adjacency_diffusion_cycles_planar_low_k8.json",
            "discrete_adjacency_diffusion_cycles_planar_none_k8.json",
            "discrete_scaled_up_planar.json",
        ],
        key="max validity over the discrete reports and the recovered scaled-up runs",
        fn=lambda s: 100.0
        * max(
            [
                path_get(s.load(f), "evaluation.vun/valid")
                for f in (
                    "discrete_adjacency_diffusion_cycles_planar_low_k8.json",
                    "discrete_adjacency_diffusion_cycles_planar_none_k8.json",
                    "discrete_adjacency_diffusion_planar_none_k8.json",
                )
            ]
            + [r["validity"] for r in s.load("discrete_scaled_up_planar.json")["runs"]]
        ),
        tol=0.0,
        seeds=[0],
    )

    # ---------------- S4.5 end-to-end ----------------
    for cid, quantity, paper_value, key, fn, tex in (
        ("s45.er_reference", "ER reference for the 4-metric scale", 161.67,
         "adjacency_diffusion_training.high_k32.er_ratio",
         lambda s: path_get(s.load("stage8_results.json"), "adjacency_diffusion_training.high_k32.er_ratio"),
         "ER reference of $161.67$"),
        ("s45.oracle_high", "oracle high k=32 Ratio", 5.92,
         "adjacency_diffusion_training.high_k32.ratio",
         lambda s: path_get(s.load("stage8_results.json"), "adjacency_diffusion_training.high_k32.ratio"),
         "Ratio $5.92$"),
        ("s45.oracle_low", "oracle low k=8 Ratio", 74.86,
         "adjacency_diffusion_training.low_k8.ratio",
         lambda s: path_get(s.load("stage8_results.json"), "adjacency_diffusion_training.low_k8.ratio"),
         "$74.86$"),
        ("s45.e2e_high", "end-to-end high k=32 Ratio at gs=1", 159.72,
         "end_to_end_sampling['high_k32_gs1.0'].ratio",
         lambda s: s.load("stage8_results.json")["end_to_end_sampling"]["high_k32_gs1.0"]["ratio"],
         "$159.72$"),
        ("s45.e2e_low", "end-to-end low k=8 Ratio at gs=1", 295.64,
         "end_to_end_sampling['low_k8_gs1.0'].ratio",
         lambda s: s.load("stage8_results.json")["end_to_end_sampling"]["low_k8_gs1.0"]["ratio"],
         "$295.64$"),
        ("s45.best_gs", "optimal guidance scale", 1.5,
         "guidance_scale_sweep.best_scale",
         lambda s: float(path_get(s.load("stage8_results.json"), "guidance_scale_sweep.best_scale")),
         "an optimum at $1.5$"),
        ("s45.degradation", "oracle-to-end-to-end degradation at the optimum (x)", 24.0,
         "guidance_scale_sweep.best_ratio / adjacency_diffusion_training.high_k32.ratio",
         lambda s: path_get(s.load("stage8_results.json"), "guidance_scale_sweep.best_ratio")
         / path_get(s.load("stage8_results.json"), "adjacency_diffusion_training.high_k32.ratio"),
         "$24\\times$ degradation"),
    ):
        add(
            id=cid,
            location="S4.5 body",
            quantity=quantity,
            paper=paper_value,
            tex=tex,
            files=["stage8_results.json"],
            key=key,
            fn=fn,
            tol=0.5 if cid == "s45.degradation" else 0.005,
            seeds=[0],
            breakdown="ratio-only",
        )
    add(
        id="s45.discrete_none_scaled",
        location="S4.5 body",
        quantity="scaled-up discrete unconditioned Ratio",
        paper=39.60,
        tex="$39.60$",
        files=["discrete_scaled_up_planar.json"],
        key="headline.matched_scaled_up_pair.unconditioned",
        fn=lambda s: path_get(s.load("discrete_scaled_up_planar.json"), "headline.matched_scaled_up_pair.unconditioned"),
        tol=0.005,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s45.discrete_low_scaled",
        location="S4.5 body",
        quantity="scaled-up discrete low k=8 Ratio",
        paper=15.40,
        tex="to $15.40$",
        files=["discrete_scaled_up_planar.json"],
        key="headline.matched_scaled_up_pair.low_k8",
        fn=lambda s: path_get(s.load("discrete_scaled_up_planar.json"), "headline.matched_scaled_up_pair.low_k8"),
        tol=0.005,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s45.discrete_relative",
        location="S4.5 body",
        quantity="scaled-up discrete conditioning benefit (%)",
        paper=-61.0,
        tex="($-61\\%$)",
        files=["discrete_scaled_up_planar.json"],
        key="headline.matched_scaled_up_pair.relative_improvement x100",
        fn=lambda s: 100.0
        * path_get(s.load("discrete_scaled_up_planar.json"), "headline.matched_scaled_up_pair.relative_improvement"),
        tol=0.5,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s45.discrete_gap_narrow",
        location="S4.5 body",
        quantity="scaled-up discrete conditioned-vs-unconditioned gap (Ratio points)",
        paper=24.0,
        tex="to ${\\sim}24$",
        files=["discrete_scaled_up_planar.json"],
        key="unconditioned - low_k8 in the matched scaled-up pair",
        fn=lambda s: path_get(s.load("discrete_scaled_up_planar.json"), "headline.matched_scaled_up_pair.unconditioned")
        - path_get(s.load("discrete_scaled_up_planar.json"), "headline.matched_scaled_up_pair.low_k8"),
        tol=0.5,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s45.discrete_pilot_baseline",
        location="S4.5 body",
        quantity="discrete pilot unconditioned Ratio the '~327' refers to",
        paper=327.0,
        tex="from ${\\sim}327$ to",
        files=["discrete_scaled_up_planar.json"],
        key="runs[name='unconditioned (pilot scale)'].ratio -- the paper's ~327 is Table 1's "
        "CONTINUOUS none mean (327.23), not this file's discrete pilot (306.78)",
        fn=lambda s: row(s.load("discrete_scaled_up_planar.json"), "runs", name="unconditioned (pilot scale)")["ratio"],
        tol=5.0,
        seeds=[0],
        breakdown="ratio-only",
    )
    add(
        id="s45.discrete_gap_before",
        location="S4.5 body",
        quantity="pilot-scale conditioned-vs-unconditioned gap (Ratio points)",
        paper=220.0,
        tex="from ${\\sim}220$",
        files=["discrete_scaled_up_planar.json"],
        key="saved_json_counterparts: cycles none (262.47) - cycles low (40.01). NOT derivable "
        "from this file's own `runs` block, whose pilot pair gives 306.78 - 30.07 = 276.71.",
        fn=lambda s: path_get(s.load("discrete_scaled_up_planar.json"), "saved_json_counterparts.discrete_adjacency_diffusion_cycles_planar_none_k8")
        - path_get(s.load("discrete_scaled_up_planar.json"), "saved_json_counterparts.discrete_adjacency_diffusion_cycles_planar_low_k8"),
        tol=5.0,
        seeds=[0],
        breakdown="full",
    )

    # ---------------- S4.6 downstream / Figure 2 ----------------
    for cid, quantity, paper_value, key, tex in (
        ("s46.n20_uncond", "n=20 unconditioned augmentation accuracy (%)", 89.8, "results.n20.real_plus_unconditioned.mean", "unconditioned augmentation scores $89.8\\%$"),
        ("s46.n20_real", "n=20 real-only accuracy (%)", 94.8, "results.n20.real_only.mean", "against $94.8\\%$ real-only"),
        ("s46.n20_cond", "n=20 conditioned augmentation accuracy (%)", 92.0, "results.n20.real_plus_conditioned.mean", "conditioned augmentation $92.0\\%$"),
        ("s46.n80_real", "n=80 real-only baseline (%)", 97.2, "results.n80.real_only.mean", "below the $97.2\\%$ baseline"),
        ("s46.n80_true_sbm", "n=80 true-SBM control accuracy (%)", 98.8, "results.n80.real_plus_random.mean", "($98.8\\%$ at $n{=}80$)"),
    ):
        add(
            id=cid,
            location="S4.6 body / Figure 2 (figures/downstream.pdf, generated but cut from main.tex)",
            quantity=quantity,
            paper=paper_value,
            tex=tex,
            files=["stage9_downstream.json"],
            key=f"{key} x100",
            fn=(lambda kk: lambda s: 100.0 * path_get(s.load("stage9_downstream.json"), kk))(key),
            tol=0.05,
            seeds=[0, 1, 2, 3, 4],
        )
    add(
        id="s46.n80_both_below",
        location="S4.6 body",
        quantity="n=80 best diffusion-augmented arm (%), must be below real-only",
        paper=96.2,
        tex="at $n{=}80$ both stay below the $97.2\\%$ baseline",
        files=["stage9_downstream.json"],
        key="max(real_plus_unconditioned.mean, real_plus_conditioned.mean) at n80, x100",
        fn=lambda s: 100.0
        * max(
            path_get(s.load("stage9_downstream.json"), "results.n80.real_plus_unconditioned.mean"),
            path_get(s.load("stage9_downstream.json"), "results.n80.real_plus_conditioned.mean"),
        ),
        tol=0.05,
        seeds=[0, 1, 2, 3, 4],
    )
    add(
        id="s46.n_seeds",
        location="S4.6 body",
        quantity="downstream seeds",
        paper=5.0,
        tex="five seeds\nper arm",
        files=["stage9_downstream.json"],
        key="n_seeds",
        fn=lambda s: float(path_get(s.load("stage9_downstream.json"), "n_seeds")),
        tol=0.0,
    )

    # ---------------- S5 discussion ----------------
    add(
        id="s5.none_with_orca",
        location="S5 body",
        quantity="none k=8 seed-0 Ratio with ORCA",
        paper=321.21,
        tex="\\texttt{none} is $321.21$ with ORCA",
        files=[k8_file("none", 0)],
        key="evaluation.ratio",
        fn=lambda s: path_get(s.load(k8_file("none", 0)), "evaluation.ratio"),
        tol=0.005,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.none_without_orca",
        location="S5 body",
        quantity="none k=8 seed-0 Ratio without ORCA",
        paper=125.25,
        tex="$125.25$ without",
        files=[k8_file("none", 0)],
        key="mean of ratio/{degree,clustering,spectral,wavelet} (orbit excluded)",
        fn=lambda s: mean(
            path_get(s.load(k8_file("none", 0)), f"evaluation.ratio/{m}")
            for m in ("degree", "clustering", "spectral", "wavelet")
        ),
        tol=0.005,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.orca_gap",
        location="S5 body",
        quantity="ORCA-vs-no-ORCA scale gap (x)",
        paper=2.56,
        tex="$2.56\\times$ gap",
        files=[k8_file("none", 0)],
        key="5-metric mean / 4-metric mean",
        fn=lambda s: path_get(s.load(k8_file("none", 0)), "evaluation.ratio")
        / mean(
            path_get(s.load(k8_file("none", 0)), f"evaluation.ratio/{m}")
            for m in ("degree", "clustering", "spectral", "wavelet")
        ),
        tol=0.005,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.er_5metric",
        location="S5 body",
        quantity="ER reference on the 5-metric scale",
        paper=337.91,
        tex="$337.91$",
        files=[k8_file("none", 0)],
        key="er_ratio",
        fn=lambda s: path_get(s.load(k8_file("none", 0)), "er_ratio"),
        tol=0.005,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.orbit_share",
        location="S5 body",
        quantity="orbit's share of the aggregate improvement (%)",
        paper=84.0,
        tex="orbit alone contributes $84.0\\%$",
        files=["report_breakdown.json"],
        key="headline_improvement.contribution_share.orbit x100",
        fn=lambda s: 100.0 * path_get(s.load("report_breakdown.json"), "headline_improvement.contribution_share.orbit"),
        tol=0.05,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.aggregate_change",
        location="S5 body",
        quantity="aggregate k=8 improvement, seed 0 (%)",
        paper=-52.9,
        tex="$-52.9\\%$ aggregate",
        files=["report_breakdown.json"],
        key="headline_improvement.relative_change x100",
        fn=lambda s: 100.0 * path_get(s.load("report_breakdown.json"), "headline_improvement.relative_change"),
        tol=0.05,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.degree_wrong_way",
        location="S5 body",
        quantity="degree delta, low minus none (positive = worse)",
        paper=101.81,
        tex="\\emph{degree moves the wrong way}",
        files=["report_breakdown.json"],
        key="headline_improvement.delta_per_metric.degree (sign check only)",
        fn=lambda s: path_get(s.load("report_breakdown.json"), "headline_improvement.delta_per_metric.degree"),
        tol=0.05,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.geometric_mean_change",
        location="S5 body",
        quantity="k=8 improvement under a geometric mean (%)",
        paper=-34.8,
        tex="a geometric mean puts it at $-34.8\\%$",
        files=["report_breakdown.json"],
        key="100 * (geo(low seed0) / geo(none seed0) - 1)",
        fn=lambda s: 100.0
        * (
            path_get(s.load("report_breakdown.json"), "decompositions.adjacency_diffusion_planar_low_k8.geometric_mean")
            / path_get(s.load("report_breakdown.json"), "decompositions.adjacency_diffusion_planar_none_k8.geometric_mean")
            - 1.0
        ),
        tol=0.05,
        seeds=[0],
        breakdown="full",
    )
    add(
        id="s5.wins_four_of_five",
        location="S5 body",
        quantity="metrics on which low beats none at k=8 (seed 0)",
        paper=4.0,
        tex="it wins on four of five metrics",
        files=[k8_file("low", 0), k8_file("none", 0)],
        key="count of ratio/<metric> where low < none",
        fn=lambda s: float(
            sum(
                path_get(s.load(k8_file("low", 0)), f"evaluation.ratio/{m}")
                < path_get(s.load(k8_file("none", 0)), f"evaluation.ratio/{m}")
                for m in COMPONENT_METRICS
            )
        ),
        tol=0.0,
        seeds=[0],
        breakdown="full",
    )
    return C


# --------------------------------------------------------------------------------------
# Evaluation of the claim table
# --------------------------------------------------------------------------------------


def _decimals(value: float) -> int:
    """Decimal places the paper printed, inferred from the declared value."""
    text = f"{value!r}"
    return len(text.split(".")[1]) if "." in text else 0


def _ulp(value: float) -> float:
    """One unit in the last place the paper printed."""
    return 10.0 ** (-_decimals(value))


def check_tex_present(claims: list[dict]) -> dict[str, bool]:
    """Confirm each claim's `tex` anchor really occurs in paper/main.tex."""
    if not PAPER.is_file():
        return {claim["id"]: False for claim in claims}
    source = PAPER.read_text()
    normalized = " ".join(source.split())
    found = {}
    for claim in claims:
        anchor = " ".join(claim["tex"].split())
        found[claim["id"]] = anchor in normalized
    return found


def audit_claims(store: Store, claims: list[dict]) -> list[dict]:
    anchors = check_tex_present(claims)
    out = []
    for claim in claims:
        record = {
            "id": claim["id"],
            "location": claim["location"],
            "quantity": claim["quantity"],
            "paper_value": None if isinstance(claim["paper"], float) and math.isnan(claim["paper"]) else claim["paper"],
            "tex_anchor": claim["tex"],
            "tex_anchor_found_in_main_tex": anchors[claim["id"]],
            "source_files": claim["files"],
            "source_key": claim["key"],
            "seeds": claim["seeds"],
            "n_seeds": len(claim["seeds"]) if claim["seeds"] else None,
            "breakdown": claim["breakdown"],
            "tolerance": claim["tol"],
        }
        if claim["fn"] is None:
            record.update(
                file_value=None,
                status="UNTRACEABLE",
                delta=None,
                note="no committed results file backs this number",
            )
        else:
            try:
                value = float(claim["fn"](store))
            except Exception as exc:  # noqa: BLE001 - the failure itself is the finding
                record.update(
                    file_value=None,
                    status="ERROR",
                    delta=None,
                    note=f"{type(exc).__name__}: {exc}",
                )
            else:
                paper = claim["paper"]
                if isinstance(paper, float) and math.isnan(paper):
                    record.update(file_value=value, status="INFO", delta=None)
                else:
                    delta = value - paper
                    if abs(delta) <= claim["tol"]:
                        status, note = "MATCH", None
                    elif abs(delta) <= _ulp(paper):
                        status = "ROUNDING"
                        note = (
                            f"within one unit of the paper's last printed digit; the backing "
                            f"value rounds to {round(value, _decimals(paper))}"
                        )
                    else:
                        status, note = "MISMATCH", None
                    record.update(file_value=value, delta=delta, status=status)
                    if note:
                        record["note"] = note
        out.append(record)
    return out


# --------------------------------------------------------------------------------------
# Component metrics (the reviewer's second request)
# --------------------------------------------------------------------------------------


def component_metrics(store: Store) -> dict:
    per_run = []
    for band in K8_BANDS:
        for seed in K8_SEEDS:
            name = k8_file(band, seed)
            payload = store.load(name)
            if payload is None:
                per_run.append({"file": name, "band": band, "seed": seed, "status": "MISSING"})
                continue
            evaluation = payload["evaluation"]
            per_run.append(
                {
                    "file": name,
                    "band": band,
                    "k": payload["k"],
                    "seed": payload["seed"],
                    "status": "OK",
                    "best_epoch": path_get(payload, "best.epoch"),
                    "validation_loss": path_get(payload, "best.validation_loss"),
                    "n_generated": evaluation["n_generated"],
                    "n_reference": evaluation["n_reference"],
                    "er_ratio": payload.get("er_ratio"),
                    "gate_passed": payload.get("gate_passed"),
                    "ratio_aggregate": evaluation["ratio"],
                    "ratio": {m: evaluation[f"ratio/{m}"] for m in COMPONENT_METRICS},
                    "mmd": {m: evaluation[f"mmd/{m}"] for m in COMPONENT_METRICS},
                    "vun": {f: evaluation[f"vun/{f}"] for f in VUN_FIELDS},
                    "history_epochs_recorded": len(payload.get("history") or []),
                    "samples_path": payload.get("samples_path"),
                }
            )

    per_arm: dict[str, dict] = {}
    for band in K8_BANDS:
        runs = [r for r in per_run if r["band"] == band and r["status"] == "OK"]
        entry: dict[str, Any] = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs]}
        for metric in COMPONENT_METRICS:
            values = [r["ratio"][metric] for r in runs]
            entry[f"ratio/{metric}"] = {
                "per_seed": values,
                "mean": mean(values),
                "std_sample_ddof1": sd(values, 1),
                "std_population_ddof0": sd(values, 0),
            }
        for metric in COMPONENT_METRICS:
            values = [r["mmd"][metric] for r in runs]
            entry[f"mmd/{metric}"] = {"per_seed": values, "mean": mean(values)}
        for field in VUN_FIELDS:
            values = [r["vun"][field] for r in runs]
            entry[f"vun/{field}"] = {
                "per_seed": values,
                "mean": mean(values),
                "std_sample_ddof1": sd(values, 1),
            }
        aggregates = [r["ratio_aggregate"] for r in runs]
        losses = [r["validation_loss"] for r in runs]
        entry["ratio_aggregate"] = {
            "per_seed": aggregates,
            "mean": mean(aggregates),
            "std_sample_ddof1": sd(aggregates, 1),
            "std_population_ddof0": sd(aggregates, 0),
        }
        entry["validation_loss"] = {
            "per_seed": losses,
            "mean": mean(losses),
            "std_sample_ddof1": sd(losses, 1),
        }
        entry["er_ratio_per_seed"] = [r["er_ratio"] for r in runs]
        per_arm[band] = entry

    # The reviewer asked specifically whether uniqueness and novelty are 1.000 everywhere.
    unique_values = [r["vun"]["unique"] for r in per_run if r["status"] == "OK"]
    novel_values = [r["vun"]["novel"] for r in per_run if r["status"] == "OK"]
    valid_values = [r["vun"]["valid"] for r in per_run if r["status"] == "OK"]
    uniqueness_check = {
        "n_runs_checked": len(unique_values),
        "unique_all_exactly_1.0": all(v == 1.0 for v in unique_values),
        "novel_all_exactly_1.0": all(v == 1.0 for v in novel_values),
        "valid_all_exactly_0.0": all(v == 0.0 for v in valid_values),
        "distinct_unique_values": sorted(set(unique_values)),
        "distinct_novel_values": sorted(set(novel_values)),
        "interpretation": (
            "Confirmed: uniqueness and novelty are exactly 1.000 in all 12 k=8 runs, and "
            "validity exactly 0.000. With 32 samples of a 64-node graph drawn from a "
            "continuous relaxation, no two generated graphs collide and none coincides with a "
            "training graph, so unique and novel are uninformative here -- they carry no "
            "signal separating the arms and should not be read as a quality result. The "
            "joint vun metric is therefore pinned at 0.000 purely by the validity term."
        ),
    }

    # Cross-arm ordering per component metric, so the reviewer can see where low actually wins.
    ordering = {}
    for metric in COMPONENT_METRICS:
        means = {band: per_arm[band][f"ratio/{metric}"]["mean"] for band in K8_BANDS}
        ordering[metric] = {
            "means": means,
            "best_arm": min(means, key=means.get),
            "low_beats_none": means["low"] < means["none"],
        }
    ordering["_aggregate"] = {
        "means": {band: per_arm[band]["ratio_aggregate"]["mean"] for band in K8_BANDS},
        "low_beats_none": per_arm["low"]["ratio_aggregate"]["mean"] < per_arm["none"]["ratio_aggregate"]["mean"],
    }

    return {
        "scope": "the twelve k=8 Planar arms -- the only runs in the project with a full "
        "per-metric evaluation block",
        "metrics": list(COMPONENT_METRICS),
        "vun_fields": list(VUN_FIELDS),
        "per_run": per_run,
        "per_arm": per_arm,
        "uniqueness_novelty_check": uniqueness_check,
        "per_metric_ordering": ordering,
    }


# --------------------------------------------------------------------------------------
# Lost breakdowns
# --------------------------------------------------------------------------------------


def lost_breakdowns(store: Store) -> dict:
    entries = []
    for name in ("k_sweep_planar.json", "k_sweep_planar_seedcheck.json", "k_sweep_planar_summary.json"):
        payload = store.load(name)
        if payload is None:
            continue
        provenance = payload.get("provenance", {})
        rows = payload.get("rows") or []
        fields = sorted({key for r in rows for key in r})
        entries.append(
            {
                "file": name,
                "n_rows": len(rows),
                "fields_present": fields,
                "has_per_metric_breakdown": any(f.startswith("ratio/") for f in fields),
                "has_validation_loss": "validation_loss" in fields,
                "provenance": provenance,
                "orca_available_flag": payload.get("orca_available"),
            }
        )
    for name in ("discrete_scaled_up_planar.json",):
        payload = store.load(name)
        if payload is None:
            continue
        entries.append(
            {
                "file": name,
                "n_rows": len(payload.get("runs") or []),
                "fields_present": sorted({key for r in payload.get("runs", []) for key in r}),
                "has_per_metric_breakdown": False,
                "has_validation_loss": False,
                "provenance": payload.get("provenance", {}),
                "orca_available_flag": None,
            }
        )
    return {
        "statement": (
            "The k-sweep and the recovered discrete runs carry ONLY an aggregate Ratio and a "
            "validity fraction. Their degree / clustering / spectral / wavelet / orbit "
            "components are unrecoverable: the Colab VM was recycled before the JSON reports "
            "were downloaded, and the generated samples were never committed. This audit does "
            "NOT reconstruct them from the aggregate -- a mean of five numbers does not "
            "determine the five numbers, and inventing a split would be fabrication."
        ),
        "affected_paper_content": [
            "Figure 1(a) and (b) -- every point",
            "Table 2 -- all three arms",
            "S4.2 -- the entire flat-surface analysis (23-cell mean, sd, z-scores, control spans)",
            "abstract -- the 2.3x and 29.7x improvements",
            "S4.5 -- the discrete 39.60 / 15.40 pair",
        ],
        "files": entries,
    }


# --------------------------------------------------------------------------------------
# Table 1 vs Table 2 reconciliation
# --------------------------------------------------------------------------------------


def _welch(a: list[float], b: list[float]) -> dict:
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se else float("nan")
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return {
        "mean_a": ma,
        "mean_b": mb,
        "difference": ma - mb,
        "std_error_of_difference": se,
        "welch_t": t,
        "welch_df": num / den if den else float("nan"),
        "difference_in_se_units": abs(ma - mb) / se if se else float("nan"),
    }


def reconcile_tables(store: Store) -> dict:
    sweep = store.load("k_sweep_planar.json")
    seedcheck = store.load("k_sweep_planar_seedcheck.json")

    table1 = {band: k8_ratios(store, band) for band in K8_BANDS}
    table2_low = [row(seedcheck, band="low", k=8, seed=s)["ratio"] for s in K8_SEEDS]
    table2_none = [row(seedcheck, band="none", k=2, seed=s)["ratio"] for s in K8_SEEDS]
    table2_high32 = [row(seedcheck, band="high", k=32, seed=s)["ratio"] for s in K8_SEEDS]

    # --- test 1: seed-0 agreement arm by arm, same nominal config ---------------------
    seed0 = {}
    for band, sweep_key in (("low", ("low", 8)), ("high", ("high", 8)), ("random", ("random", 8)), ("none", ("none", 2))):
        sweep_value = row(sweep, band=sweep_key[0], k=sweep_key[1])["ratio"]
        table1_value = table1[band][0]
        seed0[band] = {
            "table1_file": k8_file(band, 0),
            "table1_seed0": table1_value,
            "sweep_cell": f"{sweep_key[0]} k={sweep_key[1]}",
            "sweep_seed0": sweep_value,
            "absolute_difference": sweep_value - table1_value,
            "relative_difference_pct": 100.0 * (sweep_value - table1_value) / table1_value,
            "note": "band=none carries an all-zero condition, so the sweep's k=2 cell and "
            "Table 1's k=8 cell differ only in the (uninformative) condition width"
            if band == "none"
            else None,
        }

    # --- test 2: is `low` simply the noisiest arm? ------------------------------------
    noise = {}
    for band in K8_BANDS:
        values = table1[band]
        noise[f"table1_{band}"] = {
            "per_seed": values,
            "mean": mean(values),
            "std_ddof1": sd(values, 1),
            "coefficient_of_variation_pct": 100.0 * sd(values, 1) / mean(values),
        }
    for label, values in (("seedcheck_low_k8", table2_low), ("seedcheck_none", table2_none), ("seedcheck_high_k32", table2_high32)):
        noise[label] = {
            "per_seed": values,
            "mean": mean(values),
            "std_ddof1": sd(values, 1),
            "coefficient_of_variation_pct": 100.0 * sd(values, 1) / mean(values),
        }
    cvs_table1 = {band: noise[f"table1_{band}"]["coefficient_of_variation_pct"] for band in K8_BANDS}
    low_is_noisiest_t1 = max(cvs_table1, key=cvs_table1.get) == "low"
    cv_ratio_t1 = cvs_table1["low"] / max(v for b, v in cvs_table1.items() if b != "low")

    # --- test 3: is the cross-set difference distinguishable from seed noise? ---------
    welch_low = _welch(table1["low"], table2_low)
    welch_none = _welch(table1["none"], table2_none)
    paired_low = [t1 - t2 for t1, t2 in zip(table1["low"], table2_low)]
    paired = {
        "per_seed_difference_table1_minus_table2": paired_low,
        "mean": mean(paired_low),
        "std_ddof1": sd(paired_low, 1),
        "t": mean(paired_low) / (sd(paired_low, 1) / math.sqrt(len(paired_low))),
        "caveat": "Pairing by seed index assumes seed 0 in one set corresponds to seed 0 in the "
        "other. Across different machines and library versions a fixed seed does not "
        "reproduce the same realization, so this pairing is nominal, not real.",
    }

    # --- test 4: what the git history does and does not explain -----------------------
    table1_commit = (_git_ok("log", "-1", "--format=%h %ad %s", "--date=short", "--", f"results/{k8_file('low', 0)}") or "").strip()
    sweep_commit = (_git_ok("log", "-1", "--format=%h %ad %s", "--date=short", "--", "results/k_sweep_planar.json") or "").strip()
    seedcheck_commit = (_git_ok("log", "-1", "--format=%h %ad %s", "--date=short", "--", "results/k_sweep_planar_seedcheck.json") or "").strip()
    between = (
        _git_ok(
            "log",
            "--format=%h %s",
            f"{table1_commit.split()[0]}..{sweep_commit.split()[0]}",
            "--",
            "fald/",
            "scripts/train_adjacency_diffusion.py",
        )
        if table1_commit and sweep_commit
        else None
    )

    wall_clock = {
        "table1_training_minutes_per_run": [
            store.load(k8_file(b, s))["training_minutes"] for b in K8_BANDS for s in K8_SEEDS
        ],
        "sweep_runtime_minutes_per_run": path_get(sweep, "provenance.runtime_minutes_per_run"),
        "reading": "Nominally the same 200-epoch run takes ~2.1 min in the Table 1 set and 7.6 "
        "min in the sweep. That is a ~3.6x wall-clock difference and is independent "
        "evidence the two sets did not execute on the same hardware.",
    }

    hardware = {
        "table1": {
            "declared_hardware_field": None,
            "inferred_from_samples_path": store.load(k8_file("low", 0)).get("samples_path"),
            "inference": "a Windows OneDrive path belonging to a teammate, so the Table 1 set "
            "was produced on a local Windows machine",
        },
        "sweep": {"declared_hardware_field": path_get(sweep, "provenance.hardware")},
        "seedcheck": {"declared_hardware_field": path_get(seedcheck, "provenance.hardware")},
    }

    # --- test 2b: WHY is low the noisiest arm?  The component metrics can answer this,
    # because Ratio is the unweighted mean of five components, so each component's
    # covariance with the aggregate is its exact share of the aggregate's seed variance.
    variance_decomposition = {}
    for band in K8_BANDS:
        components = {
            metric: [
                path_get(store.load(k8_file(band, seed)), f"evaluation.ratio/{metric}")
                for seed in K8_SEEDS
            ]
            for metric in COMPONENT_METRICS
        }
        aggregate = table1[band]
        var_aggregate = statistics.variance(aggregate)
        shares = {}
        for metric, values in components.items():
            scaled = [v / len(COMPONENT_METRICS) for v in values]
            covariance = sum(
                (a - mean(scaled)) * (b - mean(aggregate)) for a, b in zip(scaled, aggregate)
            ) / (len(aggregate) - 1)
            shares[metric] = {
                "per_seed": values,
                "range": max(values) - min(values),
                "share_of_aggregate_seed_variance": covariance / var_aggregate if var_aggregate else None,
            }
        variance_decomposition[band] = {
            "aggregate_per_seed": aggregate,
            "aggregate_variance": var_aggregate,
            "aggregate_std_ddof1": sd(aggregate, 1),
            "per_metric": shares,
        }

    return {
        "question": "Table 1 gives low k=8 = [151.29, 146.17, 176.44] (mean 158.0). The "
        "seed-check for the same nominal configuration gives [116.27, 146.79, "
        "162.67] (mean 141.9). Why?",
        "test_1_seed0_agreement_by_arm": seed0,
        "test_1_reading": (
            "high and random and none agree to within 1-5% between the two sets; low diverges "
            "by 23%. The divergence is confined to the low arm, exactly as the noisiest-arm "
            "hypothesis predicts."
        ),
        "test_2_arm_noise": noise,
        "test_2b_variance_decomposition": variance_decomposition,
        "test_2b_reading": (
            "Ratio is the unweighted mean of five components, so each component's covariance "
            "with the aggregate is exactly its share of the aggregate's seed-to-seed variance. "
            "For the low arm, orbit accounts for essentially all of it: orbit is 390.98 / "
            "388.00 / 537.44 across the three seeds, a 149-point range whose fifth (29.9) "
            "covers 30.3 of the aggregate's 30.3-point range. low's four other components are "
            "tight (spectral 7.06-7.63, clustering 14.90-14.98, wavelet 127.2-130.3). So low is "
            "not diffusely noisy -- its instability is orbit instability, one component with a "
            "~1e-4 reference floor that dominates any mean it enters. That is the same metric-"
            "scale problem the paper flags in S5, and it is the concrete mechanism behind the "
            "Table 1 / Table 2 divergence. Note this can only be checked on the k=8 arms: the "
            "sweep runs kept no per-metric breakdown, so the same decomposition cannot be run "
            "on the Table 2 values."
        ),
        "test_2_reading": {
            "low_has_highest_cv_in_table1": low_is_noisiest_t1,
            "low_cv_over_next_noisiest_arm": cv_ratio_t1,
            "statement": (
                "In Table 1's own three seeds, low's coefficient of variation is 10.3%, against "
                "2.7-3.6% for none, high and random -- roughly 3x noisier than any other arm. "
                "The seed-check reproduces this independently: low k=8 varies by 16.6% CV while "
                "none varies by 2.3% and high k=32 by 6.4%. Two independent three-seed sets "
                "agree that low is by far the least stable arm."
            ),
        },
        "test_3_is_the_gap_significant": {
            "welch_low_table1_vs_table2": welch_low,
            "welch_none_table1_vs_table2": welch_none,
            "paired_by_nominal_seed": paired,
            "statement": (
                "The 16.1-point gap between the two low means is 0.97 standard errors -- Welch "
                "t = 0.97 on ~3.5 df, nowhere near significance. Given low's own seed variance, "
                "the two sets report statistically indistinguishable values. The hypothesis "
                "that low is simply the noisiest arm is CONSISTENT with the data and no code "
                "change is needed to explain the difference of means."
            ),
        },
        "test_4_code_archaeology": {
            "table1_commit": table1_commit,
            "sweep_commit": sweep_commit,
            "seedcheck_commit": seedcheck_commit,
            "commits_touching_the_training_path_between_them": (between or "").strip().splitlines(),
            "diffs_reviewed": [
                "fald/data/conditioning.py -- adds the gaussian/shuffled bands and a derangement "
                "donor index. For band in {none,low,high,random} the donor index is the identity, "
                "the RNG stream (seed+index) is unchanged, and the crop/pad rewrite is a no-op on "
                "an equal-n dataset. NO numerical change to Planar low/high/random/none.",
                "fald/models/adjacency_diffusion.py -- replaces the DiGress import with an "
                "importlib file-based loader. Import mechanics only; NO numerical change.",
                "fald/eval/validity.py -- adds sbm_validity. Planar runs still use is_planar; NO "
                "numerical change.",
                "scripts/train_adjacency_diffusion.py -- adds the gaussian/shuffled choices, the "
                "validity diagnosis print, the metrics/orca_available report fields and the "
                "--results-subdir plumbing. NO change to the model, the optimizer, the sampler or "
                "the evaluator.",
                "fald/eval/mmd.py, fald/eval/descriptors.py, fald/eval/evaluator.py, "
                "fald/data/spectre.py, fald/data/spectral.py -- UNCHANGED between the two "
                "commits, so the metric and the split are identical in both sets.",
            ],
            "conclusion": "No code change between the two runs can account for the difference.",
        },
        "test_5_environment": {
            "hardware": hardware,
            "wall_clock": wall_clock,
            "environment_capture_in_artifacts": {
                "git_commit_recorded_in_report": False,
                "torch_version_recorded": False,
                "cuda_version_recorded": False,
                "hostname_recorded": False,
                "random_state_recorded": False,
                "statement": "Neither set records a library version, a CUDA version, a device "
                "name or a git SHA. `config` holds model hyperparameters only.",
            },
            "evaluation_side_noise": {
                "er_ratio_by_seed_in_table1": {
                    str(seed): store.load(k8_file("none", seed))["er_ratio"] for seed in K8_SEEDS
                },
                "statement": "The ER reference -- which depends only on the evaluator and the "
                "seed, not on any trained model -- itself moves 322.65 / 337.91 / "
                "343.89 across the three seeds, a 6.6% spread. A meaningful part of "
                "the run-to-run variation is evaluation-side sampling noise at "
                "n=32, not training noise.",
            },
        },
        "verdict": {
            "hypothesis_tested": "`low` is simply the noisiest arm across independent runs.",
            "supported": True,
            "evidence": [
                "Only the low arm diverges between the two sets (23%); high, random and none "
                "agree to 1-5%.",
                "low has ~3x the coefficient of variation of every other arm, in both "
                "three-seed sets independently.",
                "The difference of the two low means is 0.97 standard errors (Welch t = 0.97), "
                "i.e. indistinguishable from zero given low's own seed spread.",
            ],
            "what_is_NOT_recoverable": (
                "The specific seed-0 result -- 151.29 in Table 1 against 116.27 in the sweep, a "
                "23% swing at a nominally identical configuration AND a nominally identical "
                "seed -- is NOT recoverable. A fixed seed reproduces exactly only on the same "
                "machine and library stack; these two runs were executed on different platforms "
                "(a Windows workstation versus a Colab T4, inferred from the samples_path and "
                "confirmed by a 3.6x wall-clock difference), and neither artifact records a "
                "torch version, a CUDA version, a device, or a git SHA. The checkpoints are "
                "gone, the sweep's samples were never downloaded, and the sweep's per-metric "
                "breakdown was lost with the VM, so the run cannot be re-scored or re-executed "
                "to settle it. We can say the difference of MEANS is fully explained by low's "
                "seed variance; we CANNOT say what made seed 0 specifically differ, and no "
                "evidence in this repository would let anyone determine it. Stating a cause "
                "would be a guess."
            ),
            "recommendation": "Report Table 1 and Table 2 as two independent three-seed "
            "estimates of the same arm (158.0 +/- 16.2 and 141.9 +/- 19.3) "
            "whose difference is within noise, and stop treating either seed-0 "
            "draw as a reproducible point value.",
        },
    }


# --------------------------------------------------------------------------------------
# Provenance facts
# --------------------------------------------------------------------------------------


def provenance_facts(store: Store) -> dict:
    sample_paths = {}
    for band in K8_BANDS:
        for seed in K8_SEEDS:
            payload = store.load(k8_file(band, seed))
            if payload is None:
                continue
            declared = payload.get("samples_path")
            local = RESULTS / Path(declared.replace("\\", "/")).name if declared else None
            sample_paths[k8_file(band, seed)] = {
                "declared_samples_path": declared,
                "declared_path_is_external": bool(declared and declared.startswith("C:\\")),
                "present_in_repo": bool(local and local.is_file()),
            }
    present = [k for k, v in sample_paths.items() if v["present_in_repo"]]

    checkpoints = REPO / "checkpoints"
    checkpoint_files = sorted(p.name for p in checkpoints.iterdir()) if checkpoints.is_dir() else []

    tracked_pt = (_git_ok("ls-files", "results/") or "")
    tracked_pt_files = [line for line in tracked_pt.splitlines() if line.endswith(".pt")]

    return {
        "generated_samples": {
            "statement": "Eleven of the twelve k=8 runs declare a samples_path on a teammate's "
            "Windows OneDrive, so those .pt files are not in this repository. Only "
            "one sample file exists locally, and it is not version-controlled: "
            "`.gitignore` excludes `*.pt`, so NO generated samples are committed. "
            "Any claim requiring the generated graphs themselves -- re-scoring a "
            "run, recomputing a component metric, measuring distance-to-planarity "
            "on generated output -- cannot be reproduced from this repository.",
            "per_run": sample_paths,
            "present_in_repo": present,
            "n_present": len(present),
            "n_declared": len(sample_paths),
            "tracked_in_git": tracked_pt_files,
        },
        "checkpoints": {
            "directory": "checkpoints/",
            "files": checkpoint_files,
            "statement": "checkpoints/ is empty. The models that produced every number in the "
            "paper are gone, and the paper states that the validation split which "
            "scored the results also selected those checkpoints. No number can be "
            "re-scored on the test split without retraining.",
        },
        "environment_capture": {
            "statement": "No run report records a git commit, a torch/CUDA version, a device "
            "name or an OS. The `config` block holds model hyperparameters only, "
            "and the training budget the paper quotes (200 epochs, batch 16, lr "
            "3e-4) appears nowhere in the artifacts -- only as argparse defaults in "
            "scripts/train_adjacency_diffusion.py, which any run could have "
            "overridden without leaving a trace.",
            "fields_present_in_config": sorted((store.load(k8_file("low", 0)) or {}).get("config", {})),
        },
        "code_version": {
            "statement": "The reviewer asked for the code version behind each table. No run "
            "report records a git SHA, so the code version behind a result is only "
            "inferable from the commit that introduced that result file -- an upper "
            "bound on the code's age, not the commit the run actually used. The paper "
            "source and the figure script are tracked (both landed in 902532e), but "
            "both carry uncommitted modifications at audit time, so the text being "
            "audited is not itself a committed revision.",
            "uncommitted_at_audit_time": [
                line
                for line in (_git_ok("status", "--porcelain", "paper/", "scripts/", "docs/") or "").splitlines()
            ],
            "paper_main_tex_tracked": _tracked("paper/main.tex"),
            "make_paper_figures_tracked": _tracked("scripts/make_paper_figures.py"),
            "commit_introducing_each_result_file": {
                name: (_git_ok("log", "--diff-filter=A", "-1", "--format=%h %ad %s", "--date=short", "--", f"results/{name}") or "UNTRACKED").strip()
                for name in (
                    k8_file("low", 0),
                    k8_file("none", 0),
                    "frequency_pilot_planar_k8_summary.json",
                    "k_sweep_planar.json",
                    "k_sweep_planar_seedcheck.json",
                    "condition_leakage_planar.json",
                    "condition_leakage_sbm.json",
                    "eval_calibration_planar.json",
                    "report_breakdown.json",
                    "stage8_results.json",
                    "stage9_downstream.json",
                    "discrete_scaled_up_planar.json",
                )
            },
        },
        "metric_scale": {
            "statement": "Ratio averages whichever metrics were available at run time. The k=8 "
            "arms, the discrete arms and the sweep are five-metric (with ORCA); "
            "Stage 8's end-to-end numbers are four-metric and NOT comparable to the "
            "tables. The paper says so at the point of use.",
            "five_metric_er_reference": (store.load(k8_file("none", 0)) or {}).get("er_ratio"),
            "four_metric_er_reference": path_get(store.load("stage8_results.json"), "adjacency_diffusion_training.high_k32.er_ratio")
            if store.load("stage8_results.json")
            else None,
        },
        "std_convention_inconsistency": {
            "statement": "Table 1's standard deviations are SAMPLE standard deviations (ddof=1) "
            "computed from the three run reports; Table 2's come from "
            "k_sweep_planar_seedcheck.json's `aggregates`, which uses the "
            "POPULATION standard deviation (ddof=0). The same three-seed quantity is "
            "therefore reported under two different conventions two pages apart. "
            "Table 2's low k=8 would read 23.6 rather than 19.3 under Table 1's "
            "convention, and high k=32 would read 0.71 rather than 0.58.",
            "table2_low_k8_ddof1": sd([row(store.load("k_sweep_planar_seedcheck.json"), band="low", k=8, seed=s)["ratio"] for s in K8_SEEDS], 1),
            "table2_low_k8_ddof0_as_published": path_get(store.load("k_sweep_planar_seedcheck.json"), "aggregates.low_k8.ratio_std"),
            "table2_high_k32_ddof1": sd([row(store.load("k_sweep_planar_seedcheck.json"), band="high", k=32, seed=s)["ratio"] for s in K8_SEEDS], 1),
            "table2_high_k32_ddof0_as_published": path_get(store.load("k_sweep_planar_seedcheck.json"), "aggregates.high_k32.ratio_std"),
        },
    }


# --------------------------------------------------------------------------------------
# HEAD vs worktree
# --------------------------------------------------------------------------------------


def head_vs_worktree() -> dict:
    """Where the committed results and the working tree disagree, field by field."""
    head, work = Store("head"), Store("worktree")
    names = sorted(p.name for p in RESULTS.glob("*.json"))
    report = []
    for name in names:
        h, w = head.load(name), work.load(name)
        if h is None and w is None:
            continue
        if h is None:
            report.append({"file": name, "status": "UNCOMMITTED (working tree only)"})
            continue
        if w is None:
            report.append({"file": name, "status": "COMMITTED but absent from the working tree"})
            continue
        if h == w:
            continue
        entry: dict[str, Any] = {"file": name, "status": "DIFFERS between HEAD and working tree"}
        if isinstance(h.get("rows"), list) and isinstance(w.get("rows"), list):
            changes = []
            hrows = {(r.get("band"), r.get("k"), r.get("seed")): r for r in h["rows"]}
            wrows = {(r.get("band"), r.get("k"), r.get("seed")): r for r in w["rows"]}
            for key in sorted(set(hrows) | set(wrows), key=str):
                hr, wr = hrows.get(key), wrows.get(key)
                if hr is None or wr is None:
                    changes.append({"row": str(key), "change": "row added or removed"})
                    continue
                for field in sorted(set(hr) | set(wr)):
                    hv, wv = hr.get(field), wr.get(field)
                    if isinstance(hv, (int, float)) and isinstance(wv, (int, float)):
                        if abs(hv - wv) > 1e-12:
                            changes.append({"row": str(key), "field": field, "head": hv, "worktree": wv})
                    elif hv != wv:
                        changes.append({"row": str(key), "field": field, "head": hv, "worktree": wv})
            entry["row_level_changes"] = changes
            entry["n_row_level_changes"] = len(changes)
        entry["top_level_keys_added"] = sorted(set(w) - set(h))
        report.append(entry)
    return {
        "statement": "The audit's primary numbers come from HEAD, because that is what a "
        "reviewer can check out. This section records where the working tree has "
        "already moved, so a claim that matches HEAD today may not match tomorrow's "
        "commit.",
        "files": report,
        "uncommitted_result_files": [
            line for line in (_git_ok("status", "--porcelain", "results/") or "").splitlines()
        ],
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def in_flight_corrections() -> dict:
    """Table 3's backing file is being corrected in the working tree; quantify the change.

    Buried inside `head_vs_worktree`'s row-level diff is the single most consequential
    provenance fact in this repository, so it gets its own section: the leakage probe's
    orientation selection was a no-op, and every committed `low`-band number was measured with
    the ranking sign inverted.
    """
    head, work = Store("head"), Store("worktree")
    cells = []
    for band in ("low", "high", "random", "gaussian", "shuffled"):
        for k in (2, 4, 8, 16, 32):
            h = work_row = None
            try:
                h = row(head.load("condition_leakage_planar.json"), band=band, k=k)
            except (LookupError, TypeError):
                pass
            try:
                work_row = row(work.load("condition_leakage_planar.json"), band=band, k=k)
            except (LookupError, TypeError):
                pass
            if h is None or work_row is None:
                continue
            cells.append(
                {
                    "band": band,
                    "k": k,
                    "paper_table3_value_pct": round(100.0 * h["edge_recovery"], 1),
                    "committed_head_pct": 100.0 * h["edge_recovery"],
                    "corrected_worktree_pct": 100.0 * work_row["edge_recovery"],
                    "delta_pct_points": 100.0 * (work_row["edge_recovery"] - h["edge_recovery"]),
                    "fixed_sign_negative_pct": 100.0 * work_row.get("edge_recovery_neg", float("nan")),
                    "fixed_sign_positive_pct": 100.0 * work_row.get("edge_recovery_pos", float("nan")),
                    "fraction_of_graphs_where_positive_sign_wins": work_row.get("orientation_pos_frac"),
                }
            )
    changed = [c for c in cells if abs(c["delta_pct_points"]) > 0.5]
    return {
        "status": "UNCOMMITTED at the time of this audit -- the correction lives in the "
        "working tree (scripts/condition_leakage.py and "
        "results/condition_leakage_{planar,sbm}.json are modified; "
        "results/condition_leakage_*_single_orientation.json freeze the old numbers).",
        "what_changed_in_the_code": (
            "reconstruct_from_condition() built both sign orientations of the reconstruction "
            "and then chose between them with `hits = graph.number_of_edges()` and a strict "
            "`>`. Both orientations keep exactly m edges by construction, so the comparison was "
            "always False and the first iteration -- orientation -1.0 -- always won. The "
            "per-graph sign selection the docstring described never ran. The fix scores each "
            "orientation against the true edge set instead."
        ),
        "why_it_matters_for_the_paper": (
            "Table 3's `low` row, the abstract's '0.0% of the target's edges', Table 2's leak "
            "column and the whole of S4.3 rest on low recovering nothing. Under the corrected "
            "probe low k=8 recovers 72.7%, not 0.0%, and the anti-correlation argument in S4.3 "
            "inverts: at k=8 low leaks MORE than high, not less. This is not a rounding "
            "difference; it is the paper's central claim."
        ),
        "the_correction_is_not_an_artifact_of_the_extra_oracle_bit": (
            "The corrected rule picks the better orientation per graph using the true edge set, "
            "which is one more privileged input than the probe had before. That does not "
            "explain the change: `orientation_pos_frac` is 1.000 for every low cell on both "
            "datasets, so a single FIXED global sign (+1) already gives low 48.1-78.7% on "
            "Planar. The old 0.0% was a sign-convention artifact, not a per-graph cherry-pick."
        ),
        "cells_whose_value_changes": len(changed),
        "cells": cells,
        "newly_available_evidence_not_in_HEAD": [
            "results/condition_leakage_full_spectrum_planar.json",
            "results/condition_leakage_full_spectrum_sbm.json",
            "-- these add the k=-1 full-spectrum row, which is the backing file S3's "
            "'the full spectrum recovers 100.0%' currently lacks (that number is presently "
            "traceable only to docs/LEAKAGE.md prose).",
        ],
    }


def summarize(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "n_claims": len(records),
        "by_status": counts,
        "mismatches": [r["id"] for r in records if r["status"] == "MISMATCH"],
        "rounding_only": [r["id"] for r in records if r["status"] == "ROUNDING"],
        "untraceable": [r["id"] for r in records if r["status"] == "UNTRACEABLE"],
        "errors": [r["id"] for r in records if r["status"] == "ERROR"],
        "tex_anchors_not_found": [r["id"] for r in records if not r["tex_anchor_found_in_main_tex"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="head", choices=["head", "worktree", "both"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print", dest="do_print", action="store_true", help="summary to stdout")
    args = parser.parse_args()

    primary = "worktree" if args.source == "worktree" else "head"
    store = Store(primary)

    claims = build_claims()
    records = audit_claims(store, claims)

    payload = {
        "audit": {
            "script": "scripts/audit_provenance.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repository": str(REPO),
            "git_head": _git("rev-parse", "HEAD").strip(),
            "git_head_subject": _git("log", "-1", "--format=%s").strip(),
            "result_source": primary,
            "paper_source": "paper/main.tex",
            # paper/main.tex is under active revision.  Pinning its digest makes it obvious
            # when this audit has gone stale relative to the text it checked.
            "paper_sha256": hashlib.sha256(PAPER.read_bytes()).hexdigest() if PAPER.is_file() else None,
            "paper_committed": _tracked("paper/main.tex"),
            "missing_result_files": store.missing,
            "purpose": "Reviewer request: map each table and figure to its result files, "
            "configuration, seed and code version; report component metrics.",
        },
        "summary": summarize(records),
        "claims": records,
        "component_metrics": component_metrics(store),
        "lost_breakdowns": lost_breakdowns(store),
        "table1_vs_table2": reconcile_tables(store),
        "provenance_facts": provenance_facts(store),
    }
    payload["in_flight_corrections"] = in_flight_corrections()
    if args.source == "both":
        payload["head_vs_worktree"] = head_vs_worktree()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    summary = payload["summary"]
    print(f"wrote {args.out}")
    print(f"  claims audited : {summary['n_claims']}")
    for status, count in sorted(summary["by_status"].items()):
        print(f"  {status:<12s}: {count}")
    if args.do_print:
        for record in records:
            if record["status"] in {"MISMATCH", "ROUNDING", "UNTRACEABLE", "ERROR"}:
                print(
                    f"  [{record['status']}] {record['id']:<28s} paper={record['paper_value']} "
                    f"file={record['file_value']} :: {record['location']}"
                )
    return 1 if summary["by_status"].get("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
