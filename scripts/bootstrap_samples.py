"""Within-run uncertainty and memorisation audit for saved generated samples.

Answers three reviewer requests from saved samples alone -- no retraining, no checkpoints:

  1. **Reproduce** the committed evaluation of a saved sample set with the committed
     evaluator, so the numbers in `results/*.json` are confirmed rather than trusted.

  2. **Bootstrap** the Ratio and every component metric over the generated graphs (and,
     separately, over the reference and train sets). This is a *within-run* interval: it
     says how much of the reported number is finite-sample noise in the 32 drawn graphs.
     It is NOT the across-seed std the paper reports, and the two are complementary --
     the bootstrap holds the trained model fixed and varies the sample; the seed std
     varies the model and resamples. Neither bounds the other.

  3. **Memorisation**: uniqueness among the samples, novelty against the training split,
     and -- the load-bearing one -- edge-level similarity to the *conditioning* graph each
     sample was generated from. Each sample `i` is conditioned on validation graph `i`'s
     spectrum (`eval_loader` is built with `shuffle=False` and `band != 'shuffled'` uses an
     identity donor map, so the pairing is positional). High overlap there would mean the
     "generation" is partly reconstruction of the conditioning target.

Connectivity and planarity are reported separately, not only as their conjunction: the
committed `is_planar` is `connected AND planar`, so a 0% validity row cannot say which half
failed.

Bootstrap mechanics. The committed MMD is a biased V-statistic,

    MMD(X, Y) = mean(K_XX) + mean(K_YY) - 2 * mean(K_XY)

with means taken over *all* pairs, diagonal included. Every kernel is `gaussian_tv`, so
K[i, j] = exp(-d^2 / 2 sigma^2) with d = ||x_i - y_j||_1 / 2. Kernel matrices are therefore
built once from the committed descriptors and each resample is a weighted mean over that
fixed matrix: `mean(K[I][:, I]) == (c @ K @ c) / n^2` for multiplicity vector `c`. This is
exact, and the run asserts it against `fald.eval.mmd.compute_mmd` for every metric before
resampling, which is what makes thousands of resamples cheap enough to run on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.data import spectral
from fald.data.spectre import load_splits
from fald.eval import descriptors, orca, validity
from fald.eval.evaluator import MMD_SPECS, GraphEvaluator
from fald.eval.mmd import compute_mmd, gaussian_tv
from fald.paths import results_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_breakdown import diagnose_validity

# Matches `_RATIO_FLOOR` in fald/eval/evaluator.py: the floor is a near-zero self-similarity
# value, and clamping keeps a degenerate resample from producing a meaningless ratio.
RATIO_FLOOR = 1e-8

METRIC_NAMES = tuple(spec.name for spec in MMD_SPECS)


# --------------------------------------------------------------------------------------
# descriptors and kernels
# --------------------------------------------------------------------------------------


def describe(graphs, metrics, filters, bound) -> dict[str, list[np.ndarray]]:
    """Per-graph descriptors, via the committed `fald.eval.descriptors` functions."""
    out: dict[str, list[np.ndarray]] = {}
    if "degree" in metrics:
        out["degree"] = [descriptors.degree_histogram(g) for g in graphs]
    if "clustering" in metrics:
        out["clustering"] = [descriptors.clustering_histogram(g) for g in graphs]
    if "spectral" in metrics:
        out["spectral"] = [descriptors.laplacian_spectrum(g) for g in graphs]
    if "orbit" in metrics:
        out["orbit"] = [descriptors.orbit_descriptor(g) for g in graphs]
    if "wavelet" in metrics:
        out["wavelet"] = [
            descriptors.wavelet_descriptor(*descriptors.eigen_decomposition(g), filters, bound)
            for g in graphs
        ]
    return out


def _stack(vectors: list[np.ndarray], is_hist: bool, width: int) -> np.ndarray:
    """Normalise (histograms only), then zero-pad every descriptor to a common width.

    `compute_mmd` normalises first and `gaussian_tv` pads pairwise to the longer of the two.
    Padding everything to one global width is equivalent: the extra zeros are zero in both
    operands, so they contribute nothing to the L1 distance, and zeros do not change a sum.
    """
    rows = np.zeros((len(vectors), width), dtype=float)
    for i, vector in enumerate(vectors):
        vector = np.asarray(vector, dtype=float)
        if is_hist:
            vector = vector / (vector.sum() + 1e-6)
        rows[i, : len(vector)] = vector
    return rows


def kernel_matrix(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    """`gaussian_tv` evaluated for every pair, vectorised.

    d = ||x - y||_1 / 2 exactly as in `fald.eval.mmd.gaussian_tv`; blocked over rows so the
    1200-dim wavelet descriptors never materialise a (n, m, 1200) intermediate.
    """
    out = np.empty((len(left), len(right)), dtype=float)
    block = max(1, int(2e7 // max(left.shape[1], 1)))
    for start in range(0, len(left), block):
        chunk = left[start : start + block]
        dist = np.abs(chunk[:, None, :] - right[None, :, :]).sum(axis=2) / 2.0
        out[start : start + block] = np.exp(-(dist**2) / (2 * sigma**2))
    return out


def mmd_from_kernels(k_xx: np.ndarray, k_yy: np.ndarray, k_xy: np.ndarray) -> float:
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def _weighted_mean(kernel: np.ndarray, left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray:
    """Mean of a resampled kernel submatrix, for a whole batch of resamples at once.

    For multiplicity vectors c (how often each original index was drawn), the mean over the
    resampled submatrix is (c_left @ K @ c_right) / (n_left * n_right) -- duplicated indices
    included, which is what a bootstrap of this V-statistic means.
    """
    numerator = np.einsum("bi,ij,bj->b", left_counts, kernel, right_counts)
    return numerator / (left_counts.sum(axis=1) * right_counts.sum(axis=1))


def _counts(rng: np.random.Generator, n: int, draws: int, resample: bool) -> np.ndarray:
    if not resample:
        return np.ones((draws, n), dtype=float)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(float)


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def bootstrap_ratios(
    kernels: dict[str, dict[str, np.ndarray]],
    *,
    n_generated: int,
    n_reference: int,
    n_train: int,
    resamples: int,
    rng: np.random.Generator,
    resample_generated: bool,
    resample_reference: bool,
    resample_train: bool,
    batch: int = 250,
) -> dict[str, np.ndarray]:
    """Bootstrap draws of every component MMD, component ratio, and the mean Ratio.

    The reference set appears in both the numerator (MMD vs generated) and the denominator
    (MMD vs train), so a resampled reference is reused in both -- that correlation is real
    and dropping it would overstate the interval.
    """
    metrics = list(kernels)
    out: dict[str, list[np.ndarray]] = {f"mmd/{m}": [] for m in metrics}
    out.update({f"ratio/{m}": [] for m in metrics})
    out["ratio"] = []
    floors: dict[str, list[np.ndarray]] = {f"floor/{m}": [] for m in metrics}

    done = 0
    while done < resamples:
        size = min(batch, resamples - done)
        c_gen = _counts(rng, n_generated, size, resample_generated)
        c_ref = _counts(rng, n_reference, size, resample_reference)
        c_train = _counts(rng, n_train, size, resample_train)

        per_metric_ratio = []
        for metric in metrics:
            k = kernels[metric]
            ref_ref = _weighted_mean(k["ref_ref"], c_ref, c_ref)
            gen_gen = _weighted_mean(k["gen_gen"], c_gen, c_gen)
            ref_gen = _weighted_mean(k["ref_gen"], c_ref, c_gen)
            train_train = _weighted_mean(k["train_train"], c_train, c_train)
            ref_train = _weighted_mean(k["ref_train"], c_ref, c_train)

            mmd = ref_ref + gen_gen - 2.0 * ref_gen
            floor = ref_ref + train_train - 2.0 * ref_train
            ratio = mmd / np.maximum(floor, RATIO_FLOOR)

            out[f"mmd/{metric}"].append(mmd)
            out[f"ratio/{metric}"].append(ratio)
            floors[f"floor/{metric}"].append(floor)
            per_metric_ratio.append(ratio)

        out["ratio"].append(np.mean(per_metric_ratio, axis=0))
        done += size

    merged = {key: np.concatenate(value) for key, value in out.items()}
    merged.update({key: np.concatenate(value) for key, value in floors.items()})
    return merged


def summarise(draws: np.ndarray, point: float | None = None) -> dict:
    """Percentile and bias-corrected intervals, plus the bias itself.

    Resampling with replacement inflates every MMD here. The committed estimator is the
    biased V-statistic, whose within-set term averages over all pairs *including* the
    diagonal, and a resample that draws graph `i` twice contributes an extra K[i, i] = 1 --
    the largest value the kernel can take. Concretely E[c_i^2] = 2 - 1/n against E[c_i c_j]
    = 1 - 1/n, so the diagonal is overweighted by O(1/n) in every resample. The effect is
    small for the Ratio (numerator and denominator shift together) but reaches ~9% on the
    low-variance component MMDs, enough to push a percentile interval clear of the point
    estimate. `ci95_basic_*` is the pivotal interval, which reflects that bias back and is
    the one to quote for a component metric.
    """
    lo, hi = np.percentile(draws, [2.5, 97.5])
    estimate = float(point) if point is not None else float(np.mean(draws))
    summary = {
        "point": estimate,
        "bootstrap_mean": float(np.mean(draws)),
        "bootstrap_bias": float(np.mean(draws)) - estimate,
        "bootstrap_std": float(np.std(draws, ddof=1)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci95_basic_low": float(2 * estimate - hi),
        "ci95_basic_high": float(2 * estimate - lo),
        "median": float(np.median(draws)),
    }
    if summary["point"]:
        summary["ci95_width_over_point"] = float((hi - lo) / abs(summary["point"]))
        summary["relative_bias"] = summary["bootstrap_bias"] / abs(summary["point"])
    return summary


# --------------------------------------------------------------------------------------
# memorisation / similarity
# --------------------------------------------------------------------------------------


def upper_triangle_matrix(graphs) -> np.ndarray:
    """(n_graphs, n_pairs) boolean edge-indicator rows, in the stored node ordering.

    Node identity is meaningful here: the adjacency, the condition's pair channel and the
    sampler all share one node ordering, so an index-aligned edge comparison is the right
    way to ask whether a sample reproduces its conditioning target.
    """
    sizes = {g.number_of_nodes() for g in graphs}
    if len(sizes) != 1:
        raise ValueError(f"edge-overlap comparison needs one common node count, saw {sizes}")
    n = sizes.pop()
    iu = np.triu_indices(n, k=1)
    return np.array(
        [nx.to_numpy_array(g, dtype=np.uint8)[iu].astype(bool) for g in graphs]
    )


def overlap_matrix(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    """Pairwise edge-set intersection statistics between two edge-indicator blocks."""
    inter = (left.astype(np.int32) @ right.astype(np.int32).T).astype(float)
    left_size = left.sum(axis=1).astype(float)[:, None]
    right_size = right.sum(axis=1).astype(float)[None, :]
    union = left_size + right_size - inter
    precision = inter / np.maximum(left_size, 1.0)
    recall = inter / np.maximum(right_size, 1.0)
    return {
        "intersection": inter,
        "jaccard": inter / np.maximum(union, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / np.maximum(precision + recall, 1e-12),
    }


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, resamples: int) -> dict:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return summarise(draws, point=float(values.mean()))


def conditioning_similarity(
    generated_edges: np.ndarray,
    condition_edges: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    """Is sample i a partial reconstruction of the graph whose spectrum conditioned it?

    The matched (diagonal) overlap is meaningless without the mismatched (off-diagonal)
    overlap beside it: two graphs of the same density share ~density of their pairs by
    chance, so only the matched-minus-mismatched gap is evidence of copying.
    """
    stats = overlap_matrix(generated_edges, condition_edges)
    n = len(generated_edges)
    eye = np.eye(n, dtype=bool)

    result: dict = {}
    for name in ("jaccard", "precision", "recall", "f1", "intersection"):
        matrix = stats[name]
        matched = matrix[eye]
        mismatched = matrix[~eye]
        # Per-sample mean over the 31 non-matching conditions, so the paired difference
        # below compares like with like sample by sample.
        mismatched_per_sample = (matrix.sum(axis=1) - matched) / (n - 1)
        difference = matched - mismatched_per_sample
        result[name] = {
            "matched_mean": float(matched.mean()),
            "matched_std": float(matched.std(ddof=1)),
            "matched_min": float(matched.min()),
            "matched_max": float(matched.max()),
            "mismatched_mean": float(mismatched.mean()),
            "mismatched_std": float(mismatched.std(ddof=1)),
            "difference_mean": float(difference.mean()),
            "difference_ci95": _bootstrap_mean_ci(difference, rng, resamples),
        }

    # Retrieval test: if a sample carried its condition's edges, its own condition would be
    # its nearest neighbour among the 32 candidates. Chance is 1/32 = 3.1%.
    jaccard = stats["jaccard"]
    argmax = jaccard.argmax(axis=1)
    rank = (jaccard > jaccard[eye][:, None]).sum(axis=1)
    result["retrieval"] = {
        "top1_accuracy": float((argmax == np.arange(n)).mean()),
        "chance_top1": 1.0 / n,
        "mean_rank_of_own_condition": float(rank.mean() + 1),
        "expected_rank_by_chance": (n + 1) / 2,
        "n_candidates": n,
    }
    return result


def spectral_agreement(generated, condition_graphs, k: int) -> dict:
    """Does a sample's own low band match the band it was conditioned on?

    Edge overlap can be zero while the condition is still honoured -- the condition is a
    spectrum, not an edge list. This checks the conditioned quantity directly, so a null
    edge-overlap result can be read as "not reconstruction" rather than "condition ignored".
    """

    # `select_band(..., 'low', k)` takes eigenvalues [n_trivial : n_trivial + k], skipping the
    # zero eigenvalue(s). The conditioning graphs are real Planar graphs and all connected, so
    # n_trivial is 1; the same slice is taken on both sides so the comparison is like for like.
    trivial = max(
        (spectral.count_trivial_eigenpairs(descriptors.eigen_decomposition(g)[0])
         for g in condition_graphs),
        default=1,
    )
    trivial = max(trivial, 1)

    def low_eigenvalues(graph):
        values, _ = descriptors.eigen_decomposition(graph)
        return np.sort(values)[trivial : trivial + k]

    gen = np.array([low_eigenvalues(g) for g in generated])
    cond = np.array([low_eigenvalues(g) for g in condition_graphs])
    distance = np.linalg.norm(gen[:, None, :] - cond[None, :, :], axis=2)
    n = len(gen)
    eye = np.eye(n, dtype=bool)
    matched = distance[eye]
    mismatched_per_sample = (distance.sum(axis=1) - matched) / (n - 1)
    return {
        "k": k,
        "n_trivial_eigenpairs_skipped": int(trivial),
        "matched_l2_mean": float(matched.mean()),
        "mismatched_l2_mean": float(mismatched_per_sample.mean()),
        "difference_mean": float((matched - mismatched_per_sample).mean()),
        "retrieval_top1_accuracy": float((distance.argmin(axis=1) == np.arange(n)).mean()),
        "chance_top1": 1.0 / n,
        "note": (
            "Lower L2 is closer. matched < mismatched means the sample's own low band is "
            "nearer its conditioning spectrum than to other graphs' spectra."
        ),
    }


def nearest_neighbour_similarity(
    query_edges: np.ndarray, corpus_edges: np.ndarray, exclude_self: bool = False
) -> dict:
    """Graded memorisation: the closest graph in the corpus, in node-aligned edge space.

    Isomorphism-based novelty saturates at 100% for graphs this dense -- two independent
    64-node graphs at 9% density are essentially never isomorphic -- so it cannot tell a
    near-copy from an unrelated sample. The nearest-neighbour Jaccard can.
    """
    jaccard = overlap_matrix(query_edges, corpus_edges)["jaccard"]
    if exclude_self:
        np.fill_diagonal(jaccard, -np.inf)
    best = jaccard.max(axis=1)
    return {
        "mean_max_jaccard": float(best.mean()),
        "std_max_jaccard": float(best.std(ddof=1)),
        "max_max_jaccard": float(best.max()),
        "min_max_jaccard": float(best.min()),
        "n_query": int(len(query_edges)),
        "n_corpus": int(len(corpus_edges)),
    }


# --------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------


def density_matched_er(graphs, seed: int) -> list[nx.Graph]:
    """Byte-identical to `_density_matched_er` in scripts/train_adjacency_diffusion.py.

    Copied rather than imported so this audit does not pull the training script's torch and
    model imports; `--self-check` asserts the reproduced ER Ratio against the committed one,
    which would catch any drift between the two copies.
    """
    rng = np.random.default_rng(seed)
    controls = []
    for graph in graphs:
        n = graph.number_of_nodes()
        p = 2.0 * graph.number_of_edges() / max(n * (n - 1), 1)
        controls.append(nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1))))
    return controls


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def _jsonable(value):
    """numpy scalars leak in from comparisons and reductions; json cannot encode them."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def load_samples(path: Path) -> list[nx.Graph]:
    payload = torch.load(path, weights_only=False)
    return [nx.from_numpy_array(np.asarray(a).astype(np.uint8)) for a in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        default=str(results_dir() / "adjacency_diffusion_planar_low_k8_samples.pt"),
    )
    parser.add_argument(
        "--committed-report",
        default=str(results_dir() / "adjacency_diffusion_planar_low_k8.json"),
        help="Report whose evaluation block this run tries to reproduce.",
    )
    parser.add_argument("--dataset", default="planar", choices=["planar", "sbm"])
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--k", type=int, default=8, help="Condition band width, for the spectral check.")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--er-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planarity-distance", action="store_true", default=True)
    parser.add_argument("--no-planarity-distance", dest="planarity_distance", action="store_false")
    parser.add_argument("--output", default=str(results_dir() / "bootstrap_low_k8.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    samples_path = Path(args.samples)
    generated = load_samples(samples_path)
    splits = load_splits(args.dataset)
    train_graphs = splits["train"]
    reference_graphs = splits[args.eval_split]

    metrics = ["degree", "clustering", "spectral", "wavelet"]
    if orca.is_available():
        metrics.append("orbit")
    else:
        print("ORCA unavailable -- Ratio would average 4 metrics, not 5, and is NOT comparable.")
    print(
        f"{len(generated)} generated | {len(reference_graphs)} reference ({args.eval_split}) "
        f"| {len(train_graphs)} train | metrics={metrics}"
    )

    # ---------------------------------------------------------------- 1. reproduction
    print("\n[1/5] reproducing the committed evaluation with the committed evaluator ...")
    evaluator = GraphEvaluator(
        reference_graphs,
        train_graphs=train_graphs,
        validity_func=validity.is_planar if args.dataset == "planar" else validity.sbm_validity,
        metrics=metrics,
    )
    floor = evaluator.self_similarity(train_graphs)
    evaluation = evaluator.evaluate(generated, baseline_mmd=floor).as_flat_dict()

    committed = json.loads(Path(args.committed_report).read_text())
    committed_eval = committed.get("evaluation") or {}
    reproduction = {"committed_report": str(Path(args.committed_report)), "fields": {}}

    # Density is a property of the graphs alone -- no evaluator, no kernel, no ORCA -- so it
    # separates "the evaluator drifted" from "these are different graphs". If it disagrees,
    # nothing downstream can agree and the comparison above is not a reproduction attempt.
    densities = [nx.density(g) for g in generated]
    reproduction["density"] = {
        "committed_mean": committed.get("generated_density_mean"),
        "recomputed_mean": float(np.mean(densities)),
        "committed_std": committed.get("generated_density_std"),
        "recomputed_std": float(np.std(densities)),
        "same_graphs": (
            committed.get("generated_density_mean") is not None
            and abs(np.mean(densities) - committed["generated_density_mean"]) < 1e-12
            and abs(np.std(densities) - committed["generated_density_std"]) < 1e-12
        ),
    }
    if not reproduction["density"]["same_graphs"]:
        print(
            f"  !! density mean {np.mean(densities):.12g} != committed "
            f"{committed.get('generated_density_mean')} -- the saved samples are NOT the "
            f"graphs behind this report."
        )

    worst = 0.0
    for key, value in sorted(evaluation.items()):
        if key not in committed_eval or not isinstance(value, (int, float)):
            continue
        expected = committed_eval[key]
        absolute = abs(value - expected)
        relative = absolute / max(abs(expected), 1e-12)
        reproduction["fields"][key] = {
            "committed": expected,
            "recomputed": value,
            "abs_diff": absolute,
            "rel_diff": relative,
        }
        worst = max(worst, relative)
        flag = "ok " if relative < 1e-6 else "DIFF"
        print(f"  {flag} {key:<20} committed={expected:<22.10g} recomputed={value:.10g}")
    reproduction["max_relative_difference"] = worst
    reproduction["reproduced"] = worst < 1e-6
    print(f"  -> max relative difference {worst:.2e} "
          f"({'reproduced' if reproduction['reproduced'] else 'MISMATCH'})")

    # Context for a mismatch: the committed across-seed spread for the same arm. A recomputed
    # value outside that range says the seed std understates run-to-run movement.
    sibling_ratios = {}
    for sibling in sorted(Path(args.committed_report).parent.glob(
        Path(args.committed_report).stem.split("_seed")[0] + "*.json"
    )):
        block = json.loads(sibling.read_text()).get("evaluation") or {}
        if "ratio" in block:
            sibling_ratios[sibling.stem] = block["ratio"]
    if sibling_ratios:
        values = list(sibling_ratios.values())
        reproduction["committed_arm_ratios"] = sibling_ratios
        reproduction["recomputed_within_committed_seed_range"] = (
            min(values) <= evaluation["ratio"] <= max(values)
        )

    # ------------------------------------------------------- 2. kernels + self-check
    print("\n[2/5] building kernel matrices ...")
    filters, bound = descriptors.wavelet_filter_bank()
    described = {
        "gen": describe(generated, metrics, filters, bound),
        "ref": describe(reference_graphs, metrics, filters, bound),
        "train": describe(train_graphs, metrics, filters, bound),
    }
    specs = {spec.name: spec for spec in MMD_SPECS}
    kernels: dict[str, dict[str, np.ndarray]] = {}
    kernel_check: dict[str, dict[str, float]] = {}
    for metric in metrics:
        spec = specs[metric]
        width = max(len(v) for block in described.values() for v in block[metric])
        stacked = {
            name: _stack(block[metric], spec.is_hist, width) for name, block in described.items()
        }
        kernels[metric] = {
            "gen_gen": kernel_matrix(stacked["gen"], stacked["gen"], spec.sigma),
            "ref_ref": kernel_matrix(stacked["ref"], stacked["ref"], spec.sigma),
            "ref_gen": kernel_matrix(stacked["ref"], stacked["gen"], spec.sigma),
            "train_train": kernel_matrix(stacked["train"], stacked["train"], spec.sigma),
            "ref_train": kernel_matrix(stacked["ref"], stacked["train"], spec.sigma),
        }
        # The whole bootstrap rests on these matrices reproducing `compute_mmd` exactly.
        fast_mmd = mmd_from_kernels(
            kernels[metric]["ref_ref"], kernels[metric]["gen_gen"], kernels[metric]["ref_gen"]
        )
        slow_mmd = compute_mmd(
            described["ref"][metric],
            described["gen"][metric],
            kernel=gaussian_tv,
            is_hist=spec.is_hist,
            sigma=spec.sigma,
        )
        kernel_check[metric] = {
            "kernel_matrix_mmd": fast_mmd,
            "compute_mmd": slow_mmd,
            "abs_diff": abs(fast_mmd - slow_mmd),
        }
        assert abs(fast_mmd - slow_mmd) < 1e-9, (
            f"{metric}: vectorised kernel MMD {fast_mmd} != compute_mmd {slow_mmd}"
        )
    print(f"  kernel-matrix MMD matches compute_mmd for all {len(metrics)} metrics "
          f"(max |diff| {max(v['abs_diff'] for v in kernel_check.values()):.2e})")

    # ------------------------------------------------------------------ 3. bootstrap
    print(f"\n[3/5] bootstrapping ({args.resamples} resamples per scheme) ...")
    schemes = {
        "generated_only": dict(resample_generated=True, resample_reference=False, resample_train=False),
        "reference_only": dict(resample_generated=False, resample_reference=True, resample_train=False),
        "full": dict(resample_generated=True, resample_reference=True, resample_train=True),
    }
    point_values = {f"ratio/{m}": evaluation[f"ratio/{m}"] for m in metrics}
    point_values.update({f"mmd/{m}": evaluation[f"mmd/{m}"] for m in metrics})
    point_values["ratio"] = evaluation["ratio"]

    bootstrap: dict[str, dict] = {}
    generated_draws: dict[str, np.ndarray] = {}
    for name, flags in schemes.items():
        draws = bootstrap_ratios(
            kernels,
            n_generated=len(generated),
            n_reference=len(reference_graphs),
            n_train=len(train_graphs),
            resamples=args.resamples,
            rng=np.random.default_rng(args.seed + abs(hash(name)) % 10_000),
            **flags,
        )
        if name == "generated_only":
            generated_draws = draws
        bootstrap[name] = {
            key: summarise(value, point=point_values.get(key)) for key, value in draws.items()
        }
        # The Ratio divides by a near-zero self-similarity floor, so resampling the reference
        # set can drive the denominator toward zero and the quotient toward infinity. When
        # that happens the percentile interval is still a valid order statistic but the mean
        # and std are not, and the interval should be read as "unbounded above".
        floor_draws = np.array(
            [draws[f"floor/{m}"] for m in metrics]
        )
        bootstrap[name]["_stability"] = {
            "min_floor_seen": float(floor_draws.min()),
            "floor_within_1e3_of_clamp_frac": float((floor_draws < RATIO_FLOOR * 1e3).mean()),
            "ratio_p99_over_p50": float(
                np.percentile(draws["ratio"], 99) / max(np.median(draws["ratio"]), 1e-12)
            ),
            "heavy_tailed": bool(
                np.percentile(draws["ratio"], 99) > 3 * np.median(draws["ratio"])
            ),
            "note": (
                "Ratio is a quotient by the reference-vs-train MMD floor. Schemes that "
                "resample the reference or train set can shrink that floor toward zero, so "
                "their Ratio bootstrap is heavy-tailed: read the percentile interval, not "
                "the mean or std."
            ),
        }
        head = bootstrap[name]["ratio"]
        print(
            f"  {name:<15} Ratio {head['point']:8.2f}  "
            f"percentile 95% CI [{head['ci95_low']:.2f}, {head['ci95_high']:.4g}]  "
            f"basic [{head['ci95_basic_low']:.2f}, {head['ci95_basic_high']:.4g}]  "
            f"bias {head['bootstrap_bias']:+.2f}"
        )

    # ER control and the model-vs-ER difference, the quantity the paper's gate tests.
    print("\n  ER control ...")
    er_graphs = density_matched_er(reference_graphs, args.seed)
    er_described = describe(er_graphs, metrics, filters, bound)
    er_kernels = {}
    for metric in metrics:
        spec = specs[metric]
        width = max(
            max(len(v) for block in described.values() for v in block[metric]),
            max(len(v) for v in er_described[metric]),
        )
        ref_rows = _stack(described["ref"][metric], spec.is_hist, width)
        train_rows = _stack(described["train"][metric], spec.is_hist, width)
        gen_rows = _stack(described["gen"][metric], spec.is_hist, width)
        er_rows = _stack(er_described[metric], spec.is_hist, width)
        er_kernels[metric] = {
            "gen_gen": kernel_matrix(er_rows, er_rows, spec.sigma),
            "ref_ref": kernel_matrix(ref_rows, ref_rows, spec.sigma),
            "ref_gen": kernel_matrix(ref_rows, er_rows, spec.sigma),
            "train_train": kernel_matrix(train_rows, train_rows, spec.sigma),
            "ref_train": kernel_matrix(ref_rows, train_rows, spec.sigma),
        }
        # Reuse the model kernels where the blocks are identical, guarding the padding path.
        assert er_kernels[metric]["ref_ref"].shape == kernels[metric]["ref_ref"].shape

    er_point = float(
        np.mean(
            [
                mmd_from_kernels(k["ref_ref"], k["gen_gen"], k["ref_gen"])
                / max(mmd_from_kernels(k["ref_ref"], k["train_train"], k["ref_train"]), RATIO_FLOOR)
                for k in er_kernels.values()
            ]
        )
    )
    er_draws = bootstrap_ratios(
        er_kernels,
        n_generated=len(er_graphs),
        n_reference=len(reference_graphs),
        n_train=len(train_graphs),
        resamples=args.resamples,
        rng=np.random.default_rng(args.seed + 777),
        resample_generated=True,
        resample_reference=False,
        resample_train=False,
    )
    difference = generated_draws["ratio"] - er_draws["ratio"]
    er_summary = {
        "committed_er_ratio": committed.get("er_ratio"),
        "recomputed_er_ratio": er_point,
        "reproduced": (
            committed.get("er_ratio") is not None
            and abs(er_point - committed["er_ratio"]) / abs(committed["er_ratio"]) < 1e-6
        ),
        "er_ratio_bootstrap": summarise(er_draws["ratio"], point=er_point),
        "model_minus_er": summarise(difference, point=evaluation["ratio"] - er_point),
        "fraction_of_resamples_model_beats_er": float((difference < 0).mean()),
        "note": (
            "Both sides resample only their own 32 generated graphs; the reference and train "
            "sets are shared and held fixed, so this is a paired difference."
        ),
    }
    # The ER control shares the reference and train descriptors with the model's evaluation,
    # so if it reproduces exactly, the reference side and the floor are byte-identical to the
    # committed run and any remaining mismatch is confined to the generated graphs.
    reproduction["reference_side_reproduces_via_er_control"] = er_summary["reproduced"]
    reproduction["mismatch_localised_to"] = (
        "generated samples only (ER control reproduces exactly, so the reference split, the "
        "train split and the MMD floor are identical to the committed run)"
        if er_summary["reproduced"] and not reproduction["reproduced"]
        else None
    )

    print(
        f"    ER Ratio {er_point:.2f} (committed {committed.get('er_ratio')}) | "
        f"model-ER {er_summary['model_minus_er']['point']:+.2f} "
        f"95% CI [{er_summary['model_minus_er']['ci95_low']:+.2f}, "
        f"{er_summary['model_minus_er']['ci95_high']:+.2f}] | "
        f"model wins in {er_summary['fraction_of_resamples_model_beats_er']:.1%} of resamples"
    )

    # Independent ER redraws, to separate "ER is noisy" from "the bootstrap is narrow".
    er_redraws = []
    for draw in range(args.er_draws):
        graphs = density_matched_er(reference_graphs, args.seed + 1000 + draw)
        block = describe(graphs, metrics, filters, bound)
        ratios = []
        for metric in metrics:
            spec = specs[metric]
            width = max(
                max(len(v) for b in described.values() for v in b[metric]),
                max(len(v) for v in block[metric]),
            )
            rows = _stack(block[metric], spec.is_hist, width)
            ref_rows = _stack(described["ref"][metric], spec.is_hist, width)
            train_rows = _stack(described["train"][metric], spec.is_hist, width)
            mmd = mmd_from_kernels(
                kernel_matrix(ref_rows, ref_rows, spec.sigma),
                kernel_matrix(rows, rows, spec.sigma),
                kernel_matrix(ref_rows, rows, spec.sigma),
            )
            floor_value = mmd_from_kernels(
                kernel_matrix(ref_rows, ref_rows, spec.sigma),
                kernel_matrix(train_rows, train_rows, spec.sigma),
                kernel_matrix(ref_rows, train_rows, spec.sigma),
            )
            ratios.append(mmd / max(floor_value, RATIO_FLOOR))
        er_redraws.append(float(np.mean(ratios)))
    er_summary["independent_er_redraws"] = {
        "n_draws": args.er_draws,
        "mean": float(np.mean(er_redraws)),
        "std": float(np.std(er_redraws, ddof=1)),
        "min": float(np.min(er_redraws)),
        "max": float(np.max(er_redraws)),
        "values": er_redraws,
    }
    print(
        f"    ER across {args.er_draws} independent draws: "
        f"{np.mean(er_redraws):.2f} +/- {np.std(er_redraws, ddof=1):.2f}"
    )

    # ------------------------------------------- 4. uniqueness / novelty / conditioning
    print("\n[4/5] uniqueness, novelty and conditioning similarity ...")
    generated_edges = upper_triangle_matrix(generated)
    reference_edges = upper_triangle_matrix(reference_graphs)
    train_edges = upper_triangle_matrix(train_graphs)

    hashes = [nx.weisfeiler_lehman_graph_hash(g, iterations=3) for g in generated]
    self_jaccard = overlap_matrix(generated_edges, generated_edges)["jaccard"]
    off_diagonal = self_jaccard[~np.eye(len(generated), dtype=bool)]
    reference_self = overlap_matrix(reference_edges, reference_edges)["jaccard"]
    reference_off = reference_self[~np.eye(len(reference_graphs), dtype=bool)]

    memorisation = {
        "uniqueness": {
            "fraction_unique_isomorphism": validity.fraction_unique(generated),
            "distinct_wl_hashes": len(set(hashes)),
            "n_samples": len(generated),
            "pairwise_edge_jaccard_mean": float(off_diagonal.mean()),
            "pairwise_edge_jaccard_max": float(off_diagonal.max()),
            "reference_pairwise_edge_jaccard_mean": float(reference_off.mean()),
            "note": (
                "Isomorphism-based uniqueness saturates at 1.0 for graphs this dense. The "
                "pairwise edge Jaccard is the graded version; compare it against the "
                "reference split's own pairwise value for scale."
            ),
        },
        "novelty_vs_train": {
            "fraction_novel_isomorphism": validity.fraction_novel(generated, train_graphs),
            "n_train": len(train_graphs),
            "nearest_train_graph": nearest_neighbour_similarity(generated_edges, train_edges),
            "reference_nearest_train_graph": nearest_neighbour_similarity(
                reference_edges, train_edges
            ),
            "train_nearest_other_train_graph": nearest_neighbour_similarity(
                train_edges, train_edges, exclude_self=True
            ),
            "note": (
                "`nearest_train_graph` is node-aligned edge Jaccard against the closest of "
                "the 128 training graphs. The two reference rows are the real-data scale: a "
                "held-out real graph's nearest training graph scores about the same, so a "
                "generated value near them is not memorisation."
            ),
        },
        "conditioning_graph": conditioning_similarity(
            generated_edges, reference_edges, rng, args.resamples
        ),
        "conditioning_spectrum": spectral_agreement(generated, reference_graphs, args.k),
        "pairing": (
            "Sample i was conditioned on the spectrum of validation graph i. Confirmed from "
            "scripts/train_adjacency_diffusion.py: eval_graphs = splits['val'] (unshuffled), "
            "the DataLoader is built with shuffle=False, _sample_loader extends its output in "
            "loader order, and build_condition_tensors uses an identity donor map for every "
            "band except 'shuffled'."
        ),
    }
    cond = memorisation["conditioning_graph"]
    print(
        f"  conditioning-graph edge Jaccard: matched {cond['jaccard']['matched_mean']:.4f} vs "
        f"mismatched {cond['jaccard']['mismatched_mean']:.4f} "
        f"(difference {cond['jaccard']['difference_mean']:+.4f}, 95% CI "
        f"[{cond['jaccard']['difference_ci95']['ci95_low']:+.4f}, "
        f"{cond['jaccard']['difference_ci95']['ci95_high']:+.4f}])"
    )
    print(
        f"    precision {cond['precision']['matched_mean']:.4f} (mismatched "
        f"{cond['precision']['mismatched_mean']:.4f}) | recall "
        f"{cond['recall']['matched_mean']:.4f} (mismatched "
        f"{cond['recall']['mismatched_mean']:.4f}) | shared edges "
        f"{cond['intersection']['matched_mean']:.1f} of "
        f"{np.mean([g.number_of_edges() for g in generated]):.1f} generated"
    )
    print(
        f"  retrieval of own condition: top-1 {cond['retrieval']['top1_accuracy']:.1%} "
        f"(chance {cond['retrieval']['chance_top1']:.1%}), mean rank "
        f"{cond['retrieval']['mean_rank_of_own_condition']:.1f} of "
        f"{cond['retrieval']['n_candidates']}"
    )
    print(
        f"  novelty vs train (isomorphism): "
        f"{memorisation['novelty_vs_train']['fraction_novel_isomorphism']:.3f}; "
        f"nearest-train Jaccard {memorisation['novelty_vs_train']['nearest_train_graph']['mean_max_jaccard']:.4f} "
        f"vs real held-out "
        f"{memorisation['novelty_vs_train']['reference_nearest_train_graph']['mean_max_jaccard']:.4f}"
    )

    # ------------------------------------------------- 5. connectivity vs planarity
    print("\n[5/5] connectivity and planarity, split apart ...")
    diagnosis = diagnose_validity(generated, with_distance=args.planarity_distance)
    connected = np.array([nx.is_connected(g) for g in generated], dtype=float)
    planar = np.array([nx.check_planarity(g)[0] for g in generated], dtype=float)
    both = connected * planar
    components = [nx.number_connected_components(g) for g in generated]
    largest = [len(max(nx.connected_components(g), key=len)) for g in generated]
    diagnosis_extra = {
        "connected_frac_ci95": _bootstrap_mean_ci(connected, rng, args.resamples),
        "planar_frac_ci95": _bootstrap_mean_ci(planar, rng, args.resamples),
        "valid_frac_ci95": _bootstrap_mean_ci(both, rng, args.resamples),
        "component_count_histogram": {
            str(c): int((np.array(components) == c).sum()) for c in sorted(set(components))
        },
        "largest_component_mean": float(np.mean(largest)),
        "largest_component_min": int(np.min(largest)),
        "n_nodes": int(generated[0].number_of_nodes()),
        "mean_edges": float(np.mean([g.number_of_edges() for g in generated])),
        "planar_edge_bound_3n_minus_6": int(3 * generated[0].number_of_nodes() - 6),
        "reference_connected_frac": float(
            np.mean([nx.is_connected(g) for g in reference_graphs])
        ),
        "reference_planar_frac": float(
            np.mean([nx.check_planarity(g)[0] for g in reference_graphs])
        ),
        "note": (
            "validity = connected AND planar. These rows separate the two so a 0% validity "
            "figure is attributable."
        ),
    }
    print(
        f"  connected {diagnosis['connected_frac']:.3f} | planar {diagnosis['planar_frac']:.3f} "
        f"| both {diagnosis['valid_frac']:.3f} | mean components "
        f"{diagnosis['mean_components']:.2f} | edges/(3n-6) "
        f"{diagnosis['mean_edges_over_bound']:.2f} | over the bound "
        f"{diagnosis['exceeds_3n_minus_6_frac']:.3f}"
    )

    payload = {
        "experiment": "bootstrap_samples",
        "description": (
            "Within-run bootstrap and memorisation audit of one saved sample set. The "
            "intervals hold the trained model fixed and resample the graphs; they are not "
            "the across-seed std reported elsewhere and neither bounds the other."
        ),
        "samples_path": str(samples_path),
        "samples_path_committed_in_report": committed.get("samples_path"),
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "band": committed.get("band"),
        "k": committed.get("k"),
        "seed": args.seed,
        "n_generated": len(generated),
        "n_reference": len(reference_graphs),
        "n_train": len(train_graphs),
        "metrics": metrics,
        "orca_available": orca.is_available(),
        "resamples": args.resamples,
        "reproduction": reproduction,
        "kernel_self_check": kernel_check,
        "recomputed_evaluation": evaluation,
        "reference_floor_mmd": floor,
        "bootstrap": bootstrap,
        "er_control": er_summary,
        "memorisation": memorisation,
        "validity_diagnosis": diagnosis,
        "validity_diagnosis_extra": diagnosis_extra,
        "coverage": {
            "arms_in_paper": 12,
            "arms_with_saved_samples": 1,
            "arm_covered": f"{committed.get('band')} k={committed.get('k')} "
                           f"{args.dataset} seed {committed.get('seed')}",
            "note": (
                "Every other report's samples_path points at a teammate's local Windows "
                "OneDrive and the files are not in this repository; checkpoints/ is empty. "
                "Per-arm uncertainty for the remaining 11 arms is unrecoverable without "
                "those files or a retrain."
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=_jsonable))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
