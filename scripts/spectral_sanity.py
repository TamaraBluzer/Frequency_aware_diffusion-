"""Stage 3 gate: verify the spectral utilities and SignNet before anything depends on them.

Every later stage reads the spectrum through these functions, so a sign error or an off-by-one
in the band indexing would silently corrupt the headline result rather than crash. The checks
here have known answers independent of any model:

    T-eig   L u = lambda u to float tolerance, and lambda in [0, 2]
    T-comp  multiplicity of lambda = 0 equals the number of connected components
    T-triv  the dropped trivial eigenvector is proportional to D^1/2 1
    T-band  all arms are dimension-matched, and low/high/random select disjoint-as-expected
            index sets from the non-trivial range
    T-sign  SignNet output is unchanged by per-column sign flips (WORKPLAN.md T4)
    T-perm  SignNet is permutation equivariant
    T-cache the on-disk eigendecomposition cache round-trips exactly

Also writes the u_2 node-colouring figure for Planar and SBM, which goes in the report.

Usage:
    python scripts/spectral_sanity.py [--seed 0] [--no-figure]
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

from fald.data import load_splits
from fald.data.spectral import (
    BANDS,
    TRIVIAL_EIGVAL_TOL,
    cached_eigendecompositions,
    cluster_condition,
    count_trivial_eigenpairs,
    eigendecomposition,
    normalized_laplacian,
    select_band,
)
from fald.models import SignNet, random_sign_flip

EIG_TOL = 1e-8
SIGN_TOL = 1e-5


def check_eigendecomposition(graphs, failures: list) -> dict:
    worst_residual = 0.0
    worst_range = 0.0
    comp_mismatches = 0
    trivial_worst = 0.0
    connected_worst = 0.0
    disconnected = 0

    for graph in graphs:
        laplacian = normalized_laplacian(graph)
        eigvals, eigvecs = eigendecomposition(graph)

        residual = np.abs(laplacian @ eigvecs - eigvecs * eigvals[None, :]).max()
        worst_residual = max(worst_residual, float(residual))

        worst_range = max(worst_range, float(max(-eigvals.min(), eigvals.max() - 2.0)))

        n_components = nx.number_connected_components(graph)
        n_zero = int(np.sum(eigvals < 1e-8))
        if n_zero != n_components:
            comp_mismatches += 1

        # D^1/2 1 always lies in the null space, which is why we always drop that direction.
        # It equals u_1 only when the graph is connected: with c components the null space is
        # c-dimensional and u_1 is an arbitrary basis vector of it.
        degrees = np.array([d for _, d in graph.degree()], dtype=float)
        expected = np.sqrt(degrees)
        expected = expected / np.linalg.norm(expected)
        trivial_worst = max(trivial_worst, float(np.linalg.norm(laplacian @ expected)))

        if n_components == 1:
            observed = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
            alignment = abs(float(np.dot(expected, observed)))
            connected_worst = max(connected_worst, 1.0 - alignment)
        else:
            disconnected += 1

    if worst_residual > 1e-8:
        failures.append(f"T-eig: max |Lu - lambda u| = {worst_residual:.2e} > 1e-8")
    if worst_range > 1e-8:
        failures.append(f"T-eig: eigenvalues outside [0, 2] by {worst_range:.2e}")
    if comp_mismatches:
        failures.append(
            f"T-comp: zero-eigenvalue multiplicity != component count for "
            f"{comp_mismatches}/{len(graphs)} graphs"
        )
    if trivial_worst > 1e-6:
        failures.append(f"T-triv: ||L_norm (D^1/2 1)|| = {trivial_worst:.2e}, expected 0")
    if connected_worst > 1e-6:
        failures.append(
            f"T-triv: on connected graphs u_1 deviates from D^1/2 1 by {connected_worst:.2e}"
        )

    return {
        "max_eig_residual": worst_residual,
        "max_range_violation": worst_range,
        "component_mismatches": comp_mismatches,
        "max_null_space_residual": trivial_worst,
        "max_u1_misalignment_connected": connected_worst,
        "disconnected_graphs": disconnected,
        "n_graphs": len(graphs),
    }


def check_bands(graphs, k: int, rng, failures: list) -> dict:
    eigvals, eigvecs = eigendecomposition(graphs[0])
    n = graphs[0].number_of_nodes()
    shapes = {}

    for band in BANDS:
        if band == "cluster":
            continue
        condition = select_band(eigvals, eigvecs, band, k, rng=rng)
        shapes[band] = [list(condition.eigvals.shape), list(condition.eigvecs.shape)]
        expected_k = 0 if band == "none" else k
        if condition.eigvals.shape != (expected_k,) or condition.eigvecs.shape != (n, expected_k):
            failures.append(f"T-band: {band} produced {shapes[band]}, expected [({expected_k},), ({n}, {expected_k})]")
        if band in ("low", "high", "random") and (condition.indices == 0).any():
            failures.append(f"T-band: {band} selected the trivial eigenpair (index 0)")

    low = select_band(eigvals, eigvecs, "low", k, rng=rng)
    high = select_band(eigvals, eigvecs, "high", k, rng=rng)
    if not np.all(low.indices == np.arange(1, k + 1)):
        failures.append(f"T-band: low selected {low.indices}, expected 1..{k}")
    if not np.all(high.indices == np.arange(n - k, n)):
        failures.append(f"T-band: high selected {high.indices}, expected {n-k}..{n-1}")
    if low.eigvals.max() > high.eigvals.min():
        failures.append("T-band: low band eigenvalues are not below high band eigenvalues")

    # The gaussian control must match shape exactly while carrying no structure.
    gauss = select_band(eigvals, eigvecs, "gaussian", k, rng=rng)
    if gauss.eigvecs.shape != low.eigvecs.shape:
        failures.append("T-band: gaussian arm is not dimension-matched to low")

    clusters = cluster_condition(graphs[0], n_clusters=4)
    if clusters.shape != (n, 4) or not np.allclose(clusters.sum(axis=1), 1.0):
        failures.append("T-band: cluster arm is not a valid one-hot partition")

    return {
        "shapes": shapes,
        "low_indices": low.indices.tolist(),
        "high_indices": high.indices.tolist(),
        "low_eigval_max": float(low.eigvals.max()),
        "high_eigval_min": float(high.eigvals.min()),
    }


def check_disconnected_band(graphs, k: int, failures: list) -> dict:
    """T-disc: on a disconnected graph the band must contain no constant directions.

    SPECTRE drops exactly one eigenpair (`eigvals[1:]`), which leaves c-1 component indicators
    inside the band. We offset by the component count instead, following DiGress. Checked on a
    real disconnected graph, not a synthetic one, so the tolerance is exercised on real spectra.
    """
    disconnected = [g for g in graphs if nx.number_connected_components(g) > 1]
    if not disconnected:
        return {"skipped": "no disconnected graph in this sample"}

    graph = disconnected[0]
    n_components = nx.number_connected_components(graph)
    eigvals, eigvecs = eigendecomposition(graph)

    counted = count_trivial_eigenpairs(eigvals)
    if counted != n_components:
        failures.append(
            f"T-disc: counted {counted} trivial eigenpairs but the graph has {n_components} "
            f"components"
        )

    condition = select_band(eigvals, eigvecs, "low", k)
    if condition.indices[0] < n_components:
        failures.append(
            f"T-disc: low band starts at index {condition.indices[0]}, which is inside the "
            f"{n_components}-dimensional null space"
        )
    if condition.eigvals.min() < TRIVIAL_EIGVAL_TOL:
        failures.append(
            f"T-disc: low band contains a zero eigenvalue ({condition.eigvals.min():.2e}), "
            f"so part of the conditioning budget carries no frequency information"
        )
    if condition.n_components != n_components:
        failures.append("T-disc: SpectralCondition.n_components disagrees with networkx")

    # What SPECTRE's hardcoded slice would have produced, for the report.
    spectre_style = select_band(eigvals, eigvecs, "low", k, n_trivial=1)

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_components": n_components,
        "ours_indices": condition.indices.tolist(),
        "ours_min_eigval": float(condition.eigvals.min()),
        "spectre_style_indices": spectre_style.indices.tolist(),
        "spectre_style_min_eigval": float(spectre_style.eigvals.min()),
        "spectre_style_wasted_slots": int(
            np.sum(spectre_style.eigvals < TRIVIAL_EIGVAL_TOL)
        ),
    }


def check_signnet(graphs, k: int, failures: list) -> dict:
    torch.manual_seed(0)
    n = graphs[0].number_of_nodes()
    eigvals_np, eigvecs_np = eigendecomposition(graphs[0])
    condition = select_band(eigvals_np, eigvecs_np, "low", k)

    eigvecs = torch.tensor(condition.eigvecs, dtype=torch.float32).unsqueeze(0)
    eigvals = torch.tensor(condition.eigvals, dtype=torch.float32).unsqueeze(0)

    model = SignNet(k=k, hidden=32, out_dim=48).eval()
    with torch.no_grad():
        base = model(eigvecs, eigvals)

        # Exhaustive on a few columns, then random flips for the rest.
        worst_sign = 0.0
        for column in range(k):
            flipped = eigvecs.clone()
            flipped[:, :, column] *= -1
            delta = (model(flipped, eigvals) - base).abs().max().item()
            worst_sign = max(worst_sign, delta)

        all_flipped = -eigvecs
        worst_sign = max(worst_sign, (model(all_flipped, eigvals) - base).abs().max().item())

        generator = torch.Generator().manual_seed(1)
        for _ in range(8):
            randomly = random_sign_flip(eigvecs, generator=generator)
            worst_sign = max(worst_sign, (model(randomly, eigvals) - base).abs().max().item())

        # Permutation equivariance: permuting nodes must permute the output identically.
        perm = torch.randperm(n)
        permuted = model(eigvecs[:, perm, :], eigvals)
        worst_perm = (permuted - base[:, perm, :]).abs().max().item()

    if worst_sign > SIGN_TOL:
        failures.append(f"T-sign: SignNet changed by {worst_sign:.2e} under sign flips (> {SIGN_TOL:.0e})")
    if worst_perm > SIGN_TOL:
        failures.append(f"T-perm: SignNet is not permutation equivariant ({worst_perm:.2e})")

    return {"max_sign_deviation": worst_sign, "max_perm_deviation": worst_perm}


def check_cache(graphs, failures: list) -> dict:
    subset = graphs[:8]
    vals_a, vecs_a = cached_eigendecompositions(subset, tag="sanity")
    vals_b, vecs_b = cached_eigendecompositions(subset, tag="sanity")
    worst = max(
        max(float(np.abs(a - b).max()) for a, b in zip(vals_a, vals_b)),
        max(float(np.abs(a - b).max()) for a, b in zip(vecs_a, vecs_b)),
    )
    direct = [eigendecomposition(g)[0] for g in subset]
    worst_direct = max(float(np.abs(a - b).max()) for a, b in zip(vals_a, direct))
    if worst > 0.0 or worst_direct > EIG_TOL:
        failures.append(f"T-cache: cache mismatch (self {worst:.2e}, vs direct {worst_direct:.2e})")
    return {"cache_self_delta": worst, "cache_vs_direct_delta": worst_direct}


def write_u2_figure(planar_graphs, sbm_graphs, out_path: Path) -> None:
    """Colour nodes by the second eigenvector, the lowest non-trivial frequency.

    This is the figure that makes the whole hypothesis legible: on SBM, `u_2` splits the graph
    into its communities, which is the global structure we claim low-frequency conditioning
    supplies. On planar graphs it varies smoothly across the layout instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (name, graphs) in enumerate([("Planar", planar_graphs), ("SBM", sbm_graphs)]):
        for col in range(3):
            ax = axes[row, col]
            graph = graphs[col]
            _, eigvecs = eigendecomposition(graph)
            u2 = eigvecs[:, 1]
            layout = nx.spring_layout(graph, seed=0)
            limit = float(np.abs(u2).max())
            nodes = nx.draw_networkx_nodes(
                graph, layout, node_color=u2, cmap="coolwarm", node_size=45,
                vmin=-limit, vmax=limit, ax=ax,
            )
            nx.draw_networkx_edges(graph, layout, alpha=0.25, width=0.6, ax=ax)
            ax.set_title(f"{name} #{col} (n={graph.number_of_nodes()})", fontsize=10)
            ax.axis("off")
            plt.colorbar(nodes, ax=ax, fraction=0.045)

    fig.suptitle(
        r"Second eigenvector $u_2$ of $L_{norm}$ — the lowest non-trivial frequency", fontsize=13
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    failures: list = []
    report: dict = {"seed": args.seed, "k": args.k}

    planar = load_splits("planar")["train"]
    sbm = load_splits("sbm")["train"]
    print(f"planar train={len(planar)} (n={planar[0].number_of_nodes()})")
    print(f"sbm    train={len(sbm)} (n range {min(g.number_of_nodes() for g in sbm)}"
          f"-{max(g.number_of_nodes() for g in sbm)})")

    print("\n[T-eig/T-comp/T-triv] planar...")
    report["planar_eig"] = check_eigendecomposition(planar[:32], failures)
    print(f"  {report['planar_eig']}")

    print("[T-eig/T-comp/T-triv] sbm...")
    report["sbm_eig"] = check_eigendecomposition(sbm[:32], failures)
    print(f"  {report['sbm_eig']}")

    print("\n[T-band] band selection...")
    report["bands"] = check_bands(planar, args.k, rng, failures)
    print(f"  low indices  {report['bands']['low_indices']}")
    print(f"  high indices {report['bands']['high_indices']}")
    print(f"  low max eigval {report['bands']['low_eigval_max']:.4f} < "
          f"high min eigval {report['bands']['high_eigval_min']:.4f}")

    print("\n[T-disc] disconnected-graph band offset...")
    report["disconnected"] = check_disconnected_band(sbm, args.k, failures)
    for key, value in report["disconnected"].items():
        print(f"  {key}: {value}")

    print("\n[T-sign/T-perm] SignNet...")
    report["signnet"] = check_signnet(planar, args.k, failures)
    print(f"  max deviation under sign flips: {report['signnet']['max_sign_deviation']:.2e}")
    print(f"  max deviation under permutation: {report['signnet']['max_perm_deviation']:.2e}")

    print("\n[T-cache] eigendecomposition cache...")
    report["cache"] = check_cache(planar, failures)
    print(f"  {report['cache']}")

    repo_root = Path(__file__).resolve().parents[1]
    if not args.no_figure:
        figure_path = repo_root / "results" / "figures" / "u2_node_coloring.png"
        print(f"\nwriting {figure_path} ...")
        write_u2_figure(planar, sbm, figure_path)
        report["figure"] = str(figure_path)

    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    report["failures"] = failures
    (results_dir / "spectral_sanity.json").write_text(json.dumps(report, indent=2))

    print()
    if failures:
        print("GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("GATE PASSED: eigendecomposition, band selection, SignNet invariance and the cache "
          "all behave as specified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
