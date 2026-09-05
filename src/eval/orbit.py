"""4-node orbit-count MMD via the ORCA binary.

PLAN.md Stage 2: "Build the ORCA binary for orbit counts (needs g++; use WSL2 if Windows
fights you). If it will not build, drop Orbit and say so explicitly in the report."

This module never crashes the eval pipeline if ORCA is unavailable: `orbit_mmd` returns
`None` and callers are expected to omit the Orbit row and note the omission, per the plan
and WORKPLAN.md risk R3.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

from src.eval.mmd import mmd_from_histograms, _normalize_histogram

ORCA_SOURCE_URL = "https://github.com/thocevar/orca"


def find_orca_binary(explicit_path: str | Path | None = None) -> Path | None:
    """Locate a built ORCA binary. Returns None if not found (caller should drop Orbit)."""
    if explicit_path is not None:
        p = Path(explicit_path)
        return p if p.is_file() else None

    candidates = [
        Path("third_party/orca/orca"),
        Path("third_party/orca/orca.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    which = shutil.which("orca")
    return Path(which) if which else None


def _graph_to_orca_input(graph: nx.Graph, path: Path) -> None:
    """ORCA expects: first line 'n_nodes n_edges', then one 'u v' edge per line, 0-indexed."""
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
    with open(path, "w") as f:
        f.write(f"{graph.number_of_nodes()} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")


def compute_orbit_counts(graph: nx.Graph, orca_binary: Path, graphlet_size: int = 4) -> np.ndarray | None:
    """Run ORCA on one graph, return its 4-node orbit-count histogram (summed over nodes)."""
    if graph.number_of_nodes() == 0:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "graph.txt"
        output_path = tmpdir_path / "output.txt"
        _graph_to_orca_input(graph, input_path)
        try:
            subprocess.run(
                [str(orca_binary), "node", str(graphlet_size), str(input_path), str(output_path)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        if not output_path.exists():
            return None
        counts = np.loadtxt(output_path, dtype=np.int64)
        if counts.ndim == 1:
            counts = counts.reshape(1, -1)
        return counts.sum(axis=0).astype(np.float64)


def orbit_histogram(graph: nx.Graph, orca_binary: Path, num_bins: int = 100) -> np.ndarray | None:
    counts = compute_orbit_counts(graph, orca_binary)
    if counts is None:
        return None
    hist, _ = np.histogram(counts, bins=num_bins)
    return _normalize_histogram(hist)


def orbit_mmd(
    graphs_a: list[nx.Graph],
    graphs_b: list[nx.Graph],
    orca_binary: str | Path | None = None,
    sigma: float = 30.0,
) -> float | None:
    """Orbit-count MMD, or None if ORCA is unavailable / fails on any graph.

    Callers must treat None as "drop this row and note it in the report"
    (PLAN.md Stage 2, WORKPLAN.md risk R3) -- never silently substitute 0.
    """
    binary = find_orca_binary(orca_binary)
    if binary is None:
        return None

    hists_a, hists_b = [], []
    for g in graphs_a:
        h = orbit_histogram(g, binary)
        if h is None:
            return None
        hists_a.append(h)
    for g in graphs_b:
        h = orbit_histogram(g, binary)
        if h is None:
            return None
        hists_b.append(h)

    return mmd_from_histograms(hists_a, hists_b, sigma=sigma)
