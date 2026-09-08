"""ORCA orbit-counting subprocess wrapper.

ORCA is a C++ program; DiGress ships only `orca.cpp`. `scripts/setup_digress.sh` compiles it.
Windows CreateProcess appends `.exe` when the given path has no extension, so a single
extensionless path works on both platforms.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..paths import third_party_dir

_COUNT_START = "orbit counts:"

_ENV_VAR = "FALD_ORCA_PATH"
_DEFAULT_RELATIVE = Path("digress/src/analysis/orca/orca")


class OrcaUnavailable(RuntimeError):
    pass


def orca_path() -> Path:
    """Resolve the ORCA binary, honouring FALD_ORCA_PATH for non-default layouts."""
    override = os.environ.get(_ENV_VAR)
    candidates = [Path(override)] if override else []
    candidates.append(third_party_dir() / _DEFAULT_RELATIVE)

    for base in candidates:
        for suffix in ("", ".exe"):
            candidate = base.with_name(base.name + suffix) if suffix else base
            if candidate.is_file():
                return base
    raise OrcaUnavailable(
        f"ORCA binary not found (looked for {[str(c) for c in candidates]}). "
        f"Run scripts/setup_digress.sh, or set {_ENV_VAR}."
    )


def is_available() -> bool:
    try:
        orca_path()
    except OrcaUnavailable:
        return False
    return True


def _reindexed_edges(graph):
    idx = {node: i for i, node in enumerate(graph.nodes())}
    for u, v in graph.edges():
        yield idx[u], idx[v]


def node_orbit_counts(graph) -> np.ndarray:
    """Per-node counts of the 15 orbits of all connected graphlets up to size 4."""
    binary = orca_path()
    tmp_dir = Path(tempfile.gettempdir())
    tmp_file = tmp_dir / f"fald_orca_{secrets.token_hex(6)}.txt"

    with open(tmp_file, "w") as handle:
        handle.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for u, v in _reindexed_edges(graph):
            handle.write(f"{u} {v}\n")

    try:
        output = subprocess.check_output(
            [str(binary), "node", "4", str(tmp_file), "std"],
            stderr=subprocess.STDOUT,
        ).decode("utf8")
    finally:
        tmp_file.unlink(missing_ok=True)

    idx = output.find(_COUNT_START)
    if idx < 0:
        raise RuntimeError(f"Unexpected ORCA output: {output[:400]!r}")
    body = output[idx + len(_COUNT_START):].strip()
    return np.array([[int(v) for v in line.split()] for line in body.splitlines() if line.strip()])
