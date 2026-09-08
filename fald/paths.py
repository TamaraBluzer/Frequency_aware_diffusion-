"""Where large, regenerable artifacts live.

The repository itself sits inside a OneDrive-synced folder, which is fine for text but bad for
the things this project generates: a DiGress checkout, downloaded datasets, cached
eigendecompositions, and model checkpoints. Syncing those risks corrupting files mid-write and
produces constant upload churn.

So the *work directory* is decoupled from the *repository*. Set `FALD_WORK_DIR` to somewhere
outside the synced tree and `third_party/`, `data/` and `checkpoints/` follow it. Unset, it
defaults to the repository root, so a fresh clone on a machine without OneDrive just works.

Deliberately not implemented with directory junctions: OneDrive sometimes follows them and
syncs the target anyway, which is worse than doing nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "work_dir", "data_dir", "third_party_dir", "checkpoints_dir", "results_dir"]

_WORK_DIR_ENV = "FALD_WORK_DIR"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def work_dir() -> Path:
    override = os.environ.get(_WORK_DIR_ENV)
    return Path(override).expanduser().resolve() if override else repo_root()


def data_dir() -> Path:
    return work_dir() / "data"


def third_party_dir() -> Path:
    return work_dir() / "third_party"


def checkpoints_dir() -> Path:
    return work_dir() / "checkpoints"


def results_dir() -> Path:
    """Results stay in the repo: they are small, and we want them versioned alongside the code."""
    return repo_root() / "results"
