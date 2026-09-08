"""Evaluation harness for topology-only graph generation.

Owned by this project rather than imported from DiGress, so that it keeps working when the
gitignored third_party/ checkout is absent and so that the metric definitions are reviewable
in one place. See docs/EVALUATION.md.
"""

from .evaluator import MMD_SPECS, EvalResult, GraphEvaluator
from .mmd import compute_mmd, disc, gaussian, gaussian_tv
from .validity import fraction_novel, fraction_unique, is_planar, vun

__all__ = [
    "GraphEvaluator",
    "EvalResult",
    "MMD_SPECS",
    "compute_mmd",
    "disc",
    "gaussian",
    "gaussian_tv",
    "is_planar",
    "vun",
    "fraction_unique",
    "fraction_novel",
]
