"""Frequency-Aware Latent Graph Diffusion.

Named `fald` rather than `src` on purpose: DiGress installs itself as an editable package
whose top-level name is `src`, so a second `src` here would shadow it entirely and make
DiGress's modules unimportable from this repo.
"""

__all__ = ["data", "eval"]
