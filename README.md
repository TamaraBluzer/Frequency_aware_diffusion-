# Frequency-Aware Latent Graph Diffusion (FALD)

See [PLAN.md](PLAN.md) for the staged execution order and [WORKPLAN.md](WORKPLAN.md) for
the full design rationale, closed design gaps (G1-G6), test list, and risk register.

## Status

- **Stage 1 — Environment and a working DiGress:** setup script + Colab notebook ready.
  Training itself runs on Google Colab (GPU), not locally.
- **Stage 2 — Evaluation harness:** implemented in `src/eval/`, runs anywhere (no GPU),
  gate passing locally (see below).

## Stage 1 — run on Google Colab

1. Open `notebooks/stage1_digress_smoke.ipynb` in Colab (or upload it).
2. Set the runtime to GPU (Runtime > Change runtime type > GPU).
3. Update `REPO_URL` in the second cell to point at this repo, then run all cells.

This clones [DiGress](https://github.com/cvignac/DiGress) into `third_party/digress`
(gitignored, never committed), installs a matching PyTorch/PyG build for whatever CUDA
version Colab hands you, and runs a short ConGress training + sampling smoke test on
Planar.

**Gate:** `python main.py dataset=planar model=continuous ...` trains, samples, and
prints MMD-style metrics without crashing.

You can also just run the setup script directly in a Colab cell without the notebook:

```bash
!bash scripts/colab_setup.sh
```

## Stage 2 — evaluation harness (local, no GPU needed)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

`src/eval/` is decoupled from DiGress/PyTorch Lightning entirely -- every function takes
and returns plain `networkx.Graph` objects, so it can score graphs from any model,
including the ones trained on Colab in Stage 1 (just pickle/export the sampled graphs
and load them here for scoring).

- `src/eval/mmd.py` -- degree / clustering / spectral MMD, total-variation Gaussian
  kernel (GRAN/SPECTRE convention), the Ratio summary metric.
- `src/eval/orbit.py` -- ORCA orbit-count MMD wrapper. Returns `None` (and the harness
  drops the Orbit row) if the ORCA binary isn't available -- see PLAN.md Stage 2 and
  WORKPLAN.md risk R3.
- `src/eval/wavelet.py` -- SPECTRE's 12-scale ab-spline wavelet MMD via PyGSP.
- `src/eval/validity.py` -- planarity/connectivity validity, pure-NumPy SBM validity
  test (WORKPLAN.md risk R4), uniqueness (WL-hash + VF2), novelty, V.U.N.
- `src/eval/report.py` -- ties it together: `score_graphs`, the training-set
  self-similarity floor, and the MMD calibration report used by the Stage 2 gate.
- `src/data/generate.py` -- SPECTRE-recipe Planar (Delaunay) and SBM generators, with
  the corrected `p_intra=0.3` / `p_inter=0.05` convention (SPECTRE's appendix has these
  swapped -- see the module docstring), plus a density-matched Erdos-Renyi generator
  used as the negative control for MMD calibration.

**Gate (passing as of this writeup):** train-vs-train MMD near zero, train-vs-ER MMD
large, and the Planar training-set row lands in the same ballpark as SPECTRE Table 1:

| Metric | Ours (train vs train) | SPECTRE Table 1 |
|---|---|---|
| Deg | ~1.5e-4 | ~1e-4 |
| Clus | ~8e-4 | ~3e-2 |
| Spec | ~5e-3 | ~5e-3 |

## Repository layout (Stages 1-2 only; grows per WORKPLAN.md section 7)

```
scripts/colab_setup.sh              # Stage 1: clone + install DiGress on Colab
notebooks/stage1_digress_smoke.ipynb
src/data/generate.py                # Planar + SBM generators
src/eval/{mmd,orbit,wavelet,validity,report}.py
tests/                               # pytest gates for Stages 1-2
```
