# Environment — what is installed, and every deviation from the published recipes

This machine is Windows 11 without administrator rights, which rules out the setup DiGress
documents. Everything below is the *actual* working configuration, plus the reasons each
deviation was unavoidable. Read this before debugging an environment problem.

## Hardware and toolchain

| Component | Value |
|---|---|
| GPU | NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU |
| Compute capability | `sm_120` |
| VRAM | 8.5 GB |
| Driver / CUDA | 595.71 / 13.2 |
| Python | 3.11 (conda env `fald`) |
| PyTorch | 2.11.0+cu128 |
| Conda | Miniconda, per-user install at `C:\Users\dhalperin\miniconda3` |
| Compiler | MinGW g++ 5.3.0 from conda-forge `m2w64-toolchain` |

Repo lives at `C:\dev\FinalProject`, deliberately outside OneDrive: syncing `.git` and
checkpoints corrupts them, and the short path avoids the 260-character Windows path limit.

Conda is configured for **conda-forge only** (`channel_priority: strict`). The Anaconda
default channels require accepting Anaconda's Terms of Service, which is a licensing question
on corporate hardware; conda-forge avoids it entirely.

Exact pinned versions: `requirements-lock.txt`.

## Why we cannot use DiGress's documented stack

DiGress specifies PyTorch 2.0.1 + CUDA 11.8 + `torch_geometric` 2.3.1 + `numpy` 1.23 +
`networkx` 2.8.7. Three of those are impossible or unwise here:

1. **PyTorch 2.0.1 cannot drive this GPU.** Blackwell `sm_120` requires a wheel built with
   CUDA >= 12.8, which means PyTorch >= 2.7. Older cu118/cu124 builds fail at runtime with
   "no kernel image is available for execution on the device." We therefore run PyTorch
   2.11.0+cu128 and patch the resulting incompatibilities (below).
2. **`graph-tool` has no Windows build.** conda-forge ships `linux-64` and `osx-arm64` only,
   and upstream's official Windows answer is "use WSL." See the known gap below.
3. **The `numpy`/`networkx` pins conflict** with a modern PyTorch/PyG stack, so they are
   relaxed to `numpy` 2.4.6 and `networkx` 3.6.1.

## Patches applied to DiGress

Upstream commit `780242b8d3e7d78316bb5cf90c639fb0cd4c6079`, patched by
`patches/digress-windows-modern-torch.patch`. Re-bootstrap with `scripts/setup_digress.sh`.
`third_party/` is gitignored, so that patch file is the only durable record of these changes.

| File | Change | Reason |
|---|---|---|
| `src/main.py` | Drop `import graph_tool as gt` | Imported only to force load order on Linux; unused on Windows |
| `src/main.py` | `strategy="ddp_..."` only when `gpus > 1`, else `"auto"` | DDP needs NCCL, which is absent from Windows PyTorch wheels |
| `src/analysis/spectre_utils.py` | `graph_tool` import wrapped in `try/except`, `is_sbm_graph` raises a clear `NotImplementedError` | Keeps every other metric usable instead of failing at import |
| `src/analysis/dist_helper.py` | `pyemd` imported lazily via `_pyemd()` | No Windows wheel past py3.9; only reachable via `compute_emd=True`, which we never set |
| `src/datasets/*.py`, `src/analysis/spectre_utils.py` | `torch.load(..., weights_only=False)` | PyTorch >= 2.6 flipped this default to `True`, which rejects PyG's `DataEdgeAttr` |
| `src/diffusion_model.py` | Pass `local_rank=self.local_rank` to `sampling_metrics` | Upstream bug: `SpectreSamplingMetrics.forward` requires it and the discrete model passes it, but the continuous ConGress path never did |

That last one is a genuine upstream bug in the continuous path, not a Windows issue. Our plan
starts from ConGress, so we are the ones who hit it.

## Known gaps

**SBM validity is unavailable.** `PLAN.md` claimed `spectre_utils.py` carried a pure-Python SBM
validity test. It does not: `is_sbm_graph` calls `graph_tool.minimize_blockmodel_dl` for
Bayesian blockmodel inference plus a merge-split MCMC refinement, then runs a Wald test on the
recovered parameters. With no `graph-tool` on Windows, SBM validity — and therefore SBM
V.U.N. — cannot be computed yet.

This does not affect Planar, which carries the headline result. It does affect the SBM half of
Stage 7 and the Stage 9 downstream task. The intended replacement is spectral clustering
(recover blocks with `sklearn.cluster.SpectralClustering`, then apply the same Wald test), which
is deterministic and dependency-light but *not* identical to graph-tool's Bayesian inference and
so must be reported as a deviation rather than compared directly to SPECTRE's SBM numbers.

**`pyemd` is absent**, so the legacy GraphRNN-style EMD kernels cannot run. This costs us
nothing: DiGress and SPECTRE both report `gaussian_tv` MMD, and the smoke test confirms the
pipeline logs `emd computation: False`.

## Reproducing from scratch

```bash
conda create -n fald --override-channels -c conda-forge python=3.11 -y
conda install -n fald --override-channels -c conda-forge m2w64-toolchain -y
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.txt
bash scripts/setup_digress.sh
python -m pip install -e third_party/digress
```

## Stage 1 gate evidence

Command (run from `third_party/digress/src`):

```
python main.py dataset=planar model=continuous general.wandb=disabled \
  general.name=smoketest general.gpus=1 train.batch_size=8 train.n_epochs=2 \
  general.number_chain_steps=10 model.diffusion_steps=50 model.n_layers=2
```

Trained a 1.9 M-parameter ConGress model on Planar (train 128 / val 32 / test 40), sampled, and
printed metrics:

```
{'spectre': 0.1509, 'clustering': 0.7094, 'orbit': 1.2635, 'planar_acc': 0.0,
 'sampling/frac_unique': 1.0, 'sampling/frac_unique_non_iso': 1.0,
 'sampling/frac_unic_non_iso_valid': 0.0, 'sampling/frac_non_iso': 1.0}
```

The values are meaningless — two epochs of a deliberately tiny model — but the gate is that the
loop runs and reports MMD without crashing, which it does. The `orbit` entry being finite
confirms the ORCA subprocess works.
