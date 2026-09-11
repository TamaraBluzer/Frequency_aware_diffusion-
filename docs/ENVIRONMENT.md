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

## Repository location and `FALD_WORK_DIR`

The repo stays at its original OneDrive path:

```
C:\Users\dhalperin\OneDrive - NVIDIA Corporation\Documents\University\ML With Graphs\FinalProject
```

`PLAN.md` Stage 1 called for moving it to `C:\dev\FinalProject`, and it was moved — then moved
back. The reason is Cursor: chat history is scoped to the workspace path, it is **not** stored in
the project folder, and there is no supported way to migrate it. Cursor's docs never document the
storage layout, and Cursor staff have stated the sidebar is rebuilt from a workspace-storage
session index rather than the on-disk transcripts, so a manual file copy would not restore it.
Moving the folder silently discards the project's entire conversation history.

The concerns behind the original decision are addressed differently. Two never materialized: the
260-character path limit is a non-issue because PyG installed from wheels, so there are no deep
nested build directories. What remains is sync churn and mid-write corruption of large files —
handled by keeping the large, regenerable artifacts out of the synced tree:

| Env var | Value here | Governs |
|---|---|---|
| `FALD_WORK_DIR` | `C:\dev\fald-work` | `third_party/`, `data/`, `checkpoints/` |

`fald/paths.py` resolves those three under `FALD_WORK_DIR`, falling back to the repo root when
it is unset, so a fresh clone on a machine without OneDrive works with no configuration. This
moved 192 MB of churn (166 MB DiGress checkout, 26 MB datasets and eigendecomposition caches)
out of OneDrive; what syncs is 11 MB, almost entirely the four reference PDFs.

`results/` deliberately stays in the repo — it is small and worth versioning next to the code.

Not implemented with directory junctions on purpose: OneDrive sometimes follows them and syncs
the target anyway, which is worse than doing nothing.

The env var is set persistently at user scope, so **new** terminals pick it up automatically;
one already open when it was set will not have it.

### Run DiGress with `MPLBACKEND=Agg`

DiGress's `visualization.py` plots during sampling. On Windows matplotlib defaults to the Tk
backend, which is torn down from a non-main thread and floods the log with
`RuntimeError: main thread is not in main loop` and `Tcl_AsyncDelete: async handler deleted by
the wrong thread`. The errors are cosmetic — they happen after metrics are computed and the exit
code is still 0 — but they bury the output. Prefix runs with `MPLBACKEND=Agg`. It is not set
globally on purpose, since that would disable interactive plots in unrelated projects.

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

**SBM validity uses a spectral-clustering stand-in.** `PLAN.md` claimed `spectre_utils.py`
carried a pure-Python SBM validity test. It does not: `is_sbm_graph` calls
`graph_tool.minimize_blockmodel_dl` for Bayesian blockmodel inference plus a merge-split MCMC
refinement, then runs a Wald test on the recovered parameters. `graph-tool` has no Windows
build, so that path is unavailable.

`sbm_validity` now implements the planned replacement: recover blocks with
`sklearn.cluster.SpectralClustering`, scanning block counts in `[2, 5]`, then check the same
structural conditions directly (every block at least 10 nodes, intra-block density >= 0.25,
inter-block density <= 0.10, graph connected).

Calibration on the SBM validation split: **90.6% of real graphs accepted (29/32), 0% of
density-matched Erdos-Renyi graphs accepted (0/32)**. The discriminator has real power, and
its ~9% false-negative rate on real graphs is a conservative bias -- it understates validity
rather than inflating it.

This is deterministic and dependency-light but *not* identical to graph-tool's Bayesian
inference, so SBM numbers produced with it must be reported as a deviation and never compared
directly against SPECTRE's published SBM figures.

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
