# Frequency-Aware Latent Graph Diffusion (FALD) — Staged Implementation Plan

**Goal.** Build a topology-only latent graph diffusion model whose denoiser is conditioned on
the first `k` non-trivial eigenpairs of the normalized graph Laplacian, and measure how
generation quality varies with `k` and with which frequency band the conditioning comes from
(low / high / random / none).

Design rationale, the six closed design gaps (G1–G6), the full test list, and the risk register
live in [WORKPLAN.md](WORKPLAN.md). **This document is the execution order:** ten small stages,
each ending in a gate that must pass before moving on.

---

## Stage checklist

- [x] **Stage 1** — Environment and a working DiGress
- [x] **Stage 2** — Evaluation harness we own
- [x] **Stage 3** — Spectral utilities and SignNet
- [ ] **Stage 4** — Graph autoencoder
- [ ] **Stage 5** — Unconditional latent diffusion (`none` baseline)
- [ ] **Stage 6** — Spectral conditioning with oracle spectra **(KILL GATE)**
- [ ] **Stage 7** — The frequency sweep (headline result)
- [ ] **Stage 8** — Learned spectral prior
- [ ] **Stage 9** — Downstream experiment and remaining controls
- [ ] **Stage 10** — Report

---

## Codebase decision

Base: [DiGress](https://github.com/cvignac/DiGress). We take its Planar/SBM datasets, its
evaluation suite (vendored from SPECTRE), its PyTorch Lightning training loop, and its graph
transformer denoiser. We port in [LGD](https://github.com/zhouc20/LatentGraphDiffusion)'s
two-stage structure (`pretrain.py` for the autoencoder, `train_diffusion.py` for latent DDPM)
rather than forking LGD, which is molecule-only and on an old dependency stack.

Three DiGress files carry most of the value:

- `src/datasets/spectre_dataset.py` — Planar and SBM graph generation and loaders
- `src/analysis/spectre_utils.py` — degree/clustering/orbit/spectral MMD, planarity check,
  SBM validity test, V.U.N. Note: the SBM validity test is *not* pure Python — it needs
  `graph_tool.minimize_blockmodel_dl`, which does not exist on Windows. See
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- `src/diffusion/extra_features.py` — Laplacian eigenvalue and eigenvector computation,
  already batched and masked

DiGress also ships **ConGress**, a continuous-diffusion variant, which is architecturally
closer to our latent Gaussian diffusion than the discrete DiGress model. Start from ConGress.

**Related work note.** Cite GGSD (*Generating Graphs via Spectral Diffusion*) as closest prior
work — it is diffusion-based, evaluates on Planar and SBM, and sweeps low-vs-high bands
(16 smallest eigenvalues for Planar, largest 32 for SBM). Our distinction is architectural:
GGSD diffuses the spectrum *itself* and rebuilds the graph from it, so `k` controls how much
information the representation carries. In our setup the latent already carries the full graph
and `k` controls only a side-channel hint. Also note that DiGress itself already feeds
Laplacian eigenfeatures to its denoiser without studying which band or how many.

---

## Stage 1 — Environment and a working DiGress ✅

- [x] ~~Move the repo out of `OneDrive - NVIDIA Corporation` to `C:\dev\FinalProject`.~~
  **Reversed.** The repo stays on the OneDrive path because Cursor scopes chat history to the
  workspace path with no supported migration, so moving the folder discards the project's entire
  conversation history. Instead, the heavy artifacts (`third_party/`, `data/`, `checkpoints/`)
  live at `C:\dev\fald-work` via `FALD_WORK_DIR` — 192 MB of churn out of the synced tree, 11 MB
  left in it. The 260-character path limit turned out to be moot since PyG installs from wheels.
  See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- [x] `.gitignore` for `results/`, `checkpoints/`, `*.pt`, `wandb/`, `data/`, `third_party/`.
- [x] Detect GPU and CUDA version *before* pinning the PyTorch build.
- [x] Create the conda environment, clone DiGress into `third_party/digress`, install.
- [x] Train ConGress on Planar for a few hundred steps to prove the loop runs end to end.

**Gate:** ✅ passed — trains, samples, and prints MMD numbers without crashing.

The environment diverges substantially from DiGress's documented recipe, because this is
Windows without admin rights and a Blackwell (`sm_120`) GPU that requires PyTorch ≥ 2.7 with
CUDA ≥ 12.8. Six patches were needed; all are recorded in
`patches/digress-windows-modern-torch.patch` and explained in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), which is the reference for anything
environment-related. **Known gap:** SBM validity needs `graph-tool` and is unavailable on
Windows, so SBM V.U.N. is blocked until we implement the spectral-clustering replacement.
Planar, which carries the headline result, is unaffected.

## Stage 2 — Evaluation harness we own ✅

Build the evaluator before any of our own models. Every model after this point is immediately
measurable.

- [x] Extract the DiGress evaluation into **`fald/eval/`** (not `src/eval/` — DiGress installs
  itself editable as a top-level package named `src`, which our `src` shadowed completely),
  decoupled from their Lightning module so it scores any list of `networkx` graphs.
- [x] Build the ORCA binary for orbit counts. Built with conda-forge MinGW g++ 5.3.0, no WSL
  and no admin needed, so **Orbit is available** and does not have to be dropped.
- [x] Add SPECTRE's Wavelet MMD (12 ab-spline kernels via PyGSP). Correction: DiGress does not
  omit it — it vendors `spectral_filter_stats` and leaves the call commented out, so this was
  re-enabling existing code.
- [x] Add the training-set self-similarity row and the Ratio summary metric.

**Gate:** ✅ passed — `python scripts/calibrate_eval.py --dataset planar`. Ordering
`real-vs-real ≈ 0 < train-vs-test << ER` holds, the ER control is density-matched and scores
3035× the floor, and our Planar training row (Deg 1.4e-5, Clus 1.7e-2, Spec 4.0e-3) matches
SPECTRE Table 1's order of magnitude. Details and the full table in
[docs/EVALUATION.md](docs/EVALUATION.md).

Also found: DiGress's `process()` appends each graph to `data_list` twice, duplicating its
processed splits (its logs report 80 test graphs for a 40-graph split). MMD is invariant to
duplication so their published numbers stand, but `fald.data` does its own processing to avoid
it.

## Stage 3 — Spectral utilities and SignNet ✅

- [x] `fald/data/spectral.py`: normalized Laplacian, eigendecomposition, and band selection for
  all six arms (`low`, `high`, `random`, `gaussian`, `cluster`, `none`). Written directly
  against `L_norm` rather than adapted from DiGress `extra_features.py`, which is built for
  batched masked tensors inside their denoiser rather than per-graph numpy analysis.
- [x] Cache eigendecompositions to disk, keyed by graph structure so a changed split misses the
  cache instead of silently returning the wrong spectra.
- [x] Implement SignNet for sign-invariant eigenvector encoding, plus random sign-flip
  augmentation.

**Gate:** ✅ passed — `python scripts/spectral_sanity.py`. `‖Lu − λu‖ = 1.2e-15`, `λ ∈ [0,2]`,
zero-eigenvalue multiplicity equals component count on all 64 graphs, SignNet deviation under
sign flips is exactly **0.00e+00** (invariant by construction, not by tolerance), permutation
equivariance 3e-08, cache round-trips bit-exactly. The `u₂` node-colouring figure is at
`results/figures/u2_node_coloring.png`. Details in [docs/SPECTRAL.md](docs/SPECTRAL.md).

**Caveat the gate uncovered, now fixed:** 3 of 128 SBM graphs (2.3%) are disconnected, so their
`L_norm` null space is `c`-dimensional and dropping *one* trivial eigenpair is not enough — the
`low` band then encodes component membership rather than community structure (visible in the
figure). Since extra zero eigenvalues sit at the bottom of the spectrum this biased the `low`
arm only, i.e. exactly the low-versus-high comparison that is the headline claim. `select_band`
now offsets the band by the component count, following DiGress's `get_eigenvalues_features`.
**SPECTRE does not do this** — its `eigvals[1:]` is hardcoded — so this is a small but genuine
methodological improvement over the closest prior work, and worth a sentence in the report.

## Stage 4 — Graph autoencoder

- Node latents `Z ∈ R^(n×d_v)` plus pair latents `W ∈ R^(n×n×d_e)`, LGD-style.
- Reuse DiGress's graph transformer block as the encoder backbone; it already handles node,
  edge, and global features with masking for variable `n`.
- Decoder: small MLP per pair with BCE loss, symmetrized, zero diagonal.
- LayerNorm on encoder outputs, no KL. LGD reports strong KL hurts downstream diffusion.

**Gate:** at least 99% edge accuracy on held-out Planar and SBM. Record this as the ceiling and
plot it on every later results figure. If this fails, nothing downstream is interpretable.

## Stage 5 — Unconditional latent diffusion (the `none` baseline)

- Port LGD's latent DDPM into the DiGress Lightning loop. Continuous Gaussian diffusion,
  `x₀`-prediction, `T=1000`, cosine schedule, DDIM-200 at sampling.
- Freeze the Stage 4 autoencoder.

**Gate:** diffusion round-trip tests pass (`t=0` is identity, `t=T` is approximately standard
normal), the model overfits a 4-graph batch, and generated graphs beat a density-matched
Erdős–Rényi baseline on Ratio.

## Stage 6 — Spectral conditioning with oracle spectra (KILL GATE)

This is the stage that decides whether the project's hypothesis is alive.

- Inject the condition three ways: SignNet eigenvector embedding added to node tokens,
  eigenvalue MLP as adaLN scale/shift alongside the timestep, and a `U_k diag(λ_k) U_kᵀ`
  channel added to pair tokens.
- Classifier-free guidance: drop the condition with probability 0.1 during training.
- Train at `low`, `k=8`, using real spectra from held-out test graphs (oracle mode, labelled as
  such — it is a diagnostic, not a generative model).

**Gate:** permutation equivariance and condition-sensitivity tests pass, and oracle-conditioned
`k=8` beats the Stage 5 `none` baseline on Ratio with non-overlapping 3-seed error bars.

If this gate is red, **stop**. Conditioning on perfect spectra failing means no downstream work
will save the hypothesis, and we re-plan rather than push forward.

## Stage 7 — The frequency sweep (headline result)

- Six arms × `k ∈ {2, 4, 8, 16, 32}` on Planar, 3 seeds each, still oracle-conditioned so the
  frequency question is isolated from the spectral-prior question.
- Reduced grid on SBM: `low`, `high`, `none` at `k ∈ {2, 8, 32}`.
- **Condition-information floor:** train a deterministic decoder from `C_k` alone at every `k`.
  At `k=32` on 64-node planar graphs the condition nearly determines the graph, so the curve is
  only interpretable relative to this floor.

**Gate:** the headline figure exists — Ratio and V.U.N. against `k`, one line per band, floor as
a shaded region — with the `gaussian` capacity control included.

## Stage 8 — Learned spectral prior (turns it into a real generative model)

- Small Stage-1 diffusion over `(n, λ_k, U_k)` with an orthogonality penalty and QR retraction
  at the final step.
- Sample end to end with no oracle inputs. Compare against oracle mode; the gap is the cost of
  not having real spectra, directly comparable to SPECTRE's "real spectra" rows.
- Sweep the guidance scale.

**Gate:** end-to-end sampling works with no ground-truth inputs and beats the `none` baseline.

## Stage 9 — Downstream experiment and remaining controls

- SBM community-count classification, 4-way, at 20/40/80 real training graphs, 5 seeds,
  comparing real-only vs real-plus-unconditioned vs real-plus-low-band augmentation. This
  replaces the planar-vs-SBM task from the proposal, which is separable by mean degree alone
  and cannot show a difference between methods.
- DualDiff-style cluster-conditioning arm, combinatorial-vs-normalized Laplacian ablation,
  SignNet ablation.
- Memorization audit: nearest training graph by edit distance for every final sample set.

**Gate:** downstream table complete with error bars.

## Stage 10 — Report

- All figures regenerated from `results/` by a single script. No hand-typed numbers.
- Related work covers SPECTRE, LGD, DualDiff, SDMG, plus GGSD and DiGress's use of spectral
  features.

**Gate:** a clean clone plus one command regenerates every figure.

---

## Statistical rules that apply from Stage 5 onward

- **Three seeds minimum per arm**, mean and standard deviation reported.
- **Paired bootstrap confidence intervals** on MMD differences between arms. "Low beats high"
  without an interval excluding zero is reported as inconclusive.
- **Equal compute per arm** — same epochs, batch size, schedule, early stopping. Log wall-clock
  to prove it.
- Sampler settings fixed once on validation, then frozen for all arms.
- Every results table includes the training-set MMD row (the self-similarity floor). Without
  it, MMD numbers are meaningless.

## Compute

Planar is cheap and runs locally. SBM reaches about 200 nodes so its `n²` pair tensor is roughly
ten times heavier per graph. Benchmark one SBM run locally at Stage 5, then decide between
renting a cloud GPU and the reduced grid. Before spending money: mixed precision, gradient
accumulation instead of large batches, and the Stage 3 eigendecomposition cache.

## If scope must shrink

Cut in this order:

1. Stage 9 extras (cluster-conditioning arm, Laplacian variant, SignNet ablation)
2. The SBM half of Stage 7 — keep Planar, which carries the headline figure
3. Stage 8, reporting oracle-conditioned results only and labelling them as such throughout

**Never cut Stage 2 or the Stage 6 gate.** Everything else is negotiable; those two are what
separate a measured result from an anecdote.

---

## Glossary for the team

**Graph / node / edge.** Dots connected by lines. Dots are nodes, lines are edges. Written down
as an **adjacency matrix**: a grid of 0s and 1s where entry `(i,j)` is 1 if nodes `i` and `j`
are connected.

**Generative model.** A program that learns from many examples and invents new ones that look
similar but are not copies.

**Diffusion model.** One way to do that, the same technique behind image generators. Take a real
example, add a little random static, then more, until it is pure noise. Train a network to undo
one step. Once it can undo one step, start from pure noise and run backwards many times — a
realistic sample appears.

**Latent / latent space.** "Compressed." Instead of diffusing the big adjacency matrix directly,
first squeeze each graph into a short list of numbers and diffuse in that smaller space.

**Autoencoder.** The compressor plus decompressor pair. The **encoder** squeezes, the **decoder**
expands back. Autoencoder plus diffusion in the compressed space = **latent diffusion**, the
family this project belongs to.

**Laplacian / eigenvalues / eigenvectors / spectrum.** A matrix derived from the graph, and the
standard linear-algebra decomposition of it. Intuition: the spectrum is the bass/mid/treble of
a song. **Low-frequency** components (small eigenvalues) describe the big picture — how many
clusters, where the graph splits. **High-frequency** components describe fine local detail.

**Conditioning.** Giving the model a hint while it works. Unconditioned means it generates
freely; conditioned means we feed it the spectrum as it goes.

**MMD (Maximum Mean Discrepancy).** A number saying how different two *collections* of graphs
are. Low is good. Always set-against-set, never one generated graph against one real graph —
there is no single "correct" output.

**V.U.N.** Valid, Unique, and Novel. Valid = actually the right type of graph (checkable on the
graph itself). Unique = we did not emit the same graph twice (checked among our own outputs).
Novel = not memorized from the training set (here we *want* difference, the opposite direction
from every other metric).

**Baseline.** The simpler version we compare against. **Arm** = one experimental variant.

**Seed.** The starting random number. Running three seeds and getting similar answers is how we
show a result is not luck.

**Oracle.** A deliberately unfair version where we hand the model real answers as *input*. Used
as a measuring stick, not a shippable model.

**Gate.** A checkpoint where we verify something worked before spending more effort.
