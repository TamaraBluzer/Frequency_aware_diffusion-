# Workplan — Frequency-Aware Spectral Conditioning for Latent Graph Diffusion (FALD)

**Team:** Tamara Bluzer (315287441), Dan Shamia (208004119), Daniel Halperin (207826314), Itamar Kolodny (211490362)
**Course:** ML with Graphs, Tel Aviv University

---

## 0. One-paragraph statement of the project

We build a topology-only latent graph diffusion model whose denoiser is conditioned on the
first `k` non-trivial eigenpairs of the normalized graph Laplacian, and we measure how
generation quality varies as a function of `k` and as a function of *which* frequency band
the conditioning comes from (low / high / random / none). The scientific claim we are
trying to establish or refute is: **latent graph diffusion benefits specifically from
low-frequency spectral conditioning, with a measurable and non-monotone optimum in `k`.**
Every engineering decision below exists to make that claim falsifiable rather than
decorative.

---

## 1. Positioning: how we differ from each cited paper

This section is the spine of the report's novelty argument. Read it before writing any code,
because several implementation choices exist *only* to preserve these distinctions.

| Paper | What it does | What we do differently |
|---|---|---|
| **SPECTRE** (Martinkus et al., ICML'22) | GAN. Generates `(λ_k, U_k)` with a Stiefel-manifold GAN, then a PPGN generator builds `A` from `L⁽⁰⁾ = U_k diag(λ_k) U_kᵀ`. Picks a *single* `k` per dataset by early-stopping heuristic. | Diffusion, in a **learned latent space**, not GAN, not on raw `A`. We treat `k` as the *object of study*, not a hyperparameter to be tuned away — SPECTRE explicitly says it picks "the lowest `k` which resulted in good samples" and never reports the curve. We also replace their sign-canonicalization hack (flip so max-abs is positive) with a **sign/basis-invariant encoder (SignNet)**, and replace their Stiefel GAN with a **cascaded diffusion prior over the spectrum**. |
| **LGD** (Zhou et al., NeurIPS'24) | Latent diffusion with per-node + per-pair latents, cross-attention conditioning, evaluated on molecules (QM9/MOSES/ZINC). Never evaluated on Planar or SBM. | We use the LGD latent-space *paradigm* but apply it to **topology-only generic graphs**, add **spectral conditioning** (LGD conditions on scalar properties or masked graphs, never on spectrum), and inject the condition via **per-node feature fusion + global adaLN**, not cross-attention. We are, in effect, running the Planar/SBM experiment LGD skipped. |
| **DualDiff** (Xie & Pan, ICLR'26) | Two coupled EDM branches over per-node latents `Z_l` and per-**cluster** latents `Z_g`; global info is *cluster-based* (K-means / spectral clustering → pooled latents), fused with FiLM. | Their "global" signal is a **hard partition into K clusters**. Ours is a **continuous, ordered, frequency-indexed** signal. This matters: a cluster assignment collapses the spectrum into one number (`K`), whereas eigenpairs let us sweep a *band*. We will include DualDiff-style cluster conditioning as a **baseline arm** to show the frequency view is not just a re-parameterization of clustering. |
| **SDMG** (Zhu et al., ICML'25) | Shows low-frequency reconstruction beats full-spectrum reconstruction — but for **representation learning** (node/graph classification), with a multi-scale smoothing loss. | SDMG's thesis is that high-frequency detail *hurts* discriminative representations. Generation is the opposite regime: you cannot emit a valid planar graph without high-frequency detail. Our contribution is testing whether SDMG's low-frequency preference **transfers to generation**, where it is *a priori* likely to fail. We expect a non-monotone curve, and we will use SDMG's framing to explain it. We do **not** adopt their smoothing loss (it is a representation-learning objective). |

**The honest novelty sentence for the report:**
> "Spectral conditioning for graph generation exists (SPECTRE) and latent graph diffusion
> exists (LGD), but no prior work measures the *frequency-band sensitivity* of a diffusion
> generator. We provide that measurement, with matched-capacity controls that isolate the
> contribution of low-frequency structure from the contribution of merely having extra
> conditioning channels."

---

## 2. Gaps in the proposal that must be closed before coding

These are real holes. Each one has a decision attached; do not start Stage 3 until they are settled.

### G1 — Where does `C_k` come from at sampling time? *(the critical one)*
The proposal writes `ε_θ(z_t, t, C_k)` but never says how `C_k` is obtained when generating a
*new* graph. There is no graph yet, so there is no Laplacian to decompose. This is the single
biggest way the project can fail. Three modes, all of which we implement:

- **M0 — Oracle conditioning.** `C_k` taken from held-out *real* test graphs. This is not a
  generative model, it is a **diagnostic upper bound**. SPECTRE reports this ("real spectra")
  and so will we. Cheap, and it isolates "can the denoiser *use* the condition?" from
  "can we *sample* a valid condition?".
- **M1 — Cascaded spectral prior (primary).** A small Stage-1 diffusion model generates
  `(n, λ_k, U_k)`; Stage-2 latent diffusion is conditioned on it. We do **not** use SPECTRE's
  Stiefel-manifold machinery (rotation layers, Gumbel-softmax bank of learned Stiefel points).
  Instead we diffuse `U_k` in ambient `R^{n×k}` and project onto the Stiefel manifold with a
  **QR retraction at the final denoising step**, plus a soft orthogonality penalty
  `‖U_kᵀU_k − I_k‖_F²` during training. Simpler, diffusion-native, and a clean point of
  difference.
- **M2 — Joint conditioning (ablation).** Concatenate the spectral embedding onto the node
  latent and diffuse `[z_i ‖ ψ_i]` as one token. Removes the cascade entirely. Worth one run
  because if it works it is the cleanest story; if it doesn't, that's a finding.

**Decision:** implement M0 first (Stage 4), M1 as the headline method (Stage 6), M2 last.
All reported "main results" must be M1. M0 numbers must be clearly labelled as an oracle.

### G2 — At `k = 32`, the condition nearly determines the graph
Planar graphs have `n = 64`. Conditioning on 32 eigenpairs means handing the model half the
spectrum. A generator that "wins" at `k = 32` may just be **inverting the condition**, not
learning a distribution. Without controlling for this, the frequency-cutoff curve is
uninterpretable.

**Decision:** add a mandatory **condition-information baseline**: train a purely
deterministic decoder `A_hat = g(C_k)` (no diffusion, no noise) and report its reconstruction
quality at every `k`. This quantifies how much of the graph the condition already carries.
The cutoff curve is only meaningful *relative to this floor*. To our knowledge no cited paper
reports this, so it is a genuine contribution and it protects us from an embarrassing
reviewer question.

### G3 — Eigenvector sign and basis ambiguity
`u_i` and `−u_i` are the same eigenvector; for repeated eigenvalues *any* rotation within the
eigenspace is valid. SBM graphs have near-degenerate eigenvalues by construction, so this is
not a corner case. SPECTRE's fix (make the max-abs entry positive) is discontinuous and breaks
under multiplicity.

**Decision:** encode eigenvectors with **SignNet** — `ρ(Σ_i [φ(u_i) + φ(−u_i)])` — which is
sign-invariant by construction. Also apply **random sign flipping as training augmentation**
as a cheap secondary defence. Add a unit test (see §8, T4) that verifies model output is
invariant to sign flips of the conditioning eigenvectors.

### G4 — Matched-capacity controls
"Low-frequency conditioning beats no conditioning" is a weak result if the conditioned model
simply has more input dimensions and more parameters. The high-frequency and random arms must
be **dimension-matched and parameter-matched** to the low-frequency arm — same `k`, same
encoder, same training budget, only the *selection of eigenpairs* differs.

**Decision:** all conditioning arms share one code path; the arm is a config flag
(`band: low | high | random | shuffled | gaussian | none`). Add a
**`gaussian` arm** (condition on `k` i.i.d. Gaussian node features) — this is the strictest
control, since it has identical shape but zero structural information. If low-frequency does
not beat `gaussian`, we have no result.

### G5 — The proposed downstream task is too easy
"Train a GNN to distinguish planar from SBM graphs" is separable by mean degree and clustering
coefficient alone. A logistic regression on two scalars will hit ~100%, so no augmentation
method can show a difference. This experiment as written cannot produce a signal.

**Decision:** replace with a task that is genuinely hard in the low-data regime:
**predict the number of communities (2–5) in an SBM graph**, 4-way classification, with only
20/40/80 real training graphs. Community count is exactly what low-frequency eigenvalues
encode, so this task is *aligned with our hypothesis* and will show a difference if one exists.
Keep planar-vs-SBM only as a sanity check that the pipeline runs, and say explicitly in the
report why it is uninformative.

### G6 — Combinatorial vs normalized Laplacian
The proposal offers both. They are not interchangeable: `L = D − A` mixes degree (a local
property) into the low-frequency components, while `L_norm` eigenvalues are bounded in `[0,2]`
and comparable across graph sizes.

**Decision:** use **`L_norm = I − D^{-1/2} A D^{-1/2}`** throughout, matching SPECTRE and SDMG.
Drop the first eigenpair (`λ₁ = 0`, `u₁ ∝ D^{1/2}1`) since it only carries degree information.
Run one small ablation with `L = D − A` and report it in the appendix.

---

## 3. Method specification

### 3.1 Notation
Graph `G = (V, E)`, `n = |V|`, adjacency `A ∈ {0,1}^{n×n}`, degree matrix `D`.
`L_norm = I − D^{-1/2} A D^{-1/2} = U Λ Uᵀ`, eigenvalues sorted `0 = λ₁ ≤ … ≤ λ_n ≤ 2`.

Spectral condition, band-parameterised:

```
low     C_k = (λ_2..λ_{k+1},   u_2..u_{k+1})      # skip trivial eigenpair
high    C_k = (λ_{n-k+1}..λ_n, u_{n-k+1}..u_n)
random  C_k = k eigenpairs sampled uniformly without replacement from indices 2..n
gaussian C_k = (k random scalars, N(0,1) matrix in R^{n×k})   # capacity control
none    C_k = ∅
```

### 3.2 Stage 0 — Graph autoencoder
Encoder `E_φ: A → (Z, W)`, `Z ∈ R^{n×d_v}` per-node, `W ∈ R^{n×n×d_e}` per-pair
(LGD-style augmented edge tensor; `n ≤ 187` so `n²` is affordable at `d_e = 4–8`).
Backbone: augmented-edge graph transformer, 4–5 layers, hidden 96–128.

Decoder `D_ξ: W → A_hat`, a linear/2-layer-MLP head per pair with BCE loss.
Symmetrise as `(W_ij + W_ji)/2` before decoding, zero the diagonal.

> **CORRECTION (measured).** This decoder specification is unsafe as written and was implemented
> literally, with the predicted result. It is LGD's reconstruction task (ii) alone; LGD's paper
> lists **five** tasks, prefaced "to force the encoder to learn meaningful representations", and
> task (ii) is the only one that permits the encoder to copy `A_ij` straight into `W_ij` down a
> private per-pair lane. Measured on Planar: F1 1.000 with Cohen's d of 13302 between the edge
> and non-edge latent clouds — a two-point binary code, not a manifold, and 32,768 numbers for
> 2,016 binary decisions, so no compression either.
>
> Tasks (i), (iv) and (v) are vacuous for us because our node features are constant, leaving only
> (iii) — decode `e_ij` from `(z_i, z_j)`. That removes the copy path but exposes a second
> problem: a permutation-equivariant encoder cannot separate structurally similar nodes on
> featureless near-regular graphs, so the node latents collapse (effective rank 1.04 of 32).
> LGD's fix is RRWP positional encodings, which are powers of the normalized adjacency and would
> leak spectral structure into the latent, contaminating our own `none` arm.
>
> See [docs/AUTOENCODER.md](docs/AUTOENCODER.md). Do not re-specify a single per-pair
> reconstruction head without reading it.

Regularization: **LayerNorm on encoder outputs, no KL.** LGD reports that strong KL hurts
downstream diffusion quality and that LayerNorm works better; we follow that and note it.
Add a *small* `σ₀` Gaussian jitter (DualDiff-style) to keep the latent manifold smooth.

**Gate:** the AE must exceed **99% edge-level accuracy and ≥0.98 F1 on Planar and SBM
reconstruction** before we train any diffusion model. If the AE cannot reconstruct, nothing
downstream is interpretable. Record the AE reconstruction quality as the **ceiling** for all
generation metrics and plot it as a horizontal line on every results figure.

### 3.3 Stage 1 — Spectral prior (for sampling mode M1)
Diffuse `(λ_k ∈ R^k, U_k ∈ R^{n×k})` conditioned on `n`.
- `n` is sampled from the empirical training distribution (SPECTRE and GRAN do the same).
- Denoiser: small permutation-equivariant transformer over `n` tokens of width `k`, plus a
  1-D branch for `λ_k`.
- Losses: standard `x₀`-prediction MSE + `μ_ortho · ‖U_kᵀU_k − I_k‖_F²` + monotonicity penalty
  on `λ` (`relu(λ_i − λ_{i+1})`).
- Final step: QR retraction of `U_k`, sort and clamp `λ_k` into `[0, 2]`.

### 3.4 Stage 2 — Conditional latent diffusion
Continuous Gaussian diffusion (DDPM formulation, `x₀`-prediction, `T = 1000`, cosine schedule),
DDIM sampler with 200 steps at inference. Denoiser is a permutation-equivariant
augmented-edge graph transformer over `(Z_t, W_t)`.

**Conditioning injection (our specific design, distinct from all three papers):**
1. Eigenvectors → SignNet → per-node vector `ψ_i ∈ R^{d_ψ}`, **added** to node token `z_i`.
2. Eigenvalues → MLP → global vector `c_λ`, injected as **adaLN** (scale/shift on every
   layer norm) together with the timestep embedding.
3. A rank-1 pair-level term `(U_k diag(λ_k) U_kᵀ)_{ij}` added as an extra channel to the pair
   token `W_ij`. This is the one idea we borrow directly from SPECTRE's `L⁽⁰⁾`, and we should
   cite it as such.

**Classifier-free guidance:** drop the condition with probability 0.1 during training so we can
sweep guidance scale `w` at sampling time. None of the four cited papers do CFG on spectral
conditions; it gives us a free extra axis ("how *hard* should we push the spectral condition?")
that pairs naturally with the frequency-band question.

### 3.5 Tier-0 de-risking baseline
Before the latent model works, implement **continuous diffusion directly on the dense
adjacency** (EDP-GNN/GDSS style, no autoencoder). It is ~150 lines, trains in minutes on
Planar, and gives us (a) a working evaluation harness, (b) a published-comparable number, and
(c) an insurance policy if the autoencoder stalls. **Do not skip this.**

---

## 4. Datasets

| Dataset | Spec | Source |
|---|---|---|
| **Planar** | 200 graphs, `n = 64`, Delaunay triangulation of uniform random points in the unit square | Regenerate from SPECTRE's recipe |
| **SBM** | 200 graphs, 2–5 communities (uniform), 20–40 nodes per community (uniform) → `n ∈ [40, 200]`, `p_intra = 0.3`, `p_inter = 0.05` | Regenerate from SPECTRE's recipe |

> **Note a real typo in SPECTRE.** Appendix C states "inter-community edge probability is 0.3
> and the intra-community edge probability is 0.05," which is inverted — that would produce
> anti-communities. The community convention (and their own results) require `p_intra = 0.3`,
> `p_inter = 0.05`. Use the corrected values and put a footnote in the report; catching this
> is a small but genuine sign of careful work.

**Splits:** 64% train / 16% val / 20% test, fixed seed, serialized to disk as `.pt`, committed
as a hash so all four of us evaluate on byte-identical data. Model selection **only** on val;
test touched once per arm at the end.

Generate 40 samples per arm for Planar (= test set size) and 40 for SBM, matching the
convention "generate as many graphs as there are in the test split."

---

## 5. Evaluation protocol

### 5.1 Distributional metrics (MMD, lower is better)
Reimplement the GRAN/SPECTRE suite so numbers are comparable to published tables:

- **Deg.** — degree histogram MMD
- **Clus.** — clustering coefficient histogram MMD
- **Orbit** — 4-node orbit counts MMD (requires the ORCA binary — see §9 R3)
- **Spec.** — normalized Laplacian eigenvalue histogram MMD
- **Wavelet** — SPECTRE's eigenspace MMD via 12 ab-spline graph wavelet kernels (PyGSP)
- **Ratio** — mean of (our MMD / training-set MMD) across the five, the headline scalar

Kernel: **total-variation Gaussian kernel** (GRAN convention, and what SPECTRE used), not
Gaussian-EMD. Fix and document the bandwidth.

### 5.2 Validity / Uniqueness / Novelty
- **Planar valid** = `networkx.check_planarity` AND connected.
- **SBM valid** = recovered community structure consistent with the generating parameters.
  SPECTRE uses graph-tool Bayesian inference + merge-split MCMC + a Wald test at `p ≥ 0.9`.
  graph-tool is painful on Windows — use the **DiGress-style pure-NumPy SBM test** as the
  primary implementation, and report the graph-tool version only if we get it running in WSL.
  Whichever we use, state it explicitly; the two are not numerically comparable.
- **Unique** = fraction in distinct isomorphism classes (`nx.is_isomorphic` / VF2).
- **Novel** = fraction whose isomorphism class is absent from the training set.
- **V.U.N.** = fraction that is simultaneously valid, unique, and novel. Report this as the
  second headline scalar alongside Ratio.

### 5.3 Our own metrics (not in any cited paper)
These are what make the project a *study* rather than a reimplementation.

- **Spectral consistency error (SCE).** For each generated graph `G_hat`, recompute its first-`k`
  eigenpairs and measure the distance to the condition it was generated from:
  `SCE_λ = ‖λ_k(G_hat) − λ_k^cond‖₂` and
  `SCE_U = 1 − (1/k)·‖U_k(G_hat)ᵀ U_k^cond‖_F² / k` (subspace alignment, sign/basis-invariant).
  This directly answers "is the model actually *obeying* the condition, or ignoring it?"
  A model with great MMD and terrible SCE is not doing what we claim it is doing.
- **Condition-information floor** (from G2): reconstruction quality of `A_hat = g(C_k)` at each `k`.
- **Conditioning sensitivity.** Generate two graphs from the same noise seed with two different
  conditions; measure edit distance. Near-zero means the condition is being ignored.

### 5.4 Statistical rigor
This is where most course projects lose credibility, so it is non-negotiable:

- **3 seeds minimum per arm.** Report mean ± std. A single run of a diffusion model on 200
  graphs is noise.
- **Paired bootstrap CIs** on MMD differences between arms (resample the generated set 1000×).
  "Low beats high" must come with an interval that excludes zero, or it is reported as
  inconclusive.
- **Equal compute per arm.** Same epochs, same batch size, same LR schedule, same early-stop
  criterion. Log wall-clock and step counts to prove it.
- **No cherry-picking the sampler.** Fix DDIM steps and guidance scale using the *validation*
  set once, then freeze for all arms.
- Report the **training-set MMD row** in every table (the self-similarity floor). Without it,
  MMD numbers are meaningless.

---

## 6. Experiment matrix

Primary grid, run on **both** Planar and SBM:

| Arm | `k` | Purpose |
|---|---|---|
| `none` | – | Unconditioned latent diffusion baseline |
| `low` | 2, 4, 8, 16, 32 | **The frequency-cutoff curve — the headline result** |
| `high` | 2, 4, 8, 16, 32 | Is it the *frequency band* or just extra info? |
| `random` | 2, 4, 8, 16, 32 | Band-agnostic control |
| `gaussian` | 2, 4, 8, 16, 32 | Strict capacity control (zero structural information) |
| `cluster` (DualDiff-style) | K = 4, 8 | Is the frequency view distinct from hard clustering? |

Secondary (single `k*` = best from the low curve):
- Sampling mode: M0 oracle vs M1 cascade vs M2 joint
- Guidance scale `w ∈ {0, 0.5, 1, 2, 4}`
- SignNet vs SPECTRE-style sign canonicalization vs no handling
- `L_norm` vs `L = D − A`
- Tier-0 (adjacency-space) vs latent diffusion, at `none` and `low, k*`

Downstream (see G5):
- SBM community-count classification (4-way), real-only vs +unconditioned vs +low-`k*`
  augmentation, at 20/40/80 real training graphs, 5 seeds each.

**Budget estimate.** The primary grid is `(1 + 5×4 + 2) = 23` configs × 2 datasets × 3 seeds
= **138 training runs**. Planar is cheap (fixed `n = 64`, small dense tensors) and fits
comfortably on a single local GPU. SBM is the expensive half: `n` reaches ~200, so the `n²`
pair tensor is roughly 10× larger per graph.

Plan for it in two tiers. Run the **full `k` sweep on Planar locally** — that is the headline
figure and it does not need cloud. For SBM, first measure the per-run cost locally at `k = 8`;
if a single seed exceeds a few hours, either rent a larger cloud GPU for the SBM half or fall
back to the reduced grid (`k ∈ {2, 8, 32}` for `low` / `high` / `none` only). If we reduce,
state it explicitly in the report rather than quietly dropping cells.

Before renting anything, exhaust the cheap wins: mixed precision, gradient accumulation instead
of large batches, and caching eigendecompositions to disk (Stage 1) — the last one alone is a
large constant-factor saving since the spectrum never changes during training.

---

## 7. Repository structure

```
FinalProject/
├── WORKPLAN.md                  # this file
├── README.md                    # how to reproduce, one command per figure
├── environment.yml / requirements.txt
├── configs/
│   ├── base.yaml                # shared defaults
│   ├── ae/{planar,sbm}.yaml
│   ├── diffusion/{planar,sbm}.yaml
│   └── arms/{none,low,high,random,gaussian,cluster}.yaml
├── src/
│   ├── data/
│   │   ├── generate.py          # Planar + SBM generators (SPECTRE recipe, corrected)
│   │   ├── dataset.py           # splits, caching, dense padding + node masks
│   │   └── spectral.py          # L_norm, eigendecomposition, band selection, caching
│   ├── models/
│   │   ├── layers.py            # augmented-edge graph transformer block, adaLN
│   │   ├── signnet.py
│   │   ├── autoencoder.py
│   │   ├── denoiser.py
│   │   ├── spectral_prior.py    # Stage-1 cascade
│   │   └── adjacency_diffusion.py   # Tier-0
│   ├── diffusion/
│   │   ├── schedules.py
│   │   ├── ddpm.py              # training objective
│   │   └── sampler.py           # DDIM + CFG
│   ├── eval/
│   │   ├── mmd.py               # TV-Gaussian kernel, the 5 statistics
│   │   ├── orbit.py             # ORCA wrapper
│   │   ├── validity.py          # planarity, SBM test, iso-class uniqueness/novelty
│   │   ├── spectral_consistency.py   # SCE, our metric
│   │   └── report.py            # tables + figures straight to LaTeX
│   ├── downstream/
│   │   └── community_count.py   # G5 experiment
│   └── utils/{seed,logging,checkpoint}.py
├── tests/                       # see §8 — these are load-bearing, not decoration
├── scripts/
│   ├── run_sweep.py
│   └── make_all_figures.py
├── results/                     # CSV per run, never edited by hand
└── report/
```

Config-driven, one entry point (`python -m src.train --config ... --override arm=low k=8`).
Every run writes `results/<run_id>/{config.yaml, metrics.csv, samples.pt, git_sha.txt}`.
Never report a number that isn't in a `metrics.csv`.

---

## 8. Correctness checks (`tests/`)

Diffusion models fail silently — they produce plausible-looking garbage. These tests catch
the failure modes that are otherwise invisible until you have already built everything on top of them.

| ID | Test | Why |
|---|---|---|
| **T1** | Eigendecomposition sanity: `L_norm @ u_i ≈ λ_i * u_i`; `λ ∈ [0,2]`; `λ₁ ≈ 0`; multiplicity of `λ=0` equals number of connected components | Catches normalization bugs, isolated-node division-by-zero |
| **T2** | **Permutation equivariance** of encoder and denoiser: for random permutation `P`, `f(PAPᵀ) == P f(A) Pᵀ` up to float tolerance | The whole method is invalid without this. Run it in CI. |
| **T3** | Masking correctness for variable `n`: padded nodes contribute nothing to any pooled quantity or loss | SBM has `n ∈ [40,200]`; silent mask bugs are the classic killer here |
| **T4** | **Sign/basis invariance**: SignNet output unchanged under `u_i → −u_i`; subspace-level invariance under rotation within a degenerate eigenspace | Closes G3 |
| **T5** | MMD calibration: `MMD(train, train_split_2)` is near zero; `MMD(train, Erdős–Rényi)` is large; metric is symmetric and non-negative | An MMD implementation that reports 0 for everything is easy to write and hard to notice |
| **T6** | Validity oracles: real planar graphs → 100% valid; real SBM graphs → ≥95% valid; a random ER graph of matched density → low validity | If our own training data fails the validity test, the test is wrong |
| **T7** | AE reconstruction gate: ≥99% edge accuracy on held-out graphs before diffusion training starts | Closes the "is the ceiling the problem?" question permanently |
| **T8** | Diffusion round-trip: `q(x_t \| x_0)` at `t=0` returns `x_0`; at `t=T` returns approximately `N(0,I)` (check mean/var); denoiser at `t=0` is near-identity | Catches schedule/indexing off-by-one, the most common DDPM bug |
| **T9** | **Condition-ignoring detector**: same seed, two different conditions → outputs must differ (§5.3). Fails loudly if the model learned to ignore `C_k` | This is the failure mode that would silently produce "no significant difference between arms" |
| **T10** | Overfit-one-batch: model can drive loss to ~0 on 4 graphs | Standard, 5 minutes, catches a dozen bugs |
| **T11** | Memorization audit: for each generated graph, nearest training graph by edit distance; flag exact isomorphs | Guards against the GRAN-style "perfect validity, zero novelty" trap SPECTRE calls out |

---

## 9. Risk register

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| **R1** | Autoencoder can't reconstruct planar graphs well enough; every downstream number is capped | High | Tier-0 adjacency diffusion (§3.5) as fallback; increase `d_e`; report AE ceiling on every plot |
| **R2** | Stage-1 spectral prior generates invalid spectra, so M1 looks worse than `none` and the whole story inverts | Medium-high | This is what M0 (oracle) is for — it separates "denoiser can't use the condition" from "we can't sample conditions." SPECTRE hit exactly this (their Table 3 shows large eigenvector MMD ratios); if we hit it too, that *is* a reportable finding, not a failure |
| **R3** | ORCA (orbit counts) needs a C++ binary; Windows toolchain pain | Medium | Build under WSL2, or use `networkx`-based graphlet counting for `n ≤ 200` (slower but tractable), or drop Orbit and report the other four with a clear note |
| **R4** | graph-tool for the SBM validity test won't install on Windows | High | Use the pure-NumPy DiGress-style SBM test as primary (already the plan in §5.2) |
| **R5** | OneDrive syncs `.git`, checkpoints, and sample tensors → corruption and quota | **High — already applicable** | Move the repo out of `OneDrive - NVIDIA Corporation`, or at minimum exclude `results/`, `checkpoints/`, and `.git` from sync. Add them to `.gitignore` today |
| **R6** | 138 runs don't fit on the local GPU | High | Planar sweep runs locally; SBM is the expensive half — measure one run first, then decide between cloud rental and the reduced grid in §6 |
| **R7** | All arms come out statistically indistinguishable | Medium | Still publishable as a negative result *if* T9 passes and SCE shows the model is genuinely using the condition. The controls are what make a null result credible |
| **R8** | Four people editing one codebase, merge chaos | Medium | Branch per phase, PR review by one other member, `main` always green on `tests/` |

---

## 10. Ordered stages with gates

Each stage ends in a **gate**: a concrete, checkable condition. Do not proceed past a red gate;
fix it or invoke the fallback. The ordering matters more than any calendar — in particular,
evaluation is built before models (Stage 2) and the hypothesis is stress-tested early (Stage 4).

### Stage 0 — Setup
- Move repo out of OneDrive (R5). `.gitignore` for `results/`, `checkpoints/`, `*.pt`, `wandb/`.
- Environment: PyTorch + PyG, NetworkX, PyGSP, SciPy, hydra/OmegaConf for configs.
- Local GPU is the default target. Write down its VRAM, and make the training script
  device-agnostic and checkpoint-resumable from the start so a cloud instance is a drop-in
  swap rather than a port. Enable mixed precision by default.
- **Gate:** every member can run `pytest tests/` (empty is fine) and train a toy MLP on the GPU.

### Stage 1 — Data + spectral utilities
- `generate.py` for Planar and SBM with the corrected probabilities.
- Fixed splits, hashed, committed.
- `spectral.py`: `L_norm`, eigendecomposition, band selection for all six arms, **cached to disk**
  (recomputing eigendecompositions every epoch is a silent 10× slowdown).
- **Gate:** T1, T3 pass. Visual sanity plot of 8 real graphs per dataset + their `u₂` colorings
  (reproduce the intuition in SPECTRE's Figure 4 on *our* data — this figure goes in the report).

### Stage 2 — Evaluation harness *(build this before any model)*
Building the evaluator first is the single highest-leverage sequencing decision. It means every
model you train from that point on is immediately measurable.
- Full MMD suite, validity, uniqueness, novelty, V.U.N., Ratio.
- SCE and the condition-sensitivity probes.
- `report.py` emitting LaTeX tables directly.
- **Gate:** T5, T6 pass. Reproduce the "Training set" row of SPECTRE's Table 1 to within an
  order of magnitude on regenerated Planar/SBM data. Exact match is not expected (different
  random data), but Deg ≈ 1e-4, Clus ≈ 3e-2, Spec ≈ 5e-3 should be roughly recovered.

### Stage 3 — Tier-0 adjacency diffusion
- Unconditional continuous diffusion on dense `A`, on Planar only.
- **Gate:** T8, T10 pass. Generated graphs are visually graph-like and beat an Erdős–Rényi
  baseline of matched density on Ratio. This proves the diffusion plumbing works.

### Stage 4 — Autoencoder + unconditional latent diffusion + M0 oracle
- Train AE to the T7 gate. Freeze it.
- Unconditional latent diffusion (`none` arm) — this is Baseline #1 for the whole paper.
- Add conditioning path (SignNet + adaLN + rank-1 pair channel), run **M0 oracle** at `low, k=8`.
- **Gate:** T2, T4, T7, T9 pass. M0-oracle at `k=8` beats `none` on Ratio with non-overlapping
  3-seed error bars. **This is the kill-gate.** If conditioning on *perfect* spectra cannot beat
  the unconditioned baseline, the hypothesis is dead and no amount of downstream work will save
  it. This stage is placed early for exactly that reason — a red gate here means pivot, and it
  is far cheaper to pivot before the sweep than after.

### Stage 5 — The frequency sweep
- All six arms × `k ∈ {2,4,8,16,32}` on Planar, 3 seeds, still under M0 oracle conditioning
  (this isolates the *frequency* question from the *spectral prior* question).
- Reduced grid on SBM.
- Condition-information floor (G2) at every `k`.
- **Gate:** the frequency-cutoff curve exists and is interpretable *relative to the floor*.
  Produce the headline figure: Ratio and V.U.N. vs `k`, one line per band, floor as a shaded
  region.

### Stage 6 — Cascaded spectral prior (M1)
- Stage-1 diffusion over `(n, λ_k, U_k)` with orthogonality penalty + QR retraction.
- End-to-end sampling; compare M1 vs M0 at `k*` — the gap *is* the cost of not having real spectra,
  and it is directly comparable to SPECTRE's "real spectra" rows.
- CFG guidance sweep.
- **Gate:** M1 is a functioning generative model (no oracle inputs) and beats `none`.

### Stage 7 — Downstream + controls
- SBM community-count augmentation experiment (G5), 5 seeds.
- DualDiff-style cluster-conditioning arm.
- Laplacian variant ablation, SignNet ablation, M2 joint-conditioning if time permits.
- T11 memorization audit on every final sample set.
- **Gate:** downstream table complete with error bars.

### Stage 8 — Writeup
- Report, figures regenerated from `results/` by `make_all_figures.py` (no hand-made numbers).
- README with exact reproduction commands.
- **Gate:** a clean clone + `bash scripts/reproduce.sh` regenerates every figure.

### Scope-reduction order

If the project has to shrink, cut in this order and say so in the report:

1. Stage 7 extras (cluster-conditioning arm, Laplacian variant, M2 joint conditioning)
2. The SBM half of the frequency sweep — keep Planar, which carries the headline figure
3. Stage 6, reporting M0-oracle results only and labelling the study as oracle-conditioned
   throughout

**Never cut Stage 2 (evaluation) or the gates in Stage 4.** Everything else is negotiable;
those two are what separate a measured result from an anecdote.

---

## 11. Division of labour

Four people, minimal blocking dependencies:

| Owner | Stages | Primary artifacts |
|---|---|---|
| **A** | 1, 5 | Data generation, `spectral.py`, band selection, the frequency sweep driver |
| **B** | 2, 7 | Full evaluation harness, SCE, validity, downstream experiment |
| **C** | 3, 4 | Tier-0 diffusion, autoencoder, denoiser architecture, SignNet |
| **D** | 6, 0/8 | Spectral prior cascade, CFG, infra/configs/CI, report assembly |

Everyone writes tests for their own module. Cross-review: A↔B, C↔D.
Short recurring sync; the only standing agenda item is "which gate are we at and is it red."

---

## 12. Report structure (draft the skeleton early — it exposes missing experiments)

1. **Introduction** — the frequency-band question, why it is unanswered
2. **Related work** — the four papers, framed by the §1 positioning table
3. **Method** — §3, with the three sampling modes made explicit
4. **Experimental setup** — data, metrics, and *especially* the controls from G2/G4
5. **Results**
   - 5.1 The frequency-cutoff curve (headline figure)
   - 5.2 Band controls: low vs high vs random vs gaussian
   - 5.3 The condition-information floor — how much of the curve is just leakage
   - 5.4 Spectral consistency: is the model obeying the condition?
   - 5.5 Oracle (M0) vs learned prior (M1)
   - 5.6 Downstream augmentation
6. **Discussion** — reconciling with SDMG (low-frequency helps recognition) vs generation
   (needs high frequency for validity); why the optimum is where it is
7. **Limitations** — small datasets, `n ≤ 200`, single architecture family, compute-limited grid
8. **Conclusion**

---

## 13. Immediate next actions

1. Move the repo off OneDrive (R5) and add `.gitignore`.
2. Agree on the six decisions in §2 (G1–G6) — 30-minute meeting, record the outcomes in this file.
3. Assign owners per §11.
4. Start Stage 0 and Stage 1 in parallel; Stage 2 (evaluation) starts as soon as Stage 1's data
   generator produces graphs.
5. Benchmark one SBM training run on the local GPU early, so the cloud-vs-reduced-grid decision
   in §6 is made on measured numbers rather than guesses.
