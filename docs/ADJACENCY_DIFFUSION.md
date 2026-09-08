# Tier-0 adjacency diffusion — the primary frequency experiment

**Decision (5 September 2026).** Direct adjacency diffusion is now the primary path for the
frequency-band study. The latent model is a transfer ablation, not a dependency of the main
result.

## Why we originally chose latent diffusion

The original motivation was sound. Dense graph models process all `n²` possible node pairs even
though most are non-edges. A fixed-width node representation `Z ∈ R^(n×d)` could reduce the
diffusion state from quadratic to linear size, provide a smooth continuous target for Gaussian
diffusion, and isolate graph-specific decoding inside a reusable autoencoder. LGD supplied a
published two-stage template.

The implemented design did not deliver those benefits:

- `W ∈ R^(n×n×8)` is an expansion, not compression. On 64-node Planar graphs it uses 32,768
  floating-point values to represent 2,016 independent binary edge decisions.
- Removing `W` produced an honest node-only bottleneck, but the current encoder and endpoint MLP
  could not reconstruct featureless near-regular graphs.
- A perfect binary pair code is not mathematically incompatible with Gaussian diffusion. It is
  simply equivalent to diffusing adjacency through a learned reparameterization, while retaining
  the same O(`n²`) cost.

## New primary method

The diffusion state is the centered adjacency itself:

`x₀ = 2A − 1`.

A cosine forward process adds symmetric Gaussian noise independently to each undirected node
pair. A DiGress `XEyTransformerLayer` stack predicts the clean adjacency state `x₀` from `x_t`.
The diagonal and padded pairs are masked at every step.

The explicit oracle condition has two parts:

1. `U_k diag(λ_k) U_kᵀ`, supplied as a pair channel and scaled to unit RMS.
2. The selected normalized-Laplacian eigenvalues, mapped from `[0,2]` to `[-1,1]` and supplied
   globally with the timestep.

The pair channel is invariant to eigenvector sign flips and to basis rotations within an exactly
repeated eigenspace. This avoids making the primary path depend on the current SignNet, which is
sign-invariant but not basis-invariant.

`none`, `low`, `high`, and `random` all use the same model and diffusion state. Only the condition
tensor changes. Classifier-free condition dropout is applied during training, and the sampler
supports guidance.

Implementation:

- `fald/models/adjacency_diffusion.py`
- `fald/data/conditioning.py`
- `scripts/train_adjacency_diffusion.py`
- `tests/test_adjacency_diffusion.py`

## What the experiment now claims

The `none` arm means **no explicit graph-specific spectral side channel**. It does not mean the
denoiser is incapable of learning spectral information from adjacency; that would be impossible
for any model that learns the full graph.

The narrow claim is:

> Given the same full-graph diffusion state and matched architecture, does a target-derived
> spectral side condition improve generation, and does that incremental benefit depend on the
> selected frequency band?

GGSD already measured low versus high frequencies when eigenpairs are the generated
representation itself. Our remaining distinction is the controlled *side-channel* experiment,
including `none`, unrelated-condition, and condition-information controls.

## Gates

The mathematical gate currently passes:

- exact `t=0` forward-process identity;
- terminal state approximately standard normal;
- symmetric noise and output with a zero diagonal;
- permutation equivariance when adjacency and the pair condition are permuted together.

The quality gates are:

- generated Planar graphs must beat density-matched Erdős–Rényi on Ratio;
- then oracle `low, k=8` must beat `none` across at least three seeds;
- only after that do we run the full frequency sweep.

The first weighted-loss smoke run exposed an important objective bug. Positive-class weighting is
appropriate for an imbalanced reconstruction classifier, but not for diffusion regression: under
heavy noise, a weight of 10.34 changes the optimal edge threshold to the dataset base rate and
therefore encourages too many edges. Removing that weight moved generated density from 0.246 to
0.082–0.089.

## Three-seed `k=8` result

Planar validation split, 200 epochs, 100 diffusion steps, four transformer layers. Ratio is the
five-metric mean including ORCA orbit MMD; lower is better.

| arm | Ratio, mean ± std | best validation loss, mean ± std | Planar validity |
|---|---:|---:|---:|
| `none` | 327.23 ± 8.80 | 0.13232 ± 0.01197 | 0% |
| `low` | **157.97 ± 16.20** | **0.05349 ± 0.00436** | 0% |
| `high` | 339.07 ± 9.18 | 0.09376 ± 0.00860 | 0% |
| `random` | 326.48 ± 11.75 | 0.12832 ± 0.01170 | 0% |

Paired hierarchical bootstrap, 1,000 resamples:

- `low − none`: 95% CI `[-189.43, -152.09]`
- `low − high`: 95% CI `[-206.68, -152.42]`
- `low − random`: 95% CI `[-195.10, -142.21]`

All intervals exclude zero by a wide margin. T9 also passes: under identical diffusion noise,
adding the low-frequency condition changes 12.95% of edge decisions and produces a mean absolute
clean-prediction change of 0.162.

Artifacts:

- `results/frequency_pilot_planar_k8_summary.json`
- `results/adjacency_diffusion_planar_*_k8*.json`
- `results/adjacency_diffusion_planar_low_k8_condition_sensitivity.json`

## What is solved, and what is not

This resolves the Stage 4 dependency problem: a working frequency experiment no longer requires
an autoencoder, and the low-frequency effect survives the three-seed kill gate.

It does not yet produce valid Planar graphs. All arms have 0% Planar validity, which is consistent
with ConGress's published 0% Planar V.U.N. Continuous Gaussian edge noise is therefore useful as
the diagnostic frequency experiment but is not the best final generator. The next architecture
step is to carry the same explicit condition into direct **discrete DiGress**, which reports 75%
Planar V.U.N., rather than returning to the failed latent design.

## Stage 6.5 discrete pilot

A binary marginal-transition D3PM was implemented in
`fald/models/discrete_adjacency_diffusion.py`. Its reverse posterior is tested against explicit
enumeration. DiGress-style cycle counts for lengths 3–6 are computed from each noisy graph and
supplied as band-neutral node/global features.

Reduced local configuration: six layers, 200 categorical diffusion steps, 500 epochs, batch 16.
This is much smaller than published Planar DiGress (ten layers, 1,000 steps, all auxiliary
features, up to 100,000 epochs).

| process | condition | Ratio | Planar validity |
|---|---|---:|---:|
| discrete, no auxiliary cycles | `none` | 318.57 | 0% |
| discrete + cycle features | `none` | 262.47 | 0% |
| discrete + cycle features | `low, k=8` | **40.01** | 0% |

Categorical diffusion and cycle features improve distributional fidelity, and the low-frequency
condition becomes even more effective. They do not repair exact planarity at the reduced budget.
Stage 6.5 therefore remains red: Ratio passes, validity fails.

The honest next decision is between:

1. reproduce the full published DiGress architecture/training budget and then add the condition;
2. report 0% V.U.N. as a limitation while keeping the statistically strong frequency result;
3. treat discrete validity as compute-blocked and prioritize the remaining controls on the
   continuous diagnostic model.

Do not claim that the reduced discrete implementation reproduces published DiGress.

## Latent follow-up

If the primary experiment succeeds, use LG-Flow's LG-VAE design for a reduced latent transfer
experiment. Its adjacency-identifying Laplacian positional encoder and row-wise DeepSet decoder
are a substantially better starting point than further tuning the current endpoint MLP. Because
that autoencoder has a low-frequency inductive bias, its result must be described as transfer
evidence rather than the clean primary comparison.
