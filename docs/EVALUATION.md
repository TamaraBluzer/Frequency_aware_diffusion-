# Evaluation harness (`fald/eval`)

Built before any of our own models, so that every model from Stage 4 onward is immediately
measurable against a harness whose behaviour we have already verified.

## Why it is ours and not DiGress's

The metric functions in DiGress's `spectre_utils.py` are sound, but they are bundled into an
`nn.Module` (`SpectreSamplingMetrics`) that expects a Lightning datamodule, a `local_rank`, and
wandb. Ours takes two lists of `networkx` graphs and returns numbers, so it can be called from a
training loop, a notebook, or the calibration script. It also has no import dependency on the
gitignored `third_party/` checkout — only the optional ORCA binary is external.

## Package name

`fald`, not `src`. DiGress installs itself editable as a top-level package literally named
`src`; a second `src` here shadows it completely (verified: `src.__path__` resolved to this repo
only, making `src.analysis` unimportable). Stage 4 needs to import DiGress's graph transformer
*and* our code in the same process, so the names had to be disjoint.

## Metrics

Kernels and sigmas are copied from DiGress/SPECTRE deliberately. Changing them silently breaks
comparability with published tables.

| Metric | Descriptor | Kernel | sigma | Normalized to pmf |
|---|---|---|---|---|
| `degree` | degree histogram | `gaussian_tv` | 1.0 | yes |
| `clustering` | clustering coefficients, 100 bins on [0, 1] | `gaussian_tv` | 0.1 | yes |
| `spectral` | normalized-Laplacian eigenvalues, 200 bins on [0, 2] | `gaussian_tv` | 1.0 | yes |
| `wavelet` | 12 ab-spline scales, 100 energy bins each | `gaussian_tv` | 1.0 | yes |
| `orbit` | per-node-averaged 4-node orbit counts (15 dims, ORCA) | `gaussian_tv` | 30.0 | no |

`orbit` passes `is_hist=False` because its descriptor is already a per-node average rather than
a histogram.

**On the wavelet metric:** `PLAN.md` said to add SPECTRE's Wavelet MMD because "DiGress omits"
it. DiGress actually *vendors* it — `spectral_filter_stats`, with the same 12 `pygsp`
`Abspline` filters — but leaves the call commented out. So Stage 2 re-enabled existing code
rather than writing a new metric.

## Ratio

MMD alone is uninterpretable: nothing tells you whether 0.01 is close or far. `Ratio` divides
each metric by the same metric measured between two *real* samples (the train-vs-test
self-similarity floor) and averages, so 1.0 means "indistinguishable from real data at this
sample size" and 10 means "ten times worse than the floor."

Every results table must carry the floor row. The denominator is clamped at 1e-8 so a
degenerate reference cannot produce a meaningless astronomical ratio.

## V.U.N.

`vun()` reports the joint fraction that is simultaneously valid, in a distinct isomorphism class
from earlier samples, and not isomorphic to any training graph — plus each marginal, because a
low joint score is otherwise unattributable.

Planarity is exact (`nx.check_planarity` plus connectivity). **SBM validity raises
`NotImplementedError`**: it needs graph-tool, which has no Windows build. See
[ENVIRONMENT.md](ENVIRONMENT.md).

## Data loading

`fald.data.load_splits` downloads SPECTRE's released tensors and reproduces DiGress's split
exactly (200 graphs, 20% test, 80% of the rest train, `manual_seed(0)`) — but does its own
processing, because DiGress's `process()` appends every graph to `data_list` twice, so its
processed splits are duplicated. That is why its logs say "80 test graphs" for a 40-graph test
split. MMD is invariant to duplicating a sample so their published numbers stand, but it doubles
descriptor cost and misreports set sizes.

## The gate

```
python scripts/calibrate_eval.py --dataset planar
```

Three cases with known answers, verified in this order: `real-vs-real ≈ 0 < train-vs-test <<
ER-vs-test`. The Erdős–Rényi control is **density-matched** per graph, so passing it cannot be
explained by MMD merely noticing a different mean degree.

Result (seed 0, Planar, train 128 / val 32 / test 40, all training graphs planar):

| row | degree | clustering | spectral | wavelet | orbit | Ratio |
|---|---|---|---|---|---|---|
| train vs test (floor) | 1.42e-05 | 1.65e-02 | 3.98e-03 | 6.01e-04 | 1.36e-04 | 1.00 |
| train[:h] vs train[h:] | 9.56e-05 | 3.09e-02 | 3.22e-03 | 1.03e-03 | 1.98e-03 | — |
| ER (matched) vs test | 5.50e-02 | 3.66e-01 | 8.44e-02 | 3.69e-01 | 1.45e+00 | 3035.21 |

The floor row is the same order of magnitude as SPECTRE Table 1's Planar training row
(degree ~2e-4, clustering ~3.1e-2, spectral ~5e-3). ER is 3035× the floor on average.

V.U.N. sanity: training graphs score `valid=1.0, unique=1.0, novel=0.0` (correct — they *are*
the training set, so nothing is novel), density-matched ER scores `valid=0.0, novel=1.0`
(correct — random graphs of this density are essentially never planar).

The gate asserts SPECTRE's order of magnitude for degree, clustering and spectral only. Orbit
and wavelet are excluded from that particular check because we have no published training-row
value for either; they are still covered by the real-vs-real and ER assertions, which require no
external reference. An earlier draft asserted an invented orbit target of 6e-3 and "failed" at
1.36e-04, which was a bad test, not a bad measurement.

Runtime is ~12 s for the full five-metric calibration.
