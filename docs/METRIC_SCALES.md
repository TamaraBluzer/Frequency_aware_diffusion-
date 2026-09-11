# The Ratio is not one scale, and orbit dominates it

Two reporting problems, both found by decomposing saved reports (`scripts/report_breakdown.py`).
Neither requires retraining to fix.

## 1. The Erdos-Renyi reference shift is a metric-set change, not seed noise

The ER reference is 334.82 in Table 1 and 161.67 in Stage 8, on the same validation split. A
reviewer flagged this as unexplained and correctly judged that a 2x move is too large for seed
noise.

The cause is in the evaluation setup, not the data. Both `train_adjacency_diffusion.py` and
`sample_end_to_end.py` build their metric list conditionally:

```python
metrics = ["degree", "clustering", "spectral", "wavelet"]
if orca.is_available():
    metrics.append("orbit")
```

The Ratio is an unweighted mean over whichever metrics that produced. **A run on a machine
without ORCA averages four metrics; a run with ORCA averages five, and is on a different
scale.** Nothing in the saved report records which happened.

Orbit is the metric that moves the scale, because its reference floor is by far the smallest:

| Metric | train-vs-test floor |
|---|---|
| degree | 1.4e-05 |
| orbit | 1.4e-04 |
| wavelet | 6.0e-04 |
| spectral | 4.0e-03 |
| clustering | 1.6e-02 |

Dividing by a tiny floor produces a large ratio, so orbit dominates any mean it enters. On the
`none` arm:

| Metric set | Mean ratio |
|---|---|
| 5-metric (with orbit) | 321.21 |
| 4-metric (no orbit) | 125.25 |

That is a 2.56x scale difference, which brackets the observed ER gap of 337.91 / 161.67 =
2.09x.

**Consequence: Stage 8's ratios are not comparable to Table 1's.** Stage 8's reports store only
summary scalars, so the metric list it used cannot be recovered from the file directly; the
`er_ratio` of 161.67 sitting ~2x below the 5-metric ER is the evidence that ORCA was absent
(Stage 8 ran in a Colab session where ORCA was not compiled). Any sentence comparing a Stage 8
ratio to a Table 1 ratio is comparing two different quantities.

**Fix going forward:** record `metrics` in every saved report and refuse to compare across sets.
`audit_comparability` in `scripts/report_breakdown.py` flags this automatically.

## 1b. The notebook-vs-JSON mismatch is different configurations, not overwrites

A reviewer noticed that `notebooks/FALD_Colab.ipynb` and the saved reports disagree for
similar run names -- discrete unconditioned 306.78 in the notebook vs 262.47 in the JSON,
discrete low 30.07 vs 40.01 -- and inferred that Colab runs had overwritten each other's files.

They are different experiments, not different versions of one experiment
(`scripts/recover_scaled_up.py` re-extracts this from git history):

| Source | Layers | Steps | Epochs | Minutes | uncond | low k=8 |
|---|---|---|---|---|---|---|
| `results/*.json` | 6 | 200 | -- | ~8.5 | 262.47 | 40.01 |
| notebook (pilot) | 6 | 200 | 500 | -- | 306.78 | 30.07 |
| notebook (scaled) | 10 | 500 | 1000 | ~86 | **39.60** | **15.40** |

Neither notebook pair was ever written to `results/`. The scaled-up pair is the project's best
matched comparison and the best Ratio recorded anywhere in it, and the paper omits it.

**Why it matters for the argument.** At the scaled-up configuration the *unconditioned*
baseline improves roughly 8x (327 -> 39.60). Conditioning still helps by -61%, but the
absolute gap narrows from ~220 Ratio points to ~24. The pilot's headline overstates how much
conditioning contributes at a serious training budget, so the scaled-up pair belongs in the
paper -- it is a weaker but far more honest number.

Zero validity persists at every scale, including the 86-minute runs.

Recovered into `results/discrete_scaled_up_planar.json`. Recovered from stdout, so per-metric
ratios and validation losses are unavailable for these runs; the ER of 337.91 matches Table
1's 5-metric scale, so ORCA was present.

## 2. The headline improvement is mostly one metric

The reported `none -> low` improvement is a change in a mean over five metrics with wildly
different scales, so it is worth asking which metric produced it. Decomposing the seed-0 pair:

| Metric | Delta | Share of total improvement |
|---|---|---|
| orbit | -714.11 | **+84.0%** |
| wavelet | -228.59 | +26.9% |
| degree | **+101.81** | **-12.0%** |
| spectral | -5.51 | +0.6% |
| clustering | -3.23 | +0.4% |

(Shares exceed 100% because degree moves the wrong way and offsets the rest.)

Two things follow:

* **Orbit carries the result.** 84% of the improvement is one metric, the same one whose tiny
  floor makes it dominate the mean in the first place.
* **Degree gets worse under conditioning**, by enough to cancel a tenth of the gain. The paper
  reports only the aggregate, so this is invisible.

An independent review computed 79% / 27% / -7% for the same three metrics against the 3-seed
mean; this table is the seed-0 pair. The agreement is close and the conclusion identical.

**What to report instead:** per-metric ratios alongside the aggregate, and a geometric mean,
which is the appropriate summary for quantities spanning orders of magnitude and is far less
sensitive to a single small-floor metric. Both are emitted into
`results/report_breakdown.json`.

This does not overturn the low-band result -- the `low` arm still wins on four of five metrics,
and on validation loss, which is metric-free. But "-51.7% aggregate" overstates how broad the
improvement is, and the degree regression deserves to be stated rather than averaged away.
