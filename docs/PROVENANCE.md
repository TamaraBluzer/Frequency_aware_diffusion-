# Provenance audit

A reviewer asked us to "map each table and figure to its result files, configuration, seed, and
code version where available" and to "report component metrics: extract existing degree,
clustering, spectrum, wavelet, and orbit results." This is that map, plus what it turned up.

Everything below is **computed, not transcribed**. `scripts/audit_provenance.py` declares each of
the paper's 141 numeric claims alongside the file and key it should come from, recomputes the
backing value, and writes `results/provenance_audit.json`. Regenerate with:

```bash
python scripts/audit_provenance.py --source both --print
```

It audits the **committed** results by default (`git show HEAD:results/...`), because that is what
a reviewer can check out. `--source both` adds a HEAD-versus-working-tree diff.

**Pinned to:** `git HEAD` = `902532e`, `paper/main.tex` sha256 `ae5c07c5…`, audit run
2026-09-13. `paper/main.tex` is untracked and under active revision, so re-run the script
before trusting the claim list.

**Headline:** of 141 claims, **128 match**, 2 differ only in the last printed digit, 1 is a real
mismatch, and 10 have no backing result file. One further finding dominates all of them: Table 3's
source file is being corrected in the working tree, and the correction changes the paper's central
claim.

---

## 1. Mismatches, in order of consequence

### 1.1 Table 3's `low` row is a sign-convention artifact — the paper's central claim

Not a mismatch against the committed file (the paper matches it exactly), but a mismatch against
what that file is **about to become**. At audit time `scripts/condition_leakage.py` and
`results/condition_leakage_{planar,sbm}.json` are modified-but-uncommitted, and the modification is
a bug fix.

`reconstruct_from_condition()` built the reconstruction under both sign orientations and then chose
between them with `hits = graph.number_of_edges()` and a strict `>`. Both orientations keep exactly
`m` edges by construction, so the comparison was always false, the first iteration always won, and
the per-graph sign selection the docstring promised never ran. Every committed number was measured
at orientation `-1.0`.

| band | k | Table 3 / committed | corrected | Δ (pts) | fixed sign −1 | fixed sign +1 | graphs where +1 wins |
|---|---|---|---|---|---|---|---|
| `low` | 2 | 0.0 | **48.1** | +48.1 | 0.0 | 48.1 | 100% |
| `low` | 4 | 0.0 | **54.0** | +54.0 | 0.0 | 54.0 | 100% |
| `low` | 8 | **0.0** | **72.7** | **+72.7** | 0.0 | 72.7 | 100% |
| `low` | 16 | 0.0 | **78.7** | +78.7 | 0.0 | 78.7 | 100% |
| `low` | 32 | 0.6 | **47.9** | +47.3 | 0.6 | 47.9 | 100% |
| `high` | 2–32 | 21.8 … 98.5 | unchanged | +0.0 | same | 2.5 … 0.0 | 0% |
| `random` | 2–32 | 14.8 … 43.6 | unchanged | +0.0 | same | 4.9 … 0.6 | 0–9% |
| `gaussian` | 2, 8 | 7.9, 8.2 | 9.2, 9.2 | +1.3, +0.9 | 7.9, 8.2 | 9.2, 9.2 | 56%, 59% |
| `shuffled` | 2–32 | 8.3 … 8.8 | unchanged | +0.0 | same | 8.4 … 8.7 | 38–50% |

Seven cells change; five are the whole `low` row.

**The correction is not an artifact of the extra oracle bit.** The corrected rule uses the true edge
set to pick the orientation per graph, which is one more privileged input than the probe had. That
does not explain the change: `orientation_pos_frac` is exactly 1.000 for every `low` cell on both
Planar and SBM, so a single **fixed** global sign already gives `low` 48.1–78.7% on Planar. The old
0.0% was the wrong sign convention, not a per-graph cherry-pick.

What depends on the 0.0%:

- abstract — "recovering $0.0\%$ of the target's edges"
- Table 2 — the `leak` column for `low` $k{=}8$, and the caption "only one of them is leak-free"
- Table 3 — the entire `low` row
- §4.3 — "the objection is refuted in the strongest possible way: leakage is *anti*-correlated with
  quality", and "`low`, $k{=}8$ improves generation while leaking nothing measurable"
- §4.4 — "`low` at $k{=}16$ is already $71.9\%$ connected on $0.0\%$ edge recovery"
- §5 — "a condition a privileged probe cannot use to recover one edge above chance"

Under the corrected probe the ordering at $k{=}8$ is the *same*, not opposite: `low` leaks 72.7% and
wins, `high` leaks 49.3% and loses. `docs/LEAKAGE.md` and `docs/K_SWEEP.md` have already been
rewritten around this; `paper/main.tex` has not. That is the one number in the paper we would stop
the presses for.

### 1.2 §4.5, "from ${\sim}327$ to $39.60$ … the gap narrows from ${\sim}220$ to ${\sim}24$"

`s45.discrete_pilot_baseline`, **MISMATCH**. This sentence quotes three baselines from three
different configurations without saying so.

| quantity | paper | nearest backing value | source |
|---|---|---|---|
| pilot unconditioned | ~327 | **306.78** | `discrete_scaled_up_planar.json` → `runs[unconditioned (pilot scale)]` |
| | | 327.23 | Table 1's **continuous** `none` 3-seed mean — a different diffusion process |
| pilot gap | ~220 | **222.46** | `saved_json_counterparts`: cycles `none` 262.47 − cycles `low` 40.01 (6 layers, 200 steps) |
| | | 276.71 | this file's own `runs` pair: 306.78 − 30.07 |
| scaled-up baseline | 39.60 | 39.60 ✓ | `headline.matched_scaled_up_pair.unconditioned` |
| scaled-up gap | ~24 | 24.20 ✓ | 39.60 − 15.40 |

The "~8×" holds either way (306.78/39.60 = 7.75; 327.23/39.60 = 8.26). The "~220", though, comes
from the 6-layer/200-step cycles pair while "~24" comes from the 10-layer/500-step pair, so the
sentence compares a gap narrowing across two different budget changes at once. The paper inherited
this wording verbatim from the `interpretation` field inside `discrete_scaled_up_planar.json`, so
the file is not wrong — it is the same loose sentence, one level down.

### 1.3 Two display-rounding slips

| claim | paper | file | correctly rounded |
|---|---|---|---|
| §4.2 flat-cell mean | 327.2 | 327.1470 | **327.1** |
| Table 2, `low` $k{=}8$ std | 19.3 | 19.25 | 19.2 or 19.3 (half-up) |

Cosmetic. Noted only so the next person recomputing them does not think they have found something.

### 1.4 Table 1 and Table 2 use different standard deviations

Not flagged by any single claim, because both match their own source — which is the problem.

- **Table 1**'s `± 8.8 / 16.2 / 9.2 / 11.8` are **sample** standard deviations (ddof = 1),
  computed across the three run reports and stored in
  `results/frequency_pilot_planar_k8_summary.json`.
- **Table 2**'s `± 0.6 / 19.3 / 6.2` come from `k_sweep_planar_seedcheck.json`'s `aggregates`
  block, which uses the **population** standard deviation (ddof = 0).

The same three-seed quantity is reported under two conventions two pages apart. Under Table 1's
convention Table 2 would read `low` $k{=}8$ = 141.9 ± **23.6** (not 19.3) and `high` $k{=}32$ =
11.1 ± **0.71** (not 0.58). The paper's prose repeats the smaller one — "std $0.58$ on a mean of
$11.11$" — so the stability claim is stated with the more flattering estimator. Pick one and say
which.

---

## 2. Traceability map

Every row verified by `scripts/audit_provenance.py`; the `claims` array in
`results/provenance_audit.json` carries the per-number detail.

| Paper location | Backing file(s) | Key | Config | Seeds | Breakdown |
|---|---|---|---|---|---|
| Abstract, sweep size / 23 of 25 | `k_sweep_planar.json` | `rows` | 5 bands × 5 k, 200 ep | 0 | ratio-only |
| Abstract, 2.3× / 29.7× | `k_sweep_planar_seedcheck.json` | `aggregates.*.ratio_mean` | k=8 low, k=32 high, k=2 none | 0,1,2 | ratio-only |
| Abstract, 0.0% / 98.5% | `condition_leakage_planar.json` | `rows[*].edge_recovery` | model-free probe, val split | 0 | n/a |
| §3 model (1.83M, T=100, 4 layers, drop 0.1) | any `adjacency_diffusion_planar_*_k8*.json` | `n_parameters`, `config.*` | — | 0 | full |
| §3 calibration (floor 1.00, ER 3035×) | `eval_calibration_planar.json` | `rows.*.ratio` | — | 0 | full |
| §3 bootstrap (1,000 resamples) | `frequency_pilot_planar_k8_summary.json` | `bootstraps` | — | 0,1,2 | full |
| **Table 1** (4 arms × Ratio/loss/validity) | 12 × `adjacency_diffusion_planar_{band}_k8{,_seed1,_seed2}.json` | `evaluation.ratio`, `best.validation_loss`, `evaluation.vun/valid` | k=8, 200 ep, T=100, 4 layers | 0,1,2 | **full** |
| §4.1 bootstrap CIs | `frequency_pilot_planar_k8_summary.json` | `paired_hierarchical_bootstrap.low_minus_*.ci_95` | — | 0,1,2 | full |
| §4.1 condition sensitivity (12.95%) | `adjacency_diffusion_planar_low_k8_condition_sensitivity.json` | `same_noise_edge_edit_fraction` | low k=8 | 0 | n/a |
| **Figure 1(a)**, §4.2 flat surface | `k_sweep_planar.json` | `rows[*].ratio` | one run per cell, 200 ep | 0 | ratio-only |
| **Figure 1(b)** | `k_sweep_planar.json` + `condition_leakage_planar.json` | `ratio` × `edge_recovery` | — | 0 | ratio-only |
| **Table 2** | `k_sweep_planar_seedcheck.json` | `aggregates` | low k=8, high k=32, none k=2 | 0,1,2 | ratio-only |
| **Table 3**, §4.3 | `condition_leakage_planar.json` | `rows[*].edge_recovery` | k ∈ {2,4,8,16,32}, val split | 0 | n/a |
| §4.3 SBM replication | `condition_leakage_sbm.json` | `rows[*].edge_recovery`, `chance_recovery` | — | 0 | n/a |
| §4.4 connected / planar split | `condition_leakage_planar.json` | `connected_frac`, `planar_frac` | — | 0 | n/a |
| §4.5 oracle vs e2e, guidance | `stage8_results.json` | `adjacency_diffusion_training.*`, `end_to_end_sampling.*`, `guidance_scale_sweep.*` | **4-metric, no ORCA** | 0 | ratio-only |
| §4.5 discrete budgets | `discrete_scaled_up_planar.json` | `headline.matched_scaled_up_pair` | 10 layers, 500 steps, 1000 ep | 0 | ratio-only |
| §4.6 / `figures/downstream.pdf` | `stage9_downstream.json` | `results.n{20,40,80}.*.mean` | high k=32, gs 1.5 | 0–4 | n/a |
| §5 metric-set audit | `report_breakdown.json` + `adjacency_diffusion_planar_none_k8.json` | `headline_improvement.*`, `decompositions.*` | — | 0 | full |

`figures/downstream.pdf` is generated by `scripts/make_paper_figures.py` but is not
`\includegraphics`'d in `main.tex`; §4.6 states its numbers in prose instead.

---

## 3. Component metrics

The reviewer asked for degree, clustering, spectrum, wavelet and orbit results. **Only the twelve
$k{=}8$ arms have them.** Every other run in the project kept an aggregate Ratio and nothing else
(§4).

Ratio per metric (lower is better; 1.0 = indistinguishable from real at this sample size).

### Per run

| file | band | seed | degree | clustering | spectral | wavelet | orbit | Ratio | val loss | valid | unique | novel | vun |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `adjacency_diffusion_planar_none_k8.json` | none | 0 | 110.70 | 18.21 | 13.14 | 358.93 | 1105.08 | 321.21 | 0.1302 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_none_k8_seed1.json` | none | 1 | 94.60 | 17.58 | 12.30 | 356.62 | 1134.63 | 323.15 | 0.1216 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_none_k8_seed2.json` | none | 2 | 209.65 | 17.93 | 13.76 | 353.54 | 1091.80 | 337.34 | 0.1452 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_low_k8.json` | low | 0 | 212.51 | 14.98 | 7.63 | 130.35 | 390.98 | 151.29 | 0.0526 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_low_k8_seed1.json` | low | 1 | 191.61 | 14.90 | 7.06 | 129.25 | 388.00 | 146.17 | 0.0496 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_low_k8_seed2.json` | low | 2 | 195.50 | 14.90 | 7.18 | 127.19 | 537.44 | 176.44 | 0.0582 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_high_k8.json` | high | 0 | 157.84 | 20.40 | 13.93 | 361.14 | 1167.47 | 344.16 | 0.0920 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_high_k8_seed1.json` | high | 1 | 171.73 | 20.20 | 12.35 | 367.29 | 1151.35 | 344.58 | 0.0862 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_high_k8_seed2.json` | high | 2 | 166.24 | 18.88 | 12.37 | 353.79 | 1091.07 | 328.47 | 0.1031 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_random_k8.json` | random | 0 | 172.58 | 19.13 | 13.06 | 366.57 | 1128.63 | 339.99 | 0.1266 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_random_k8_seed1.json` | random | 1 | 108.93 | 17.88 | 12.62 | 353.39 | 1100.27 | 318.62 | 0.1175 | 0.000 | 1.000 | 1.000 | 0.000 |
| `adjacency_diffusion_planar_random_k8_seed2.json` | random | 2 | 160.64 | 17.63 | 13.25 | 354.35 | 1058.33 | 320.84 | 0.1408 | 0.000 | 1.000 | 1.000 | 0.000 |

All twelve: `n_generated` = `n_reference` = 32, `k` = 8, 200 epochs, T = 100, 4 layers, 1,833,473
parameters, condition dropout 0.1, Planar validation split, 5-metric scale (ORCA present).

### Per arm — mean ± sample std (ddof = 1), three seeds

| band | degree | clustering | spectral | wavelet | orbit | **Ratio** | val loss |
|---|---|---|---|---|---|---|---|
| `none` | 138.3 ± 62.3 | 17.9 ± 0.3 | 13.1 ± 0.7 | 356.4 ± 2.7 | 1110.5 ± 21.9 | **327.2 ± 8.8** | 0.132 ± 0.012 |
| `low` | 199.9 ± 11.1 | **14.9 ± 0.0** | **7.3 ± 0.3** | **128.9 ± 1.6** | **438.8 ± 85.4** | **158.0 ± 16.2** | **0.053 ± 0.004** |
| `high` | 165.3 ± 7.0 | 19.8 ± 0.8 | 12.9 ± 0.9 | 360.7 ± 6.8 | 1136.6 ± 40.3 | **339.1 ± 9.2** | 0.094 ± 0.009 |
| `random` | 147.4 ± 33.8 | 18.2 ± 0.8 | 13.0 ± 0.3 | 358.1 ± 7.4 | 1095.7 ± 35.4 | **326.5 ± 11.8** | 0.128 ± 0.012 |

Raw MMD, per-arm mean (for anyone who wants the pre-Ratio numbers):

| band | degree | clustering | spectral | wavelet | orbit |
|---|---|---|---|---|---|
| `none` | 0.0310 | 0.4431 | 0.0920 | 0.3728 | 1.5777 |
| `low` | 0.0448 | 0.3694 | 0.0514 | 0.1349 | 0.6234 |
| `high` | 0.0371 | 0.4906 | 0.0907 | 0.3774 | 1.6148 |
| `random` | 0.0331 | 0.4506 | 0.0914 | 0.3746 | 1.5567 |

### What the components say

1. **The paper's "wins on four of five metrics" holds on the three-seed means, not just seed 0.**
   `low` is the best arm on clustering, spectral, wavelet and orbit, and the **worst** on degree
   (199.9 against `none`'s 138.3). Degree is the one metric where conditioning actively hurts.
2. **The other four wins are unanimous across seeds; the degree loss is not.** `low` beats `none`
   on clustering, spectral, wavelet and orbit in 3 of 3 seeds each. On degree it loses in 2 of 3 —
   seed 2 is the exception, and only because `none`'s own degree spikes there (209.65 against 110.70
   and 94.60). Degree is both the metric conditioning hurts and the metric with the widest seed
   spread, so "degree moves the wrong way" is the right call on the mean but rests on two seeds.
3. **Orbit dominates everything.** Its reference floor is ~1e-4, so its Ratio runs 3× the next
   largest component and sets the aggregate. This is the effect §5 already flags; the component
   table quantifies it.
4. **The ER reference is not a constant.** It is 337.91 / 322.65 / 343.89 across seeds 0/1/2 — a
   6.6% spread — even though it depends only on the evaluator and the seed, never on a trained
   model. At n = 32 a meaningful part of every number here is evaluation-side sampling noise.

### Uniqueness and novelty are 1.000 everywhere — verified, and uninformative

Checked across all twelve runs: `vun/unique` is exactly 1.000 in every run, `vun/novel` is exactly
1.000 in every run, `vun/valid` is exactly 0.000 in every run. The distinct-value sets are `{1.0}`,
`{1.0}` and `{0.0}` respectively.

This is not a result. With 32 samples of a 64-node graph drawn from a continuous relaxation, no two
generated graphs collide and none coincides with a training graph, so both metrics are pinned at
their ceiling by construction and carry no signal separating the arms. They should be reported as
"1.000 by construction at this sample size" or dropped. The joint `vun` is pinned at 0.000 purely
by the validity term, so it adds nothing to the validity column either. The paper is already
right not to lean on them — this confirms it and says why.

---

## 4. Component metrics that no longer exist

**The sweep runs kept only `ratio` and `validity`.** `k_sweep_planar.json` and
`k_sweep_planar_seedcheck.json` have no `ratio/degree`, no `ratio/orbit`, no `validation_loss`, no
`validity_diagnosis`. Their own `provenance.caveat` records why:

> The Colab VM was recycled before the JSON reports were downloaded, so per-metric ratios,
> validation losses and validity diagnoses are unavailable. Ratio and validity are transcribed from
> stdout.

The same applies to `discrete_scaled_up_planar.json`, recovered from notebook stdout in commits
`649fc7d` and `2e0ac83`.

**These are not reconstructed anywhere in this audit, and must not be.** Ratio is the mean of five
components; a mean does not determine its terms. Any per-metric split for these runs would be
invented. The samples were never downloaded either, so they cannot be re-scored.

What the loss costs the paper:

- **Figure 1(a) and 1(b)** — every point
- **Table 2** — all three arms
- **§4.2** — the entire flat-surface analysis (23-cell range, mean, spread, control spans)
- **Abstract** — the 2.3× and 29.7× improvements
- **§4.5** — the discrete 39.60 / 15.40 pair

Which is to say: the headline sweep result cannot be decomposed, cross-checked against validation
loss, or re-scored under a different metric set. The paper states this in Limitation (ii); it is
worth knowing how much rests on it.

One partial consolation, verified here: `orca_available: true` is recorded in both sweep files and
the ER reference in `discrete_scaled_up_planar.json` is 337.91, matching Table 1's five-metric
scale. The claim that these numbers are comparable to Table 1 is supported by the ER value even
though the components are gone.

---

## 5. Table 1 versus Table 2

Both report band `low`, k = 8, Planar, 200 epochs, three seeds. They disagree:

| | seed 0 | seed 1 | seed 2 | mean | sd (ddof=1) |
|---|---|---|---|---|---|
| Table 1 (`adjacency_diffusion_planar_low_k8*.json`) | 151.29 | 146.17 | 176.44 | 157.97 | 16.20 |
| Table 2 (`k_sweep_planar_seedcheck.json`) | 116.27 | 146.79 | 162.67 | 141.91 | 23.58 |

### The "low is just the noisiest arm" hypothesis — tested, and supported

**Test 1: the divergence is confined to `low`.** Same nominal config, seed 0, both sets:

| arm | Table 1 | sweep | Δ | Δ% |
|---|---|---|---|---|
| `low` k=8 | 151.29 | 116.27 | −35.02 | **−23.1%** |
| `high` k=8 | 344.16 | 340.55 | −3.61 | −1.0% |
| `random` k=8 | 339.99 | 324.53 | −15.46 | −4.5% |
| `none` | 321.21 (k=8) | 335.74 (k=2) | +14.53 | +4.5% |

`high`, `random` and `none` agree to within 1–5%. Only `low` moves 23%.

**Test 2: `low` is the least stable arm in both sets independently.** Coefficient of variation
across three seeds:

| | `none` | `low` | `high` | `random` |
|---|---|---|---|---|
| Table 1 set | 2.7% | **10.3%** | 2.7% | 3.6% |
| seed-check set | 2.3% (none) | **16.6%** (low k=8) | 6.4% (high k=32) | — |

`low` is ~3× noisier than any other arm, and two independent three-seed sets agree on that.

**Test 2b: and the component metrics say exactly *why*.** Ratio is the unweighted mean of five
components, so each component's covariance with the aggregate is its exact share of the aggregate's
seed variance:

| arm | degree | clustering | spectral | wavelet | orbit |
|---|---|---|---|---|---|
| `none` | +1.375 | −0.000 | +0.012 | −0.058 | −0.329 |
| `low` | −0.026 | −0.000 | −0.001 | −0.017 | **+1.044** |
| `high` | −0.015 | +0.018 | +0.009 | +0.133 | +0.855 |
| `random` | +0.411 | +0.013 | +0.002 | +0.125 | +0.449 |

`low`'s seed-to-seed instability is **entirely orbit**: its orbit component reads 390.98 / 388.00 /
537.44 (a 149-point range) while its other four components are tight (spectral 7.06–7.63, clustering
14.90–14.98, wavelet 127.2–130.3). A fifth of that 149-point range is 29.9, which covers the
aggregate's entire 30.3-point range. `low` is not diffusely noisy — it has one unstable component
with a ~1e-4 reference floor, and that component sets the mean. This is the same metric-scale
problem §5 flags, showing up as run-to-run variance.

**Test 3: the gap is not statistically distinguishable from zero.** The 16.1-point difference of
means is 0.97 standard errors (Welch t = 0.97 on 3.5 df). Pairing by nominal seed gives t = 1.55.
Neither is close to significant. `none` agrees between the sets at t = −0.42.

**Verdict on the hypothesis: supported.** Given `low`'s own seed spread, the two sets report
statistically indistinguishable values, and no code change is needed to explain the difference of
means.

### Code archaeology: nothing in the history explains it

Table 1's files land in `3875e8e` (2026-09-08); the sweep lands in `4c51e47` (2026-09-11). Every
commit in between that touches the training path was read:

| file | change | numerically relevant? |
|---|---|---|
| `fald/data/conditioning.py` | adds `gaussian`/`shuffled` and a derangement donor index | **No.** For `none`/`low`/`high`/`random` the donor index is the identity, the RNG stream (`seed + index`) is unchanged, and the crop/pad rewrite is a no-op on an equal-n dataset. |
| `fald/models/adjacency_diffusion.py` | importlib file-based DiGress loader | **No.** Import mechanics only. |
| `fald/eval/validity.py` | adds `sbm_validity` | **No.** Planar still uses `is_planar`. |
| `scripts/train_adjacency_diffusion.py` | new band choices, validity-diagnosis print, `metrics`/`orca_available` fields, `--results-subdir` | **No.** Model, optimizer, sampler and evaluator untouched. |
| `fald/eval/mmd.py`, `descriptors.py`, `evaluator.py`, `fald/data/spectre.py`, `spectral.py` | **unchanged** between the two commits | metric and split are identical in both sets |

**No code change between the two runs can account for the difference.**

### What is NOT recoverable — say so plainly

The **difference of means** is fully explained by `low`'s seed variance. The **specific seed-0
result** — 151.29 in Table 1 against 116.27 in the sweep, a 23% swing at a nominally identical
configuration *and* a nominally identical seed — **is not recoverable**, and no evidence in this
repository would let anyone determine the cause.

A fixed seed reproduces exactly only on the same machine and library stack, and these two sets did
not run on the same one:

- Table 1's `samples_path` fields point at a Windows OneDrive directory (`C:\Users\dhalperin\…`), so
  that set was produced on a teammate's local Windows machine.
- The sweep declares `provenance.hardware = "Google Colab Tesla T4"`.
- Wall clock corroborates: the same nominal 200-epoch run takes ~2.1 min in the Table 1 set and 7.6
  min in the sweep, a ~3.6× difference.

And nothing was captured that could settle it: **no run report records a git SHA, a torch version, a
CUDA version, a device name or an OS.** The checkpoints are gone, the sweep's samples were never
downloaded, and the sweep's per-metric breakdown was lost with the VM, so the run can be neither
re-scored nor re-executed. Attributing the seed-0 gap to any particular cause would be a guess, and
we are not making one.

**Recommendation.** Report Table 1 and Table 2 as two independent three-seed estimates of the same
arm — 158.0 ± 16.2 and 141.9 ± 19.3 — whose difference is within noise, and stop treating either
seed-0 draw as a reproducible point value. The paper currently says "its seed-0 draw of $116.27$ was
its most favourable, so $141.91$ is the number to quote"; the fuller statement is that the seed-0
draw is not reproducible at all, in either direction.

---

## 6. Numbers with no backing result file

Ten claims resolve to no committed `results/*.json`. None is likely wrong; all are currently
un-auditable.

| Paper location | Number | Where it actually lives |
|---|---|---|
| §4.4 | mean misplaced edges 2.7 | `docs/ZERO_VALIDITY.md` prose only |
| §4.4 | max misplaced edges 6 | `docs/ZERO_VALIDITY.md` prose only |
| §4.4 | zero-misplaced fraction 3.1% | `docs/ZERO_VALIDITY.md` prose only |
| §4.4 | ~177 edges per validation graph | `docs/ZERO_VALIDITY.md` prose only |
| §4.4 | `high` k=32 median distance-to-planarity 4 | `docs/ZERO_VALIDITY.md` prose only |
| §4.4 | `high` k=16 / `low` k=32 need >12 removals | `docs/ZERO_VALIDITY.md` prose only |
| §3 | full spectrum recovers 100.0% | `docs/LEAKAGE.md` prose only |
| §3 | 12 ab-spline wavelet scales | `fald/eval/descriptors.py` docstring |
| §3 | split 128/32/40 | derived by `fald/data/spectre.py:load_splits` at run time |
| §3 | 200 epochs, batch 16, lr 3e-4 | argparse **defaults** in `scripts/train_adjacency_diffusion.py` |

Two of these deserve action rather than a shrug:

- **The §4.4 cluster.** `scripts/report_breakdown.py` has `distance_to_planar()` and
  `diagnose_validity()`, and `train_adjacency_diffusion.py` now calls the latter — but no committed
  run report contains a `validity_diagnosis` block, because every run predates that change. The
  probe reconstructions these numbers describe would have to be regenerated to re-derive them. Six
  paper numbers currently rest on a markdown file.
- **The training budget.** 200 / 16 / 3e-4 are argparse defaults, and the `config` block a run
  writes records model hyperparameters only — not epochs, batch size or learning rate. Any run could
  have overridden them without leaving a trace. This is the cheapest fix in the list: add the three
  values to the report dict.

The full-spectrum 100.0% is about to become traceable: `results/condition_leakage_full_spectrum_{planar,sbm}.json`
exist in the working tree (uncommitted) and carry a `k = -1` row using every non-trivial eigenpair.

---

## 7. Provenance facts

**Generated samples: 1 of 12 present, 0 of 12 committed.** Eleven of the twelve $k{=}8$ runs declare
a `samples_path` under `C:\Users\dhalperin\OneDrive - NVIDIA Corporation\…`, a teammate's machine,
so those `.pt` files are not in this repository. Exactly one exists locally
(`results/adjacency_diffusion_planar_low_k8_samples.pt`) and it is **not version-controlled** —
`.gitignore` excludes `*.pt`, and `git ls-files results/` returns no `.pt` at all. Anything needing
the generated graphs themselves — re-scoring a run, recomputing a component metric, measuring
distance-to-planarity on generated output — cannot be reproduced from this repository.

**Checkpoints: gone.** `checkpoints/` is empty. The models behind every number in the paper no
longer exist, and the validation split that scored the results also selected those checkpoints. No
number can be moved to the test split without retraining from scratch.

**Environment: not captured.** No run report records a git commit, torch version, CUDA version,
device name or OS. `config` holds model hyperparameters only.

**Code version: only a bound.** No run report records the commit it ran under, so "the code version
behind Table 1" is recoverable only as *the commit that introduced Table 1's result files*
(`3875e8e`) — an upper bound on the code's age, not the revision the run actually used. The full
per-file list is in `provenance_audit.json` → `provenance_facts.code_version`:

| result file | introduced by |
|---|---|
| `adjacency_diffusion_planar_*_k8*.json`, `frequency_pilot_planar_k8_summary.json`, `eval_calibration_planar.json` | `3875e8e` 2026-09-08 |
| `stage8_results.json` | `d0d1487` 2026-09-10 |
| `stage9_downstream.json` | `037c545` 2026-09-10 |
| `condition_leakage_planar.json` | `b678ee7` 2026-09-11 |
| `condition_leakage_sbm.json` | `ac97631` 2026-09-11 |
| `report_breakdown.json` | `2b6f081` 2026-09-11 |
| `discrete_scaled_up_planar.json` | `5545cdf` 2026-09-11 |
| `k_sweep_planar.json` | `4c51e47` 2026-09-11 |
| `k_sweep_planar_seedcheck.json` | `ff5f167` 2026-09-12 |

`paper/main.tex` and `scripts/make_paper_figures.py` are tracked (both landed in `902532e`), but at
audit time both carry uncommitted modifications, as do nine files under `docs/` and both figure
PDFs. The text audited here is therefore not itself a committed revision — hence the sha256 pinned
at the top of this document.

**Metric scale.** The $k{=}8$ arms, the discrete arms and the sweep are five-metric (ORCA present,
ER = 337.91). Stage 8's end-to-end numbers are four-metric (ER = 161.67) and are not comparable to
the tables. The paper says so at the point of use, and `report_breakdown.json`'s
`comparability_audit` confirms all fifteen saved reports sit on the five-metric scale.
