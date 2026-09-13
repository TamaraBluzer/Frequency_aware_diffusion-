# The frequency cutoff gradient

The proposal called a sweep over k the key experiment; the paper reported a single k on a
single dataset. This is that sweep: 5 conditioning arms x 5 values of k, plus the unconditioned
baseline (26 runs), seed 0, 200 epochs, ~3.3 GPU-hours on a Colab T4.

## Results

Ratio (lower is better). The unconditioned baseline is **335.74**.

| Band | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| low | 320.92 | 316.37 | **116.27** | 270.48 | 317.01 |
| high | 346.80 | 347.89 | 340.55 | 281.23 | **10.49** |
| random | 328.81 | 344.72 | 324.53 | 334.10 | 339.95 |
| gaussian | 325.27 | 332.54 | 322.05 | 331.26 | 330.41 |
| shuffled | 332.82 | 331.01 | 341.98 | 328.79 | 334.89 |

Validity was 0% everywhere except **high k=32, at 3.1%** -- the only arm in the sweep that
produced a single valid planar graph.

## Three things this establishes

### 1. high k=32 is the best configuration in the project

Ratio **10.49**, against the paper's headline pilot of 157.97 and the best number previously
recorded anywhere here, 15.40 from an 86-minute scaled-up discrete run. This took 7.6 minutes.

### 2. The effect is sharply tuned, not a gradient

The paper's framing implies frequency content varies smoothly with the band. It does not. The
surface is flat at ~320-348 -- statistically indistinguishable from the 335.74 baseline --
except at two isolated points, low k=8 and high k=32. Neighbouring values of k at the same band
show nothing: low k=4 is 316.37 and low k=16 is 270.48, while low k=8 is 116.27.

Whatever produces the effect is a property of specific (band, k) pairs, not of "low frequency"
or "high frequency" as such.

Quantitatively, the 23 **conditioned** non-spike cells span **270.5 to 347.9**, mean 327.2,
spread 18.5. The unconditioned baseline of 335.74 falls **inside that range**, so the flat
region simply is the baseline. The two minima fall far outside it: low k=8 at 116.27 is 154.2
Ratio points below the lowest other cell and 219.5 below the baseline; high k=32 at 10.49 is
260.0 below the lowest other cell and 325.3 below the baseline. Whether those two are
reproducible is a question about seeds, and the seed confirmation below answers it directly.

**Two corrections to how this used to be stated.**

*The population.* An earlier version used 24 cells, mean 327.50, spread 17.78. That population
had the `none` baseline folded into it, which makes "the baseline sits inside the spread"
circular -- the baseline cannot be its own reference. The 23-cell numbers above exclude it and
match `results/k_sweep_planar.json`.

*The yardstick.* That earlier version also expressed these distances in standard deviations
(+0.46 sd for the baseline, -11.9 and -17.8 sd for the minima) and argued that "no plausible
run-to-run variance spans twelve to eighteen standard deviations." **That framing is withdrawn
and should not be reintroduced with corrected numbers.** These 23 cells are 23 *different
treatments*, each run once at seed 0. Their spread mixes between-configuration variation with
run-to-run noise, and with no replication there is nothing to separate the two. It is not a
sampling distribution, so a distance measured in its standard deviations is not a significance
statement -- fixing the population makes the descriptive spread cleaner, but it does not make
the sigma yardstick valid. Distances belong in Ratio points; evidence that the minima are not
noise comes from the three-seed re-runs, not from this spread.

### 3. The controls do their job

`gaussian` (pure noise, shape-matched) spans 322-333 and `shuffled` (another graph's real low
band) spans 329-342. Both sit flat on the baseline at every k.

So neither **having a condition** nor **having real spectral statistics** is sufficient. The
signal requires the matching graph's own spectrum, at a particular band and cutoff. That is
exactly what these two arms were added to test, and it is the cleanest evidence in the project
that the conditioning mechanism is doing something real rather than acting as extra capacity.

These controls separate *information* from *capacity*. They do not separate *frequency* from
*information* -- "needs the matching graph's own spectrum" is equally what a leakage account
predicts. See the corrected leakage section below.

## What it does to the paper's claim

The paper argues low-frequency conditioning helps. The sweep supports a narrower and more
interesting claim, and complicates the original one:

* The strongest arm is **high**, not low.
* high k=32 is the **highest-leakage** arm measured, recovering 98.5% of true edges from the
  condition alone (see [LEAKAGE.md](LEAKAGE.md)).

**Corrected: low k=8 is not the leak-free minimum this section used to claim.** The earlier
version of this document said leakage and quality were *anti*-correlated at k=8 -- low leaking
0.0% and winning, high leaking 49.3% and losing -- and that this refuted the reviewer's
objection. That 0.0% was a probe bug: `reconstruct_from_condition` never applied the per-graph
orientation selection its docstring promised, so every low-band cell was measured with the
ranking sign inverted. Corrected, low k=8 recovers **72.7%** of the target's edges, not 0.0%.

So at k=8 leakage and quality are ordered the *same* way, not oppositely: low leaks 72.7% and
wins, high leaks 49.3% and loses, random leaks 19.2% and loses. Both minima now sit high on the
information axis, and neither can be attributed to frequency on the strength of the probe:

* **high k=32** supplies a near-complete specification of the target spectrum (98.5% edge
  recovery), so it is closer to handing the model the answer than to selecting a frequency.
  Unchanged -- this cell's number was already the per-graph maximum.
* **low k=8** improves generation while also leaking heavily. The leakage objection is a live
  confound for it rather than a refuted one.

What keeps low k=8 alive is that leakage does not track quality across this band at all:

| low, k | leakage | Ratio |
|---|---|---|
| 2 | 48.1% | 320.92 |
| 4 | 54.0% | 316.37 |
| 8 | **72.7%** | **116.27** |
| 16 | 78.7% | 270.48 |
| 32 | 47.9% | 317.01 |

Leakage swings 31 points across the band while Ratio sits on the 335.74 baseline at four of the
five cutoffs. The leakiest cell (k=16) generates at 270.48; the two least leaky (k=2, k=32) are
indistinguishable from the middle of the band. Only k=8 departs, and it is not the leakiest. A
model copying edges out of its condition would track that first column, and nothing here does.

That is the surviving argument, and it is *not* sufficient to establish a mechanism: five points
in one band, one seed, one dataset, and a curve with a single outlier. In particular this probe
measures *how many* edges leak, not *which*, so it cannot rule out that k=8 leaks a
differently-useful subset. See
[LEAKAGE.md](LEAKAGE.md#what-survives-leakage-does-not-track-quality-monotonically).

The honest framing is that the sweep found two isolated minima and the probe places both of
them high on the information axis. Reporting high k=32's 10.49 without its 98.5% leakage would
be misleading; so, now, would reporting low k=8's 116.27 as a clean frequency effect.

The corrected leakage grid (5 arms x 5 cutoffs, Planar and SBM) is committed in
`results/condition_leakage_{planar,sbm}.json` under `"orientation_rule": "max"`. Anything quoted
from `*_single_orientation.json` is the pre-fix measurement and should not be used.

## Seed confirmation

Both minima were re-run on seeds 1 and 2 (`results/k_sweep_planar_seedcheck.json`):

| Arm | seed 0 | seed 1 | seed 2 | mean +/- sd | vs baseline |
|---|---|---|---|---|---|
| high k=32 | 10.49 | 10.94 | 11.89 | **11.11 +/- 0.58** | **29.7x** |
| low k=8 | 116.27 | 146.79 | 162.67 | **141.91 +/- 19.25** | 2.3x |
| none | 335.74 | 332.93 | 321.50 | 330.06 +/- 6.16 | -- |

**Both reproduce.** high k=32 is extremely stable (sd 0.58 on a mean of 11.11) and produced
valid planar graphs in every seed: 3.1%, 6.25%, 3.1%. low k=8 is noisier but never comes near
baseline.

One correction to the seed-0 sweep: **116.27 was low k=8's most favourable draw.** The
three-seed mean of 141.91 is the number to report, and it sits much closer to the paper's
existing 157.97 pilot than the single-seed figure suggested.

## Caveats

**Provenance.** The Colab VM was recycled before the JSON reports were downloaded. Ratio and
validity were recovered from the notebook's stored stdout; per-metric ratios, validation losses
and validity diagnoses for these runs are lost. ORCA was verified present before the sweep
started, so these are 5-metric ratios on the same scale as Table 1 (see
[METRIC_SCALES.md](METRIC_SCALES.md)).

**Not yet run.** The generation half of the same sweep on SBM, which the proposal also promised.
The leakage half has been re-run under the corrected orientation rule and does replicate the
Planar pattern -- but the *corrected* pattern, in which `low` is a high-leakage band on both
datasets, not the anti-leakage pattern previously reported ([LEAKAGE.md](LEAKAGE.md#sbm)).
