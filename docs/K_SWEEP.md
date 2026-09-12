# The frequency cutoff gradient

The proposal called a sweep over k the key experiment; the paper reported a single k on a
single dataset. This is that sweep: 6 bands x 5 values of k, seed 0, 200 epochs, ~3.3 GPU-hours
on a Colab T4.

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

Quantitatively, over the 24 non-spike cells the mean Ratio is 327.50 with sd 17.78. The
unconditioned baseline of 335.74 sits **+0.46 sd inside that spread**, so the flat region
really is the baseline. The two minima sit **-11.9 sd** (low k=8) and **-17.8 sd**
(high k=32) below it. Effects that large are not seed noise: no plausible run-to-run variance
spans twelve to eighteen standard deviations. Seed confirmation is still worth running, but it
is checking reproducibility, not whether the spikes are real.

### 3. The controls do their job

`gaussian` (pure noise, shape-matched) spans 322-333 and `shuffled` (another graph's real low
band) spans 329-342. Both sit flat on the baseline at every k.

So neither **having a condition** nor **having real spectral statistics** is sufficient. The
signal requires the matching graph's own spectrum, at a particular band and cutoff. That is
exactly what these two arms were added to test, and it is the cleanest evidence in the project
that the conditioning mechanism is doing something real rather than acting as extra capacity.

## What it does to the paper's claim

The paper argues low-frequency conditioning helps. The sweep supports a narrower and more
interesting claim, and complicates the original one:

* The strongest arm is **high**, not low.
* high k=32 is also the **highest-leakage** arm, recovering 98.5% of true edges from the
  condition alone (see [LEAKAGE.md](LEAKAGE.md)).

At k=8 leakage and quality are *anti*-correlated -- low leaks 0.0% and wins, high leaks 49.3%
and loses -- which is what refutes the reviewer's objection. At k=32 they *coincide*. These are
not contradictory; they point to two different mechanisms:

* **low k=8** improves generation while leaking nothing, so it must be supplying coarse global
  structure the model cannot otherwise infer.
* **high k=32** supplies a near-complete specification of the target spectrum (98.5% edge
  recovery), so it is closer to handing the model the answer than to selecting a frequency.

The honest framing is that this project found one genuine frequency effect (low k=8) and one
near-oracle regime (high k=32), and the sweep is what separates them. Reporting high k=32's
10.49 without its 98.5% leakage would be misleading.

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

**Not yet run.** The same sweep on SBM, which the proposal also promised. The leakage half of
that is already done and replicates the Planar pattern.
