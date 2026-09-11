# Does the condition leak the target graph?

## The objection

Each generated graph is conditioned on validation graph *i*'s spectrum, then scored against
those same 32 validation graphs. If the `low` band simply hands the model more of the target
than `high` or `random` do, Table 1's ordering measures **information content**, not
**frequency**, and the paper's central claim collapses.

An external reviewer raised this and reported a supporting probe on freshly generated
SPECTRE-style Planar graphs:

| Condition | Their reported recovery |
|---|---|
| low, k=8 | 73% |
| high, k=8 | 50% |
| random, k=8 | 19% |
| high, k=32 | 99% |

They flagged their own caveat: *"I used freshly generated graphs, not your exact split."*
That caveat turns out to be decisive.

## The measurement

`scripts/condition_leakage.py` reconstructs edges from the condition alone -- no model, no
training, no learned parameters. It scores `U_k diag(lambda_k) U_k^T`, which is exactly the
pair channel `build_condition_tensors` passes to the denoiser, and keeps the *m* highest
pairs where *m* is the true edge count.

Using the true edge count is privileged information, and so is the node count. This is
deliberate: the probe is an **upper bound** on what any perfect decoder could extract from
the side channel. A low number is therefore strong evidence (the band cannot leak much),
while a high number means the band gives the answer away.

Run on the **actual validation split** (`load_splits("planar")["val"]`, 32 graphs, n=64):

| Band | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| low | 0.0% | 0.0% | **0.0%** | 0.0% | 0.6% |
| high | 21.8% | 31.4% | **49.3%** | 76.7% | **98.5%** |
| random | 14.8% | 15.6% | **19.2%** | 26.3% | 43.6% |
| gaussian | 7.9% | 8.5% | 8.2% | 8.7% | 9.4% |

Chance recovery is 8.8% (density of a Planar graph at n=64).

### Probe validation

Three independent checks confirm the probe measures what it claims:

1. **Pure noise scores at chance.** The `gaussian` arm lands at 0.90-1.06x lift across every
   k. A probe that hallucinated structure would score it above chance.
2. **The full spectrum recovers perfectly.** Given all n-1 eigenpairs, recovery is 100.0%,
   as it must be -- `U diag(lambda) U^T` *is* `L_norm`.
3. **The reviewer's anomaly reproduces exactly.** `high k=32` gives 98.5% recovery and 3.1%
   validity here, against their independently-computed 99% and 3%. Two implementations
   agreeing on an unusual number is good evidence both are correct.

`low`'s near-zero score is not a numerical artifact. Its pair channel has mean off-diagonal
magnitude 8.6e-03 (k=8) to 3.7e-02 (k=32), the same order as `high`'s 4.3e-02 to 5.2e-02 --
and the model receives it RMS-normalized (`_normalize_pair_channel`), so absolute scale is
stripped before training regardless.

## The result: leakage is *anti*-correlated with quality

Joining leakage against Table 1 at k=8:

| Band | Leakage | Ratio (lower better) | Val loss |
|---|---|---|---|
| low | **0.0%** | **157.97** | **0.053** |
| high | 49.3% | 339.07 | 0.094 |
| random | 19.2% | 326.48 | 0.128 |
| none | -- | 327.23 | 0.132 |

**The band that leaks the most produces the worst graphs. The band that leaks nothing
produces the best.** If Table 1 were a leakage artifact, this ordering would be reversed.
It is not, so the objection does not hold on this split.

(The direction is what matters. A correlation coefficient over three points is not a
statistic worth quoting.)

## Why the reviewer's probe disagreed

They generated their own Planar graphs rather than using the released split. Planar graphs
built from 2D point positions have low-frequency eigenvectors that encode those positions,
so on *that* construction the low band really does leak geometry. The SPECTRE Planar split
this project uses does not share that property. The disagreement is a genuine dataset
difference, not an error on either side -- and it is a good argument for probing the exact
split rather than a reconstruction of it.

## What this does and does not establish

**Does:** the `low` > `high`/`random` ordering in Table 1 is not explained by the condition
leaking the target graph. `low` leaks less than chance while winning by ~170 Ratio points.

**Does not:** explain *why* low-frequency conditioning helps. The leading account is that the
low band carries coarse global structure (connectivity, community layout) that constrains
generation usefully, while the high band carries fine local detail that identifies specific
edges without describing the shape -- consistent with `high k=32` achieving near-perfect edge
recovery yet only 3.1% validity. Testing that account is what the k-sweep is for.

**Also worth noting:** `high k=32` recovers 98.5% of edges and still produces valid graphs
only 3.1% of the time. Near-perfect information about which pairs are edges is not sufficient
for a valid planar graph. That is an independently interesting negative result about the
sampler, separate from the frequency question.
