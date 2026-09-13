# Does the condition leak the target graph?

> **Status: this document was wrong. The objection it claimed to refute stands.**
>
> `reconstruct_from_condition` promised to resolve the ranking sign per graph and did not, so
> every number previously published here was measured under one fixed orientation. Corrected,
> the `low` band recovers **72.7%** of the target's edges at k=8, not 0.0%. The external
> reviewer's numbers, which this document explained away as a dataset difference, were right.
>
> The corrected grid has since been regenerated and is committed in
> `results/condition_leakage_{planar,sbm}.json` (`"orientation_rule": "max"`). The frozen
> pre-fix numbers are kept beside it in `*_single_orientation.json`.

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
This document previously treated that caveat as decisive. It was not. Their numbers are ours.

## The measurement

`scripts/condition_leakage.py` reconstructs edges from the condition alone -- no model, no
training, no learned parameters. It scores `U_k diag(lambda_k) U_k^T`, which is exactly the
pair channel `build_condition_tensors` passes to the denoiser, and keeps the *m* highest
pairs where *m* is the true edge count.

Using the true edge count is privileged information, and so is the node count. This is
deliberate: the probe is an **upper bound** on what any perfect decoder could extract from
the side channel. A low number is therefore strong evidence (the band cannot leak much),
while a high number means the band gives the answer away.

That upper-bound reading only holds if the probe is actually taking the better of the two
ranking orientations, which is what the next section is about.

### Probe validation

Two known-answer checks, both of which still pass under the corrected rule:

1. **The full spectrum recovers perfectly.** Given all non-trivial eigenpairs, recovery is
   100.0% on both Planar and SBM, as it must be -- `U diag(lambda) U^T` *is* `L_norm`.
   (`results/condition_leakage_full_spectrum_{planar,sbm}.json`.)
2. **Pure noise scores flat.** The `gaussian` arm is flat across every k on both datasets, with
   no trend in k. A probe hallucinating structure would not be. Its absolute level is TODO --
   see [The chance line](#the-chance-line-is-being-re-derived).

What neither check caught, and what no check in this project was looking for, is a band scoring
reliably *below* the null. That is the next two sections.

## The bug

`reconstruct_from_condition`'s docstring says:

> *Sign is resolved empirically per graph rather than assumed: whichever orientation recovers
> more edges is used, which again biases the probe upward and keeps it an upper bound.*

The code does not do this:

```python
for orientation in (-1.0, 1.0):
    ranked = np.argsort(orientation * scores)[::-1][:n_edges]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(rows[ranked], cols[ranked]))
    hits = graph.number_of_edges()      # <-- always n_edges, in both orientations
    if hits > best_hits:                # <-- strict >, so the second pass never fires
```

`graph` is built with exactly `n_edges` edges by construction, so `hits` is the same number
twice and the strict `>` always keeps the first iteration. **Every committed number is
orientation −1.** The comparison should have scored the reconstruction against the target --
`len(true_edges & rebuilt_edges)` -- not counted the edges it had just put in itself.

## Why the 0.0% should have been caught

Chance recovery on Planar is 8.8% (the density at n=64). The published `low` row read 0.0% at
k=2, 4, 8 and 16: **below chance, and not marginally -- exactly zero, across 32 graphs, at
four separate cutoffs.**

A condition carrying no information about the target scores *at* chance. That is what the
`gaussian` control is for, and it did exactly that (7.9--9.4%). Scoring reliably *below*
chance is a categorically different thing: it means the ranking is informative and pointed the
wrong way. Selecting 177 pairs out of 2,016 and hitting none of the 177 true edges is not
ignorance -- it requires knowing precisely which pairs to avoid. Exactly-zero is the extreme
case of it. The probe was not failing to find edges in the low band. It was finding them and
returning the complement.

**The rule this project should have been applying: a systematically below-chance score on a
probe with a well-defined chance rate is a sign error until proven otherwise. It is never
evidence of absence.** The previous version of this document treated four below-chance cells
as the strongest evidence in the project.

The substantive origin of the error is in the docstring's own reasoning. `L_norm`'s
off-diagonal is `-1/sqrt(d_i d_j)` where an edge exists, so the score is negated before
ranking. That is correct for the full Laplacian and, empirically, for the `high` band. For the
`low` band at these cutoffs it is backwards -- which is precisely why a per-graph orientation
rule was specified, and precisely what its absence cost.

## Corrected numbers

First the diagnostic: the same five cells scored under each fixed orientation separately, which
is what makes the failure legible. Planar validation split (32 graphs, n=64, single-ranking
chance 8.8%).

| band, k | orientation −1 (published) | orientation +1 | per-graph max (correct) |
|---|---|---|---|
| low, k=8 | 0.00% | 72.72% | **72.72%** |
| low, k=16 | 0.00% | 78.72% | **78.72%** |
| high, k=8 | 49.27% | 0.26% | **49.27%** |
| high, k=32 | 98.49% | 0.00% | **98.49%** |
| random, k=8 | 19.22% | 3.27% | **19.22%** |

Two things read directly off this table:

1. **The two bands use opposite sign conventions.** `high` ranks edges positively under −1,
   `low` under +1, and each is near-zero under the other's orientation. No single fixed
   convention can measure both bands, which is the entire reason the per-graph rule existed.
2. **`high`'s numbers are unchanged.** 49.27% and 98.49% were already the per-graph maximum,
   so every argument in this project resting on `high k=32`'s 98.5% recovery -- including
   [ZERO_VALIDITY.md](ZERO_VALIDITY.md)'s planarity-fragility result -- is unaffected. Only
   the `low` rows move, and they move from below chance to far above it.

### The full corrected grid

`results/condition_leakage_planar.json`, regenerated under the per-graph-max rule:

| Band | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| low | 48.1% | 54.0% | **72.7%** | **78.7%** | 47.9% |
| high | 21.8% | 31.4% | 49.3% | 76.7% | **98.5%** |
| random | 15.6% | 15.8% | 19.2% | 26.3% | 43.6% |
| gaussian | TODO | TODO | TODO | TODO | TODO |
| shuffled | TODO | TODO | TODO | TODO | TODO |

Three things change for the signal bands:

* **`low` is a high-leakage band at every cutoff**, 48--79%. Every cell of it was previously
  reported at 0.0--0.6%.
* **`low` is not monotone in k.** It rises to a peak at k=16 and falls back to 47.9% at k=32,
  where `high` is still climbing. The two bands are not two ends of one axis.
* **`high` and `random` are unchanged.** −1 was already their per-graph maximum.

### The chance line is being re-derived

8.8% is the null for *one* fixed ranking, and it is **not** the right floor for any row in the
grid above. Taking a per-graph maximum over two orientations is a two-sided estimator with a
higher null, and it is also optimistic per graph: it lets a pure-noise condition pick whichever
of its two rankings happened to score better on *that* graph.

The probe is being re-run with the orientation chosen **per band by majority vote** instead of
per graph, which removes that per-graph optimism and should return the noise arms to the
analytic chance rate. Until it lands:

* **`gaussian` and `shuffled` are TODO.** Do not quote the per-graph-max values for them.
* **The signal-band numbers above may tick down slightly**, since a single per-band orientation
  cannot beat the per-graph maximum on any graph. The substantive picture -- `low` in the
  48--79% range, `high` climbing to 98.5% -- does not depend on which of the two rules is used,
  because both bands score near-zero under their wrong orientation (see the diagnostic table
  above), so the majority vote is near-unanimous.

## The reviewer was right, and this document explained them away

The previous version said: *"They generated their own Planar graphs rather than using the
released split... The disagreement is a genuine dataset difference, not an error on either
side."* That explanation is wrong. Compare:

| Condition | Reviewer | Ours, corrected |
|---|---|---|
| low, k=8 | 73% | 72.7% |
| high, k=8 | 50% | 49.3% |
| random, k=8 | 19% | 19.2% |
| high, k=32 | 99% | 98.5% |

Four cells agreeing to within a point is not two datasets. It is one dataset measured twice,
once with a sign error. The reviewer's own caveat was the hook the explanation hung on, and the
agreement of the other three cells should have overridden it: a dataset difference large enough
to move `low` from 73% to 0.0% could not have left `high` and `random` within a point.

## The objection now stands

At k=8 the corrected ordering is `low` 72.7% > `high` 49.3% > `random` 19.2%, which is **the
same ordering as Table 1's quality ranking**. Leakage and quality are positively ordered at
k=8, not anti-correlated. The band that wins is also the band that leaks most.

This is a live confound for the headline k=8 result and must be described as one. Two points
worth keeping straight:

* The Ratio-side controls still do their job. `gaussian` and `shuffled` sit flat on the
  unconditioned baseline at every k ([K_SWEEP.md](K_SWEEP.md)), so the effect genuinely
  requires the matching graph's own spectrum rather than extra conditioning capacity.
* But "requires the target's own spectrum" is exactly what a leakage account predicts as well.
  Those controls separate *information* from *capacity*. They do not separate *frequency* from
  *information*, and nothing currently in the project does.

## What survives: leakage does not track quality monotonically

**The single comparison that carries this section:**

| low, k | leakage | Ratio, seed 0 (lower better) |
|---|---|---|
| 8 | 72.7% | **116.27** |
| **16** | **78.7%** — *more* | **270.48** — *2.3x worse* |

`low k=16` recovers **more** of the target than `low k=8` does and generates **2.3x worse**. If
the k=8 result were the condition handing the model its answer, k=16 -- which hands over six
points more of it -- could not be the worse generator. That is the one argument standing between
this paper and the reviewer's confound, and it is a direct head-to-head, not an aggregate.

The rest of the band says the same thing less sharply. Recovery rises and then falls while
quality stays flat with one isolated spike:

| low, k | leakage | Ratio, seed 0 |
|---|---|---|
| 2 | 48.1% | 320.92 |
| 4 | 54.0% | 316.37 |
| 8 | **72.7%** | **116.27** |
| 16 | 78.7% | 270.48 |
| 32 | 47.9% | 317.01 |

Leakage is **non-monotone in k** -- it climbs to a peak at k=16 and falls back to 47.9% at
k=32 -- while Ratio sits on the 335.74 unconditioned baseline at four of the five cutoffs. The
two least leaky cells (k=32 at 47.9%, k=2 at 48.1%) generate at 317.01 and 320.92,
indistinguishable from each other and from the leakier middle of the band. Only k=8 departs, and
it is not the leakiest.

A generator reading edges out of its condition would track that first column. Nothing in this
band does.

**This is still not sufficient to establish a mechanism and must not be reported as if it
were.** It is five points, in one band, on one seed, on one dataset, and it is a correlational
argument about a curve with one outlier. It rules out the strictest form of the leakage story
-- that recovery percentage alone determines quality -- and nothing beyond that. It remains
consistent with leakage being necessary but not sufficient; with k=8's condition leaking in a
form the denoiser can exploit where the others' is not; and with the sharp k-tuning in
[K_SWEEP.md](K_SWEEP.md) being a property of *which* edges leak rather than how many, which
this probe does not measure at all.

### The test that would actually settle it

The confound exists because the conditioning set and the scoring reference are the same 32
validation graphs. Generate as now, conditioned on validation spectra, then score the same
samples against the **disjoint test split**. A generator winning by reproducing its own
conditioning targets loses that advantage immediately; a generator that learned something
general keeps it. `--eval-split test` is already implemented ([EVALUATION.md](EVALUATION.md)),
but the checkpoints from the original runs are gone, so this needs samples regenerated first.
Not run.

Note that the `shuffled` arm is *not* this test. Conditioning on another graph's low band
removes the information and lands on baseline, which is consistent with both accounts.

## SBM

The previous claim -- that SBM replicates the anti-leakage pattern, with `low` at or below
3.5% against 9.4% chance -- rested on the same single-orientation numbers, and `low`'s
0.9--3.5% was the same below-chance tell. **That replication claim is withdrawn.** The
corrected grid (`results/condition_leakage_sbm.json`, 32 graphs, n = 55--172):

| Band | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| low | 33.8% | 33.6% | 42.7% | 61.8% | 74.4% |
| high | 23.0% | 35.5% | 51.8% | 70.6% | **87.7%** |
| random | 15.7% | 16.7% | 18.3% | 22.4% | 30.3% |
| gaussian | TODO | TODO | TODO | TODO | TODO |
| shuffled | TODO | TODO | TODO | TODO | TODO |

SBM replicates the *corrected* Planar picture, not the old one: `low` is a high-leakage band
here too, rising monotonically with k and actually **out-leaking `high` at k=2**. So the low
band's leakage is not a quirk of Planar's construction from 2D point sets -- which was the
specific thing the original objection singled out, and which the old numbers were read as
exonerating.

Note that `low` is monotone in k on SBM and non-monotone on Planar. The non-monotonicity the
k=8 defence rests on is a Planar finding, and there is no SBM generation sweep to pair it
against.

### The SBM `shuffled` caveat still applies

On the pre-fix numbers `shuffled` drifted above chance as k grew on SBM, where on Planar it
stayed flat. The values are TODO pending the orientation re-run, but the mechanism is a property
of the dataset and does not depend on them: SBM graphs vary from 55 to 172 nodes, size
correlates strongly with density (r = -0.87), and donors differ from their targets by ~44 nodes
on average, so cropping or padding a donor channel to the target's size leaks *size and density*
even though it carries no information about which specific pairs are edges.

So on SBM the shuffled arm is expected to remain a weaker control than on Planar: still free of
target-specific edge information, but not free of target-specific *scale*. Prefer `gaussian` as
the strict floor on this dataset, and re-check the gap between them once the re-run lands.

## What this does and does not establish

**Does:** the published leakage grid was measured under a single sign convention and is not
usable. Corrected, `low` is a high-leakage band on Planar, and the reviewer's objection to
Table 1 is a live confound rather than a refuted one.

**Does not:** show that the k=8 low-band result *is* leakage. Recovery does not track quality
across the low band -- k=16 leaks more and generates 2.3x worse -- and on the Ratio side the
`gaussian` and `shuffled` arms remain flat on the unconditioned baseline. What the correction
removes is the *certificate*: the project no longer holds a measurement saying the low band
cannot be feeding the model the answer.

**Still true, and untouched by the bug:** `high k=32` recovers 98.5% of edges and produces
valid graphs only 3.1% of the time. Near-perfect information about which pairs are edges is not
sufficient for a valid planar graph. That is an independently interesting negative result about
the sampler, separate from the frequency question -- see
[ZERO_VALIDITY.md](ZERO_VALIDITY.md).
