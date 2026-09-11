# Why validity is zero

Every continuous-model arm reports 0% validity, and the paper states this without diagnosing
it. `is_planar` is `connected AND planar`, so the single 0.000 cannot say which half failed.
Separating them changes the picture.

## Connectivity is not the problem; planarity is

From the model-free probe (`scripts/condition_leakage.py`), which reconstructs graphs from the
condition alone and therefore isolates the difficulty of the *target* from any model:

| Band | k | Edge recovery | Connected | Planar | Valid |
|---|---|---|---|---|---|
| low | 16 | 0.0% | 71.9% | 0.0% | 0.0% |
| low | 32 | 0.6% | **100.0%** | 0.0% | 0.0% |
| high | 16 | 76.7% | **100.0%** | 0.0% | 0.0% |
| high | 32 | 98.5% | **100.0%** | **3.1%** | 3.1% |

Connectivity is reached easily -- by k=32 every reconstruction is connected, in both bands.
Planarity essentially never is. **Validity is a planarity failure, not a connectivity
failure**, and any explanation has to be about planarity specifically.

## Planarity is far more fragile than the near-perfect recovery suggests

The `high k=32` row is the informative one. Recovery is 98.5%, yet only 3.1% of
reconstructions are planar. Quantifying the gap on the 32 validation graphs (177 edges each on
average):

* misplaced edges at 98.5% recovery: **mean 2.7, max 6**
* reconstructions with **zero** misplaced edges: **3.1%**

Those two numbers are the same 3.1%. **Every reconstruction with even one wrong edge is
non-planar; only the perfect ones are valid.** Getting 174 of 177 edges right is not close --
it is a non-planar graph.

This is a property of planar graphs, not a quirk of the probe. A planar graph on n=64 with 177
edges sits near the 3n-6 = 186 maximum, so it is close to maximally triangulated: almost any
added chord creates a K5 or K3,3 minor. The target class has essentially no error tolerance.

## What this means for the model

The usual explanation for DiGress-style models failing on Planar is missing structural input
features (cycle counts, spectral features of the noisy graph). That may well contribute, and
the continuous model here does lack them while only the discrete pilot adds cycle features.

But the probe shows something stronger and prior to it: **even a decoder handed 98.5% of the
answer produces valid graphs 3.1% of the time.** Zero validity is therefore not, on its own,
evidence that the conditioning failed or that the architecture lacks a particular feature. It
is what near-miss generation looks like on a graph class with near-zero error tolerance.

Two consequences for how results should be read:

1. **Validity is close to a binary test of exactness on this dataset**, not a graded measure of
   quality. Ratio and validity are measuring very different things, and a model can improve
   substantially on Ratio while staying pinned at 0% validity -- which is exactly what the
   `low` arm does.
2. **The counter-evidence against the structural-features fix is already in this repo.** The
   Stage 6.5 discrete pilot *does* add cycle features and still reports zero validity at most
   scales. Adding features to the continuous model is therefore a plausible but unproven fix,
   not a known one.

## What would actually test it

Ordered by cost:

1. **Report the split metrics** (connected %, planar %, components, edges vs 3n-6) on generated
   samples, not just reconstructions. `diagnose_validity` now runs automatically in
   `scripts/train_adjacency_diffusion.py`, so any new run records this.
2. **Measure distance-to-planarity** rather than the boolean: the number of edge removals that
   make a sample planar. `distance_to_planar` implements this (greedy Kuratowski-subgraph
   deletion, verified to give 0 for a grid, 1 for K5 and K3,3, and 3 for K6), and
   `diagnose_validity` now reports it. It already separates arms the boolean cannot:

   | Reconstruction | Planar | Median distance | Beyond 12 removals |
   |---|---|---|---|
   | high k=32 | 3.1% | **4** | 90.6% |
   | high k=16 | 0.0% | -- | 100% |
   | low k=32 | 0.0% | -- | 100% |
   | real validation graphs | 100% | 0 | 0% |

   All three failing rows read 0% planar, but `high k=32` is measurably close while the other
   two are not. That is exactly the progress a 0/1 metric erases.
3. **Then, and only then, add structural features** to the continuous model and check whether
   that distance shrinks. Doing this first risks spending days on a fix whose own precedent in
   this repo is negative.

This is a documented limitation with a measured mechanism and a concrete next step -- not an
unexplained zero.
