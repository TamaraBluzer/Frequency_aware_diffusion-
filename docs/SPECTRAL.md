# Spectral utilities and SignNet (`fald/data/spectral.py`, `fald/models/signnet.py`)

Everything downstream reads the spectrum through these functions, so an off-by-one in the band
indexing or a sign convention error would silently corrupt the headline result instead of
crashing. Hence Stage 3's gate is a set of checks with model-independent known answers.

## Choices, and why

**`L_norm = I − D^{−1/2} A D^{−1/2}`, never `L = D − A`** (WORKPLAN G6). The combinatorial
Laplacian folds degree — a local property — into its low-frequency components, and its
eigenvalues are not comparable across graph sizes. `L_norm` is bounded in `[0, 2]`.

**The trivial eigenpair is always dropped.** `λ₁ = 0` with `u₁ ∝ D^{1/2}1` carries only degree
information, so keeping it would leak a local statistic into every band. Index 0 is never
eligible for selection, which the gate asserts directly.

**Dense `scipy.linalg.eigh`, not a sparse solver.** `n ≤ 200` here, the `high` band needs the
top of the spectrum anyway, and a dense symmetric solve is both faster and more numerically
trustworthy at this size. Observed residual `‖Lu − λu‖∞ ≈ 1.2e-15`.

**All arms are dimension-matched by construction.** Every band returns a `(k,)` eigenvalue
vector and an `(n, k)` eigenvector matrix, so the only difference between `low`, `high`,
`random` and `gaussian` is *which* eigenpairs are selected. This is what makes the frequency
claim falsifiable rather than a statement about parameter count (WORKPLAN G4).

## Bands

| Arm | `C_k` |
|---|---|
| `low` | `(λ₂..λ_{k+1}, u₂..u_{k+1})` — the headline arm |
| `high` | `(λ_{n−k+1}..λ_n, u_{n−k+1}..u_n)` |
| `random` | `k` eigenpairs sampled without replacement from indices `2..n` |
| `gaussian` | `k` random scalars and an `N(0,1)` matrix in `R^{n×k}` — strictest control, identical shape, zero structural information |
| `cluster` | DualDiff-style hard partition, one-hot `(n, K)` via spectral clustering. A *baseline*, not a band: it collapses the spectrum into one integer `K` |
| `none` | empty |

## SignNet

`ψ = ρ([φ(u_j) + φ(−u_j)]_j)`, with `φ` acting on `(u_j[i], λ_j)` per node per eigenvector.

Sign invariance is structural, not learned: `φ(u) + φ(−u)` is symmetric in the sign of `u`, so
flipping any column cannot change the output. The gate measures a deviation of exactly
`0.00e+00` — bitwise identical, not merely within tolerance. SPECTRE's alternative (force the
max-magnitude entry positive) is discontinuous and breaks under eigenvalue multiplicity, which
SBM graphs exhibit by construction.

`φ` never sees a node index, so the encoder is permutation equivariant (measured `3e-08`,
float32 rounding). Outputs are concatenated over `j` rather than summed, because the frequency
index is meaningful and ordered — summing would discard exactly the structure being measured.

`random_sign_flip` is available as training augmentation: a secondary defence so no
*downstream* component quietly learns a sign convention.

## Caching

`cached_eigendecompositions` memoizes to `data/cache/` as compressed `.npz`. The spectrum of a
training graph never changes, so recomputing it every epoch is a large silent cost. The cache
key hashes graph *structure* (Weisfeiler–Lehman hash plus node count), not object identity, so a
changed split misses the cache rather than silently returning the wrong spectra. Verified to
round-trip bit-exactly.

## Gate

```
python scripts/spectral_sanity.py
```

| Check | Result (32 Planar + 32 SBM graphs) |
|---|---|
| `‖Lu − λu‖∞` | 1.2e-15 |
| `λ ∈ [0, 2]` | violated by ≤ 8.9e-16 |
| zero-eigenvalue multiplicity = component count | 0 mismatches |
| `‖L_norm (D^{1/2}1)‖` | 2.2e-16 |
| `u₁ ∝ D^{1/2}1` on connected graphs | deviation 3.3e-16 |
| band indices, low/high ordering, dimension match | pass |
| SignNet under per-column and full sign flips | **0.00e+00** |
| SignNet under node permutation | 3.0e-08 |
| eigendecomposition cache round-trip | exact |

![u2 node colouring](../results/figures/u2_node_coloring.png)

The figure is the one that makes the hypothesis legible. On Planar, `u₂` is a smooth spatial
gradient across the layout. On SBM it splits the graph into its communities — precisely the
global structure we claim low-frequency conditioning supplies.

## Caveat found by the gate: disconnected SBM graphs

1 of 32 sampled SBM training graphs is disconnected (SBM #2 in the figure). This matters more
than the count suggests:

* With `c` components the null space of `L_norm` is `c`-dimensional, so `λ₁ = … = λ_c = 0` and
  `u₁` is an **arbitrary basis vector of that space** rather than `D^{1/2}1`. An earlier version
  of the gate asserted `u₁ ∝ D^{1/2}1` unconditionally and failed at 3.8e-01 on this graph. The
  test was wrong, not the code: `D^{1/2}1` is still *in* the null space (`‖Lv‖ = 1.4e-16`). The
  gate now asserts the invariant that always holds, and the stronger alignment claim only for
  connected graphs.
* Consequently, "drop the first eigenpair" removes only *one* of `c` trivial directions. For a
  disconnected graph the `low` band's leading eigenvectors encode **component membership, not
  community structure** — visible in the figure, where SBM #2's `u₂` puts one entire component
  at ≈0 and shows no within-component structure.

This is a real confound for SBM low-band conditioning, not a cosmetic issue: for those graphs
the conditioning signal answers a different question than it does for connected graphs. Options
when we reach the SBM arms — decide then, and report the choice — are to drop disconnected
graphs from the SBM splits, to drop all `c` zero eigenpairs instead of exactly one, or to keep
them and report the affected fraction. Planar graphs are all connected (0 of 32 disconnected),
so the headline result is unaffected.
