# DanielNewWork

Additive only. `paper/` and `scripts/` are untouched and byte-identical to
`origin/main`; nothing here is wired into the build.

Two things worth acting on are in here, and they are not the figure:

1. **`results/*.pt` is gitignored, so the paper's copying numbers are not
   reproducible and do not match this machine's sample files.**
2. **Twelve saved sample files exist here**, not one. The audit the paper
   restricts to a single arm runs across all twelve, and it holds in every one.

Details below.

---

## 1. The copying numbers do not reproduce across machines

`.gitignore` line 31 is `*.pt`, so no sample file is in the repository. Each
machine holds its own copy, and the two copies of `low k=8 seed 0` are not the
same graphs. Running the committed `bootstrap_samples.py` functions on this
machine's file:

| quantity | `results/bootstrap_low_k8.json` | here |
|---|---|---|
| matched edge $F_1$ | 0.6529 | **0.6223** |
| matched intersection | 115.44 | **110.25** |
| matched Jaccard | 0.4853 | **0.4521** |
| `connected_frac` | 0.46875 | **0.6875** |

The mismatched (off-diagonal) values agree to within 0.3%, so this is not a
pairing or formula difference — the two files' generated graphs differ. The
formula, the pairing (`sample i` ↔ `val graph i`) and the loader are identical;
I ran *her* `upper_triangle_matrix` and `overlap_matrix` to check.

`connected_frac` is the one that matters for the paper, because §4.4 quotes
`46.9%` connected with a bootstrap interval of `[31.3%, 62.5%]`, and this
machine's file gives `68.75%` — outside that interval. `connected_frac` depends
on nothing but the sample file, so this is not a platform effect like the ORCA
discrepancy. It is two different files with one name.

`results/adjacency_diffusion_planar_low_k8.json` records `samples_path` as a
Windows OneDrive path on this machine, while `bootstrap_low_k8.json` records
`/Users/tamarabluzer/...`. The audit ran against a file other than the one the
report it audits points to.

**This does not touch the conclusion.** Copying replicates on this machine's
file and on eleven others (below). Only the digits move. But any digit quoted
from a `.pt` file should be regenerated on one machine, from files committed or
hashed, before submission.

## 2. The audit covers twelve arms, not one

§4.3 calls itself "1 arm of 12" and names the missing sample files as the
blocker. Twelve are present here: 4 bands × 3 seeds at k=8.

```
python DanielNewWork/audit_copying_all_arms.py   ->  copying_all_arms.json
```

| arm | matched $F_1$ | mismatched | gap | top-1 |
|---|---|---|---|---|
| `low` k=8, seeds 0/1/2 | 0.622 / 0.613 / 0.617 | 0.088 | 0.53 | **32/32** each |
| `high` k=8, seeds 0/1/2 | 0.342 / 0.336 / 0.332 | 0.086 | 0.25 | **32/32** each |
| `none` k=8, seeds 0/1/2 | 0.082 / 0.090 / 0.084 | 0.085 | ~0.00 | 2/32, 0/32, 0/32 |
| `random` k=8, seeds 0/1/2 | 0.105 / 0.111 / 0.109 | 0.085 | 0.02 | 7/32, 6/32, 6/32 |

What this settles, none of which needs retraining:

- **Copying is not a one-run artifact.** Perfect top-1 retrieval in all three
  seeds of both conditioned bands, 6 runs, 192 samples.
- **`high` copies too**, at half the edge overlap but the same perfect
  retrieval. The paper could not say this.
- **`none` is a clean negative control**: matched indistinguishable from
  mismatched, retrieval at chance. The effect requires the condition.
- **`random` sits slightly above chance** (top-1 ~6/32 against 1/32, gap 0.02,
  consistent across seeds). Small, but it is a donor-spectrum arm and should not
  identify the target at all. Worth a look; it may be the same node-count defect
  already found in the SBM `shuffled` row.

The uncomfortable part, stated plainly: at k=8, `low` copies more than `high`
(gap 0.53 against 0.25) *and* generates better. Across bands at fixed k,
copying tracks quality — which cuts against the surviving defence rather than
for it. The non-monotonicity argument is about varying k, and these files cannot
speak to it.

**Still unanswerable.** Every saved file is k=8. Whether k=16 reconstructs less
than k=8 cannot be checked; those samples were never written.

## 3. The figure

```
python DanielNewWork/make_copying_figure.py    ->  figures/copying.pdf
```

A conditioning graph beside its own generated copy, with shared edges in green.
`low k=8` reproduces 119 of the graph's 179 edges and visibly traces its
outline; `none`, which never saw the spectrum, overlaps at the mismatched rate.
Sample index 0 of each arm, fixed before inspection. All panels share one layout
computed on the conditioning graph — that is what makes the comparison legible,
and the caption says so rather than implying each graph was drawn on its own
terms.

`copying_figure_snippet.tex` is the drop-in for §4.3. It quotes only
figure-local counts and defers the 32-graph numbers to the text, so it carries
none of the discrepancy in §1. Needs `cp figures/copying.pdf paper/figures/`.
Note the paper is at exactly 8 content pages, so this costs space it does not
have.

## 4. Superseded

`make_samples_figure.py` and `figures/samples.pdf` were built to argue that
low-frequency conditioning teaches mesh density — `low k=8` lands at 116
triangles against the real graph's 118. That reading is dead: the two panels are
a graph and its own partial copy (119 of 179 edges shared, $F_1$ 0.65 against
0.10 for any other validation graph). Kept only so the superseded figure is
traceable. Use `make_copying_figure.py` instead.

`main_daniel.tex` / `main_daniel.pdf` are a pre-correction pass that argues
`low k=8` is leak-free. Refuted twice over — by the probe fix (72.7% recovery)
and by the copying result. Do not merge. Reusable parts are framing-independent
only: the Related Work / Method / Discussion compression and the
source-naming captions, driven by a 5-page limit. Diff against `paper/main.tex`
at `902532e`, not at `HEAD`, or the edits mix with the corrections.
`paper_README.patch` is the matching unapplied `paper/README.md` diff.
