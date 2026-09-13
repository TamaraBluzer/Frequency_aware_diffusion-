# DanielNewWork

Parked work from a paper-editing pass that ran against the pre-correction paper.
Nothing here is wired into the build. `paper/` and `scripts/` are untouched and
byte-identical to `origin/main`; this directory is additive and self-contained.

## Why it is parked instead of merged

The pass started from commit `902532e`, before the three commits ending at
`13dbf6f` fixed the orientation-selection bug in the leakage probe. That fix
inverted the paper's headline: `low` `k=8` went from a reported `0.0%` edge
recovery to `72.7%`, which makes it the *highest*-leaking arm at its own cutoff
rather than a leak-free one. `paper/main.tex` now says so directly and withdraws
the old claim.

So `main_daniel.tex` argues a conclusion the data no longer supports. It is kept
for the prose and layout work, not the argument. Do not merge it.

## Contents

| file | state |
|---|---|
| `figures/samples.pdf` | Reusable. Regenerates from committed artifacts. |
| `make_samples_figure.py` | Reusable. Standalone; reproduces the above. |
| `main_daniel.tex` | Framing is wrong. Layout/prose trims may be worth lifting. |
| `main_daniel.pdf` | Build of the above, 5 body pages. |
| `paper_README.patch` | Unapplied diff against the old `paper/README.md`. |

### The figure is the part worth keeping

`paper/main.tex` currently has one figure (`sweep.pdf`) and no qualitative one,
so this fills a real gap. It puts a real Planar validation graph beside the
`none` and `low` `k=8` samples, annotated with edge and triangle counts:

```
real 118 triangles | low k=8 116 | none 15 | high 15 | random 13
```

Two anti-cherry-picking rules, both stated in the caption: sample index 0 from
every arm, and one shared Kamada-Kawai layout, so the real graph is not handed
back the coordinates its triangulation came from.

Regenerate with:

```
python DanielNewWork/make_samples_figure.py
```

It reads only the seed-0 sample tensors, the k=8 arm reports, and the dataset
split — none of which the leakage commits changed, which is why it still runs.

**Caveat to carry into any caption.** The counts are measurements and they
stand. The *reading* does not. Pre-correction, `low` `k=8` landing at 116
triangles against the real graph's 118 looked like evidence that low-frequency
conditioning teaches mesh density. At `72.7%` edge recovery that is confounded:
the condition may simply be carrying the edges, and the triangle match may be
transcription rather than learning. The honest version of this figure shows the
confound instead of claiming a win — which is the same direction
`paper/main.tex` already took with Table 3. Also worth noting on the same slide:
validity is `0%` in every leakage cell but one, so recovering edges is not
recovering graphs.

### On `main_daniel.tex`

Reusable, framing-independent: compression of Related Work, Method and
Discussion; folding Conclusion into Discussion; caption rewrites that name each
number's source file. These were driven by a 5-page body limit.

Not reusable: anything about leakage, the `none` baselines, or the standard
deviations, all of which the later commits revisited. Diff against
`paper/main.tex` at `902532e` rather than at `HEAD` to isolate the edits, since
diffing against `HEAD` mixes them with the corrections.
