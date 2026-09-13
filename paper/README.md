# The paper

`main.pdf` is the compiled ACL-format writeup: **5 pages of body, references on page 6**, which
meets the 5-pages-excluding-references limit. `main.tex` is the source.

## Rebuilding it

The PDF in this directory was built with [Tectonic](https://tectonic-typesetting.github.io)
against the official ACL style files vendored in `acl-style/`:

```bash
cd paper && cp acl-style/acl.sty acl-style/acl_natbib.bst . && tectonic -X compile main.tex
```

Tectonic downloads the TeX packages it needs on first run, so nothing else has to be installed.

## Building it on Overleaf instead

1. Open the ACL 2023 proceedings template:
   <https://www.overleaf.com/latex/templates/acl-2023-proceedings-template/qjdgcrdwcnwp>
   and click **Open as Template**.
2. Upload `main.tex`, `custom.bib` and the `figures/` folder, replacing the template's own
   `custom.bib`. Do not upload `acl-style/` — the template has its own copy.
3. Set `main.tex` as the main document and compile.

`main.tex` opens with
`\IfFileExists{acl2023.sty}{\usepackage{acl2023}}{\usepackage{acl}}`, so it works with either
style file without editing. For a review copy with line numbers, change that to
`\usepackage[review]{acl2023}`.

**Check the page count after compiling there.** The 5-page result was measured against
`acl-style/acl.sty`; the Overleaf template ships `acl2023.sty`, an older revision of the same
style, and small differences in float or caption spacing could push a line or two over. The
layout is tight by design — the body ends about two lines from the bottom of page 5.

## If it does run over

Trim in this order. Each removes roughly a quarter page without losing a result:

1. Cut the second half of the "Two caveats on our own reporting" paragraph (the per-metric
   decomposition), keeping only the geometric-mean sentence.
2. Drop the last sentence of §4.4 (distance-to-planarity) — it is a measurement refinement, not
   a result.
3. Shorten the "Others" paragraph of Related Work to one sentence per paper.

Do **not** cut Table 3 (leakage) or Figure 1 — they carry the paper's central claim.

## Figures

Figure 1 is regenerated from the committed run artifacts by

```bash
python scripts/make_paper_figures.py
```

which writes `paper/figures/sweep.pdf` from `results/*.json`. It contains no hand-typed number,
and the script raises rather than skipping if a source file is missing. It also still emits
`figures/downstream.pdf`; that figure was cut from the paper for space (its numbers are stated in
full in §4.6) and can be reinstated with a `\begin{figure}` block if you free up a column.

## Where every number comes from

| Paper location | Source file |
|---|---|
| §3 calibration (floor, ER $3035\times$) | `results/eval_calibration_planar.json` |
| Table 1 ($k{=}8$ bands, bootstrap CIs) | `results/frequency_pilot_planar_k8_summary.json` |
| §4.1 condition sensitivity (12.95%) | `results/adjacency_diffusion_planar_low_k8_condition_sensitivity.json` |
| Figure 1(a), flat-surface statistics | `results/k_sweep_planar.json` |
| Table 2 (three-seed confirmation) | `results/k_sweep_planar_seedcheck.json` |
| Table 3, Figure 1(b) (edge recovery) | `results/condition_leakage_planar.json` |
| §4.3 SBM replication | `results/condition_leakage_sbm.json` |
| §4.4 connected / planar split, misplaced edges | `results/condition_leakage_planar.json`, `docs/ZERO_VALIDITY.md` |
| §4.5 oracle vs end-to-end, guidance sweep | `results/stage8_results.json` |
| §4.5 discrete budgets (39.60 / 15.40) | `results/discrete_scaled_up_planar.json` |
| §4.6 downstream accuracies | `results/stage9_downstream.json` |
| §5 metric-set and per-metric audit | `results/report_breakdown.json`, `docs/METRIC_SCALES.md` |

Longer writeups of the four reviewer-response investigations are in `docs/LEAKAGE.md`,
`docs/METRIC_SCALES.md`, `docs/K_SWEEP.md` and `docs/ZERO_VALIDITY.md`. `docs/REPORT.md` is the
internal stage-by-stage report; **it predates the leakage/k-sweep work and still leads with the
older framing**, so the paper supersedes it.

## Known gaps, stated in the paper

- Every number is scored on the **validation** split, which also selected checkpoints. The
  original checkpoints did not survive, so test numbers would require retraining.
- The $k$-sweep's Ratio and validity values were transcribed from a recycled Colab session's
  stdout; per-metric breakdowns for those runs are lost. ORCA was verified present beforehand, so
  they are on the same five-metric scale as Table 1.
- Stage 8's Ratios are four-metric (no ORCA) and are **not** comparable to the other tables. The
  paper says so at the point of use.
