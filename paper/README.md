# The paper

`main.tex` is the source for **Frequency or Information? A Diagnostic Study of Spectral
Side-Channel Conditioning in Graph Diffusion**. The local ACL build has **five body pages
and one reference page**. Recheck after any text, figure, or template change.

The manuscript focuses on generation with supplied spectral conditions. The learned-prior
follow-up has been removed from the paper; its code and `results/stage8_results.json` remain
in the repository as historical work, not evidence reported in the current manuscript.
The separate larger-budget discrete comparison and the copying figure remain included.
All reported generation Ratios now use the five-metric evaluation, including ORCA.

## CPU-only verification and build

Run from the repository root, using the project's Python environment:

```bash
python scripts/paper_cpu_checks.py
python scripts/make_paper_figures.py
python -m pytest tests/ -q
tectonic -X compile paper/main.tex -Z search-path=paper/acl-style --keep-logs --reruns 3
python scripts/check_paper_layout.py paper/main.pdf --max-body-pages 5
```

The numerical audit uses the portable JSON snapshot by default: no checkpoint, local `.pt`
file, GPU, ORCA executable, or diffusion training is needed. It also regenerates the lightweight
node-count-only downstream classifier using the existing SBM data-generation code.

On this Windows machine, the Python environment is
`C:\Users\dhalperin\miniconda3\envs\fald`; invoke its `python.exe` or activate it first.
Tectonic is under that environment's `Library/bin`. Set `CUDA_VISIBLE_DEVICES` to an empty
string to hide GPUs from all verification processes, and use `MPLBACKEND=Agg`.
The full test suite also exercises existing CPU model tests and conditionally uses local data.

The local build tools are Tectonic 0.15.0, PyMuPDF 1.26.4 and pytest 8.4.2. The figure script
uses the project's existing Matplotlib/NumPy/NetworkX/PyTorch stack. Tectonic downloads TeX
resources on its first invocation. The vendored ACL style selects `acl_natbib` itself; do not
add a duplicate `bibliographystyle` command. The explicit three reruns avoid Tectonic 0.15's
spurious `.bbl` change-tracking rerun loop on this machine; the PDF checks still verify that
references resolve.

For visual inspection:

```bash
python scripts/check_paper_layout.py paper/main.pdf --render-dir _preview --max-body-pages 5
```

This renders page PNGs and checks page count, unresolved references and text outside page
boundaries. It supplements, rather than replaces, visual inspection of text and plots.

## Figures

`python scripts/make_paper_figures.py` writes vector PDFs:

- `figures/sweep.pdf`: full-width cutoff and recovery plots. Bands have distinct colors and
  markers; controls also have distinct line styles. Hollow points are single runs, stars are
  selected three-seed means, and every uncertainty bar uses sample SD (`ddof=1`). The gray
  baseline band is descriptive SD, not a confidence interval or an equivalence threshold.
- `figures/copying.pdf`: full-width, fixed-index graph comparison. All three panels share one
  layout computed from the reference graph. Shared edges are solid green, non-reference edges
  dashed gray; all counts and F1 scores are computed from the snapshot.
- `figures/downstream.pdf`: supplementary, not included in `main.tex`. It explicitly identifies
  high-frequency conditioning, unverified synthetic labels, and sample-SD error bars. It is not
  evidence of low-frequency augmentation utility.

Figures read their inputs from JSON and fail on missing sources. No raw result aggregates are
silently substituted. Sweep SDs and downstream SDs are recomputed from individual observations,
not taken from the historical population-SD fields.

## Source map

| Paper evidence | Source |
|---|---|
| Table 1: pilot Ratio and validation loss | `results/adjacency_diffusion_planar_{band}_k8{suffix}.json`, four bands and three seeds |
| Pilot bootstrap intervals | `results/frequency_pilot_planar_k8_summary.json` (archived within-protocol analysis) |
| Condition sensitivity | `results/adjacency_diffusion_planar_low_k8_condition_sensitivity.json` |
| Table 2: component Ratios | The twelve pilot reports; recomputed in `results/paper_cpu_checks.json` |
| Figure 1: cutoff scores | `results/k_sweep_planar.json` |
| Table 3 and Figure 1: selected repeats | Individual rows of `results/k_sweep_planar_seedcheck.json`; sample SD recomputed |
| Table 4 and Figure 1: recovery | `results/condition_leakage_planar.json` |
| SBM recovery | `results/condition_leakage_sbm.json`, excluding its defective shuffled control |
| Table 5, Figure 2, pilot connectivity/planarity | `results/paper_sample_snapshot.json` and `results/paper_cpu_checks.json` |
| Discrete budget comparison | `results/discrete_scaled_up_planar.json` (five metrics) |
| Table 6: downstream GNN test accuracies and true-SBM control | `results/stage9_downstream.json` and the split protocol in `scripts/downstream_classifier.py` |
| Node-count-only test accuracies | `results/paper_cpu_checks.json`, regenerated by `scripts/paper_cpu_checks.py` |

`tests/test_paper_cpu_checks.py` verifies all six manuscript tables against their inputs,
packed-graph round trips, tie-aware retrieval, reproducibility of the CPU audit and node-count
baseline, and the sweep's sample-SD bars and readable axis settings.

## Sample provenance and scope

`results/paper_sample_snapshot.json` contains all 384 saved continuous pilot samples (four
bands, three seeds, 32 samples per run) and the 32 ordered validation references. Each graph
stores its upper-triangle adjacency as packed hexadecimal bits, with node count and encoding
specified. Original `.pt` filenames and SHA-256 hashes are recorded. JSON is written with LF
line endings so its recorded SHA-256 survives cross-platform Git checkout.

The snapshot was created using:

```bash
python scripts/paper_cpu_checks.py --capture-samples
```

Only run that option deliberately: it replaces the snapshot with the local sample files and
requires the original Planar data. Ordinary reproduction must use the default snapshot mode.
Raw sample and checkpoint files are unchanged.

The earlier `results/bootstrap_low_k8.json` audit used a different machine's sample file with
the same nominal run name. Its F1 and connectivity values differ. The paper now consistently
uses the snapshot for sample-level claims, rather than mixing the two artifacts. The original
MMD reports did not record sample hashes, so their exact relationship to this snapshot cannot
be reconstructed retrospectively. The pilot and sweep-confirmation batches are also separate
executions; their differing estimates are explicitly identified in the text.

Historical result JSONs, including population SD aggregates, are retained unchanged. The new
CPU report and figure code recompute sample SD. `docs/PROVENANCE.md`,
`results/provenance_audit.json`, `scripts/audit_provenance.py`, `docs/REPORT.md`, and the
`DanielNewWork` draft/patch files describe earlier manuscript versions; they are not the
validation authority for the current manuscript. In particular, the historical provenance
script compares a hard-coded old claim list, not the current LaTeX automatically.

Remaining gaps are explicit: no untouched-test generation scores, no SBM generation sweep,
no saved cutoff-sweep samples/checkpoints/component reports, and no clean downstream label
validation. The downstream classifier uses separately generated test graphs, unlike the
validation-only generation comparisons. These gaps are not repaired by the CPU audit.

## Overleaf

Use the ACL 2023 template:
<https://www.overleaf.com/latex/templates/acl-2023-proceedings-template/qjdgcrdwcnwp>.
Upload `main.tex`, `custom.bib` and the `figures/` folder; the template supplies its own style.
The source accepts either `acl2023.sty` or `acl.sty`. Both template selection and float placement
can change pagination, so check the five-body-page limit after compiling. Keep the figures at
full width and shorten prose before reducing their labels or panels.
