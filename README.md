# Spectral Conditioning for Graph Diffusion

A course project for **ML with Graphs, Tel Aviv University**, developed under the project
name **FALD**. The current study uses direct adjacency diffusion with spectral information as
an additional input.

## Paper

**Frequency or Information? A Diagnostic Study of Spectral Side-Channel Conditioning in Graph Diffusion**

- [Read the paper (PDF)](paper/main.pdf)
- [LaTeX source](paper/main.tex)
- [Build instructions, figure sources and evaluation scope](paper/README.md)

The ACL-format paper has five body pages and one reference page. It includes the frequency
comparison, reconstruction analysis, a larger-budget experiment and a secondary graph
classification experiment.

## What we study

Does giving a graph diffusion model spectral hints improve its output, and how much of that
improvement may come from reconstructing the graph that supplied the hint?

The model denoises the full adjacency matrix. We vary the extra input at
`k = 2, 4, 8, 16, 32`:

| Input | What the model receives |
|---|---|
| `low` | The graph's first `k` nonzero Laplacian eigenpairs |
| `high` | Its last `k` eigenpairs |
| `random` | `k` randomly selected nonzero eigenpairs of the same graph |
| `gaussian` | Random numbers replacing the eigenvalue and eigenvector inputs |
| `shuffled` | Another graph's low-frequency condition |
| `none` | No spectral hint; the model still receives the noisy graph, timestep and graph size |

Band comparisons use the same parameter count at a fixed cutoff. A model-free recovery probe
measures what the condition reveals, and a separate copying check compares generated graphs
with their conditioning graphs.

## Reading the results

- The main generation study uses **Planar validation graphs and oracle conditions**: real
  graph spectra are supplied as hints. These are not untouched-test or independent-generation
  benchmarks.
- Selected low- and high-frequency settings improve distributional scores across repeated
  runs, but also reveal substantial target structure. The paper therefore interprets gains
  alongside recovery and copying, rather than attributing them to frequency alone.
- The copying analysis covers **384 generated graphs from 12 pilot runs** at `k=8`, with an
  accompanying portable snapshot and source-file hashes.
- The secondary SBM experiment asks whether diffusion-generated training examples help a
  separate community-count classifier. It does not demonstrate a low-frequency augmentation
  benefit; the paper explains the labeling and graph-size limitations.

Generation quality uses degree, clustering, spectral, wavelet and orbit MMD. **Ratio is lower
when these statistics are closer to the reference**; a Ratio of 1 matches the empirical
real-data reference, not a statistical indistinguishability test. Validity separately checks
connectivity and planarity. Recovery percentage and edge F1 describe reconstruction and are
not substitutes for generation quality.

## Reproduce the paper analysis on CPU

Use Python 3.10 or newer and a PyTorch build suitable for your platform. A CPU build is
sufficient for the commands below. The project dependencies are declared in
[pyproject.toml](pyproject.toml) and [requirements.txt](requirements.txt).

```bash
python -m pip install -e . -r requirements.txt
python scripts/paper_cpu_checks.py
python scripts/make_paper_figures.py
python -m pytest tests/test_paper_cpu_checks.py -q
```

These commands recompute the reported CPU analysis, rebuild the vector figures, and check all
six manuscript tables against the saved results. They require **no GPU, diffusion checkpoint,
ORCA executable or local sample `.pt` files**. The node-count-only classification baseline is
also regenerated on CPU.

Key outputs:

- [CPU analysis](results/paper_cpu_checks.json): seed statistics, copying, validity and the
  node-count-only baseline.
- [Portable graph snapshot](results/paper_sample_snapshot.json): 384 generated samples and
  32 ordered validation references, with packed adjacency data and original file hashes.
- [Cutoff/recovery figure](paper/figures/sweep.pdf).
- [Aligned graph comparison](paper/figures/copying.pdf).
- [Supplementary classification figure](paper/figures/downstream.pdf).

The default audit reads the versioned snapshot. `--capture-samples` is only for deliberately
replacing that snapshot from local sample files; it is not needed for ordinary reproduction.
The audit does not recreate the original diffusion training runs.

## Build the PDF

Install **Tectonic 0.15.0** separately and make it available on your PATH. PyMuPDF is used for
page and reference checks:

```bash
python -m pip install "PyMuPDF==1.26.4"
tectonic -X compile paper/main.tex -Z search-path=paper/acl-style --keep-logs --reruns 3
python scripts/check_paper_layout.py paper/main.pdf --max-body-pages 5
```

The first Tectonic build downloads its TeX resources. See [paper/README.md](paper/README.md)
for Overleaf instructions, exact local tool versions, visual inspection and the full source
map. Keep plots at full width when editing, and recheck pagination after compiling.

## Training and the full test suite

New generator training requires the DiGress checkout, its optional dependencies, dataset
artifacts and suitable compute. See [environment setup](docs/ENVIRONMENT.md) and the
[Colab notebook](notebooks/FALD_Colab.ipynb); the
[initial smoke notebook](notebooks/stage1_digress_smoke.ipynb) checks the basic setup.
`FALD_WORK_DIR` can keep datasets, checkpoints and the DiGress checkout outside the repository.

With that environment configured, run the full suite:

```bash
python -m pytest tests/ -q
```

The focused paper checks above can run without DiGress; the full suite also exercises model
code and locally available benchmark data. Training instructions do not imply that every
historical experiment is reproducible from its saved summaries; the paper README specifies
what evidence and artifacts are available.

## Repository guide

| Location | Contents |
|---|---|
| `paper/` | Current paper, LaTeX source, figures and reproduction guide |
| `fald/` | Active graph data, conditioning, models, diffusion and evaluation code |
| `scripts/` | Training, analysis, CPU checks and figure generation |
| `results/` | Experiment summaries and portable analysis artifacts |
| `tests/` | Numerical, model and manuscript regression checks |
| `notebooks/` | Colab training and smoke-test workflows |
| `docs/` | Detailed method, evaluation and environment notes |

[PLAN.md](PLAN.md), [WORKPLAN.md](WORKPLAN.md), and earlier drafts record the project's
development. The current paper and [paper/README.md](paper/README.md) define the final reported
scope. Historical experiments, including the learned spectral prior, remain in the repository
but are not all included in the paper.
