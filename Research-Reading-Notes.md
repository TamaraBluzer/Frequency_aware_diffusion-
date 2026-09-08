# Frequency-aware diffusion: reading record

Review date: 5 September 2026. This records source reading and static code inspection, not
reproduced experimental results. **Historical-status correction:** implementation has now
started, and direct adjacency diffusion is the primary experiment. The original requested
reading remains incomplete because DualDiff was inaccessible at review time.

Subsequent literature review found two load-bearing references that must be added to the report:

- **GGSD (ICLR 2025)** already sweeps smallest versus largest eigenpairs on Planar and SBM. Our
  novelty claim is therefore the incremental value of a matched spectral *side channel*, not the
  first frequency-band sensitivity study.
- **LG-Flow (2026)** demonstrates near-lossless, linear-size node latents on Planar using
  adjacency-identifying Laplacian positional encodings and a set-aware decoder. It is the starting
  point for the optional latent transfer ablation.

## Coverage and sources

| Proposal reference | Paper coverage | Repository coverage |
|---|---|---|
| [Latent Graph Diffusion (LGD)](https://arxiv.org/abs/2402.02518) | All 30 pages of arXiv v2, including proofs, references and experimental appendices; selected equations/figures checked visually | [zhouc20/LatentGraphDiffusion](https://github.com/zhouc20/LatentGraphDiffusion): README and tree, core encoder, conditioning, diffusion loss/sampling, synthetic dataset, generic generation trainer and representative configuration |
| [SPECTRE](https://proceedings.mlr.press/v162/martinkus22a.html) | All 21 pages, including appendices and visual sample panels | [KarolisMart/SPECTRE](https://github.com/KarolisMart/SPECTRE): README and tree, planar/SBM datasets and splits, spectral and adjacency generator forward paths, conditioning schedule, training/generation orchestration and evaluation functions |
| [SDMG](https://proceedings.mlr.press/v267/zhu25g.html) | All 21 pages, including proofs, references and appendices; key equations and appendix plots checked visually | [JYZHU03/SDMG](https://github.com/JYZHU03/SDMG): README and tree, node/graph model paths, conditioning, losses, graph training/evaluation, node evaluation/model-selection paths, representative configurations and dependencies |
| [DualDiff](https://openreview.net/forum?id=IZV9k5BGxi) | **Not read in full.** Metadata and indexed snippets only; these are insufficient to assess the method | Advertised repository [Xyhi/DualDiff](https://github.com/Xyhi/DualDiff) returned Not Found through GitHub access |

Repository review covers the relevant implementation paths, not every vendored dependency, every configuration or every line in the repositories. No training, checkpoint execution or experimental reproduction was performed.

The [project repository](https://github.com/TamaraBluzer/Frequency_aware_diffusion-) contained the three-page proposal PDF, which was read previously. Its four references determine the scope above. The attached course brief was also read previously.

## LGD: what to carry into this project

LGD learns a graph autoencoder and then trains diffusion over its continuous representations. It maintains both node latents and dense pairwise edge latents, including absent edges. Its denoiser updates these jointly and supports conditioning. The pretrained first stage is frozen during diffusion training. The paper covers graph generation and prediction; its large-graph prediction examples should not be interpreted as demonstrations of equally large dense graph generation.

The paper's regularization ablations are useful: stronger KL or vector-quantization regularization was not automatically beneficial, and output normalization was effective. Its theory concerns conditional prediction under assumptions; it does not establish that low-frequency conditioning improves topology generation.

**Static implementation findings and implications:**

- `lgd/loader/dataset/synthetic_dataset.py`, `SpectreGraphDataset.process`: the data object is appended before filtering and again after transformation. Accepted graphs are therefore duplicated in this path; a rejected object has already been appended. Fix and verify dataset cardinalities before using it.
- That loader uses a split seed of 0, while SPECTRE uses 1234. Matching dataset names does not ensure matching splits.
- `lgd/model/SyntheticGraphTransformerEncoder.py` explicitly maintains all node pairs. GPU budgeting must account for quadratic edge storage.
- `lgd/ddpm/LGD.py` and `lgd/model/DenoisingTransformer.py` do not supply a finished spectral-conditioning switch. The `pe` pathway is unfinished and is not in the denoiser's allowed conditioning list. An explicit spectral adapter would be required.
- Generic inference carries a batch derived from clean graphs before replacing latent fields with noise. This is not, by itself, proof of leakage; every clean structural field consumed by the denoiser must be traced before declaring an unconditional baseline.
- Generic graph decoding removes self-loops and isolated nodes, and substitutes a node for an empty graph. This changes the sampled node-count distribution. Report raw output quality and any cleanup separately.
- `lgd/train/generic_generation.py` prepares reference graphs from the test loader and evaluates against them during training. Its validation-history logic also warrants repair before use: some branches index an uninitialized history. A clean validation-only selection protocol is needed.
- The molecular configuration is a reference for wiring the two stages, not a ready planar/SBM experiment. Node/edge vocabularies, checkpoints, latent dimensions, batch sizes and structural encodings require deliberate adaptation.

Read paths include `lgd/ddpm/LGD.py`, `lgd/model/DenoisingTransformer.py`, `lgd/model/SyntheticGraphTransformerEncoder.py`, `lgd/loader/dataset/synthetic_dataset.py`, `lgd/train/generic_generation.py`, and `cfg/QM9_unconditional_generation_diffusion_simple.yaml`.

## SPECTRE: closest methodological and evaluation reference

SPECTRE generates eigenvalues, then eigenvectors, then an adjacency matrix using a GAN architecture. Its spectral conditioning uses the normalized Laplacian. The eigenvector generator uses orthogonality-preserving rotations, and the adjacency generator combines spectral information with noise through a powerful graph network. Training mixes real and generated spectral conditions.

The paper already studies **k = 2, 4, 8, 16, 32**, the same sweep proposed for this project. It selects small useful values for planar and SBM graphs. Consequently, repeating that sweep alone is not a new contribution. The project's potential contribution is a controlled frequency-band study inside latent diffusion, subject to comparison with DualDiff and other spectral diffusion work.

**Static implementation findings and implications:**

- `data.py`, `PlanarDataset`: planar graphs use Delaunay triangulations of random points. The standard experiment uses 64 nodes and 200 graphs.
- `data.py`, `SBMDataset`: 2–5 communities, 20–40 nodes per community, within-community probability **0.3**, between-community probability **0.005**. These differ from the probability description in the paper appendix. Use a recorded dataset artifact and an explicit specification rather than copying the prose.
- `GraphDataModule.setup` splits non-QM9 datasets into 64% training, 16% validation and 20% test with seed 1234; 200 graphs become 128/32/40.
- `full_gan.py` distinguishes `all_fake`, `fake_eigvec`, and `fake_adj`. These use different amounts of real spectral information. Results conditioned on a held-out graph's true eigenpairs are a different task from generation using independently sampled conditions.
- `model/ppgn_gan.py` constructs a spectral pairwise input and learns adjacency probabilities; low-rank Laplacian truncation is not itself an adjacency decoder.
- `model/SON_gan.py` preserves orthogonality using matrix exponentials. Deterministic eigenvector sign handling does not resolve arbitrary basis rotations within repeated eigenvalues.
- `util/eval_helper.py` defines planar validity as connectedness plus planarity. Its combined uniqueness/novelty/validity function uses exact isomorphism checks after inexpensive filters. The separate uniqueness function defaults to a heuristic unless `precise=True`.
- Appendix visualizations show why exact novelty alone can miss near-memorization. Include a structural diversity diagnostic in addition to isomorphism-based novelty.

Read paths include `data.py`, generator forward paths in `model/lambda_gan.py`, `model/SON_gan.py`, `model/ppgn_gan.py`, conditioning/training/test paths in `full_gan.py`, and relevant functions in `util/eval_helper.py`.

## SDMG: motivation, with important limits

SDMG studies representation learning by denoising features on a known graph. It combines topology and feature conditions and uses a multi-scale smoothing objective. Its embeddings are evaluated on classification tasks. This supports investigating frequency bias, but it is not evidence that low frequencies alone suffice to generate new adjacency matrices.

**Mathematical observations from reading, not author claims:**

1. Equation (11), page 6, minimizes the Frobenius distance between the normalized Laplacian and a positive-semidefinite low-rank factorization. As written, the optimal unconstrained rank-q approximation retains the **largest** Laplacian eigenvalues, not the smallest ones claimed in the adjacent explanation. For example, approximating diag(0.1, 1.8) with rank one favors retaining 1.8. Do not reuse this objective as a justified low-frequency extractor without resolving the discrepancy.
2. The appendix argues that the response `(1 − λ)^k` attenuates high frequencies across λ in [0, 2]. Its magnitude is not monotone over that interval: it increases again above 1. Additional spectral assumptions are needed for that explanation.

**Static implementation findings and implications:**

- Node and graph experiment folders implement different MSS losses. The node loss uses normalized, squared similarity terms with one minus their product. The graph loss uses a different transformed distance/product formula, and aggregated features are not renormalized at every scale. They should not be treated as identical implementations.
- `NodeExp/models/SDMG.py` applies its matrix-factorization loss only to stored sparse Laplacian coordinates and scales by the squared node count. This is not the full dense Frobenius objective, which also penalizes nonzero predictions at zero entries. The graph model does not use that same loss.
- Noise construction depends on clean-feature statistics and signs; this differs from simply drawing independent standard Gaussian noise. These paths are tailored to representation learning, not standalone graph synthesis.
- Evaluation scripts consult test performance across pretraining epochs; the graph script also initializes its best-accuracy accumulator outside the seed loop. Any new experiment should use validation-only selection and independent per-seed bookkeeping.
- The dependency file includes local build paths and multiple DGL package variants. It is not a portable environment specification as written.

Read paths include both `NodeExp/models/SDMG.py` and `GraphExp/models/SDMG.py`, their `MS_SSIM_loss.py` and conditioning modules, graph `main_graph.py`/`evaluator.py`, node selection/evaluation paths, representative YAMLs and `requirements.txt`.

## Decisions that must precede a revised implementation plan

These are reading-derived requirements, not a finalized project plan:

1. **Finish DualDiff.** Obtain the full PDF and appendices; obtain code if available. No method comparison or novelty claim should assume its contents from its title or abstract.
2. **Define the generation task.** Separate independently sampled conditions, training-donor spectral conditions, and held-out-donor conditional reconstruction. If using empirical conditions, draw them only from training graphs and label that setup accurately.
3. **Specify the spectral object.** Fix normalized versus combinatorial Laplacian, zero-mode handling, disconnected graphs, padding and node counts. For normalized Laplacians, the zero-mode eigenvector of a connected graph is proportional to square-root degree, not generally constant.
4. **Handle spectral ambiguity.** Eigenvectors have sign ambiguity; repeated eigenvalues permit basis rotations. Low/high/random comparisons must use matched dimensions, conditioning capacity, normalization, and a defined random-band or random-feature control. Avoid cutting a repeated eigenspace without documenting it.
5. **Establish a trustworthy baseline.** Repair dataset duplication and evaluation selection, verify frozen autoencoder reconstruction, then audit unconditional inference before adding a spectral condition.
6. **Evaluate beyond spectral similarity.** Include degree, clustering and orbit statistics, spectral statistics, graph validity, exact uniqueness/novelty, node-count distribution and diversity. Spectral MMD alone can favor the condition by construction.
7. **Keep augmentation optional.** A low-data classifier study requires a defensible labeling mechanism for generated graphs and training-only generator fitting. Topology generation does not automatically supply valid class labels.
8. **Check adjacent work before novelty claims.** SDMG cites additional spectral diffusion work, including *Generating Graphs via Spectral Diffusion*. Those papers were not among the four proposal references and have not been read fully in this review.

## Remaining access requirement

OpenReview's DualDiff page/PDF required browser verification; direct PDF/API access was denied. No accessible full author copy was found in the searches performed. The advertised GitHub location returned Not Found, which does not establish whether it is private, unpublished, moved, or unavailable through this connection.

Needed from the user: the **DualDiff full PDF including appendices**, and, if available, a **code archive or accessible repository link**. The three-paper review is complete at the coverage stated above; the four-paper prerequisite remains open.
