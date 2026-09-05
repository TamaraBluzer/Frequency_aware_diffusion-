# Stage 4 — graph autoencoder: what we found

**Status: blocked on a design decision, not on code.** Both decoder designs were built and
measured. One passes the gate but is degenerate; the other is honest but does not train. There
is no cheap middle ground, and the reason is structural.

## Results

Planar, 128 train / 40 test, 64 nodes, 8.83% of node pairs are edges.

All rows at `d_v=32`, 120 epochs (60 for `pair`, which converges by 20), batch 32, lr 5e-4,
`pos_weight` 10.33, decoder bias initialised to `logit(0.0883)`.

| decoder | node inputs | F1 | edge acc | `Z` eff. rank | Cohen's d | verdict |
|---|---|---|---|---|---|---|
| `pair` (WORKPLAN 3.2 as written) | degree | **1.000** | **1.000** | 1.02 / 32 | **50799** | photocopy — **gate passes** |
| `node_mlp` | degree | 0.194 | 0.574 | 1.04 / 32 | n/a | rank collapse |
| `node_mlp` | degree + 16 random | 0.161 | 0.157 | 10.35 / 32 | n/a | symmetry broken, still at chance |

The `pair` row is the only one that passes `PLAN.md`'s gate (acc > 0.99, F1 ≥ 0.98), and it
passes it by cheating — the diagnostics label the very same run `degenerate-binary-code`. That
single fact is the strongest argument that the gate, not the model, is what needs changing.

Note the `pair` run *also* shows `Z` effective rank 1.02 / 32: when the decoder never needs to
tell nodes apart, the encoder never learns to. That independently corroborates the symmetry
diagnosis below.

(An earlier measurement of the `pair` row gave Cohen's d = 13302 under a weaker `pos_weight` of
3.21 and no bias initialisation. The degeneracy is not an artefact of those settings — fixing
them made it three times more extreme.)

## Why `pair` is degenerate

Each pair has a private lane `A_ij → W_ij → Â_ij` that never mixes with any other pair; there
are as many pair slots out as pairs in; and the transformer's residual connections carry the
input edge indicator straight through. So copying the bit is a global optimum on the shortest
gradient path, and that is what gradient descent finds.

Measured consequences: Cohen's d of **50799** between the edge and non-edge latent clouds (a
"large" effect is 0.8), and F1 of 0.9989 under latent noise of σ=1.0 — noise as large as the
latent's own standard deviation. That is a two-point binary code, not a manifold. Gaussian
diffusion in it would have nowhere to land, and it is 32,768 numbers encoding 2,016 binary
decisions, so it provides no compression either.

`WORKPLAN.md` §3.2 specified exactly one reconstruction objective. LGD's paper lists **five**,
introduced with "to force the encoder to learn meaningful representations" — they knew this trap
existed. Four of the five force information across the node/pair boundary and destroy the private
lane. Our plan kept only the one that permits copying.

## Why `node_mlp` does not train

Removing the per-pair lane makes copying arithmetically impossible, which was the goal. It also
exposes a harder problem the `pair` decoder was hiding.

**A permutation-equivariant encoder must give structurally similar nodes similar latents.** Our
graphs are topology-only (constant node features) and near-regular (planar degrees 3–10, mean
5.5), so with degree as the only node input every latent collapses onto one direction:
effective rank **1.04 of 32**, across-node standard deviation 0.007, mean pairwise distance
0.05. The decoder then has nothing to distinguish pairs with and can only emit a constant.

Two optimization bugs of ours initially masked this, both now fixed and worth keeping:

* `pos_weight` was computed as the *square root* of the inverse class frequency (3.21 instead of
  10.33), leaving the majority class dominant enough that predicting it everywhere was stable.
* The decoder output bias started at 0, so the fastest early loss reduction was to drive the
  bias to the base rate. The model reached that flat region within ~50 steps and stalled with
  vanishing gradients (measured `|grad|` 1.3 → 1e-5, F1 0). `init_output_bias` now starts it at
  `logit(density)` so gradients go to the input-dependent part.

After fixing both, the model learns *something* (F1 0.194 vs exactly 0) but the rank collapse
remains, confirming symmetry rather than optimization is the binding constraint.

Adding 16 random node channels breaks the symmetry (effective rank 10.35 / 32) and still does
not reconstruct: F1 0.161 with precision 0.088, which *is* the base rate — predictions
uncorrelated with the truth. Recovering a Delaunay triangulation from 32-dim endpoint codes
requires the latent to encode something like the 2-D geometry, and 4 layers with this budget
does not get there.

## The leakage problem with the standard fix

LGD's answer to the symmetry problem is positional encodings — their config sets
`posenc_RRWP: enable: True, ksteps: 20`. RRWP is relative random-walk probabilities, i.e.
powers of the normalized adjacency, which are **literally spectral filters**.

That is a direct confound for this project. If the autoencoder input already carries random-walk
spectral structure, the latent carries it too, the `none` arm is no longer spectrum-free, and the
frequency-band study measures "extra explicit spectral conditioning on top of implicit spectral
features" rather than the effect of spectral conditioning. Laplacian-eigenvector PEs would be
worse still, being exactly the quantity under study.

## Options

1. **Tier-0 adjacency diffusion.** Continuous diffusion directly on the dense adjacency, no
   autoencoder, ~150 lines. Sidesteps all of the above: no latent to collapse, no PEs, no
   leakage, and the denoiser sees exactly the conditioning we choose to give it. `WORKPLAN.md`
   §3.5 already specifies this and says **"Do not skip this"** — and `PLAN.md` omits it entirely
   from the ten-stage sequence. Latent diffusion then becomes the ablation `WORKPLAN.md` line 293
   already plans, rather than a prerequisite.
2. **Node latent + random node features.** Does not leak spectral information, and is viable in
   principle because the encoder is not needed at sampling time — only the decoder is. Costs:
   the same graph maps to many latents, broadening the diffusion target, and it does not
   currently reach the gate.
3. **Node latent + RRWP PEs** (LGD-faithful). Most likely to train, but contaminates the
   headline experiment as above. Would need the `none` arm relabelled honestly.
4. **Keep the `pair` latent**, document that it is a recoding rather than a compression, and let
   the Tier-0 comparison quantify what the latent buys (probably nothing).

Recommendation: **option 1**, with the current `pair` and `node_mlp` results kept as a measured
ablation. The research question does not require a latent at all.

## A gate that rewards the wrong thing

`PLAN.md`'s Stage 4 gate is accuracy > 99% and F1 ≥ 0.98. Only the degenerate photocopy meets
it. Perfect reconstruction and a smooth latent are in genuine tension — this is rate–distortion —
so a gate on reconstruction alone will always select the lookup table. If we pursue any latent
design, the gate needs the latent-quality diagnostics (`Cohen's d`, effective rank, noise curve)
as pass/fail conditions alongside reconstruction.

Reproduce any row with:

```
python scripts/train_autoencoder.py --dataset planar --decoder {pair,node_mlp,node_dot} \
    --d-v 32 --node-random-dim {0,16} --epochs 120 --batch-size 32 --lr 5e-4
```
