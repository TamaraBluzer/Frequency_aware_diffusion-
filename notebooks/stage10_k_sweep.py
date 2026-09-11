"""Stage 10: Frequency cutoff gradient (k-sweep) + leakage join — Colab cell.

Run in Google Colab with a GPU runtime.

What this answers
-----------------
The proposal called a k-sweep over {2,4,8,16,32} the key experiment; the paper reported a
single k on a single dataset. This restores it, and adds the two controls the original design
was missing:

  * gaussian  -- shape-matched pure noise. The floor a real band must beat.
  * shuffled  -- another graph's real low band. Separates "low-frequency structure helps"
                 from "the *matching* condition helps".

It then joins the sweep against `scripts/condition_leakage.py` and plots Ratio against
measured leakage. That is the test of whether "frequency" is a real axis or a proxy for how
much of the target each condition hands over.

ORCA matters
------------
The Ratio averages whichever metrics are available. Without ORCA it averages 4 instead of 5
and lands on a ~2.6x different scale, which is what made Stage 8's numbers incomparable to
Table 1 (see docs/METRIC_SCALES.md). colab_setup.sh builds ORCA; run_k_sweep.py refuses to
start without it.

Runtime
-------
78 runs (6 bands x 5 k x 3 seeds, with `none` collapsed to one run per seed).
At the pilot config (~2 min/run) that is roughly 2.5-3 GPU-hours.
Use SEEDS = [0] first (~50 min) to confirm the shape before committing to all three.
"""

# %% [markdown]
# ## Setup

# %%
import os
import subprocess
import sys

REPO_DIR = "/content/fald"
if not os.path.isdir(REPO_DIR):
    subprocess.check_call(
        ["git", "clone", "https://github.com/TamaraBluzer/Frequency_aware_diffusion-.git", REPO_DIR]
    )
os.chdir(REPO_DIR)
subprocess.check_call(["git", "pull", "--ff-only"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
subprocess.check_call(["bash", "scripts/colab_setup.sh"])

sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## Confirm ORCA, or the sweep is not comparable to Table 1

# %%
from fald.eval import orca

print("ORCA available:", orca.is_available())
assert orca.is_available(), (
    "ORCA is missing: Ratio would average 4 metrics instead of 5 and would not be comparable "
    "to Table 1. Check the g++ build in scripts/colab_setup.sh before continuing."
)

# %% [markdown]
# ## Measure leakage first (CPU, ~1 min)
#
# This is the reference the sweep is joined against, and it needs no GPU.

# %%
subprocess.check_call(
    [sys.executable, "scripts/condition_leakage.py", "--dataset", "planar", "--split", "val"]
)

# %% [markdown]
# ## Run the sweep
#
# Start with `SEEDS = [0]` to see the shape (~50 min), then widen to `[0, 1, 2]`.

# %%
SEEDS = [0]  # widen to [0, 1, 2] once the single-seed shape looks right
BANDS = ["low", "high", "random", "gaussian", "shuffled", "none"]
K_VALUES = [2, 4, 8, 16, 32]
EPOCHS = 200

subprocess.check_call(
    [
        sys.executable, "scripts/run_k_sweep.py",
        "--dataset", "planar",
        "--bands", *BANDS,
        "--k-values", *map(str, K_VALUES),
        "--seeds", *map(str, SEEDS),
        "--epochs", str(EPOCHS),
    ]
)

# %% [markdown]
# ## Analyze: frequency view vs information view

# %%
subprocess.check_call([sys.executable, "scripts/analyze_sweep.py", "--dataset", "planar"])

from IPython.display import Image, display

display(Image("results/figures/k_sweep_planar.png"))

# %% [markdown]
# ## How to read the right-hand panel
#
# * **Points fall on one curve** -> frequency is a proxy for information content. The
#   honest framing becomes "how much the condition reveals", not "which band it came from".
# * **Bands stay separated at equal leakage** -> frequency carries something of its own,
#   and the paper's claim survives with a measured control behind it.
#
# On the validation split, `low` already measures at ~0% leakage while scoring best, and
# `high` measures at 49-98% while scoring worst, so the expected result is separation --
# but the sweep is what turns that from two points into a curve.

# %% [markdown]
# ## Commit results back

# %%
subprocess.check_call(["git", "add", "results/"])
subprocess.check_call(
    ["git", "-c", "user.email=bluzertamara2@gmail.com", "-c", "user.name=TamaraBluzer",
     "commit", "-m", "Stage 10: k-sweep results with leakage join"]
)
print("Committed. Push with your credentials:  !git push")
