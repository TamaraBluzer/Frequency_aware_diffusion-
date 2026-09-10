"""Stage 9: Downstream SBM Community-Count Classification — Colab cell.

Run this entire cell in Google Colab with a T4 GPU runtime.

Steps:
  1. Install dependencies and clone repo
  2. Train two SBM adjacency diffusion models (~7 min each):
     - unconditioned (band=none)
     - spectrally-conditioned (band=high, k=32)
  3. Run downstream classifier experiment:
     - real-only, real+random, real+unconditioned, real+conditioned
     - Training sizes: 20, 40, 80
     - 5 seeds each
  4. Print results table

Expected runtime: ~25-30 minutes on T4.
"""

# %% [markdown]
# ## Setup

# %%
import subprocess, sys, os

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "networkx", "numpy", "scipy"])

# Clone repo if needed
REPO_DIR = "/content/fald"
if not os.path.isdir(REPO_DIR):
    subprocess.check_call(["git", "clone",
        "https://github.com/TamaraBluzer/Frequency_aware_diffusion-.git",
        REPO_DIR])
os.chdir(REPO_DIR)

# Install the package
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

# Clone DiGress for the transformer layer
DIGRESS_DIR = os.path.join(REPO_DIR, "third_party", "digress")
if not os.path.isdir(DIGRESS_DIR):
    os.makedirs(os.path.join(REPO_DIR, "third_party"), exist_ok=True)
    subprocess.check_call(["git", "clone",
        "https://github.com/cvignac/DiGress.git", DIGRESS_DIR])

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, DIGRESS_DIR)

import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Step 1: Train SBM Diffusion Models

# %%
import time
import json
import numpy as np
import networkx as nx
from pathlib import Path

# Train unconditioned SBM diffusion model
print("="*60)
print("Training UNCONDITIONED SBM adjacency diffusion (band=none)")
print("="*60)

cmd_none = [
    sys.executable, "scripts/train_adjacency_diffusion.py",
    "--dataset", "sbm",
    "--band", "none",
    "--k", "32",
    "--epochs", "200",
    "--batch-size", "8",
    "--timesteps", "100",
    "--n-layers", "4",
    "--guidance-scale", "1.0",
    "--no-full-eval",
    "--allow-gate-failure",
]
t0 = time.time()
result = subprocess.run(cmd_none, capture_output=True, text=True)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
print(f"Unconditioned training: {(time.time()-t0)/60:.1f} min")

# %%
# Train conditioned SBM diffusion model (high band, k=32)
print("="*60)
print("Training CONDITIONED SBM adjacency diffusion (band=high, k=32)")
print("="*60)

cmd_cond = [
    sys.executable, "scripts/train_adjacency_diffusion.py",
    "--dataset", "sbm",
    "--band", "high",
    "--k", "32",
    "--epochs", "200",
    "--batch-size", "8",
    "--timesteps", "100",
    "--n-layers", "4",
    "--guidance-scale", "1.5",
    "--no-full-eval",
    "--allow-gate-failure",
]
t0 = time.time()
result = subprocess.run(cmd_cond, capture_output=True, text=True)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
print(f"Conditioned training: {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## Step 2: Run Downstream Classifier Experiment

# %%
print("="*60)
print("Running downstream classifier experiment")
print("="*60)

# Checkpoint paths
ckpt_dir = Path("checkpoints")
ckpt_none = ckpt_dir / "adjacency_diffusion_sbm_none_k32.pt"
ckpt_cond = ckpt_dir / "adjacency_diffusion_sbm_high_k32.pt"

cmd_downstream = [
    sys.executable, "scripts/downstream_classifier.py",
    "--train-sizes", "20", "40", "80",
    "--n-seeds", "5",
    "--classifier-epochs", "100",
    "--cond-band", "high",
    "--cond-k", "32",
    "--guidance-scale", "1.5",
]

if ckpt_none.exists():
    cmd_downstream.extend(["--diffusion-checkpoint-none", str(ckpt_none)])
    print(f"Using unconditioned checkpoint: {ckpt_none}")
else:
    print(f"WARNING: {ckpt_none} not found — skipping unconditioned arm")

if ckpt_cond.exists():
    cmd_downstream.extend(["--diffusion-checkpoint-cond", str(ckpt_cond)])
    print(f"Using conditioned checkpoint: {ckpt_cond}")
else:
    print(f"WARNING: {ckpt_cond} not found — skipping conditioned arm")

t0 = time.time()
result = subprocess.run(cmd_downstream, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
print(f"Classifier experiment: {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## Step 3: Results

# %%
results_path = Path("results/stage9_downstream.json")
if results_path.exists():
    with open(results_path) as f:
        data = json.load(f)

    print("="*60)
    print("FINAL RESULTS: SBM Community-Count Classification (4-way)")
    print("="*60)
    print(f"\nMetric: Test accuracy (mean ± stderr over {data['n_seeds']} seeds)")
    print()

    # Pretty table
    arms = set()
    for n_key in data["results"]:
        arms.update(data["results"][n_key].keys())
    arms = sorted(arms)

    header = f"{'N_train':>8s}"
    for arm in arms:
        header += f"  {arm:>30s}"
    print(header)
    print("-" * len(header))

    for n_key in sorted(data["results"].keys()):
        row = f"{n_key:>8s}"
        for arm in arms:
            if arm in data["results"][n_key]:
                stats = data["results"][n_key][arm]
                row += f"  {stats['mean']:.4f} ± {stats['stderr']:.4f}     "
            else:
                row += f"  {'—':>30s}"
        print(row)

    print()
    print("DONE — Stage 9 downstream experiment complete.")
else:
    print("ERROR: results file not found")
