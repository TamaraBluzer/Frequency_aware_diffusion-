#!/usr/bin/env bash
# Stage 1 setup (see PLAN.md) — run this from inside a Google Colab GPU runtime.
#
# Usage in a Colab cell:
#   !bash scripts/colab_setup.sh
#
# What it does:
#   1. Detects the CUDA build Colab shipped with, so we pin torch/PyG to match
#      (PLAN.md: "Detect GPU and CUDA version before pinning the PyTorch build").
#   2. Clones DiGress into third_party/digress (gitignored — never committed).
#   3. Installs DiGress's Python dependencies plus PyTorch Geometric wheels that
#      match the detected CUDA build.
#   4. Installs this repo's own eval-harness requirements (networkx, pygsp, scipy).
#
# Idempotent: safe to re-run in the same Colab session.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
DIGRESS_DIR="${THIRD_PARTY_DIR}/digress"
DIGRESS_URL="https://github.com/cvignac/DiGress.git"

echo "== Stage 1: environment setup =="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. This script expects a Colab GPU runtime"
  echo "(Runtime > Change runtime type > GPU). Continuing anyway."
else
  echo "-- GPU detected --"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

TORCH_CUDA_TAG="cu121"
if command -v nvcc >/dev/null 2>&1; then
  NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
  echo "-- nvcc reports CUDA ${NVCC_VER} --"
  case "${NVCC_VER}" in
    12.1*|12.2*|12.3*|12.4*) TORCH_CUDA_TAG="cu121" ;;
    12.5*|12.6*|12.*)        TORCH_CUDA_TAG="cu124" ;;
    11.8*)                   TORCH_CUDA_TAG="cu118" ;;
    *)                       echo "Unrecognized CUDA ${NVCC_VER}, defaulting to cu121" ;;
  esac
fi
echo "-- Using torch/PyG wheel tag: ${TORCH_CUDA_TAG} --"

mkdir -p "${THIRD_PARTY_DIR}"

if [ -d "${DIGRESS_DIR}/.git" ]; then
  echo "-- DiGress already cloned at ${DIGRESS_DIR}, pulling latest --"
  git -C "${DIGRESS_DIR}" pull --ff-only
else
  echo "-- Cloning DiGress into ${DIGRESS_DIR} --"
  git clone --depth 1 "${DIGRESS_URL}" "${DIGRESS_DIR}"
fi

echo "-- Installing PyTorch Geometric extension wheels (torch-scatter/sparse/cluster) --"
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "2.3.0")"
pip install --quiet torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_CUDA_TAG}.html" || \
  echo "NOTE: PyG extension wheel install failed — check torch version (${TORCH_VER}) against https://data.pyg.org/whl/ and retry manually."

pip install --quiet torch-geometric

if [ -f "${DIGRESS_DIR}/requirements.txt" ]; then
  echo "-- Installing DiGress requirements --"
  pip install --quiet -r "${DIGRESS_DIR}/requirements.txt" || \
    echo "NOTE: some DiGress requirements failed to install; inspect output above."
fi

echo "-- Installing this repo's eval-harness / spectral-utility requirements --"
pip install --quiet networkx pygsp scipy numpy pandas hydra-core omegaconf pytorch-lightning

echo "-- Installing DiGress package (editable) --"
pip install --quiet -e "${DIGRESS_DIR}"

echo "== Stage 1 setup complete =="
echo "DiGress is at: ${DIGRESS_DIR}"
echo "Next: run the Stage 1 smoke test, e.g."
echo "  cd ${DIGRESS_DIR}/src && python main.py dataset=planar general.name=stage1_smoke train.n_epochs=5"
