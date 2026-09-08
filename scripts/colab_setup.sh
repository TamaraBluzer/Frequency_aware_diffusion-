#!/usr/bin/env bash
# Full project setup for Google Colab GPU runtimes.
#
# Usage (in a Colab cell):
#   !bash scripts/colab_setup.sh
#
# What it does:
#   1. Detects the CUDA build Colab shipped with and selects matching torch/PyG wheels.
#   2. Clones DiGress into third_party/digress at the pinned commit and applies the
#      Colab-compatible patch (Linux: no Windows-specific changes needed, but the
#      weights_only, pyemd, graph-tool, and local_rank fixes still apply).
#   3. Builds the ORCA binary for orbit-count MMD (g++ is available on Colab).
#   4. Installs this repo's evaluation/spectral/training dependencies.
#   5. Installs the fald package in editable mode so all imports work.
#
# Idempotent: safe to re-run in the same Colab session.

set -euo pipefail

DIGRESS_COMMIT="780242b8d3e7d78316bb5cf90c639fb0cd4c6079"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
DIGRESS_DIR="${THIRD_PARTY_DIR}/digress"
PATCH_FILE="${REPO_ROOT}/patches/digress-windows-modern-torch.patch"

echo "== FALD Colab setup =="

# --- GPU detection ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. This script expects a Colab GPU runtime"
  echo "(Runtime > Change runtime type > GPU). Continuing anyway."
else
  echo "-- GPU detected --"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# --- Detect CUDA version for PyG wheel selection ---
TORCH_CUDA_TAG="cu121"
if command -v nvcc >/dev/null 2>&1; then
  NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
  echo "-- nvcc reports CUDA ${NVCC_VER} --"
  case "${NVCC_VER}" in
    12.1*|12.2*|12.3*|12.4*) TORCH_CUDA_TAG="cu121" ;;
    12.5*|12.6*|12.7*|12.8*|12.*) TORCH_CUDA_TAG="cu124" ;;
    11.8*)                   TORCH_CUDA_TAG="cu118" ;;
    *)                       echo "Unrecognized CUDA ${NVCC_VER}, defaulting to cu121" ;;
  esac
fi
echo "-- Using PyG wheel tag: ${TORCH_CUDA_TAG} --"

# --- Clone and patch DiGress ---
mkdir -p "${THIRD_PARTY_DIR}"

if [ -d "${DIGRESS_DIR}/.git" ]; then
  echo "-- DiGress already present at ${DIGRESS_DIR}, checking commit --"
  CURRENT="$(git -C "${DIGRESS_DIR}" rev-parse HEAD)"
  if [ "${CURRENT}" != "${DIGRESS_COMMIT}" ]; then
    echo "   Commit mismatch (have ${CURRENT:0:12}, want ${DIGRESS_COMMIT:0:12}); re-cloning."
    rm -rf "${DIGRESS_DIR}"
  else
    echo "   Correct commit, skipping clone."
  fi
fi

if [ ! -d "${DIGRESS_DIR}/.git" ]; then
  echo "-- Cloning DiGress at pinned commit --"
  git clone https://github.com/cvignac/DiGress.git "${DIGRESS_DIR}"
  git -C "${DIGRESS_DIR}" checkout "${DIGRESS_COMMIT}"
  echo "-- Applying patches --"
  git -C "${DIGRESS_DIR}" apply --reject --no-backup "${PATCH_FILE}" 2>&1 || true
  find "${DIGRESS_DIR}" -name '*.rej' -delete 2>/dev/null || true
fi

# --- Build ORCA (C++ orbit counter) ---
ORCA_DIR="${DIGRESS_DIR}/src/analysis/orca"
ORCA_BIN="${ORCA_DIR}/orca"
if [ -x "${ORCA_BIN}" ]; then
  echo "-- ORCA already built --"
elif command -v g++ >/dev/null 2>&1; then
  echo "-- Building ORCA --"
  g++ -O2 -std=c++11 -o "${ORCA_BIN}" "${ORCA_DIR}/orca.cpp"
  echo "   Built at ${ORCA_BIN}"
else
  echo "WARNING: g++ not found; ORCA will be unavailable (orbit MMD disabled)." >&2
fi

# --- Install PyTorch Geometric extension wheels ---
echo "-- Installing PyTorch Geometric extension wheels --"
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "2.3.0")"
pip install --quiet torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_CUDA_TAG}.html" 2>/dev/null || \
  echo "NOTE: Some PyG extension wheels failed. Check torch version (${TORCH_VER}) against https://data.pyg.org/whl/"

pip install --quiet torch-geometric 2>/dev/null || true

# --- Install DiGress requirements ---
if [ -f "${DIGRESS_DIR}/requirements.txt" ]; then
  echo "-- Installing DiGress requirements --"
  pip install --quiet -r "${DIGRESS_DIR}/requirements.txt" 2>/dev/null || \
    echo "NOTE: Some DiGress requirements failed; inspect output above."
fi

# --- Install DiGress editable ---
echo "-- Installing DiGress package (editable) --"
pip install --quiet -e "${DIGRESS_DIR}"

# --- Install this project (fald) as an editable package ---
echo "-- Installing FALD package (editable) --"
pip install --quiet -e "${REPO_ROOT}"

# Also install graph-tool on Linux (available via apt on Colab)
if [ "$(uname)" = "Linux" ]; then
  if python -c "import graph_tool" 2>/dev/null; then
    echo "-- graph-tool already available --"
  else
    echo "-- Installing graph-tool (for SBM validity) --"
    pip install --quiet graph-tool 2>/dev/null || \
      echo "NOTE: graph-tool pip install failed. SBM validity will be unavailable."
  fi
fi

# --- Install sklearn (for spectral clustering) ---
pip install --quiet scikit-learn 2>/dev/null || true

echo ""
echo "== Setup complete =="
echo "DiGress: ${DIGRESS_DIR} (commit ${DIGRESS_COMMIT:0:12}, patched)"
echo "ORCA:    $([ -x "${ORCA_BIN}" ] && echo 'available' || echo 'unavailable')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
