#!/usr/bin/env bash
# Reproduces the third_party/digress working tree from scratch.
# third_party/ is gitignored, so the pinned commit + patch file in patches/ are the
# only durable record of how our DiGress checkout differs from upstream.
set -euo pipefail

DIGRESS_COMMIT="780242b8d3e7d78316bb5cf90c639fb0cd4c6079"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIGRESS_DIR="${REPO_ROOT}/third_party/digress"
PATCH_FILE="${REPO_ROOT}/patches/digress-windows-modern-torch.patch"

if [[ -d "${DIGRESS_DIR}" ]]; then
    echo "third_party/digress already exists; remove it first to re-bootstrap." >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/third_party"
git clone https://github.com/cvignac/DiGress.git "${DIGRESS_DIR}"
git -C "${DIGRESS_DIR}" checkout "${DIGRESS_COMMIT}"
git -C "${DIGRESS_DIR}" apply "${PATCH_FILE}"

# ORCA is a C++ orbit counter invoked as a subprocess; DiGress ships only the source.
# Windows CreateProcess appends .exe automatically, so the extension needs no patch.
ORCA_DIR="${DIGRESS_DIR}/src/analysis/orca"
if command -v g++ >/dev/null 2>&1; then
    g++ -O2 -std=c++11 -o "${ORCA_DIR}/orca.exe" "${ORCA_DIR}/orca.cpp"
    echo "Built ORCA."
else
    echo "WARNING: g++ not on PATH; ORCA not built and orbit MMD will be unavailable." >&2
    echo "         conda install -n fald -c conda-forge m2w64-toolchain" >&2
fi

echo "DiGress ready at ${DIGRESS_DIR} (commit ${DIGRESS_COMMIT}, patched)."
