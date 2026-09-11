#!/usr/bin/env bash
# Reproduces the third_party/digress working tree from scratch.
# third_party/ is gitignored, so the pinned commit + patch file in patches/ are the
# only durable record of how our DiGress checkout differs from upstream.
set -euo pipefail

DIGRESS_COMMIT="780242b8d3e7d78316bb5cf90c639fb0cd4c6079"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Mirrors fald/paths.py: heavy artifacts live outside the OneDrive-synced repo when
# FALD_WORK_DIR is set, and inside it otherwise.
WORK_DIR="${FALD_WORK_DIR:-${REPO_ROOT}}"
DIGRESS_DIR="${WORK_DIR}/third_party/digress"
PATCH_FILE="${REPO_ROOT}/patches/digress-windows-modern-torch.patch"

if [[ -d "${DIGRESS_DIR}" ]]; then
    echo "${DIGRESS_DIR} already exists; remove it first to re-bootstrap." >&2
    exit 1
fi

mkdir -p "${WORK_DIR}/third_party"
git clone https://github.com/cvignac/DiGress.git "${DIGRESS_DIR}"
git -C "${DIGRESS_DIR}" checkout "${DIGRESS_COMMIT}"
git -C "${DIGRESS_DIR}" apply "${PATCH_FILE}"

# ORCA is a C++ orbit counter invoked as a subprocess; DiGress ships only the source.
# Windows CreateProcess appends .exe automatically, so the extension needs no patch.
ORCA_DIR="${DIGRESS_DIR}/src/analysis/orca"
if command -v g++ >/dev/null 2>&1; then
    if [[ "$(uname)" == "Darwin" ]]; then
        # A partially-installed Command Line Tools can leave usr/include/c++/v1 nearly empty
        # while the SDKs still carry a complete libc++, and clang searches the broken path
        # first -- so <cstdio> is "not found" even though it exists. Pick the newest SDK that
        # actually has headers and point clang at it explicitly.
        SDK=""
        for candidate in /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk; do
            if [[ -d "${candidate}/usr/include/c++/v1" ]] \
               && [[ -n "$(ls -A "${candidate}/usr/include/c++/v1" 2>/dev/null)" ]] \
               && [[ -f "${candidate}/usr/include/c++/v1/cstdio" ]]; then
                SDK="${candidate}"
            fi
        done
        if [[ -n "${SDK}" ]]; then
            clang++ -O2 -std=c++11 -nostdinc++ \
                -isystem "${SDK}/usr/include/c++/v1" -isysroot "${SDK}" \
                -o "${ORCA_DIR}/orca" "${ORCA_DIR}/orca.cpp"
            echo "Built ORCA against ${SDK}."
        else
            echo "WARNING: no macOS SDK with complete libc++ headers found." >&2
            echo "         Try: xcode-select --install" >&2
        fi
    else
        g++ -O2 -std=c++11 -o "${ORCA_DIR}/orca" "${ORCA_DIR}/orca.cpp"
        echo "Built ORCA."
    fi
else
    echo "WARNING: g++ not on PATH; ORCA not built and orbit MMD will be unavailable." >&2
    echo "         conda install -n fald -c conda-forge m2w64-toolchain" >&2
fi

echo "DiGress ready at ${DIGRESS_DIR} (commit ${DIGRESS_COMMIT}, patched)."
