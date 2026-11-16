#!/bin/bash
#  PySTARC - Clean install
#  Usage:  bash install_PySTARC.sh   (Linux host with an NVIDIA GPU, stops otherwise)
#  This script:
#    1. Deactivates any active conda env
#    2. Removes existing PySTARC env (if any)
#    3. Creates a fresh PySTARC env
#    4. Installs all dependencies (conda + pip; CuPy with the CUDA toolkit from PyPI)
#    5. Installs PySTARC from wheel
#    6. Runs tests to verify
set -e
ENV_NAME="PySTARC"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEEL="$SCRIPT_DIR/dist/pystarc-1.1.0-py3-none-any.whl"

# --- GPU check ---
# PySTARC runs on NVIDIA GPUs, so the install stops here, before anything is
# removed or installed, when no NVIDIA GPU is visible (macOS, a login node, a job
# without a GPU) or when the driver is older than CUDA 12. cupy-cuda12x[ctk]
# brings the CUDA 12 runtime and NVRTC from PyPI, so no CUDA module has to be loaded.
OS_KIND="$(uname -s)"
if [ "$OS_KIND" != "Linux" ] || ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: no NVIDIA GPU visible on $(uname -n) ($OS_KIND). PySTARC needs an NVIDIA GPU with CUDA."
    echo "  Run this script on a GPU node. Nothing was removed or installed."
    exit 1
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\).*/\1/p' | head -1)"
if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -lt 12 ]; then
    echo "ERROR: the NVIDIA driver on $(uname -n) supports CUDA $CUDA_MAJOR. PySTARC needs a driver for CUDA 12 or newer."
    echo "  Nothing was removed or installed."
    exit 1
fi
CUPY_PKG="cupy-cuda12x[ctk]>=14"   # the ctk extra exists from CuPy 14 on

echo ""
echo "  PySTARC - Clean install"
echo "  Date: $(date)"
echo "  GPU: $GPU_NAME    CuPy: $CUPY_PKG"
# 1. Deactivate current env
echo ""
echo "[1/7] Deactivating current conda environment."
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
echo " Deactivated"
# 2. Remove existing PySTARC env (if found)
echo ""
echo "[2/7] Removing existing '$ENV_NAME' environment."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
# Also remove the directory if it lingers
rm -rf "$HOME/.conda/envs/$ENV_NAME" 2>/dev/null || true
echo "Clean slate"
# 3. Create fresh env
echo ""
echo "[3/7] Creating fresh conda env: $ENV_NAME (Python $PYTHON_VERSION)."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
# Pin to the env's interpreter by absolute path so PATH/activation quirks
# (Flatiron module Python sitting ahead on PATH) can't redirect us.
ENV_PY="$CONDA_PREFIX/bin/python"
echo "Created and activated: $("$ENV_PY" --version)"
echo "Using interpreter: $ENV_PY"
"$ENV_PY" -m pip --version
# 4. Conda dependencies
echo ""
echo "[4/7] Installing conda dependencies (ambertools 24.8, apbs)"
# AmberTools pinned to 24.8, the version PySTARC was validated with. AmberTools 26.0
# bundles 'proprep', whose metadata pins numpy<2, biopython<1.86 and pyyaml (pip conflicts).
conda install -c conda-forge "numpy>=2.0,<2.6" "ambertools=24.8" apbs rdkit openbabel -y
echo "ambertools and apbs installed"
# 5. GPU + pip dependencies
echo ""
echo "[5/7] Installing pip dependencies ($CUPY_PKG, matplotlib, pytest)."
"$ENV_PY" -m pip install "$CUPY_PKG" matplotlib pytest-cov pytest-sugar
echo "pip dependencies installed"
# 6. Install PySTARC
echo ""
echo "[6/7] Installing PySTARC from current source."
"$ENV_PY" -m pip install -e "$SCRIPT_DIR"
echo "PySTARC installed (editable, from current source)"
# 7. Verify
echo ""
echo "[7/7] Verifying installation."
echo ""
"$ENV_PY" -c "import pystarc; print(f'PySTARC {pystarc.__version__}')"
"$ENV_PY" -c "import numpy; print(f'NumPy {numpy.__version__}')"
"$ENV_PY" -c "import scipy; print(f'SciPy {scipy.__version__}')"
"$ENV_PY" -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
# CuPy imports even when CUDA does not work, so compile and run one kernel. If
# that fails PySTARC cannot run, so the environment is removed.
if ! "$ENV_PY" -c "import cupy as cp; cp.arange(3).sum(); print(f'CuPy {cp.__version__} (GPU ready)')"; then
    echo "ERROR: CuPy ($CUPY_PKG) cannot run a kernel on the GPU (error above). The driver reports CUDA ${CUDA_MAJOR:-unknown}."
    echo "  Removing the $ENV_NAME environment."
    conda deactivate 2>/dev/null || true
    conda env remove -n "$ENV_NAME" -y || true
    exit 1
fi
which cpptraj  >/dev/null 2>&1 && echo "cpptraj"  || echo "cpptraj not found"
which ambpdb   >/dev/null 2>&1 && echo "ambpdb"   || echo "ambpdb not found"
which tleap    >/dev/null 2>&1 && echo "tleap"    || echo "tleap not found"
which apbs     >/dev/null 2>&1 && echo "apbs"     || echo "apbs not found"
which obabel   >/dev/null 2>&1 && echo "obabel"   || echo "obabel not found"
"$ENV_PY" -c "from rdkit import Chem; print(f'RDKit {Chem.rdBase.rdkitVersion}')" 2>/dev/null || echo "RDKit not available"
# Run tests
echo ""
echo "Running tests."
cd "$SCRIPT_DIR"
"$ENV_PY" -m pytest tests/ --tb=short || true
echo ""
echo "  Installation complete!"
echo ""
echo "  To use PySTARC:"
echo "    conda activate $ENV_NAME"
echo "    cd examples/two_charged_spheres"
echo "    python ../../run_pystarc.py input.xml"
echo ""
echo "  To run all examples:"
echo "    cd examples && cat README.md"
