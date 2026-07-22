#!/bin/bash
#  PySTARC - Clean install
#  Usage:  bash install_PySTARC.sh
#  This script:
#    1. Deactivates any active conda env
#    2. Removes existing PySTARC env (if any)
#    3. Creates a fresh PySTARC env
#    4. Installs all dependencies (conda + pip; CuPy auto-detected for GPU)
#    5. Installs PySTARC from wheel
#    6. Runs tests to verify
set -e
ENV_NAME="PySTARC"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEEL="$SCRIPT_DIR/dist/pystarc-1.1.0-py3-none-any.whl"

# --- Platform and accelerator autodetection ---
# OS_KIND is Darwin (macOS) or Linux. GPU_MODE becomes "cuda" only when an
# NVIDIA GPU is actually visible on Linux; otherwise it stays "cpu". macOS never
# has a CUDA GPU and CuPy ships no macOS wheel, so macOS is always CPU. On a CUDA
# machine the matching CuPy wheel (11x or 12x) is picked from the driver's
# reported CUDA major version, defaulting to 12x when it cannot be read.
OS_KIND="$(uname -s)"
GPU_MODE="cpu"
CUPY_PKG=""
if [ "$OS_KIND" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_MODE="cuda"
    CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\).*/\1/p' | head -1)"
    if [ "$CUDA_MAJOR" = "11" ]; then
        CUPY_PKG="cupy-cuda11x[ctk]"
    else
        CUPY_PKG="cupy-cuda12x[ctk]"   # CUDA 12+ (or unreadable) -> 12x wheel
    fi
fi

echo ""
echo "  PySTARC - Clean install"
echo "  Date: $(date)"
echo "  Platform: $OS_KIND    Accelerator: $GPU_MODE${CUPY_PKG:+ ($CUPY_PKG)}"
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
ENV_PY="$HOME/.conda/envs/$ENV_NAME/bin/python"
echo "Created and activated: $("$ENV_PY" --version)"
echo "Using interpreter: $ENV_PY"
"$ENV_PY" -m pip --version
# 4. Conda dependencies
echo ""
echo "[4/7] Installing conda dependencies (ambertools, apbs)"
conda install -c conda-forge "numpy>=2.0,<2.6" ambertools apbs rdkit openbabel -y
echo "ambertools and apbs installed"
# OpenEye Toolkits: needed for hsp90/ttk examples (AM1-BCC charges). Needs OE_LICENSE to run.
conda install -c openeye openeye-toolkits -y || echo "  WARNING: OpenEye install failed; inhibitor examples will not run."
# 5. GPU + pip dependencies (CuPy only when a GPU was detected)
echo ""
echo "[5/7] Installing pip dependencies (matplotlib, pdb2pqr, pytest; CuPy if GPU)."
if [ "$GPU_MODE" = "cuda" ]; then
    echo "  NVIDIA GPU detected -> installing $CUPY_PKG for GPU acceleration."
    "$ENV_PY" -m pip install "$CUPY_PKG" matplotlib pdb2pqr pytest-cov pytest-sugar
else
    echo "  No NVIDIA GPU detected ($OS_KIND) -> CPU install; GPU code paths use the NumPy fallback."
    "$ENV_PY" -m pip install matplotlib pdb2pqr pytest-cov pytest-sugar
fi
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
"$ENV_PY" -c "import pdb2pqr; print(f'pdb2pqr')" 2>/dev/null || echo "pdb2pqr not available"
# On a GPU install, confirm CuPy can actually compile a kernel (it imports fine
# even when the CUDA toolkit headers are missing). On CPU/macOS CuPy is absent
# by design and the GPU code paths take the NumPy fallback.
if [ "$GPU_MODE" = "cuda" ]; then
    if "$ENV_PY" -c "import cupy as cp; cp.arange(3).sum(); print(f'CuPy {cp.__version__} (GPU ready)')" 2>/dev/null; then
        :
    else
        echo "CuPy present but the GPU is not usable yet (CUDA toolkit headers not found)."
        echo "  -> load CUDA (e.g. 'module load cuda') or set CUDA_PATH before the tests so the GPU tests compile and pass."
    fi
else
    echo "CPU mode (no NVIDIA GPU) -> GPU tests run through the NumPy fallback."
fi
which cpptraj  >/dev/null 2>&1 && echo "cpptraj"  || echo "cpptraj not found"
which ambpdb   >/dev/null 2>&1 && echo "ambpdb"   || echo "ambpdb not found"
which tleap    >/dev/null 2>&1 && echo "tleap"    || echo "tleap not found"
which apbs     >/dev/null 2>&1 && echo "apbs"     || echo "apbs not found"
which obabel   >/dev/null 2>&1 && echo "obabel"   || echo "obabel not found"
"$ENV_PY" -c "from rdkit import Chem; print(f'RDKit {Chem.rdBase.rdkitVersion}')" 2>/dev/null || echo "RDKit not available"
"$ENV_PY" -c "from openeye import oechem; print('OpenEye', oechem.OEChemGetRelease())" 2>/dev/null || echo "OpenEye not available (hsp90/ttk only; check OE_LICENSE)"
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
if [ "$GPU_MODE" = "cuda" ]; then
echo "    module load cuda                  # GPU node: load CUDA so kernels compile"
fi
echo "    cd examples/two_charged_spheres"
echo "    python ../../run_pystarc.py input.xml"
echo ""
echo "  To run all examples:"
echo "    cd examples && cat README.md"
