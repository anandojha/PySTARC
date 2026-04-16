#!/bin/bash
#  PySTARC - Clean install
#  Usage:  bash install_PySTARC.sh

#  This script:
#    1. Deactivates any active conda env
#    2. Removes existing PySTARC env (if any)
#    3. Creates a fresh PySTARC env
#    4. Installs all dependencies (conda + pip + GPU)
#    5. Installs PySTARC from wheel
#    6. Runs tests to verify

set -e
ENV_NAME="PySTARC"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEEL="$SCRIPT_DIR/dist/pystarc-1.1.0-py3-none-any.whl"

echo ""
echo "  PySTARC - Clean install"
echo "  Date: $(date)"

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
echo "Created and activated: $(python --version)"
# 4. Conda dependencies
echo ""
echo "[4/7] Installing conda dependencies (ambertools, apbs)"
conda install -c conda-forge ambertools apbs rdkit openbabel openmm -y
echo "ambertools and apbs installed"
# 5. GPU + pip dependencies
echo ""
echo "[5/7] Installing pip dependencies (cupy, matplotlib, pdb2pqr)."
pip install cupy-cuda12x matplotlib pdb2pqr
echo "pip dependencies installed"
# 6. Install PySTARC
echo ""
echo "[6/7] Installing PySTARC."
if [ ! -f "$WHEEL" ]; then
    echo "ERROR: Wheel not found: $WHEEL"
    echo "Make sure you run this from the PySTARC directory."
    exit 1
fi
pip install "$WHEEL" --force-reinstall
echo "PySTARC installed"
# 7. Verify
echo ""
echo "[7/7] Verifying installation."
echo ""
python -c "import pystarc; print(f'PySTARC {pystarc.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
python -c "import pdb2pqr; print(f'pdb2pqr')" 2>/dev/null || echo "pdb2pqr not available"
python -c "import cupy; print(f'CuPy {cupy.__version__} (GPU ready)')" 2>/dev/null || echo "CuPy not available (no GPU on this node)"
which cpptraj  >/dev/null 2>&1 && echo "cpptraj"  || echo "cpptraj not found"
which ambpdb   >/dev/null 2>&1 && echo "ambpdb"   || echo "ambpdb not found"
which tleap    >/dev/null 2>&1 && echo "tleap"    || echo "tleap not found"
which apbs     >/dev/null 2>&1 && echo "apbs"     || echo "apbs not found"
which obabel   >/dev/null 2>&1 && echo "obabel"   || echo "obabel not found"
python -c "from rdkit import Chem; print(f'RDKit {Chem.rdBase.rdkitVersion}')" 2>/dev/null || echo "RDKit not available"
python -c "import openmm; print(f'OpenMM {openmm.__version__}')" 2>/dev/null || echo "OpenMM not available"
# Run tests
echo ""
echo "Running tests."
cd "$SCRIPT_DIR"
python -m pytest tests/ -q --tb=short 2>&1 | tail -3
echo ""
echo "  Installation complete!"
echo ""
echo "  To use PySTARC:"
echo "    conda activate $ENV_NAME"
echo "    module load cuda                  # HPC only"
echo "    cd examples/two_charged_spheres"
echo "    python ../../run_pystarc.py input.xml"
echo ""
echo "  To run all examples:"
echo "    cd examples && cat README.md"
