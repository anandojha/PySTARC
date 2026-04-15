[![CI](https://github.com/anandojha/PySTARC/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/PySTARC/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/anandojha/PySTARC/graph/badge.svg)](https://codecov.io/gh/anandojha/PySTARC) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![CodeFactor](https://www.codefactor.io/repository/github/anandojha/pystarc/badge)](https://www.codefactor.io/repository/github/anandojha/pystarc) [![PyPI](https://img.shields.io/pypi/v/pystarc.svg)](https://pypi.org/project/pystarc/) [![Downloads](https://img.shields.io/pypi/dm/pystarc.svg)](https://pypi.org/project/pystarc/) [![Dependabot](https://img.shields.io/badge/dependabot-enabled-blue.svg?logo=dependabot)](https://github.com/anandojha/PySTARC/network/updates) [![Lines of Code](https://img.shields.io/badge/lines_of_code-NUMBER-blue)](https://github.com/anandojha/PySTARC) [![DeepSource](https://app.deepsource.com/gh/anandojha/PySTARC.svg/?label=active+issues&show_trend=true)](https://app.deepsource.com/gh/anandojha/PySTARC/)

# PySTARC - Python Simulation Toolkit for Association Rate Constants

**PySTARC** computes bimolecular association rate constants (k<sub>on</sub>) via GPU-accelerated rigid-body Brownian dynamics.

## Features

- **GPU batch simulation** - All trajectories run simultaneously on GPU via CuPy.
- **Physics** - Ermak-McCammon integrator, RPY hydrodynamics, Born desolvation, APBS electrostatics, adaptive time step, and Yukawa monopole fallback.
- **Brownian bridge** - Catches mid-step reaction crossings.
- **Multi-GPU workflow** - Split simulations across N GPUs with automatic grid generation, symlinked DX files, and pooled result combining.
- **Automated system setup** - From a PDB and topology file to a ready-to-run simulation in one command via `setup.py`.
- **Convergence analysis** - Wilson score CI, relative SE, and trajectory-count estimates for target precision.
- **Output files** - 14 structured files, including trajectories, encounters, radial density, angular maps, and transition matrices.
- **Checkpointing** - Automatic save/resume for long production runs.
- **Live progress** - k<sub>on</sub> and P<sub>rxn</sub> printed at configurable intervals.
- **Temperature scaling** - Correct thermodynamics at any temperature.

## Installation

**GPU (Linux/HPC):**
```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install_PySTARC.sh
```

**On Mac/CPU:**
```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
conda create -n PySTARC python=3.11 -y
conda activate PySTARC
conda install -c conda-forge ambertools apbs -y
pip install matplotlib pdb2pqr
pip install dist/pystarc-1.1.0-py3-none-any.whl --force-reinstall
```

## Testing

```bash
python -m pytest tests/ -v          
```

## Quick start

```bash
conda activate PySTARC
module load cuda                # HPC only, skip on local machines
cd examples/two_charged_spheres
chmod +x run.sh
bash run.sh
```

## Examples

See `examples/README.md` for complete instructions.

## License

MIT

## Citation

When using PySTARC, please cite:

Ojha, A. A. et al. PySTARC: GPU-accelerated Brownian dynamics for bimolecular association rate constants (2026).

## Requirements

- Python 3.11+
- AmberTools (tleap, cpptraj, ambpdb)
- APBS
- CuPy (GPU) or NumPy (CPU fallback)
- NVIDIA GPU with CUDA 12+ (recommended)
