<div align="center">

<pre>
██████╗ ██╗   ██╗███████╗████████╗ █████╗ ██████╗  ██████╗
██╔══██╗╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝
██████╔╝ ╚████╔╝ ███████╗   ██║   ███████║██████╔╝██║     
██╔═══╝   ╚██╔╝  ╚════██║   ██║   ██╔══██║██╔══██╗██║     
██║        ██║   ███████║   ██║   ██║  ██║██║  ██║╚██████╗
╚═╝        ╚═╝   ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
</pre>

### Python Simulation Toolkit for Association Rate Constants

GPU-accelerated rigid body and flexible chain Brownian dynamics simulations for bimolecular association rate constants

<br>

[![CI](https://github.com/anandojha/PySTARC/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/PySTARC/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anandojha/PySTARC/graph/badge.svg)](https://codecov.io/gh/anandojha/PySTARC)
[![CodeFactor](https://www.codefactor.io/repository/github/anandojha/pystarc/badge)](https://www.codefactor.io/repository/github/anandojha/pystarc)

[![PyPI](https://img.shields.io/pypi/v/pystarc.svg)](https://pypi.org/project/pystarc/)
[![Downloads](https://img.shields.io/pypi/dm/pystarc.svg)](https://pypi.org/project/pystarc/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-blue.svg?logo=dependabot)](https://github.com/anandojha/PySTARC/network/updates)
[![Lines of Code](https://img.shields.io/badge/lines_of_code-51k-blue.svg)](https://github.com/anandojha/PySTARC)

</div>

<div align="center">

<img src="pystarc_demo.gif" width="480" alt="PySTARC Brownian dynamics demonstration">

</div>

---

## Overview

PySTARC computes the bimolecular association rate constants by implementing rigid body Brownian dynamics within the Northrup-Allison-McCammon formalism. Trajectories run in parallel on the GPU with a NumPy CPU fallback. 

## Installation

**GPU (Linux / HPC):**

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install_PySTARC.sh
```

**Mac / CPU:**

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
conda create -n PySTARC python=3.11 -y
conda activate PySTARC
conda install -c conda-forge ambertools apbs rdkit openbabel -y
conda install -c openeye openeye-toolkits -y
pip install matplotlib pdb2pqr
pip install dist/pystarc-1.1.0-py3-none-any.whl --force-reinstall
```

## Testing

```bash
python -m pytest tests/
python -m pytest tests/ -v
```

## Quick start

```bash
conda activate PySTARC
module load cuda                # HPC only
cd examples/two_charged_spheres
bash run.sh
```

## Examples

See [`examples/`](examples/) for all example systems, each with its own README.

```
examples/
├── two_charged_spheres/              Analytical validation for the exact Smoluchowski solution
├── trypsin_benzamidine/              Protein-ligand complex
├── beta_cyclodextrin_guests/         Host-guest complex
├── thrombin_thrombomodulin/          Protein-protein complex
├── p38_mapk_sb203580/                Protein-ligand complex
├── carbonic_anhydrase_inhibitors/    Protein-ligand complexes
├── hsp90_inhibitors/                 Protein-ligand complexes
├── ttk_inhibitors/                   Protein-ligand complexes
└── barnase_barstar/                  Protein-protein complex
```

## Requirements

```
Python 3.11+
AmberTools
APBS
OpenBabel
RDKit
OpenEye Toolkits
NumPy
SciPy
Click
Numba
Matplotlib
pdb2pqr
CuPy
```

## License

MIT

## Citation

When using PySTARC, please cite:

> Ojha et al. PySTARC: GPU-accelerated Brownian dynamics for bimolecular association rate constants (2026).
