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

GPU-accelerated rigid-body and flexible chain Brownian dynamics for bimolecular association rate constants (k<sub>on</sub>)

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

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Features](#features)
- [Installation](#installation)
- [Testing](#testing)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Requirements](#requirements)
- [License](#license)
- [Citation](#citation)

---

## Overview

PySTARC computes the bimolecular association rate constants (k<sub>on</sub>) by implementing rigid body Brownian dynamics within the Northrup-Allison-McCammon formalism. k<sub>on</sub> is the diffusion limited rate of reaching an outer surface, multiplied by the probability of subsequently reaching reactive contact before escaping. Trajectories run in parallel on the GPU with a NumPy CPU fallback. Applications include protein-ligand and protein-protein association, as well as other diffusion-controlled encounters.

## Method

Forces combine Adaptive Poisson-Boltzmann Solver (APBS) electrostatic grids near the receptor, a screened-Coulomb (Yukawa) multipole far field outside the grid, Rotne-Prager-Yamakawa hydrodynamics, and Born desolvation. Reactions are captured by the closed-form Brownian bridge crossing probability, such that a crossing between two recorded positions is detected without shrinking the timestep. Diffusion constants come from a Monte Carlo hydrodynamic radius. 

## Features

### Performance

- **Batch propagation.** All trajectories advance as GPU arrays.
- **Multi-GPU.** Trajectories split across GPUs share APBS grids.
- **Memory-bounded Born forces.** Reverse direction desolvation is chunked to fit GPU memory.

### Physical model

- **Brownian-bridge reactions.** P = exp(-x₀·x₁ / (D_eff·Δt)), exact at constant cost per step.
- **Yukawa multipole far field.** Monopole, dipole, and quadrupole.
- **Rotne-Prager-Yamakawa hydrodynamics.** Valid through the sphere-overlap regime.
- **Monte Carlo hydrodynamic radius.** Solvent-excluded surface with a Kirkwood double sum within ~1% of the analytical value.
- **Bidirectional Born desolvation.** Receptor in ligand and ligand in receptor fields, coupled by Newton's third law.
- **Wilson score interval.** Valid for any P<sub>rxn</sub> and any N ≥ 1.
- **Adaptive timestep.** A user-configurable `max_dt` ceiling prevents b-surface overshoot.
- **Quaternion rotation.** Direct composition with no interpolation error.

### Automation

- **Scripted setup.** `setup.py` builds PQR files, APBS grids, reaction criteria, and `input.xml` from PDB and topology files.
- **Convergence diagnostics.** Relative standard error, Wilson interval, convergence curve, and split-half test.
- **Structured output.** Trajectories, encounters, first passage times, radial densities, occupancy maps, pose clusters, fluxes, transition matrices, commitment probabilities, and energetics.
- **Live progress.** k<sub>on</sub> and P<sub>rxn</sub> reported at a configurable interval.
- **Checkpointing.** Long runs save and resume.
- 
## Installation

**GPU (Linux / HPC):**

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install_PySTARC.sh
```

The `hsp90_inhibitors/` and `ttk_inhibitors/` examples require the OpenEye Toolkits (academic license available, https://www.eyesopen.com).

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

See [`examples/`](examples/) and [`examples/PARAMETERS.md`](examples/PARAMETERS.md) for the parameter guide.

```
examples/
├── two_charged_spheres/              Analytical validation (exact Smoluchowski solution)
├── trypsin_benzamidine/              Protein-ligand (charged ligand, surface pocket)
├── beta_cyclodextrin_guests/         Host-guest (7 neutral guests, same receptor)
├── thrombin_thrombomodulin/          Protein-protein (electrostatically steered)
├── barnase_barstar_chainbd/          Flexible chain BD (protein-protein, under active validation)
├── p38_mapk_sb203580/                Protein-ligand (neutral kinase inhibitor)
├── carbonic_anhydrase_inhibitors/    Protein-ligand (7 sulfonamides, 3 CA isozymes)
├── hsp90_inhibitors/                 Protein-ligand (6 HSP90 inhibitors)
├── ttk_inhibitors/                   Protein-ligand (8 TTK/MPS1 kinase inhibitors)
└── trypsin_benzamidine_multi_GPUs/   Cluster SLURM demo (single-GPU and multi-GPU)
```

## Requirements

- Python 3.11+
- AmberTools
- APBS
- OpenBabel
- RDKit
- OpenEye Toolkits
- NumPy
- SciPy
- Click
- Numba
- Matplotlib
- pdb2pqr
- CuPy

## License

MIT

## Citation

When using PySTARC, please cite:

> Ojha et al. PySTARC: GPU-accelerated Brownian dynamics for bimolecular association rate constants (2026).
