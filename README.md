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

PySTARC computes bimolecular association rate constants (k<sub>on</sub>) from rigid-body Brownian dynamics. It implements the Northrup-Allison-McCammon (NAM) formalism, in which the rate is written as the diffusion-limited rate of arrival at an outer b-surface multiplied by the probability that, having arrived, the pair reaches the reactive contact configuration before escaping. Trajectories are propagated in parallel on a GPU through CuPy, with a NumPy fallback on machines without a GPU.

Typical applications are protein-ligand and protein-protein association and other diffusion-controlled encounter problems for which a second-order rate constant is required.

## Method

The force model combines APBS-derived electrostatic grids near the receptor with a screened-Coulomb (Yukawa) multipole far field outside the grid, Rotne-Prager-Yamakawa hydrodynamic coupling, and a Born desolvation term. Reaction capture uses the closed-form Brownian-bridge crossing probability rather than a per-step distance test, so a reaction that occurs between two recorded positions is detected without reducing the timestep. The translational and rotational diffusion constants are obtained from a Monte Carlo estimate of the hydrodynamic radius. Setup from a PDB and topology to a runnable input is scripted, so a calculation is one setup step followed by one run.

## Features

### Performance

- **Batch trajectory propagation.** All trajectories advance together as GPU arrays of positions, quaternions, and status flags. A single RTX 6000 Ada sustains about 400,000 steps per second for a two-atom system and scales to millions of trajectories per run.
- **Multi-GPU runs.** A simulation can be split across GPUs that share APBS grids, with results pooled by `combine_data.py`.
- **Memory-bounded Born forces.** Reverse-direction Born desolvation is evaluated in chunks that fit in GPU memory, so large receptors remain tractable.

### Physical model and algorithms

- **Brownian-bridge reaction detection.** The crossing probability P = exp(-x₀·x₁ / (D_eff·Δt)) detects a reaction occurring between two recorded positions in closed form, at constant cost per step.
- **Three-term Yukawa multipole far field.** Monopole, dipole, and quadrupole terms describe the receptor potential for atoms outside the APBS grid. The dipole and quadrupole terms govern electrically neutral molecules such as β-cyclodextrin (Q = 0), where the monopole vanishes.
- **Rotne-Prager-Yamakawa hydrodynamics.** The Zuk et al. (2014) formulation covers the far-field, partial-overlap, and full-enclosure regimes, and remains valid at close approach where hydrodynamic radii overlap.
- **Monte Carlo hydrodynamic radius.** A voxelised solvent-excluded surface with a Kirkwood double sum over surface point pairs reproduces the analytical reference to within about one percent.
- **Bidirectional Born desolvation.** Born forces are computed for the receptor in the ligand field and for the ligand in the receptor field, with the reverse term obtained from Newton's third law, so mutual desolvation is captured as the molecules approach.
- **Wilson score confidence interval.** The interval on P<sub>rxn</sub> is valid for any reaction probability and any N ≥ 1, including the small-P<sub>rxn</sub> regime where the normal approximation fails.
- **Adaptive timestep with a ceiling.** A user-set `max_dt` bounds the adaptive step and prevents overshoot of the b-surface in protein-protein systems.
- **Quaternion rotational diffusion.** Orientation is updated by direct quaternion composition, with no interpolation error at any rotation magnitude.

### Automation and reproducibility

- **Scripted setup.** `setup.py` converts a PDB and topology into PQR files, APBS grids, reaction criteria, and `input.xml`, and constructs reaction criteria from crystal-structure contacts in either a polar or an all-heavy-atom mode.
- **Convergence diagnostics.** Relative standard error, the Wilson 95% interval, a cumulative convergence curve, a first-half versus second-half split test, and an estimate of the N required for a target precision.
- **Structured output.** Trajectories, encounters, near misses, first-passage times, radial densities, angular occupancy maps, pose clusters, milestone fluxes, transition matrices, commitment probabilities, and energetics are written to separate files.
- **Live progress.** k<sub>on</sub> and P<sub>rxn</sub> are reported at a configurable interval with a running Wilson interval.
- **Checkpointing.** Long runs save and resume automatically.
- **Continuous integration.** The test suite runs with coverage reporting on every commit, and the package installs from PyPI with `pip install pystarc`.

## Installation

**GPU (Linux / HPC):**

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install_PySTARC.sh
```

The `hsp90_inhibitors/` and `ttk_inhibitors/` examples use the OpenEye Toolkits for AM1-BCC ligand charges, which require a valid license (a free academic license is available from OpenEye, https://www.eyesopen.com). The other examples do not need OpenEye.

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
python -m pytest tests/                 # animated progress bar with a tick per test (pytest-sugar)
python -m pytest tests/ -v              # one PASSED/FAILED line per test with a percent counter
```

## Quick start

```bash
conda activate PySTARC
module load cuda                # HPC only, skip on local machines
cd examples/two_charged_spheres
bash run.sh
```

## Examples

See [`examples/`](examples/) for setup instructions and [`examples/PARAMETERS.md`](examples/PARAMETERS.md) for the parameter selection guide.

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
- AmberTools (tleap, cpptraj, ambpdb)
- APBS
- RDKit (ligand setup from SMILES)
- OpenBabel (mol2 conversion)
- OpenEye Toolkits (oechem, oequacpac), required only for the inhibitor examples (hsp90, ttk) for AM1-BCC ligand charges. Installable with `conda install -c openeye openeye-toolkits`; requires a valid license (OE_LICENSE).
- NumPy, SciPy, Click
- Matplotlib, pdb2pqr (setup scripts)
- CuPy (GPU) or NumPy (CPU fallback)
- NVIDIA GPU with CUDA 12+ (recommended)

## License

MIT

## Citation

When using PySTARC, please cite:

> Ojha et al. PySTARC: GPU-accelerated Brownian dynamics for bimolecular association rate constants (2026).
