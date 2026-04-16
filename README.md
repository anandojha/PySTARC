<div align="center">

<pre align="center">

╭────────────────────────────────────────────────────────────────╮
│                                                                │
│                     ◈   P y S T A R C   ◈                      │
│                                                                │
│    Python Simulation Toolkit for Association Rate Constants    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

</pre>

GPU-accelerated rigid-body Brownian dynamics for computing bimolecular association rate constants (k<sub>on</sub>)

<br>

[![CI](https://github.com/anandojha/PySTARC/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/PySTARC/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anandojha/PySTARC/graph/badge.svg)](https://codecov.io/gh/anandojha/PySTARC)
[![CodeFactor](https://www.codefactor.io/repository/github/anandojha/pystarc/badge)](https://www.codefactor.io/repository/github/anandojha/pystarc)
[![DeepSource](https://app.deepsource.com/gh/anandojha/PySTARC.svg/?label=active+issues&show_trend=true)](https://app.deepsource.com/gh/anandojha/PySTARC/)

[![PyPI](https://img.shields.io/pypi/v/pystarc.svg)](https://pypi.org/project/pystarc/)
[![Downloads](https://img.shields.io/pypi/dm/pystarc.svg)](https://pypi.org/project/pystarc/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-blue.svg?logo=dependabot)](https://github.com/anandojha/PySTARC/network/updates)
[![Lines of Code](https://img.shields.io/badge/lines_of_code-24.3k-blue)](https://github.com/anandojha/PySTARC)

</div>

---

## Table of Contents

- [What is PySTARC?](#what-is-pystarc)
- [Features](#features)
- [Installation](#installation)
- [Testing](#testing)
- [Quick start](#quick-start)
- [Examples](#examples)
- [FAQ](#faq)
- [Requirements](#requirements)
- [License](#license)
- [Citation](#citation)

---

## What is PySTARC?

PySTARC computes bimolecular association rate constants (k<sub>on</sub>) via rigid-body Brownian dynamics, implementing the Northrup-Allison-McCammon framework with modern numerics and GPU acceleration. The target use cases are drug-discovery kinetics, protein-protein association, and any diffusion-controlled encounter problem where experimental rates are needed at scale. PySTARC runs all trajectories simultaneously on GPUs via CuPy, with a complete physical model through APBS electrostatics, Rotne-Prager-Yamakawa hydrodynamics, Born desolvation, Brownian bridge reaction capture, and adaptive timestepping. Setup is automated end-to-end, so going from a PDB to a converged k<sub>on</sub> takes a single `setup.py` followed by a single `run.sh`.

## Features

### GPU-native performance

▪ **Batch trajectory propagation** - All trajectories advance simultaneously as GPU arrays for positions, quaternions, and status flags. A single RTX 6000 Ada sustains ~400,000 steps/sec for a 2-atom system and ~28,000 steps/sec for barnase-barstar (10 million trajectories in 50 minutes).
▪ **Automatic multi-GPU scaling** - Split simulations across GPUs with shared APBS grids and pooled result combining via `combine_data.py`.
▪ **Memory-safe Born force chunking** - Reverse-direction Born desolvation is batched to fit within GPU memory, enabling use with large receptors.

### Model and algorithms

▪ **Exact Brownian bridge reaction detection** — Closed-form crossing probability `P = exp(-x₀·x₁/D_eff·Δt)` captures mid-step reactions at constant cost per step with no bias, no retry loops, and no minimum-timestep floors.
▪ **Three-term Yukawa multipole far-field** - Monopole, dipole, and quadrupole analytical expansion for atoms outside the APBS grid. The dipole term is critical for electrically neutral molecules such as β-cyclodextrin (Q = 0) where the monopole contribution vanishes.
▪ **Zuk et al. (2014) RPY hydrodynamics** - Exact three-regime Rotne-Prager-Yamakawa formula covering far-field, partial overlap, and full enclosure. Accurate at close approach for protein-protein complexes where hydrodynamic radii overlap.
▪ **Hansen Monte Carlo hydrodynamic radius** - Full voxelised solvent-excluded surface with Kirkwood double-sum over 10⁶ surface point pairs. Accurate to within 1% against analytical reference.
▪ **Bidirectional Born desolvation** - Computes Born forces in both directions, receptor at ligand positions and ligand at receptor positions, with Newton's third law for the reverse. Captures mutual desolvation as both molecules approach.
▪ **Wilson score confidence interval** - Valid for any P<sub>rxn</sub> and any N ≥ 1, including the low-P<sub>rxn</sub> regime typical of tight reaction criteria where normal-approximation intervals break down.
▪ **Configurable adaptive timestep** - User-controlled `max_dt` ceiling on the adaptive timestep. Prevents trajectory overshoot past the b-surface in protein-protein systems where unchecked timestep growth produces ballistic steps.
▪ **Exact quaternion rotation** - Direct quaternion composition for rotational diffusion. No interpolation error at any rotation magnitude.

### Automation and reproducibility

▪ **End-to-end setup** - `setup.py` takes a PDB and topology and produces PQR files, APBS grids, reaction criteria, and `input.xml` in one command. Includes automated reaction-criterion construction from crystal-structure contacts with configurable polar or all-heavy-atom modes.
▪ **Convergence diagnostics** - Relative SE, Wilson 95% CI, cumulative convergence curve, first-half to second-half split test, and N-needed estimates for target precision.
▪ **14 structured output files** - Trajectories, encounters, near-misses, first-passage times, radial density, angular occupancy maps, pose clusters, milestone flux, transition matrices, commitment probabilities, and energetics.
▪ **Live progress** - k<sub>on</sub> and P<sub>rxn</sub> printed at configurable intervals with running Wilson CI.
▪ **Checkpointing** - Automatic save and resume for long production runs.

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
bash run.sh
```

## Examples

See [`examples/`](examples/) for complete setup instructions and [`examples/PARAMETERS.md`](examples/PARAMETERS.md) for the parameter selection guide.

```
examples/
├── two_charged_spheres/              Analytical validation (exact Smoluchowski solution)
├── trypsin_benzamidine/              Protein-ligand (charged ligand, surface pocket)
├── beta_cyclodextrin_guests/         Host-guest (7 neutral guests, same receptor)
├── thrombin_thrombomodulin/          Protein-protein (electrostatically steered)
├── barnase_barstar/                  Protein-protein (WT + R59A mutant)
├── p38_mapk_sb203580/                Protein-ligand (neutral kinase inhibitor)
├── carbonic_anhydrase_inhibitors/    Protein-ligand (7 sulfonamides, 3 CA isozymes)
└── trypsin_benzamidine_multi_GPUs/   Cluster SLURM demo (single-GPU and multi-GPU)
```

## FAQ

**Do I need a GPU to run PySTARC?**
No. PySTARC will fall back to NumPy on CPU if CuPy is unavailable, but single-GPU runs are typically 50 to 200 times faster than CPU for protein-scale systems. For production use, a CUDA 12+ GPU is strongly recommended.

**Which GPU hardware is supported?**
Any NVIDIA GPU with CUDA 12+ and sufficient memory (≥16 GB recommended for protein-protein complexes with fine APBS grids). PySTARC is tested on RTX 6000 Ada, A100, H100, and Quadro RTX 5000.

**How long does a typical simulation take?**
With a single RTX 6000 Ada:
- Small systems (≤50 atoms, 1M trajectories): under 20 minutes
- Protein-ligand (≤5000 atoms, 10M trajectories): 2 to 4 hours
- Protein-protein (≤10000 atoms, 5M trajectories): 6 to 12 hours

Multi-GPU runs scale linearly across 2 to 8 GPUs on one node.

**My simulation hangs at APBS grid generation. What's wrong?**
Usually an issue with the PQR file charges or grid dimensions exceeding GPU memory. Check that `receptor.pqr` and `ligand.pqr` have correct AMBER partial charges and that the APBS `dime` value is compatible with available GPU memory. Reduce `fine_grid_length` or `dime` if needed.

**How do I choose the b-surface radius and reaction criterion?**
See [`examples/PARAMETERS.md`](examples/PARAMETERS.md) for a detailed parameter selection guide covering all benchmark complexes, including b-surface sizing, reaction criterion construction, and adaptive timestep cap selection.

**I get a CUDA out-of-memory error. What should I do?**
Reduce `n_trajectories_per_batch` in `input.xml`, lower the APBS grid dimension (`dime`), or use a smaller `fine_grid_length`. GPU memory scales with both trajectory count and grid size.

## Requirements

- Python 3.11+
- AmberTools (tleap, cpptraj, ambpdb)
- APBS
- RDKit (for ligand setup from SMILES)
- OpenBabel (for mol2 conversion)
- NumPy, SciPy, Click
- Matplotlib, pdb2pqr (for setup scripts)
- CuPy (GPU) or NumPy (CPU fallback)
- NVIDIA GPU with CUDA 12+ (recommended)

## License

MIT

## Citation

When using PySTARC, please cite:

> Ojha, A. A. et al. PySTARC: GPU-accelerated Brownian dynamics for bimolecular association rate constants (2026).
