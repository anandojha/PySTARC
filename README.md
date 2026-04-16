<div align="center">

```
╭──────────────────────────╮
│   ◈  P y S T A R C  ◈   │
╰──────────────────────────╯
```

**Python Simulation Toolkit for Association Rate Constants**

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

## Features

- **GPU batch simulation** — all trajectories run simultaneously on GPU via CuPy
- **Full BD physics** — Ermak-McCammon integrator, RPY hydrodynamics, Born desolvation, APBS electrostatics, adaptive timestep, Yukawa monopole fallback
- **Brownian bridge** — catches mid-step reaction crossings
- **Multi-GPU workflow** — split simulations across N GPUs with automatic grid generation, symlinked DX files, and pooled result combining
- **Automated system setup** — from a PDB and topology file to a ready-to-run simulation in one command via `setup.py`
- **Convergence analysis** — Wilson score CI, relative SE, and trajectory-count estimates for target precision
- **14 structured output files** — trajectories, encounters, radial density, angular maps, transition matrices, and more
- **Checkpointing** — automatic save/resume for long production runs
- **Live progress** — k<sub>on</sub> and P<sub>rxn</sub> printed at configurable intervals
- **Temperature scaling** — correct thermodynamics at any temperature

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
bash run.sh
```

## Benchmark systems

PySTARC has been validated across six benchmark systems spanning analytical, protein-ligand, host-guest, and protein-protein complexes:

| System | Type | Experimental k<sub>on</sub> | PySTARC k<sub>on</sub> |
|--------|------|------------------------------|------------------------|
| Two charged spheres | Analytical | 1.56 × 10<sup>10</sup> | 1.55 × 10<sup>10</sup> (0.5% error) |
| Trypsin-benzamidine | Protein-ligand | 2.9 × 10<sup>7</sup> | 1.6 × 10<sup>7</sup> |
| p38 MAPK / SB203580 | Protein-ligand | 1.5 × 10<sup>7</sup> | 1.7 × 10<sup>7</sup> |
| β-cyclodextrin / 7 guests | Host-guest | 2.8–7.2 × 10<sup>8</sup> | ~10<sup>8</sup> (encounter rate) |
| Thrombin-thrombomodulin | Protein-protein | 6.7 × 10<sup>6</sup> | 1.5 × 10<sup>7</sup> |
| Barnase-barstar (WT) | Protein-protein | 3–6 × 10<sup>8</sup> | 3.3 × 10<sup>8</sup> |
| Barnase-barstar (R59A) | Mutant | ~6.5 × 10<sup>7</sup> | 2.2 × 10<sup>8</sup> |
| Carbonic anhydrase (7 systems) | Protein-ligand | 3.3 × 10<sup>5</sup> – 2.7 × 10<sup>6</sup> | 2.2 × 10<sup>5</sup> – 4.3 × 10<sup>6</sup> |

Overall: R² = 0.83, Spearman ρ = 0.86 across 19 systems spanning 5 orders of magnitude in k<sub>on</sub>.

See [`examples/`](examples/) for complete setup instructions and [`examples/PARAMETERS.md`](examples/PARAMETERS.md) for the parameter selection guide.

## Examples

```
examples/
├── two_charged_spheres/          Analytical validation (exact Smoluchowski solution)
├── trypsin_benzamidine/          Protein-ligand (charged ligand, surface pocket)
├── beta_cyclodextrin_guests/     Host-guest (7 neutral guests, same receptor)
├── thrombin_thrombomodulin/      Protein-protein (electrostatically steered)
├── barnase_barstar/              Protein-protein (WT + R59A mutant)
└── p38_SB2/                      Protein-ligand (neutral kinase inhibitor)
```

## Requirements

- Python 3.11+
- AmberTools (tleap, cpptraj, ambpdb)
- APBS
- CuPy (GPU) or NumPy (CPU fallback)
- NVIDIA GPU with CUDA 12+ (recommended)

## License

MIT

## Citation

When using PySTARC, please cite:

> Ojha, A. A. et al. PySTARC: GPU-accelerated Brownian dynamics for bimolecular association rate constants (2026).
