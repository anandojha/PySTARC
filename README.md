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
