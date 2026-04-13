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

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install_PySTARC.sh
```

This creates a fresh conda environment, installs all dependencies (AmberTools, APBS, CuPy, NumPy, SciPy), and runs the test suite. On machines without NVIDIA GPUs (e.g., Mac), skip the CuPy step and install manually:

```bash
pip install matplotlib pdb2pqr
pip install dist/pystarc-1.1.0-py3-none-any.whl --force-reinstall
```

PySTARC will run in CPU mode (NumPy backend) without CuPy.

## Quick start
```bash
conda activate PySTARC
module load cuda             
cd examples/two_charged_spheres
chmod +x run.sh
bash run.sh
```

## Examples
See `examples/README.md` for complete instructions.

## Testing

```bash
python -m pytest tests/ -v          
python -m pytest tests/ -q          # quiet mode
```
## License
MIT
