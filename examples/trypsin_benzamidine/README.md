# Trypsin-benzamidine (protein-ligand complex)

## System

| Property               | Value                                |
|------------------------|--------------------------------------|
| Receptor               | Trypsin (3220 atoms, Q = +6 e)       |
| Ligand                 | Benzamidine (18 atoms, Q = +1 e)     |
| b-surface              | 45.0 Angstrom                        |
| Escape sphere          | 90.0 Angstrom                        |
| Debye length           | 7.86 Angstrom (150 mM NaCl)          |
| Contact mode           | polar (N/O/S on both sides)          |
| Contact cutoff         | 6.0 Angstrom                         |
| Buffer                 | 3.0 Angstrom                         |
| Born desolvation       | enabled                              |
| Hydrodynamic radii     | receptor 22.5, ligand 5.0 Angstrom   |
| Trajectories           | 100,000                              |
| Experimental k_on      | 2.9 x 10^7 M^-1 s^-1                 |

## Input files (provided)

| File                | Description                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| `complex.pdb`       | Bound-state PDB containing trypsin, benzamidine, water, and ions.                                   |
| `complex.prmtop`    | AMBER topology file that provides partial charges, atom types, and connectivity for PQR generation. |
| `setup.py`          | Automated setup script. Reads the PDB and topology, generates files for BD simulation.              |
| `run.sh`            | Runs setup and BD simulation in one command.                                                        |

## What setup.py generates
Running `python setup.py` produces:

| Generated file   | Description                                                                                          |
|------------------|------------------------------------------------------------------------------------------------------|
| `receptor.pqr`   | Receptor PQR file. Water and ions are stripped automatically.                                        |
| `ligand.pqr`     | Ligand PQR file.                                                                                     |
| `rxns.xml`       | Reaction criterion file with atom pairs and cutoff distances identified automatically from the PDB.  |
| `input.xml`      | PySTARC input file with all simulation parameters.                                                   |

## Setup and run
```bash
conda activate PySTARC
mocule load cuda
cd examples/trypsin_benzamidine
python setup.py
python ../../run_pystarc.py input.xml
```

Or in one command:
```bash
conda activate PySTARC
module load cuda
cd examples/trypsin_benzamidine
chmod +x run.sh
bash run.sh
```

## Output files
After the simulation completes, all results are written to `bd_sims/`.

| Output file              | Description                                                                                                                    |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `results.json`           | k_on, P_rxn, Wilson 95% CI, k_b, D_rel, wall time, and GPU info.                                                              |
| `convergence.json`       | Convergence analysis: SE, relative SE, Wilson CI, convergence verdict, and trajectory estimates for target precision.           |
| `trajectories.csv`       | Per-trajectory record: number of steps, starting pose, minimum distance reached, and number of returns from the escape sphere. |
| `encounters.csv`         | Binding encounter poses for reacted trajectories: final position, orientation, and contact distances.                          |
| `near_misses.csv`        | Trajectories that approached the reaction surface but escaped.                                                                 |
| `fpt_distribution.csv`   | First-passage times for reacted trajectories.                                                                                  |
| `pose_clusters.csv`      | Clustered binding poses from encounter geometries.                                                                             |
| `paths.npz`              | Full trajectory coordinates sampled at configurable intervals.                                                                 |
| `energetics.npz`         | Per-step energies and forces along trajectories.                                                                               |
| `radial_density.csv`     | Radial probability density as a function of distance from the receptor.                                                        |
| `angular_map.npz`        | Angular occupancy map (theta, phi) on the b-surface.                                                                           |
| `contact_frequency.csv`  | Per-pair contact frequencies for the reaction criterion atom pairs.                                                            |
| `milestone_flux.csv`     | Net flux across radial shells.                                                                                                 |
| `transition_matrix.npz`  | Radial shell-to-shell transition counts.                                                                                       |
| `p_commit.npz`           | Commitment probabilities at each radial shell.                                                                                 |

A timestamped log file (`pystarc_YYYYMMDD_HHMMSS.log`) is also written to `bd_sims/` containing the full simulation output.

## Notes
- PQR files are generated from the AMBER topology via `cpptraj` and `ambpdb` and water and ions are stripped automatically.
- Reaction contacts are identified automatically by `setup.py` from the bound-state PDB structure using polar contacts (N/O/S atoms on both receptor and ligand).
