# Thrombin-thrombomodulin (protein-protein complex)

## System

| Parameter              | Value                                     |
|------------------------|-------------------------------------------|
| Receptor               | Thrombin                                  |
| Ligand                 | Thrombomodulin                            |
| b-surface              | 175.0 Angstrom                            |
| Escape sphere          | 350.0 Angstrom                            |
| Debye length           | 7.86 Angstrom (150 mM NaCl)               |
| Born desolvation       | enabled                                   |
| Hydrodynamic interactions | enabled                                |
| Overlap check          | enabled                                   |
| Trajectories           | 100,000                                   |

## Input files (provided)

| File               | Description                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------|
| `receptor.pqr`     | Thrombin PQR file (pre-computed).                                                                    |
| `ligand.pqr`       | Thrombomodulin PQR file (pre-computed).                                                              |
| `rxns.xml`         | Reaction criterion with binding-site atom pairs and cutoff distances.                                |
| `input.xml`        | PySTARC input file with simulation parameters.                                                       |
| `bb_effect.py`     | Brownian bridge A/B test: runs 4 seeds to verify BB genuinely improves P_rxn (diagnostic script).    |
| `run.sh`           | Runs BD simulation in one command.                                                                   |

## Run
`run.sh` runs the BD simulation, then runs `bb_effect.py` (4 seeds x 10k trajectories) to validate the Brownian bridge mechanism.
```bash
conda activate PySTARC
module load cuda
cd examples/thrombin_thrombomodulin
chmod +x run.sh
bash run.sh
```

## Run individual scripts (optional)
To run the scripts separately:
```bash
python ../../run_pystarc.py input.xml     # BD simulation only
python bb_effect.py                       # Brownian bridge diagnostic only
```

## Output files
After a simulation completes, all results are written to `bd_sims/`.

| Output file              | Description                                                                                                                    |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `results.json`           | k_on, P_rxn, Wilson 95% CI, k_b, D_rel, wall time, and GPU info.                                                               |
| `convergence.json`       | Convergence analysis: SE, relative SE, Wilson CI, convergence verdict, and trajectory estimates for target precision.          |
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
- This is a protein-protein complex with a large ligand (thrombomodulin), so force evaluation is automatically batched to fit GPU memory.
- The Brownian bridge diagnostic (`bb_effect.py`) is optional and only needed to validate the BB implementation.
