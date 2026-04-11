# Two charged spheres (analytical validation)

## System
This system has an exact analytical solution (Smoluchowski first-passage with return probability), making it the gold standard validation test for PySTARC.

| Parameter                 | Value                                             |
|---------------------------|---------------------------------------------------|
| Receptor                  | single atom, charge = +1 e, radius = 1.0 Angstrom |
| Ligand                    | single atom, charge = -1 e, radius = 1.0 Angstrom |
| b-surface                 | 10.0 Angstrom                                     |
| Escape sphere             | 20.0 Angstrom                                     |
| Contact criterion         | r < 2.5 Angstrom                                  |
| Debye length              | 7.828 Angstrom                                    |
| Born desolvation          | disabled                                          |
| Hydrodynamic interactions | disabled                                          |
| Overlap check             | disabled                                          |
| Trajectories              | 100,000                                           |
| Exact P_rxn               | 0.4501                                            |
| Exact k_on                | 1.56 x 10^10 M^-1 s^-1                            |

## Input files (provided)

| File               | Description                                                                                |
|--------------------|--------------------------------------------------------------------------------------------|
| `receptor.pqr`     | Single-atom receptor PQR (charge +1 e, radius 1.0 Angstrom). Hand-crafted, no PDB needed.  |
| `ligand.pqr`       | Single-atom ligand PQR (charge -1 e, radius 1.0 Angstrom). Hand-crafted, no PDB needed.    |
| `rxns.xml`         | Reaction criterion: receptor atom 1 and ligand atom 1 within 2.5 Angstrom.                 |
| `input.xml`        | PySTARC input file with simulation parameters.                                             |
| `analytical.py`    | Computes exact Smoluchowski solution and compares against simulation results.              |
| `convergence.py`   | Multi-seed convergence test (4 seeds x 10k trajectories).                                  |
| `run.sh`           | Runs BD simulation, verifies against the analytical solution, and runs the multi-seed convergence test.                           |

## Run
`run.sh` runs the BD simulation, then runs `analytical.py` to compare against the exact Smoluchowski solution, and finally runs `convergence.py` (4 seeds x 10k trajectories) to verify consistency across random seeds.
```bash
conda activate PySTARC
module load cuda
cd examples/two_charged_spheres
chmod +x run.sh
bash run.sh
```

## Run individual scripts (optional)
To run the scripts separately:
```bash
python ../../run_pystarc.py input.xml     # BD simulation only
python analytical.py                      # analytical verification only
python convergence.py                     # multi-seed test only
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
- The analytical solution is computed by `analytical.py` using Romberg integration of the Smoluchowski equation with screened Coulomb (Yukawa) potential.
- Physics is simplified for analytical comparison: no Born desolvation, no hydrodynamic interactions, no overlap check.
