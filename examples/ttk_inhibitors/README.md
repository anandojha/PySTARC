# TTK (MPS1) kinase-inhibitor complexes

## Receptor-ligand complexes
This example covers eight inhibitors of the TTK (MPS1) kinase domain. Each system is a separate crystal structure (the PDB accession is the directory name) paired with its co-crystallized inhibitor. BD models the diffusion-limited encounter of each inhibitor with the ATP-binding site of the kinase.

| Directory | PDB  | Inhibitor    | Exp k<sub>on</sub> (M<sup>-1</sup>s<sup>-1</sup>) |
|-----------|------|--------------|-----------------------------------------------------|
| `2X9E`    | 2X9E | NMS-P715     | 6.4 × 10<sup>5</sup>  |
| `3GFW`    | 3GFW | Mps1-IN-1    | 3.8 × 10<sup>5</sup>  |
| `3H9F`    | 3H9F | Mps1-IN-2    | 1.2 × 10<sup>6</sup>  |
| `5LJJ`    | 5LJJ | Reversine    | 2.1 × 10<sup>6</sup>  |
| `5N7V`    | 5N7V | MPI-0479605  | 2.0 × 10<sup>6</sup>  |
| `5N84`    | 5N84 | Mps-BAY2b    | 2.6 × 10<sup>6</sup>  |
| `5N93`    | 5N93 | TC-Mps1-12   | 2.2 × 10<sup>7</sup>  |
| `5NAD`    | 5NAD | BAY-1217389  | 3.8 × 10<sup>5</sup>  |

Experimental k<sub>on</sub> values are from Uitdehaag et al. (2017). Verify the exact citation against the primary source before external use. See the Notes on `3GFW` below.

## Shared parameters

| Parameter              | Value                                |
|------------------------|--------------------------------------|
| Receptor               | TTK (MPS1) kinase domain             |
| Ligand charge          | neutral                              |
| b-surface              | 60.0 Å                        |
| Debye length           | 7.86 Å (150 mM, 0.15 M)       |
| Reaction cutoff        | 4.5 Å (per-pair)              |
| n_needed               | 3                                    |
| Born desolvation       | enabled                              |
| Hydrodynamic interactions | enabled                           |
| Overlap check          | disabled                             |
| Trajectories           | 100,000 per complex                  |

Unlike the host-guest examples, the reaction criterion here is defined explicitly per system: each `setup.py` lists the receptor residue / ligand atom pairs taken from the crystal pose (for example, for `2X9E`: GLU603 O - N15, GLY605 O - N18, GLY605 N - N15). With `n_needed=3`, three such contacts must be satisfied simultaneously for a binding event.

## Input files (provided)
Each system directory ships only the source structure and the setup script:

| File          | Description                                                                                          |
|---------------|------------------------------------------------------------------------------------------------------|
| `<PDB>.pdb`   | Crystal structure (protein + co-crystallized inhibitor), named by PDB accession (e.g. `2X9E.pdb`).  |
| `setup.py`    | Automated setup script. Extracts protein and ligand, parameterizes from scratch, and generates all BD input files. |

The following script is in the `ttk_inhibitors/` directory:

| File       | Description                                                                                                          |
|------------|----------------------------------------------------------------------------------------------------------------------|
| `run.sh`   | Runs setup and BD simulation for all 8 complexes sequentially, then compares rates against experiment and saves `summary.txt`. |

## What setup.py generates
No parameterized files are shipped. `setup.py` builds everything from scratch each run. If the source `<PDB>.pdb` is already present (as shipped) it is used directly. Otherwise setup downloads it from the RCSB PDB. Running `python setup.py` produces:

| Generated file   | Description                                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `protein.prmtop`, `protein.rst7` | Receptor AMBER topology and coordinates from tleap (ff14SB).                                        |
| `ligand.prmtop`, `ligand.rst7`   | Ligand AMBER topology and coordinates from antechamber/tleap (GAFF2, AM1-BCC charges).             |
| `receptor.pdb`, `ligand.pdb`     | Clean receptor and ligand PDBs.                                                                     |
| `receptor.pqr`   | Receptor PQR file with atom positions, partial charges, and radii.                                                              |
| `ligand.pqr`     | Ligand PQR file with atom positions, partial charges, and radii.                                                                |
| `rxns.xml`       | Reaction criterion file with the per-system atom pairs and 4.5 Å cutoffs.                                                |
| `input.xml`      | PySTARC input file with all simulation parameters (b-surface, electrostatics, trajectories, GPU, and convergence).               |

## Setup and run (single complex)
```bash
conda activate PySTARC
module load cuda
cd examples/ttk_inhibitors/2X9E
python setup.py
python ../../../run_pystarc.py input.xml
```

## Run all 8 complexes
```bash
conda activate PySTARC
module load cuda
cd examples/ttk_inhibitors
chmod +x run.sh
bash run.sh
```
`run.sh` performs the following for each of the 8 complexes: cleans any previous output files (keeping the source PDB), runs `setup.py` to parameterize and generate the PQR files and input XMLs, then runs the BD simulation. After all simulations complete, on-rates are printed to terminal and saved to `summary.txt`. The comparison includes PySTARC k<sub>on</sub>, experimental k<sub>on</sub>, their ratio, P<sub>rxn</sub>, and a Spearman rank correlation across the systems (with `3GFW` excluded, see Notes).

## Output files
After a simulation completes, all results are written to `bd_sims/` within each receptor-ligand complex directory.

| Output file              | Description                                                                                                                    |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `results.json`           | k<sub>on</sub>, P<sub>rxn</sub>, Wilson 95% CI, k<sub>b</sub>, D<sub>rel</sub>, wall time, and GPU info.                      |
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
- The source PDB is shipped with each system, so setup runs offline. If the PDB is removed, setup downloads it from `https://files.rcsb.org/download/<PDB>.pdb`, which requires network access.
- The reaction criterion is explicit per system (defined in each `setup.py`), not auto-discovered. Each pair uses a 4.5 Å cutoff and `n_needed=3`.
- Receptor parameters are ff14SB (tleap). Ligand charges are AM1-BCC (OpenEye) and ligand parameters are GAFF2 (antechamber/parmchk2). PQR files are generated via `cpptraj` and `ambpdb`.
- `3GFW` is included for completeness but its reaction criterion is too loose: the chosen contacts do not discriminate the bound pose, producing spurious reactions and an unreliable k<sub>on</sub>. It is excluded from the rank-correlation statistic and its reported rate should not be trusted without rebuilding the criterion.
