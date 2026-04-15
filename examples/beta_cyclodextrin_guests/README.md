# Beta-cyclodextrin host-guest complexes

## Receptor-ligand complexes
All 7 receptor-ligand complexes share the same beta-cyclodextrin (BCD/MGO) receptor with different guest molecules.

| Receptor-ligand complex | Ligand            | 
|-------------------------|-------------------|
| BCD_1-propanol          | 1-propanol        |
| BCD_1-butanol           | 1-butanol         | 
| BCD_tertbutanol         | tert-butanol      |
| BCD_methyl_butyrate     | methyl butyrate   |
| BCD_aspirin             | aspirin           | 
| BCD_1-naphthylethanol   | 1-naphthylethanol |
| BCD_2-naphthylethanol   | 2-naphthylethanol |

## Shared parameters 

| Parameter              | Value                                |
|------------------------|--------------------------------------|
| Receptor               | beta-cyclodextrin (MGO), 147 atoms   |
| Ligand                 | APN                                  |
| b-surface              | 30.0 Angstrom                        |
| Escape sphere          | 60.0 Angstrom                        |
| Debye length           | 7.86 Angstrom (150 mM NaCl)          |
| Contact mode           | all (any heavy-atom contacts)        |
| Contact cutoff         | 5.0 Angstrom                         |
| Buffer                 | 2.0 Angstrom                         |
| Born desolvation       | enabled                              |
| Trajectories           | 100,000 per complex                  |

## Input files (provided)
Each `BCD_*/` directory contains the following files:

| File                | Description                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| `complex.pdb`       | Bound-state PDB containing receptor (MGO), ligand (APN), and water (WAT).                           |
| `complex.parm7`     | AMBER topology file that provides partial charges, atom types, and connectivity for PQR generation. |
| `setup.py`          | Automated setup script. Reads the PDB and topology, generates files for BD simulation.              |

The following scripts are in the `beta_cyclodextrin_guests/` directory:

| File                | Description                                                                                                                            |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `run.sh`            | Runs setup and BD simulation for all 7 complexes sequentially, then compares rates against experiment.                                 |
| `compare_rates.py`  | Collects k<sub>on</sub> from all 7 complexes, compares against experimental values, computes Spearman rank correlation, and saves `summary.txt`. |

## What setup.py generates
Running `python setup.py` produces:

| Generated file   | Description                                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `receptor.pqr`   | Receptor PQR file extracted from the topology. Contains atom positions, partial charges, and radii for the beta-cyclodextrin host. |
| `ligand.pqr`     | Ligand PQR file extracted from the topology. Contains atom positions, partial charges, and radii for the guest molecule.           |
| `rxns.xml`       | Reaction criterion file that contains atom pairs and cutoff distances identified automatically from the bound-state PDB.   |
| `input.xml`      | PySTARC input file that contains all simulation parameters (b-surface, electrostatics, trajectories, GPU, and convergence).        |

## Setup and run (single complex)
```bash
conda activate PySTARC
module load cuda
cd examples/beta_cyclodextrin_guests/BCD_1-butanol
python setup.py
python ../../../run_pystarc.py input.xml
```

## Run all 7 complexes
```bash
conda activate PySTARC
module load cuda
cd examples/beta_cyclodextrin_guests
chmod +x run.sh
bash run.sh
```
`run.sh` performs the following for each of the 7 complexes: cleans any previous output files, runs `setup.py` to generate PQR files and input XMLs, then runs the BD simulation. After all simulations complete, on-rates are printed to terminal and saved to `summary.txt`. The comparison includes PySTARC k<sub>on</sub>, experimental k<sub>on</sub>, their ratio, and a Spearman rank correlation across all 7 complexes.

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
- All 7 complexes use identical `setup.py` parameters (same receptor MGO, same ligand APN residue name).
- PQR files are generated from the AMBER topology via `cpptraj` and `ambpdb` and the water is stripped automatically.
- Reaction contacts are identified automatically by `setup.py` from the bound-state PDB structure.
- Charges come from the AMBER parm7 via `ambpdb -pqr`.
