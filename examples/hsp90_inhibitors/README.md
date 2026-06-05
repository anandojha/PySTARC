# HSP90-inhibitor complexes

## Receptor-ligand complexes
All 6 receptor-ligand complexes share the same HSP90 N-terminal domain receptor with different small-molecule inhibitors. The systems are a subset of the HSP90 inhibitor kinetics dataset of Kokh et al. (2018); the charged member of the original set is excluded so that all ligands modeled here are neutral. System indices (31, 37, ...) are dataset indices from that study.

| Receptor-ligand complex | Exp k<sub>on</sub> (M<sup>-1</sup>s<sup>-1</sup>) |
|-------------------------|-----------------------------------------------------|
| `31`                    | 1.0 x 10<sup>6</sup>  |
| `37`                    | 3.4 x 10<sup>5</sup>  |
| `43`                    | 8.4 x 10<sup>4</sup>  |
| `62`                    | 1.2 x 10<sup>5</sup>  |
| `65`                    | 2.1 x 10<sup>5</sup>  |
| `70`                    | 1.0 x 10<sup>4</sup>  |

## Shared parameters

| Parameter              | Value                                |
|------------------------|--------------------------------------|
| Receptor               | HSP90 N-terminal domain              |
| Ligand charge          | neutral (auto-detected from formal charges) |
| b-surface              | 55.0 Angstrom                        |
| Debye length           | 7.86 Angstrom (150 mM, 0.15 M)       |
| Contact mode           | all (any heavy-atom contacts)        |
| Contact cutoff         | 5.0 Angstrom                         |
| Pairs / n_needed       | up to 8 (one per residue) / 6        |
| Born desolvation       | enabled                              |
| Hydrodynamic interactions | enabled                           |
| Overlap check          | disabled                             |
| Trajectories           | 100,000 per complex                  |

## Input files (provided)
Each system directory (`31/`, `37/`, ...) ships only the source structure and the setup script:

| File          | Description                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------------|
| `complex.pdb` | Bound-state PDB containing the receptor (HSP90) and the co-crystallized inhibitor.                  |
| `setup.py`    | Automated setup script. Reads the PDB, parameterizes from scratch, and generates all BD input files. |

The following script is in the `hsp90_inhibitors/` directory:

| File       | Description                                                                                                          |
|------------|----------------------------------------------------------------------------------------------------------------------|
| `run.sh`   | Runs setup and BD simulation for all 6 complexes sequentially, then compares rates against experiment and saves `summary.txt`. |

## What setup.py generates
Unlike the pre-generated examples, no parameterized files are shipped: `setup.py` builds everything from scratch each run. Running `python setup.py` produces:

| Generated file   | Description                                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `protein.prmtop`, `protein.rst7` | Receptor AMBER topology and coordinates from tleap (ff14SB).                                        |
| `ligand.prmtop`, `ligand.rst7`   | Ligand AMBER topology and coordinates from antechamber/tleap (GAFF2, AM1-BCC charges).             |
| `receptor.pdb`, `ligand.pdb`     | Clean receptor and ligand PDBs from tleap.                                                          |
| `receptor.pqr`   | Receptor PQR file with atom positions, partial charges, and radii for the HSP90 host.                                            |
| `ligand.pqr`     | Ligand PQR file with atom positions, partial charges, and radii for the inhibitor.                                               |
| `rxns.xml`       | Reaction criterion file with atom pairs and cutoff distances identified automatically from the bound-state PDB.                   |
| `input.xml`      | PySTARC input file with all simulation parameters (b-surface, electrostatics, trajectories, GPU, and convergence).               |

## Setup and run (single complex)
```bash
conda activate PySTARC
module load cuda
cd examples/hsp90_inhibitors/31
python setup.py
python ../../../run_pystarc.py input.xml
```

## Run all 6 complexes
```bash
conda activate PySTARC
module load cuda
cd examples/hsp90_inhibitors
chmod +x run.sh
bash run.sh
```
`run.sh` performs the following for each of the 6 complexes: cleans any previous output files, runs `setup.py` to parameterize and generate the PQR files and input XMLs, then runs the BD simulation. After all simulations complete, on-rates are printed to terminal and saved to `summary.txt`. The comparison includes PySTARC k<sub>on</sub>, experimental k<sub>on</sub>, their ratio, P<sub>rxn</sub>, and a Spearman rank correlation across all 6 complexes.

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
- All 6 complexes use identical `setup.py` parameters; only the source `complex.pdb` differs per system.
- Receptor charges and parameters are from ff14SB (tleap); ligand charges are AM1-BCC (OpenEye) and ligand parameters are GAFF2 (antechamber/parmchk2).
- PQR files are generated from the AMBER topology via `cpptraj` and `ambpdb`.
- Reaction contacts are identified automatically by `setup.py` from the bound-state PDB structure (contact method, 5.0 Angstrom cutoff, n_needed=6).
- All ligands modeled here are neutral. The charged member of the original Kokh et al. set is excluded.
