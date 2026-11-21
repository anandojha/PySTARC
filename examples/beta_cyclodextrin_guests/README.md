# beta_cyclodextrin_guests

Requires a CUDA GPU.

## Complexes
```
beta_cyclodextrin_guests/
├── BCD_1-butanol/          β-cyclodextrin with 1-butanol
├── BCD_1-naphthylethanol/  β-cyclodextrin with 1-naphthylethanol
├── BCD_1-propanol/         β-cyclodextrin with 1-propanol
├── BCD_2-naphthylethanol/  β-cyclodextrin with 2-naphthylethanol
├── BCD_aspirin/            β-cyclodextrin with aspirin
├── BCD_methyl_butyrate/    β-cyclodextrin with methyl butyrate
├── BCD_tertbutanol/        β-cyclodextrin with tert-butanol
└── submit.sh               Submits all 7 complex SLURM jobs
```

## Input files for each complex
```
receptor.pqr                Prepared receptor PQR (charges and radii)
ligand.pqr                  Prepared guest PQR (charges and radii)
config.xml                  PySTARC configuration file
run.sh                      Runs PySTARC on one workstation GPU
submit.sh                   Unified SLURM script (1 GPU, 1-node multi-GPU, or multi-node GPUs)
```

## Workstation run
```bash
cd ~/PySTARC/examples/beta_cyclodextrin_guests
conda activate PySTARC
bash BCD_aspirin/run.sh
```

## SLURM cluster run
```bash
# Submit all 7 complexes
cd ~/PySTARC/examples/beta_cyclodextrin_guests
bash submit.sh

# Submit one complex
cd ~/PySTARC/examples/beta_cyclodextrin_guests/BCD_aspirin
sbatch submit.sh                                # 1 node, 4 GPUs (default)
sbatch --nodes=1 --ntasks-per-node=1 submit.sh  # 1 node, 1 GPU
sbatch --nodes=2 --ntasks-per-node=4 submit.sh  # 2 nodes, 4 GPUs per node (8 total)
```

## Output files

Outputs are written inside `bd_sims/` (see [`TUTORIAL.md`](../../TUTORIAL.md#13-results-and-output-files) for column definitions):

```
bd_sims/
├── results.json, convergence.json     Association rate constant, reaction probability, and Wilson intervals
├── input.xml, rxns.xml, build_record.json
├── receptor.pqr, ligand.pqr           Centered solute PQRs and cached hydrodynamic radii (*.cache)
├── receptor*.dx, ligand*.dx           Coarse/fine APBS electrostatic and Born desolvation grids (+ *_grids.key)
├── trajectories.csv, encounters.csv, near_misses.csv, pose_clusters.csv
├── bd_time_to_reaction.csv, contact_frequency.csv, milestone_flux.csv, radial_density.csv
├── angular_map.npz, paths.npz, energetics.npz, transition_matrix.npz
└── bd_1/ ... bd_N/, split.json        Per-GPU shard outputs and split metadata (multi-GPU runs)
```
