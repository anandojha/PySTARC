# hsp90_inhibitors

Requires a CUDA GPU.

## Complexes
```
hsp90_inhibitors/
├── HSP90-aminopyridine/     HSP90 with an aminopyridine inhibitor
├── HSP90-indazole_5LNZ/     HSP90 with an indazole inhibitor
├── HSP90-indazole_5OCI/     HSP90 with an indazole inhibitor
├── HSP90-quinazoline/       HSP90 with a quinazoline inhibitor
├── HSP90-quinazoline_6EI5/  HSP90 with a quinazoline inhibitor
├── HSP90-resorcinol/        HSP90 with a resorcinol inhibitor
└── submit.sh                Submits all 6 complex SLURM jobs
```

## Input files for each complex
```
receptor.pdb                Prepared PDB structure of HSP90
ligand.pdb                  Prepared PDB structure of the inhibitor
config.xml                  PySTARC configuration file
run.sh                      Runs PySTARC on one workstation GPU
submit.sh                   Unified SLURM script (1 GPU, 1-node multi-GPU, or multi-node GPUs)
```

## Workstation run
```bash
cd ~/PySTARC/examples/hsp90_inhibitors
conda activate PySTARC
bash HSP90-aminopyridine/run.sh
```

## SLURM cluster run
```bash
# Submit all 6 complexes
cd ~/PySTARC/examples/hsp90_inhibitors
bash submit.sh

# Submit one complex
cd ~/PySTARC/examples/hsp90_inhibitors/HSP90-aminopyridine
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
