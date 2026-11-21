# ttk_inhibitors

Requires a CUDA GPU.

## Complexes
```
ttk_inhibitors/
├── 2X9E/                   TTK kinase with ligand SVE
├── 3GFW/                   TTK kinase with ligand S22
├── 3H9F/                   TTK kinase with ligand 92M
├── 5LJJ/                   TTK kinase with ligand AD5
├── 5N7V/                   TTK kinase with ligand 8PT
├── 5N84/                   TTK kinase with ligand 8Q5
├── 5N93/                   TTK kinase with ligand 8QE
├── 5NAD/                   TTK kinase with ligand 8RH
└── submit.sh               Submits all 8 complex SLURM jobs
```

## Input files for each complex
```
receptor.pdb                Prepared PDB structure of TTK kinase
ligand.pdb                  Prepared PDB structure of the inhibitor
config.xml                  PySTARC configuration file
run.sh                      Runs PySTARC on one workstation GPU
submit.sh                   Unified SLURM script (1 GPU, 1-node multi-GPU, or multi-node GPUs)
```

## Workstation run
```bash
cd ~/PySTARC/examples/ttk_inhibitors
conda activate PySTARC
bash 2X9E/run.sh
```

## SLURM cluster run
```bash
# Submit all 8 complexes
cd ~/PySTARC/examples/ttk_inhibitors
bash submit.sh

# Submit one complex
cd ~/PySTARC/examples/ttk_inhibitors/2X9E
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
