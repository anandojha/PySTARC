# carbonic_anhydrase_inhibitors

Requires a CUDA GPU.

## Complexes
```
carbonic_anhydrase_inhibitors/
├── ca13_azm/               Carbonic anhydrase XIII with acetazolamide
├── ca13_vd1125/            Carbonic anhydrase XIII with VD1125
├── ca13_vd1126/            Carbonic anhydrase XIII with VD1126
├── ca13_vd1209/            Carbonic anhydrase XIII with VD1209
├── ca13_vd1269/            Carbonic anhydrase XIII with VD1269
├── ca1_vd1269/             Carbonic anhydrase I with VD1269
├── ca2_vd1142/             Carbonic anhydrase II with VD1142
└── submit.sh               Submits all 7 complex SLURM jobs
```

## Input files for each complex
```
receptor.pdb                Prepared PDB structure of carbonic anhydrase (with catalytic Zn2+)
ligand.pdb                  Prepared PDB structure of the sulfonamide inhibitor
config.xml                  PySTARC configuration file
run.sh                      Runs PySTARC on one workstation GPU
submit.sh                   Unified SLURM script (1 GPU, 1-node multi-GPU, or multi-node GPUs)
```

## Workstation run
```bash
cd ~/PySTARC/examples/carbonic_anhydrase_inhibitors
conda activate PySTARC
bash ca13_azm/run.sh
```

## SLURM cluster run
```bash
# Submit all 7 complexes
cd ~/PySTARC/examples/carbonic_anhydrase_inhibitors
bash submit.sh

# Submit one complex
cd ~/PySTARC/examples/carbonic_anhydrase_inhibitors/ca13_azm
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
