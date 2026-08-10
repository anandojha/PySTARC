# p38_mapk_sb203580

p38 MAPK with SB203580. Protein and ligand.

## Files
```
1A9U.pdb                    structure
config.xml                  parameters
setup.py                    builds inputs
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/p38_mapk_sb203580
conda activate PySTARC
module load cuda
bash run.sh
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/p38_mapk_sb203580
conda activate PySTARC
module load cuda
sbatch submit_SLURM_single_GPU.sh
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/p38_mapk_sb203580
conda activate PySTARC
module load cuda
sbatch submit_SLURM_multi_GPUs.sh
```

## Once simulation finishes, the following files will be generated with 1 GPU
```
input.xml  rxns.xml  receptor.pqr  ligand.pqr      (setup.py)
bd_sims/
├── bd_1/
├── results.json
├── convergence.json
├── receptor*.dx  ligand*.dx
├── *.cache
├── encounters.csv  trajectories.csv  fpt_distribution.csv
├── radial_density.csv  contact_frequency.csv  near_misses.csv
├── pose_clusters.csv  milestone_flux.csv
└── angular_map.npz  energetics.npz  paths.npz  p_commit.npz  transition_matrix.npz
```

## Once simulation finishes, the following files will be generated with multiple GPUs
```
input.xml  rxns.xml  receptor.pqr  ligand.pqr      (setup.py)
bd_sims/
├── bd_1/ ... bd_N/   one per GPU, each a full slice with its own results.json
├── results.json      pooled across GPUs, read this
├── convergence.json
├── receptor*.dx  ligand*.dx
├── *.cache
├── encounters.csv  trajectories.csv  fpt_distribution.csv
├── radial_density.csv  contact_frequency.csv  near_misses.csv
├── pose_clusters.csv  milestone_flux.csv
└── angular_map.npz  energetics.npz  paths.npz  p_commit.npz  transition_matrix.npz
```
