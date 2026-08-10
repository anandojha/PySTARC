# two_charged_spheres

Two oppositely charged spheres. Analytical validation.

## Files
```
input.xml                   PySTARC input
rxns.xml                    reaction criterion
receptor.pqr                fixed sphere
ligand.pqr                  mobile sphere
analytical.py               exact Smoluchowski rate
convergence.py              rate vs trajectory count
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/two_charged_spheres
conda activate PySTARC
module load cuda
bash run.sh
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/two_charged_spheres
conda activate PySTARC
module load cuda
sbatch submit_SLURM_single_GPU.sh
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/two_charged_spheres
conda activate PySTARC
module load cuda
sbatch submit_SLURM_multi_GPUs.sh
```

## Once simulation finishes, the following files will be generated with 1 GPU
```
bd_sims/
├── results.json
├── convergence.json
├── receptor*.dx  ligand*.dx
├── *.cache
├── encounters.csv  trajectories.csv  fpt_distribution.csv
├── radial_density.csv  contact_frequency.csv  near_misses.csv
├── pose_clusters.csv  milestone_flux.csv
└── angular_map.npz  energetics.npz  paths.npz  p_commit.npz  transition_matrix.npz
convergence_seed_*/   (convergence.py)
```

## Once simulation finishes, the following files will be generated with multiple GPUs
```
bd_sims/
├── bd_1/ ... bd_N/   one per GPU, each a full slice with its own results.json
└── results.json      pooled across GPUs, read this
```
