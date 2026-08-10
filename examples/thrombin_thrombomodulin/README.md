# thrombin_thrombomodulin

## files
```
input.xml                   PySTARC input
rxns.xml                    reaction criterion
receptor.pqr                receptor charges, radii
ligand.pqr                  ligand charges, radii
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## run
```
cd ~/PySTARC/examples/thrombin_thrombomodulin
bash run.sh                        local, slow, needs module load cuda
sbatch submit_SLURM_single_GPU.sh  1 GPU
sbatch submit_SLURM_multi_GPUs.sh  many GPUs
```

## produces
```
bd_sims/
├── results.json              k_on, P_rxn
├── convergence.json          running rate
├── bd_1 ... bd_N             one per GPU, each a full slice with its own results.json
├── receptor*.dx  ligand*.dx  APBS and Born grids
├── *.cache                   hydrodynamic radius
├── encounters.csv  trajectories.csv  fpt_distribution.csv
├── radial_density.csv  contact_frequency.csv  near_misses.csv
├── pose_clusters.csv  milestone_flux.csv
└── angular_map.npz  energetics.npz  paths.npz  p_commit.npz  transition_matrix.npz
```

## gpus
```
n_trajectories (config.xml)  split across GPUs
1 GPU                        bd_1
N GPUs                       bd_1 ... bd_N, each 1/N
pooled                       bd_sims/results.json
```
