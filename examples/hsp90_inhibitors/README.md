# hsp90_inhibitors

needs OpenEye Toolkits for setup

## folders
```
hsp90_inhibitors/
├── HSP90-aminopyridine/
├── HSP90-indazole_5LNZ/
├── HSP90-indazole_5OCI/
├── HSP90-quinazoline/
├── HSP90-quinazoline_6EI5/
└── HSP90-resorcinol/
```

## each folder
```
complex.pdb                 structure
config.xml                  parameters
setup.py                    builds inputs
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## run
```
cd ~/PySTARC/examples/hsp90_inhibitors
bash HSP90-aminopyridine/run.sh                              local, slow, needs module load cuda
cd HSP90-aminopyridine && sbatch submit_SLURM_single_GPU.sh  1 GPU
cd HSP90-aminopyridine && sbatch submit_SLURM_multi_GPUs.sh  many GPUs
all: for d in */; do bash "$d/run.sh"; done
```

## produces  (per folder)
```
input.xml  rxns.xml  receptor.pqr  ligand.pqr      (setup.py)
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
