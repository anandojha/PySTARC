# hsp90_inhibitors

6 HSP90 inhibitors. Protein and ligand.

needs OpenEye Toolkits for setup

## Complexes
```
hsp90_inhibitors/
├── HSP90-aminopyridine/
├── HSP90-indazole_5LNZ/
├── HSP90-indazole_5OCI/
├── HSP90-quinazoline/
├── HSP90-quinazoline_6EI5/
└── HSP90-resorcinol/
```

## Files within each complex
```
complex.pdb                 structure
config.xml                  parameters
setup.py                    builds inputs
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/hsp90_inhibitors
conda activate PySTARC
module load cuda
bash HSP90-aminopyridine/run.sh
all: for d in */; do bash "$d/run.sh"; done
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/hsp90_inhibitors
conda activate PySTARC
module load cuda
cd HSP90-aminopyridine && sbatch submit_SLURM_single_GPU.sh
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/hsp90_inhibitors
conda activate PySTARC
module load cuda
cd HSP90-aminopyridine && sbatch submit_SLURM_multi_GPUs.sh
```

## Once simulation finishes, the following files will be generated with 1 GPU, per complex
```
input.xml  rxns.xml  receptor.pqr  ligand.pqr      (setup.py)
bd_sims/
├── results.json
├── convergence.json
├── receptor*.dx  ligand*.dx
├── *.cache
├── encounters.csv  trajectories.csv  fpt_distribution.csv
├── radial_density.csv  contact_frequency.csv  near_misses.csv
├── pose_clusters.csv  milestone_flux.csv
└── angular_map.npz  energetics.npz  paths.npz  p_commit.npz  transition_matrix.npz
```

## Once simulation finishes, the following files will be generated with multiple GPUs, per complex
```
bd_sims/
├── bd_1/ ... bd_N/   one per GPU, each a full slice with its own results.json
└── results.json      pooled across GPUs, read this
```
