# beta_cyclodextrin_guests

7 beta cyclodextrin host guest complexes. Host and guest.

## Complexes
```
beta_cyclodextrin_guests/
├── BCD_1-butanol/
├── BCD_1-naphthylethanol/
├── BCD_1-propanol/
├── BCD_2-naphthylethanol/
├── BCD_aspirin/
├── BCD_methyl_butyrate/
└── BCD_tertbutanol/
```

## Files within each complex
```
complex.pdb                 structure
complex.parm7               topology
config.xml                  parameters
setup.py                    builds inputs
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/beta_cyclodextrin_guests
conda activate PySTARC
module load cuda
bash BCD_aspirin/run.sh
all: for d in */; do bash "$d/run.sh"; done
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/beta_cyclodextrin_guests
conda activate PySTARC
module load cuda
cd BCD_aspirin && sbatch submit_SLURM_single_GPU.sh
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/beta_cyclodextrin_guests
conda activate PySTARC
module load cuda
cd BCD_aspirin && sbatch submit_SLURM_multi_GPUs.sh
```

## Once simulation finishes, the following files will be generated with 1 GPU, per complex
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

## Once simulation finishes, the following files will be generated with multiple GPUs, per complex
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
