# carbonic_anhydrase_inhibitors

7 sulfonamide inhibitors across 3 isozymes. Protein and ligand.

## Complexes
```
carbonic_anhydrase_inhibitors/
├── ca13_azm/
├── ca13_vd1125/
├── ca13_vd1126/
├── ca13_vd1209/
├── ca13_vd1269/
├── ca1_vd1269/
└── ca2_vd1142/
```

## Files within each complex
```
<PDB>.pdb                   structure
config.xml                  parameters
setup.py                    builds inputs
run.sh                      local run
submit_SLURM_single_GPU.sh  1 GPU
submit_SLURM_multi_GPUs.sh  many GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/carbonic_anhydrase_inhibitors
conda activate PySTARC
module load cuda
bash ca13_azm/run.sh
all: for d in */; do bash "$d/run.sh"; done
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/carbonic_anhydrase_inhibitors
conda activate PySTARC
module load cuda
cd ca13_azm && sbatch submit_SLURM_single_GPU.sh
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/carbonic_anhydrase_inhibitors
conda activate PySTARC
module load cuda
cd ca13_azm && sbatch submit_SLURM_multi_GPUs.sh
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
