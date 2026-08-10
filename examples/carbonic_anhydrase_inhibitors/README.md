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
<PDB>.pdb                   PDB structure of the complex
config.xml                  Parameter file
setup.py                    Reads the config.xml file to set up the input.xml file
run.sh                      Script to run PySTARC simulations on the workstation
submit_SLURM_single_GPU.sh  Script to run PySTARC simulations on the cluster with 1 GPU
submit_SLURM_multi_GPUs.sh  Script to run PySTARC simulations on the cluster with multiple GPUs
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
input.xml                         PySTARC input file
rxns.xml                          Reaction criterion file
receptor.pqr                      PQR file for receptor charges and radii
ligand.pqr                        PQR file for ligand charges and radii
bd_sims/
├── bd_1/                         the run
├── results.json                  k_on, P_rxn, intervals
├── convergence.json              running rate vs count
├── receptor0.dx                  receptor grid, coarse
├── receptor1.dx                  receptor grid, fine
├── receptor0_born.dx             receptor Born grid, coarse
├── receptor1_born.dx             receptor Born grid, fine
├── ligand0.dx                    ligand grid, coarse
├── ligand1.dx                    ligand grid, fine
├── ligand0_born.dx               ligand Born grid, coarse
├── ligand1_born.dx               ligand Born grid, fine
├── receptor.pqr.r_hydro_*.cache  receptor hydrodynamic radius
├── ligand.pqr.r_hydro_*.cache    ligand hydrodynamic radius
├── pystarc_<timestamp>.log       run log
├── trajectories.csv              per trajectory fate
├── encounters.csv                encounter records
├── near_misses.csv               close approaches
├── contact_frequency.csv         contacts fired
├── pose_clusters.csv             bound pose clusters
├── milestone_flux.csv            milestone flux
├── radial_density.csv            radial density
├── fpt_distribution.csv          first passage times
├── angular_map.npz               angular map
├── energetics.npz                interaction energetics
├── paths.npz                     reactive paths
├── p_commit.npz                  committor probabilities
└── transition_matrix.npz         milestone transitions
```

## Once simulation finishes, the following files will be generated with multiple GPUs, per complex
```
input.xml                             PySTARC input
rxns.xml                              reaction criterion
receptor.pqr                          receptor charges and radii
ligand.pqr                            ligand charges and radii
bd_sims/
├── bd_1/                             one per GPU, each a full slice
│   ├── input.xml                     this shard input
│   ├── receptor.pqr                  receptor charges and radii
│   ├── ligand.pqr                    ligand charges and radii
│   ├── results.json                  k_on, P_rxn, intervals
│   ├── convergence.json              running rate vs count
│   ├── receptor0.dx                  receptor grid, coarse
│   ├── receptor1.dx                  receptor grid, fine
│   ├── receptor0_born.dx             receptor Born grid, coarse
│   ├── receptor1_born.dx             receptor Born grid, fine
│   ├── ligand0.dx                    ligand grid, coarse
│   ├── ligand1.dx                    ligand grid, fine
│   ├── ligand0_born.dx               ligand Born grid, coarse
│   ├── ligand1_born.dx               ligand Born grid, fine
│   ├── receptor.pqr.r_hydro_*.cache  receptor hydrodynamic radius
│   ├── ligand.pqr.r_hydro_*.cache    ligand hydrodynamic radius
│   ├── pystarc_<timestamp>.log       run log
│   ├── trajectories.csv              per trajectory fate
│   ├── encounters.csv                encounter records
│   ├── near_misses.csv               close approaches
│   ├── contact_frequency.csv         contacts fired
│   ├── pose_clusters.csv             bound pose clusters
│   ├── milestone_flux.csv            milestone flux
│   ├── radial_density.csv            radial density
│   ├── fpt_distribution.csv          first passage times
│   ├── angular_map.npz               angular map
│   ├── energetics.npz                interaction energetics
│   ├── paths.npz                     reactive paths
│   ├── p_commit.npz                  committor probabilities
│   └── transition_matrix.npz         milestone transitions
├── ...
├── bd_N/
├── results.json                      pooled across GPUs, read this
├── convergence.json                  running rate vs count
├── receptor0.dx                      receptor grid, coarse
├── receptor1.dx                      receptor grid, fine
├── receptor0_born.dx                 receptor Born grid, coarse
├── receptor1_born.dx                 receptor Born grid, fine
├── ligand0.dx                        ligand grid, coarse
├── ligand1.dx                        ligand grid, fine
├── ligand0_born.dx                   ligand Born grid, coarse
├── ligand1_born.dx                   ligand Born grid, fine
├── receptor.pqr.r_hydro_*.cache      receptor hydrodynamic radius
├── ligand.pqr.r_hydro_*.cache        ligand hydrodynamic radius
├── pystarc_<timestamp>.log           run log
├── trajectories.csv                  per trajectory fate
├── encounters.csv                    encounter records
├── near_misses.csv                   close approaches
├── contact_frequency.csv             contacts fired
├── pose_clusters.csv                 bound pose clusters
├── milestone_flux.csv                milestone flux
├── radial_density.csv                radial density
├── fpt_distribution.csv              first passage times
├── angular_map.npz                   angular map
├── energetics.npz                    interaction energetics
├── paths.npz                         reactive paths
├── p_commit.npz                      committor probabilities
└── transition_matrix.npz             milestone transitions
```
