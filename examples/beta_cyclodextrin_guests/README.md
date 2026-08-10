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
complex.pdb                 PDB structure of the complex
complex.parm7               AMBER topology file
config.xml                  Parameter file
setup.py                    Reads the config.xml file to set up the input.xml file
run.sh                      Script to run PySTARC simulations on the workstation
submit_SLURM_single_GPU.sh  Script to run PySTARC simulations on the cluster with 1 GPU
submit_SLURM_multi_GPUs.sh  Script to run PySTARC simulations on the cluster with multiple GPUs
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
input.xml                             PySTARC input file
rxns.xml                              Reaction criterion file
receptor.pqr                          PQR file for receptor charges and radii
ligand.pqr                            PQR file for ligand charges and radii
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
