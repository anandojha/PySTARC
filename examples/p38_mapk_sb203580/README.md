# p38_mapk_sb203580

## Files
```
1A9U.pdb                    PDB structure of the complex
config.xml                  PySTARC configuration file
setup.py                    Reads the config.xml file to set up the input.xml file
run.sh                      Script to run PySTARC simulations on the workstation
submit_SLURM_single_GPU.sh  Script to run PySTARC simulations on the cluster with 1 GPU
submit_SLURM_multi_GPUs.sh  Script to run PySTARC simulations on the cluster with multiple GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/p38_mapk_sb203580  # Navigate to the example directory within PySTARC
conda activate PySTARC                   # Activate the PySTARC environment
module load cuda                         # Load CUDA
bash run.sh                              # Run the simulation
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/p38_mapk_sb203580  # Navigate to the example directory within PySTARC
conda activate PySTARC                   # Activate the PySTARC environment
module load cuda                         # Load CUDA
sbatch submit_SLURM_single_GPU.sh        # Submit on 1 GPU
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/p38_mapk_sb203580  # Navigate to the example directory within PySTARC
conda activate PySTARC                   # Activate the PySTARC environment
module load cuda                         # Load CUDA
sbatch submit_SLURM_multi_GPUs.sh        # Submit on multiple GPUs
```

## Once the simulation finishes, the following files will be generated with 1 GPU
```
input.xml                         PySTARC input file
rxns.xml                          Reaction criterion file
receptor.pqr                      PQR file for receptor charges and radii
ligand.pqr                        PQR file for ligand charges and radii
bd_sims/
├── bd_1/                         Directory of the single GPU run
├── results.json                  Association rate constant, reaction probability, and confidence intervals
├── convergence.json              Running rate estimate versus the number of trajectories
├── receptor0.dx                  Coarse APBS electrostatic grid of the receptor
├── receptor1.dx                  Fine APBS electrostatic grid of the receptor
├── receptor0_born.dx             Coarse Born desolvation grid of the receptor
├── receptor1_born.dx             Fine Born desolvation grid of the receptor
├── ligand0.dx                    Coarse APBS electrostatic grid of the ligand
├── ligand1.dx                    Fine APBS electrostatic grid of the ligand
├── ligand0_born.dx               Coarse Born desolvation grid of the ligand
├── ligand1_born.dx               Fine Born desolvation grid of the ligand
├── receptor.pqr.r_hydro_*.cache  Cached hydrodynamic radius of the receptor
├── ligand.pqr.r_hydro_*.cache    Cached hydrodynamic radius of the ligand
├── pystarc_<timestamp>.log       Run log file
├── trajectories.csv              Fate and step count of each trajectory
├── encounters.csv                Records of the encounter events
├── near_misses.csv               Records of the close approaches
├── contact_frequency.csv         Frequency of each native contact
├── pose_clusters.csv             Clusters of the bound poses
├── milestone_flux.csv            Outward and inward crossings through the concentric shells
├── radial_density.csv            Radial density profile of the ligand
├── fpt_distribution.csv          Distribution of the first passage times
├── angular_map.npz               Angular map of the encounters
├── energetics.npz                Interaction energetics along the trajectories
├── paths.npz                     Samples of the reactive paths
├── p_commit.npz                  Committor probabilities
└── transition_matrix.npz         Markov transition matrix between the concentric shells
```

## Once the simulation finishes, the following files will be generated with multiple GPUs
```
input.xml                             PySTARC input file
rxns.xml                              Reaction criterion file
receptor.pqr                          PQR file for receptor charges and radii
ligand.pqr                            PQR file for ligand charges and radii
bd_sims/
├── bd_1/                             One directory per GPU, each running a share of the trajectories
│   ├── input.xml                     PySTARC input file for this slice
│   ├── receptor.pqr                  PQR file for receptor charges and radii
│   ├── ligand.pqr                    PQR file for ligand charges and radii
│   ├── results.json                  Association rate constant, reaction probability, and confidence intervals
│   ├── convergence.json              Running rate estimate versus the number of trajectories
│   ├── receptor0.dx                  Coarse APBS electrostatic grid of the receptor
│   ├── receptor1.dx                  Fine APBS electrostatic grid of the receptor
│   ├── receptor0_born.dx             Coarse Born desolvation grid of the receptor
│   ├── receptor1_born.dx             Fine Born desolvation grid of the receptor
│   ├── ligand0.dx                    Coarse APBS electrostatic grid of the ligand
│   ├── ligand1.dx                    Fine APBS electrostatic grid of the ligand
│   ├── ligand0_born.dx               Coarse Born desolvation grid of the ligand
│   ├── ligand1_born.dx               Fine Born desolvation grid of the ligand
│   ├── receptor.pqr.r_hydro_*.cache  Cached hydrodynamic radius of the receptor
│   ├── ligand.pqr.r_hydro_*.cache    Cached hydrodynamic radius of the ligand
│   ├── pystarc_<timestamp>.log       Run log file
│   ├── trajectories.csv              Fate and step count of each trajectory
│   ├── encounters.csv                Records of the encounter events
│   ├── near_misses.csv               Records of the close approaches
│   ├── contact_frequency.csv         Frequency of each native contact
│   ├── pose_clusters.csv             Clusters of the bound poses
│   ├── milestone_flux.csv            Outward and inward crossings through the concentric shells
│   ├── radial_density.csv            Radial density profile of the ligand
│   ├── fpt_distribution.csv          Distribution of the first passage times
│   ├── angular_map.npz               Angular map of the encounters
│   ├── energetics.npz                Interaction energetics along the trajectories
│   ├── paths.npz                     Samples of the reactive paths
│   ├── p_commit.npz                  Committor probabilities
│   └── transition_matrix.npz         Markov transition matrix between the concentric shells
├── ...
├── bd_N/                             Directory from the Nth GPU
├── results.json                      Association rate pooled across all GPUs, read this file
├── convergence.json                  Running rate estimate versus the number of trajectories
├── receptor0.dx                      Coarse APBS electrostatic grid of the receptor
├── receptor1.dx                      Fine APBS electrostatic grid of the receptor
├── receptor0_born.dx                 Coarse Born desolvation grid of the receptor
├── receptor1_born.dx                 Fine Born desolvation grid of the receptor
├── ligand0.dx                        Coarse APBS electrostatic grid of the ligand
├── ligand1.dx                        Fine APBS electrostatic grid of the ligand
├── ligand0_born.dx                   Coarse Born desolvation grid of the ligand
├── ligand1_born.dx                   Fine Born desolvation grid of the ligand
├── receptor.pqr.r_hydro_*.cache      Cached hydrodynamic radius of the receptor
├── ligand.pqr.r_hydro_*.cache        Cached hydrodynamic radius of the ligand
├── pystarc_<timestamp>.log           Run log file
├── trajectories.csv                  Fate and step count of each trajectory
├── encounters.csv                    Records of the encounter events
├── near_misses.csv                   Records of the close approaches
├── contact_frequency.csv             Frequency of each native contact
├── pose_clusters.csv                 Clusters of the bound poses
├── milestone_flux.csv                Outward and inward crossings through the concentric shells
├── radial_density.csv                Radial density profile of the ligand
├── fpt_distribution.csv              Distribution of the first passage times
├── angular_map.npz                   Angular map of the encounters
├── energetics.npz                    Interaction energetics along the trajectories
├── paths.npz                         Samples of the reactive paths
├── p_commit.npz                      Committor probabilities
└── transition_matrix.npz             Markov transition matrix between the concentric shells
```
