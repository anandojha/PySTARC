# thrombin_thrombomodulin

Requires a CUDA GPU.

## Input files
```
input.xml                   PySTARC input file
rxns.xml                    Reaction criterion file
receptor.pqr                PQR file for receptor charges and radii
ligand.pqr                  PQR file for ligand charges and radii
run.sh                      Script to run PySTARC simulations on the workstation
submit_SLURM_single_GPU.sh  Script to run PySTARC simulations on the cluster with 1 GPU
submit_SLURM_multi_GPUs.sh  Script to run PySTARC simulations on the cluster with multiple GPUs
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/thrombin_thrombomodulin  # Navigate to the example directory within PySTARC
conda activate PySTARC                         # Activate the PySTARC environment
module load cuda                               # Load CUDA
bash run.sh                                    # Run the simulation
```

## To run Brownian dynamics simulations on the cluster with 1 GPU
```
cd ~/PySTARC/examples/thrombin_thrombomodulin  # Navigate to the example directory within PySTARC
conda activate PySTARC                         # Activate the PySTARC environment
module load cuda                               # Load CUDA
sbatch submit_SLURM_single_GPU.sh              # Submit on 1 GPU
```

## To run Brownian dynamics simulations on the cluster with multiple GPUs
```
cd ~/PySTARC/examples/thrombin_thrombomodulin  # Navigate to the example directory within PySTARC
conda activate PySTARC                         # Activate the PySTARC environment
module load cuda                               # Load CUDA
sbatch submit_SLURM_multi_GPUs.sh              # Submit on multiple GPUs
```

## Once the simulation finishes, the following files will be generated with 1 GPU
```
bd_sims/
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
├── pose_clusters.csv             Rotation angles of the encounter poses in 8 bins
├── milestone_flux.csv            Outward and inward crossings through the concentric shells
├── radial_density.csv            BD steps and BD time per radial shell
├── bd_time_to_reaction.csv       BD time inside r_esc of each reacting trajectory
├── angular_map.npz               BD steps and BD time per angular bin
├── energetics.npz                Force and time step along the saved paths
├── paths.npz                     Samples of the reactive paths
└── transition_matrix.npz         Markov transition matrix between the concentric shells
```

## Once the simulation finishes, the following files will be generated with multiple GPUs
```
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
│   ├── pose_clusters.csv             Rotation angles of the encounter poses in 8 bins
│   ├── milestone_flux.csv            Outward and inward crossings through the concentric shells
│   ├── radial_density.csv            BD steps and BD time per radial shell
│   ├── bd_time_to_reaction.csv       BD time inside r_esc of each reacting trajectory
│   ├── angular_map.npz               BD steps and BD time per angular bin
│   ├── energetics.npz                Force and time step along the saved paths
│   ├── paths.npz                     Samples of the reactive paths
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
├── trajectories.csv                  Fate and step count of each trajectory
├── encounters.csv                    Records of the encounter events
├── near_misses.csv                   Records of the close approaches
├── contact_frequency.csv             Frequency of each native contact
├── pose_clusters.csv                 Rotation angles of the encounter poses in 8 bins
├── milestone_flux.csv                Outward and inward crossings through the concentric shells
├── radial_density.csv                BD steps and BD time per radial shell
├── bd_time_to_reaction.csv           BD time inside r_esc of each reacting trajectory
├── angular_map.npz                   BD steps and BD time per angular bin
├── energetics.npz                    Force and time step along the saved paths
├── paths.npz                         Samples of the reactive paths
└── transition_matrix.npz             Markov transition matrix between the concentric shells
```

## Expected results
The values below are from the published runs, before the soft wall and the reaction step floor now in input.xml. They will be replaced when the revised runs finish.
```
Experimental k_on  6.70e6 M⁻¹s⁻¹
PySTARC k_on       (4.14 ± 0.41)e6 M⁻¹s⁻¹
k_on is printed in the run summary and written to bd_sims/results.json
```
