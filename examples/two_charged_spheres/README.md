# two_charged_spheres

Requires a CUDA GPU.

## Input files
```
config.xml                  PySTARC configuration file
receptor.pqr                PQR file for receptor charges and radii
ligand.pqr                  PQR file for ligand charges and radii
analytical.py               Compares the rate against the exact Smoluchowski solution
convergence.py              Reports the rate against trajectory count
run.sh                      Script to run PySTARC simulations on the workstation
submit.sh                   Unified SLURM script (1 GPU, 1-node multi-GPU, or multi-node GPUs)
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/two_charged_spheres  # Navigate to the example directory within PySTARC
conda activate PySTARC                     # Activate the PySTARC environment
bash run.sh                                # Run the simulation
```

## To run Brownian dynamics simulations on a SLURM cluster
```bash
cd ~/PySTARC/examples/two_charged_spheres
conda activate PySTARC
sbatch submit.sh                                # 1 node, 4 GPUs (default)
sbatch --nodes=1 --ntasks-per-node=1 submit.sh  # 1 node, 1 GPU
sbatch --nodes=2 --ntasks-per-node=4 submit.sh  # 2 nodes, 4 GPUs per node (8 total)
```

## Once the simulation finishes, the following files will be generated with 1 GPU
```
bd_sims/
├── input.xml                     Generated PySTARC runtime input file
├── rxns.xml                      Generated reaction criterion file
├── build_record.json             Build provenance and checksums
├── receptor.pqr                  Working PQR file for receptor charges and radii
├── ligand.pqr                    Working PQR file for ligand charges and radii
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
├── transition_matrix.npz         Markov transition matrix between the concentric shells
└── convergence_seed_*/           convergence.py multi seed run written by run.sh only
```

## Once the simulation finishes, the following files will be generated with multiple GPUs
```
bd_sims/
├── input.xml                         Generated PySTARC runtime input file
├── rxns.xml                          Generated reaction criterion file
├── build_record.json                 Build provenance and checksums
├── receptor.pqr                      Working PQR file for receptor charges and radii
├── ligand.pqr                        Working PQR file for ligand charges and radii
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
```
k_b     64.68 Å³/ps
P_rxn   0.451
k_on    1.76e10 M⁻¹s⁻¹
analytical.py compares P_rxn against the exact Smoluchowski solution and reports Pass when |z| < 3
```
