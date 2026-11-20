# hsp90_inhibitors

Requires a CUDA GPU.

## Complexes
```
hsp90_inhibitors/
├── HSP90-aminopyridine/     HSP90 with an aminopyridine inhibitor
├── HSP90-indazole_5LNZ/     HSP90 with an indazole inhibitor
├── HSP90-indazole_5OCI/     HSP90 with an indazole inhibitor
├── HSP90-quinazoline/       HSP90 with a quinazoline inhibitor
├── HSP90-quinazoline_6EI5/  HSP90 with a quinazoline inhibitor
└── HSP90-resorcinol/        HSP90 with a resorcinol inhibitor
```

## Input files for each complex
```
receptor.pdb                Prepared PDB structure of the receptor
ligand.pdb                  Prepared PDB structure of the ligand
config.xml                  PySTARC configuration file
run.sh                      Script to run PySTARC simulations on the workstation
submit.sh                   Unified SLURM script (1 GPU, 1-node multi-GPU, or multi-node GPUs)
```

## To run Brownian dynamics simulations on the workstation
```
cd ~/PySTARC/examples/hsp90_inhibitors  # Navigate to the example directory within PySTARC
conda activate PySTARC                  # Activate the PySTARC environment
bash HSP90-aminopyridine/run.sh         # Run one complex
```

## To run Brownian dynamics simulations on a SLURM cluster
```bash
cd ~/PySTARC/examples/hsp90_inhibitors/HSP90-aminopyridine
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
├── receptor.pqr                  PQR file for receptor charges and radii
├── ligand.pqr                    PQR file for ligand charges and radii
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
├── input.xml                         Generated PySTARC runtime input file
├── rxns.xml                          Generated reaction criterion file
├── build_record.json                 Build provenance and checksums
├── receptor.pqr                      PQR file for receptor charges and radii
├── ligand.pqr                        PQR file for ligand charges and radii
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
