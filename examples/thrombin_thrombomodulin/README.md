# Thrombin thrombomodulin

Protein protein. Electrostatically steered.

Inputs are shipped ready to run. No setup step.

## Shipped

    thrombin_thrombomodulin/
    ├── input.xml                   PySTARC input
    ├── rxns.xml                    reaction criterion
    ├── receptor.pqr                receptor charges and radii
    ├── ligand.pqr                  ligand charges and radii
    ├── run.sh                      local run
    ├── submit_SLURM_single_GPU.sh  one GPU
    └── submit_SLURM_multi_GPUs.sh  many GPUs

## Run

Local.

    bash run.sh

One GPU.

    sbatch submit_SLURM_single_GPU.sh

Many GPUs.

    sbatch submit_SLURM_multi_GPUs.sh

## The run writes bd_sims/

    bd_sims/
    ├── results.json           k_on and P_rxn. Read this first.
    ├── convergence.json       running rate and error
    ├── bd_1 ... bd_N          one directory per GPU
    ├── receptor*.dx           receptor APBS and Born grids
    ├── ligand*.dx             ligand APBS and Born grids
    ├── *.cache                hydrodynamic radius
    ├── encounters.csv         encounter records
    ├── trajectories.csv       saved positions
    ├── fpt_distribution.csv   first passage times
    ├── radial_density.csv     radial density
    ├── contact_frequency.csv  contact frequency
    ├── near_misses.csv        near misses
    ├── pose_clusters.csv      bound pose clusters
    ├── milestone_flux.csv     milestone flux
    ├── angular_map.npz        angular occupancy
    ├── energetics.npz         energy terms
    ├── paths.npz              reactive paths
    ├── p_commit.npz           commitment probability
    └── transition_matrix.npz  transition matrix

## Expect

k_on in bd_sims/results.json.
