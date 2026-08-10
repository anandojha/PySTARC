# Beta cyclodextrin guests

Host guest. Seven neutral guests. Same host.

One folder per guest. Each folder is an independent run.

## Shipped

    beta_cyclodextrin_guests/
    ├── BCD_1-butanol/
    ├── BCD_1-naphthylethanol/
    ├── BCD_1-propanol/
    ├── BCD_2-naphthylethanol/
    ├── BCD_aspirin/
    ├── BCD_methyl_butyrate/
    └── BCD_tertbutanol/

Each guest folder.

    <guest>/
    ├── complex.pdb                 host guest structure
    ├── complex.parm7               Amber topology
    ├── config.xml                  all parameters
    ├── setup.py                    builds the inputs
    ├── run.sh                      local run
    ├── submit_SLURM_single_GPU.sh  one GPU
    └── submit_SLURM_multi_GPUs.sh  many GPUs

## Run

One guest, local.

    bash BCD_aspirin/run.sh

All guests, local.

    for d in */; do bash "$d/run.sh"; done

One guest on the cluster.

    cd BCD_aspirin && sbatch submit_SLURM_single_GPU.sh

## setup.py writes, inside each guest folder

    input.xml       PySTARC input
    rxns.xml        reaction criterion
    receptor.pqr    host charges and radii
    ligand.pqr      guest charges and radii

## The run writes bd_sims/, inside each guest folder

    bd_sims/
    ├── results.json           k_on and P_rxn. Read this first.
    ├── convergence.json       running rate and error
    ├── bd_1 ... bd_N          one directory per GPU
    ├── receptor*.dx           host APBS and Born grids
    ├── ligand*.dx             guest APBS and Born grids
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

One k_on per guest, in <guest>/bd_sims/results.json.
