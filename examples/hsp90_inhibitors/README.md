# HSP90 inhibitors

Protein ligand. Six HSP90 inhibitors.

One folder per compound. Each folder is an independent run.

Needs the OpenEye Toolkits for setup.

## Shipped

    hsp90_inhibitors/
    ├── HSP90-aminopyridine/
    ├── HSP90-indazole_5LNZ/
    ├── HSP90-indazole_5OCI/
    ├── HSP90-quinazoline/
    ├── HSP90-quinazoline_6EI5/
    └── HSP90-resorcinol/

Each compound folder.

    <compound>/
    ├── complex.pdb                 bound complex structure
    ├── config.xml                  all parameters
    ├── setup.py                    builds the inputs
    ├── run.sh                      local run
    ├── submit_SLURM_single_GPU.sh  one GPU
    └── submit_SLURM_multi_GPUs.sh  many GPUs

## Run

One compound, local.

    bash HSP90-aminopyridine/run.sh

All compounds, local.

    for d in */; do bash "$d/run.sh"; done

One compound on the cluster.

    cd HSP90-aminopyridine && sbatch submit_SLURM_single_GPU.sh

## setup.py writes, inside each compound folder

    input.xml       PySTARC input
    rxns.xml        reaction criterion
    receptor.pqr    receptor charges and radii
    ligand.pqr      ligand charges and radii

## The run writes bd_sims/, inside each compound folder

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

One k_on per compound, in <compound>/bd_sims/results.json.
