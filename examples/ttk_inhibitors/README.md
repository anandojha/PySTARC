# TTK MPS1 inhibitors

Protein ligand. Eight TTK MPS1 kinase inhibitors.

One folder per complex. Each folder is an independent run.

Needs the OpenEye Toolkits for setup.

## Shipped

    ttk_inhibitors/
    ├── 2X9E/
    ├── 3GFW/
    ├── 3H9F/
    ├── 5LJJ/
    ├── 5N7V/
    ├── 5N84/
    ├── 5N93/
    └── 5NAD/

Each complex folder.

    <PDB>/
    ├── <PDB>.pdb                   bound complex structure
    ├── config.xml                  all parameters
    ├── setup.py                    builds the inputs
    ├── run.sh                      local run
    ├── submit_SLURM_single_GPU.sh  one GPU
    └── submit_SLURM_multi_GPUs.sh  many GPUs

## Run

One complex, local.

    bash 2X9E/run.sh

All complexes, local.

    for d in */; do bash "$d/run.sh"; done

One complex on the cluster.

    cd 2X9E && sbatch submit_SLURM_single_GPU.sh

## setup.py writes, inside each complex folder

    input.xml       PySTARC input
    rxns.xml        reaction criterion
    receptor.pqr    receptor charges and radii
    ligand.pqr      ligand charges and radii

## The run writes bd_sims/, inside each complex folder

    bd_sims/
    ├── results.json           k_on and P_rxn. Read this first.
    ├── convergence.json       running rate and error
    ├── bd_1 ... bd_N          one per GPU. Each is a full slice with its own results.json
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

## Single or many GPUs

The total trajectory count is set in config.xml. It splits evenly across the GPUs. One GPU writes bd_1 only. N GPUs write bd_1 ... bd_N, each running one Nth of the trajectories. The top level bd_sims/results.json is the pooled rate over all GPUs either way. Read that, not the per GPU files.

## Expect

One k_on per complex, in <complex>/bd_sims/results.json.
