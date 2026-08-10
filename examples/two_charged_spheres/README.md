# Two charged spheres

Analytical validation. Two point charges with an exact Smoluchowski rate.

Inputs are shipped ready to run. No setup step.

## Shipped

    two_charged_spheres/
    ├── input.xml                   PySTARC input
    ├── rxns.xml                    reaction criterion
    ├── receptor.pqr                fixed sphere charge and radius
    ├── ligand.pqr                  mobile sphere charge and radius
    ├── analytical.py               exact Smoluchowski rate
    ├── convergence.py              rate against trajectory count
    ├── run.sh                      local run
    ├── submit_SLURM_single_GPU.sh  one GPU
    └── submit_SLURM_multi_GPUs.sh  many GPUs

## Run

Local. Runs the BD, then the analytical check, then the convergence check.

    bash run.sh

One GPU.

    sbatch submit_SLURM_single_GPU.sh

Many GPUs.

    sbatch submit_SLURM_multi_GPUs.sh

## The run writes bd_sims/

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

k_on in bd_sims/results.json.
analytical.py prints the exact rate. The BD rate matches it.
convergence.py prints the rate against trajectory count.
