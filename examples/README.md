# Examples

<table width="100%">
<thead><tr><th align="left">Example</th><th align="left">System</th><th align="left">Type</th></tr></thead>
<tbody>
<tr><td><code>two_charged_spheres/</code></td><td>Two oppositely charged spheres</td><td>Analytical validation</td></tr>
<tr><td><code>trypsin_benzamidine/</code></td><td>Trypsin and benzamidine</td><td>Protein and ligand</td></tr>
<tr><td><code>beta_cyclodextrin_guests/</code></td><td>7 beta cyclodextrin host guest complexes</td><td>Host and guest</td></tr>
<tr><td><code>thrombin_thrombomodulin/</code></td><td>Thrombin and thrombomodulin</td><td>Protein and protein</td></tr>
<tr><td><code>p38_mapk_sb203580/</code></td><td>p38 MAPK with SB203580</td><td>Protein and ligand</td></tr>
<tr><td><code>carbonic_anhydrase_inhibitors/</code></td><td>7 sulfonamide inhibitors across 3 isozymes</td><td>Protein and ligand</td></tr>
<tr><td><code>hsp90_inhibitors/</code></td><td>6 HSP90 inhibitors</td><td>Protein and ligand</td></tr>
<tr><td><code>ttk_inhibitors/</code></td><td>8 TTK (MPS1) kinase inhibitors</td><td>Protein and ligand</td></tr>
<tr><td><code>barnase_barstar/</code></td><td>Barnase and barstar with a flexible chain</td><td>Chain BD</td></tr>
</tbody>
</table>

A directory carries only the files that cannot be regenerated. 

**Systems built from an RCSB entry** (`carbonic_anhydrase_inhibitors`, `p38_mapk_sb203580`, `ttk_inhibitors`). `setup.py` downloads the structure from the `pdb_id` in the config and, for the sulfonamides, builds the ligand from its SMILES.

```
<system>/
├── config.xml                all parameters, including pdb_id and the reaction criterion
├── setup.py                  downloads the entry and builds every input
├── submit_SLURM_*.sh         cluster launch scripts
├── run.sh                    local single GPU run
└── <PDBID>.pdb               the downloaded structure, kept so the example runs offline
```

**Systems built from a local complex** (`hsp90_inhibitors`, `beta_cyclodextrin_guests`, `trypsin_benzamidine`). `setup.py` reads a bound state structure that must ship with the example.

```
<system>/
├── config.xml                all parameters
├── setup.py                  builds every input from the local complex
├── submit_SLURM_*.sh         cluster launch scripts
├── run.sh                    local single GPU run
├── complex.pdb               the bound state, required
└── complex.parm7             AMBER topology, prmtop source only (or complex.prmtop)
```

**Static systems** (`two_charged_spheres`, `thrombin_thrombomodulin`). No `setup.py`. The run inputs are the source and ship as is.

```
<system>/
├── input.xml                 simulation parameters
├── rxns.xml                  reaction criterion
├── receptor.pqr              receptor charges and radii
├── ligand.pqr                ligand charges and radii
├── submit_SLURM_*.sh         cluster launch scripts
├── run.sh                    local single GPU run
├── analytical.py             two_charged_spheres check versus the exact solution
├── convergence.py            two_charged_spheres multi seed convergence
└── bb_effect.py              thrombin Brownian bridge diagnostic
```

## What a run generates

Nothing below is committed. `setup.py` writes the run inputs, then the run fills `bd_sims/`.

Written by `setup.py` (for the systems that have one):

```
<system>/
├── input.xml                 simulation parameters
├── rxns.xml                  reaction criterion
├── receptor.pqr              receptor charges and radii
├── ligand.pqr                ligand charges and radii
├── receptor.pdb              cleaned receptor structure
├── ligand.pdb                cleaned ligand structure
├── protein.prmtop            receptor topology
├── protein.rst7              receptor coordinates
├── ligand.prmtop             ligand topology
└── ligand.rst7               ligand coordinates
```

Written by the run into `bd_sims/`:

```
bd_sims/
├── receptor0.dx              APBS electrostatic grid, coarse
├── receptor1.dx              APBS electrostatic grid, fine
├── receptor0_born.dx         Born desolvation grid, coarse
├── receptor1_born.dx         Born desolvation grid, fine
├── ligand0.dx                ligand electrostatic grid, coarse
├── ligand1.dx                ligand electrostatic grid, fine
├── ligand0_born.dx           ligand Born grid, coarse
├── ligand1_born.dx           ligand Born grid, fine
├── *.r_hydro_*.cache         hydrodynamic radius cache
├── pystarc_<timestamp>.log   the run log
├── results.json              k_on, P_rxn, confidence intervals, run statistics
├── convergence.json          running estimate versus trajectory count
├── trajectories.csv          per trajectory fate and step count
├── encounters.csv            encounter records
├── near_misses.csv           close approach records
├── contact_frequency.csv     which contacts fired
├── pose_clusters.csv         bound pose clusters
├── milestone_flux.csv        milestone flux
├── radial_density.csv        radial density profile
├── fpt_distribution.csv      first passage time distribution
├── angular_map.npz           angular encounter map
├── energetics.npz            interaction energetics
├── paths.npz                 reactive path samples
├── p_commit.npz              committor probabilities
└── transition_matrix.npz     milestone transition matrix
```

A multiple GPU run instead creates one shard per GPU, and the combiner pools them into the top level `results.json`, which is the number to read.

```
bd_sims/
├── bd_1/ ... bd_4/           one shard per GPU, each with its own grids and log
└── results.json              pooled result across all shards
```

## Quick start

Every system ships a `run.sh` for a local single GPU run. It builds the inputs, runs the Brownian dynamics on one GPU, and runs any verification the system provides.

```bash
conda activate PySTARC
module load cuda
cd examples/<example_name>
bash run.sh
```

The SLURM scripts are for a cluster with a `gpu` partition and are not needed for a local run:

```bash
sbatch submit_SLURM_single_GPU.sh      # one GPU
sbatch submit_SLURM_multi_GPUs.sh      # four GPUs, pooled at the end
```

The primary result is `bd_sims/results.json`, holding k<sub>on</sub>, P<sub>rxn</sub>, its confidence interval, and the run statistics.
