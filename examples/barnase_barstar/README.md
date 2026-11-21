# barnase_barstar

Runs on CPUs and needs no GPU. Total trajectories split evenly across NUMA-local tasks in `submit.sh` (default `2` Rome nodes × `8` tasks × `16` CPUs = `256` CPUs total).

## Input files
```
receptor.pdb         Prepared PDB structure of the receptor (barnase)
ligand.pdb           Prepared PDB structure of the diffusing chain (barstar)
reaction_pairs.json  Receptor atom and chain bead contact pairs
config.xml           PySTARC configuration file
run.sh               Script to run a single simulation on the workstation
make_grids.py        Builds the APBS electrostatic and Born grids
submit.sh            Multi-node CPU SLURM script (builds grids, runs NUMA-pinned shards, pools rate)
submit_shards.sh     Stages and submits independent single-node CPU shard jobs
combine_shards.py    Pools the shards into one rate
```

## To run Brownian dynamics simulations on a SLURM cluster
```bash
cd ~/PySTARC/examples/barnase_barstar
conda activate PySTARC
sbatch submit.sh                    # 2 Rome nodes, 8 tasks x 16 CPUs per node (256 CPUs total, default)
sbatch --nodes=1 submit.sh          # 1 Rome node, 8 tasks x 16 CPUs (128 CPUs total)
sbatch --nodes=10 submit.sh         # 10 Rome nodes, 8 tasks x 16 CPUs per node (1,280 CPUs total)
```

## Generated output files
```
bd_sims/
├── chain.json                Bead chain definition of the ligand
├── reaction_pairs.json       Interface contact pairs
├── receptor.pqr              Centered receptor PQR file
├── input.xml                 Generated PySTARC runtime input file
└── build_record.json         Build provenance and checksums
apbs_output/                  APBS electrostatic and Born grids of the receptor
shards/
├── shard_01/                 First NUMA-local shard with independent seed
│   └── bd_sims/results.json  Association rate from shard 1
└── shard_16/                 Sixteenth NUMA-local shard with independent seed
    └── bd_sims/results.json  Association rate from shard 16
```
