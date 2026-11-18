# barnase_barstar

Runs on CPUs and needs no GPU. The shard count and the CPUs and memory per shard are set in submit_shards.sh.

## Input files
```
barnase.pqr        PQR file for receptor charges and radii
barstar.pdb        Prepared PDB structure of the diffusing chain
config.xml         PySTARC configuration file
run.sh             Script to run a single simulation on the workstation
make_grids.py      Builds the APBS electrostatic and Born grids
submit_shards.sh   Stages and submits the 25 shards on the cluster
combine_shards.py  Pools the shards into one rate
```

## To run Brownian dynamics simulations
```
cd ~/PySTARC/examples/barnase_barstar  # Move to the example directory within PySTARC
source ~/.bashrc && conda activate PySTARC  # Activate the PySTARC environment
sbatch -p ccb -N 1 --cpus-per-task=16 --mem=128G -t 6:00:00 --wrap "source ~/.bashrc; conda activate PySTARC; python make_grids.py"  # Build the APBS grids
./submit_shards.sh  # Stage inputs and submit the 25 shards
python combine_shards.py  # Pool the shards into one rate
```

## Once the simulation finishes, the following files will be generated
```
bd_sims/
├── chain.json                Bead chain definition of the ligand
├── reaction_pairs.json       Interface contact pairs
├── receptor.pqr              Centered receptor PQR file
├── input.xml                 Generated PySTARC runtime input file
└── build_record.json         Build provenance and checksums
apbs_output/                  APBS electrostatic and Born grids of the receptor
shards/
├── shard_01/                 One independent shard with its own seed
│   └── bd_sims/results.json  Association rate from this shard
├── ...
└── shard_25/                 Last shard
logs/                         One log file per shard
```

## Shards
```
Total trajectories  n_trajectories times 25, here 200 times 25 gives 5000
Each shard          Writes its own bd_sims/results.json
Pooled rate         combine_shards.py prints k_on near 8e8 M⁻¹s⁻¹ against the experimental 6.0e8 M⁻¹s⁻¹
```
