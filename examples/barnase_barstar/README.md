# barnase_barstar

## Input files
```
1BRS.pdb           PDB structure of the complex
barnase.pqr        PQR file for receptor charges and radii
config.xml         Parameter file
setup.py           Reads config.xml to build chain.json, reaction_pairs.json, and input.xml
make_grids.py      Builds the APBS electrostatic and Born grids
submit_shards.sh   Stages and submits the 25 shards on the cluster
combine_shards.py  Pools the shards into one rate
```

## To run Brownian dynamics simulations
```
cd ~/PySTARC/examples/barnase_barstar                                                                                                # Go to the example directory
source ~/.bashrc && conda activate PySTARC                                                                                           # Activate the PySTARC environment
python setup.py                                                                                                                      # Build chain.json, reaction_pairs.json, and input.xml
sbatch -p ccb -N 1 --cpus-per-task=16 --mem=128G -t 6:00:00 --wrap "source ~/.bashrc; conda activate PySTARC; python make_grids.py"  # Build the APBS grids
./submit_shards.sh                                                                                                                   # Stage and submit the 25 shards
python combine_shards.py                                                                                                             # Pool the shards into one rate
```

## Once the simulation finishes, the following files will be generated
```
chain.json                    Bead chain definition of the ligand
reaction_pairs.json           Interface contact pairs
input.xml                     PySTARC input file
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
Total trajectories  n_trajectories times n_shards, here 200 times 25 equals 5000
Each shard          Writes its own bd_sims/results.json
Pooled rate         combine_shards.py prints k_on near 8e8
```
