# barnase_barstar

Barnase and barstar with a flexible chain. Chain BD.

## Files
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
cd ~/PySTARC/examples/barnase_barstar
source ~/.bashrc && conda activate PySTARC
python setup.py
sbatch -p ccb -N 1 --cpus-per-task=16 --mem=128G -t 6:00:00 --wrap "source ~/.bashrc; conda activate PySTARC; python make_grids.py"
./submit_shards.sh
python combine_shards.py
```

## Once simulation finishes, the following files will be generated
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
