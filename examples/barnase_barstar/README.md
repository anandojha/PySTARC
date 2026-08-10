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
chain.json  reaction_pairs.json  input.xml           (setup.py)
apbs_output/  barnase0.dx barnase1.dx *_born.dx      (make_grids.py)
shards/shard_01 ... shard_25/  each bd_sims/results.json
logs/
```

## Shards
```
n_trajectories x n_shards  200 x 25 = 5000
each shard                 own bd_sims/results.json
pooled                     combine_shards.py prints k_on ~8e8
```
