# Barnase barstar chain Brownian dynamics

Flexible chain BD for a protein protein association rate. The receptor barnase is held rigid and read from an APBS electrostatic grid, while the ligand barstar diffuses as a bead chain. A reaction is recorded when three interface pairs fall within 8.0 A of each other at once. The run is split into 25 independent single node shards of 200 trajectories, 5000 in total, and pooled into one rate with a Wilson interval.

## Settled run parameters

| parameter | value | note |
|---|---|---|
| reaction_n_needed | 3 | interface pairs within the cutoff at once |
| reaction_distance | 8.0 A | pair cutoff |
| n_trajectories | 200 | per shard |
| n_workers | 96 | one core per trajectory |
| n_shards | 25 | 5000 trajectories in total |
| max_steps | 500000 | per trajectory |
| bd_milestone_radius | 60.0 A | starting sphere |
| desolvation_alpha | 1.0 | full Born desolvation penalty |

All parameters live in `config.xml`. The PySTARC checkout is found automatically by walking up from this directory, so no paths are hardcoded.

## Steps

Activate the environment first.

```bash
source ~/.bashrc && conda activate PySTARC
```

**1. Build the inputs.** Reads `1BRS.pdb` and `barnase.pqr`, centres the receptor, and writes `chain.json`, `reaction_pairs.json`, and `input.xml` from `config.xml`.

```bash
python setup.py
```

**2. Solve the grids.** Builds the APBS electrostatic and Born grids under `apbs_output`. Only needed the first time, or after `apbs_output` is deleted.

```bash
mkdir -p logs && sbatch -p ccb -N 1 --cpus-per-task=16 --mem=128G -t 6:00:00 --job-name=barnase_grids --output=logs/grids.out --error=logs/grids.err --wrap "source ~/.bashrc; conda activate PySTARC; python make_grids.py"
```

**3. Stage and submit the shards.** Stages one shard per seed and submits all 25 jobs.

```bash
./submit_shards.sh
```

**4. Pool the results.** Reports P_rxn and k_on with a Wilson 95 percent interval once the shards finish.

```bash
python combine_shards.py
```

## Files

| file | role |
|---|---|
| `1BRS.pdb` | reference structure |
| `barnase.pqr` | receptor charges and radii |
| `config.xml` | simulation parameters |
| `setup.py` | builds `chain.json`, `reaction_pairs.json`, and `input.xml` |
| `make_grids.py` | builds the APBS grids |
| `submit_shards.sh` | stages and submits the shards |
| `combine_shards.py` | pools the shards |
