# Thrombin-thrombomodulin (protein-protein complex)

## System

<table width="100%">
<thead><tr><th width="50%" align="left">Parameter</th><th width="50%" align="left">Value</th></tr></thead>
<tbody>
<tr><td width="50%">Receptor</td><td width="50%">Thrombin</td></tr>
<tr><td width="50%">Ligand</td><td width="50%">Thrombomodulin</td></tr>
<tr><td width="50%">b-surface</td><td width="50%">175.0 Å</td></tr>
<tr><td width="50%">Escape sphere</td><td width="50%">350.0 Å</td></tr>
<tr><td width="50%">Debye length</td><td width="50%">7.86 Å (150 mM NaCl)</td></tr>
<tr><td width="50%">Born desolvation</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Hydrodynamic interactions</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Overlap check</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Trajectories</td><td width="50%">100,000</td></tr>
</tbody>
</table>

## Input files (provided)

<table width="100%">
<thead><tr><th width="50%" align="left">File</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>receptor.pqr</code></td><td width="50%">Thrombin PQR file (pre-computed).</td></tr>
<tr><td width="50%"><code>ligand.pqr</code></td><td width="50%">Thrombomodulin PQR file (pre-computed).</td></tr>
<tr><td width="50%"><code>rxns.xml</code></td><td width="50%">Reaction criterion with binding-site atom pairs and cutoff distances.</td></tr>
<tr><td width="50%"><code>input.xml</code></td><td width="50%">PySTARC input file with simulation parameters.</td></tr>
<tr><td width="50%"><code>bb_effect.py</code></td><td width="50%">Brownian bridge A/B test: runs 4 seeds to verify BB genuinely improves P<sub>rxn</sub> (diagnostic script).</td></tr>
<tr><td width="50%"><code>run.sh</code></td><td width="50%">Runs BD simulation in one command.</td></tr>
</tbody>
</table>

## Run
`run.sh` runs the BD simulation, then runs `bb_effect.py` (4 seeds × 10k trajectories) to validate the Brownian bridge mechanism.
```bash
conda activate PySTARC
module load cuda
cd examples/thrombin_thrombomodulin
chmod +x run.sh
bash run.sh
```

## Run individual scripts (optional)
To run the scripts separately:
```bash
python ../../run_pystarc.py input.xml     # BD simulation only
python bb_effect.py                       # Brownian bridge diagnostic only
```

## Output files
After a simulation completes, all results are written to `bd_sims/`.

<table width="100%">
<thead><tr><th width="50%" align="left">Output file</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>results.json</code></td><td width="50%">k<sub>on</sub>, P<sub>rxn</sub>, Wilson 95% CI, k<sub>b</sub>, D<sub>rel</sub>, wall time, and GPU info.</td></tr>
<tr><td width="50%"><code>convergence.json</code></td><td width="50%">Convergence analysis: SE, relative SE, Wilson CI, convergence verdict, and trajectory estimates for target precision.</td></tr>
<tr><td width="50%"><code>trajectories.csv</code></td><td width="50%">Per-trajectory record: number of steps, starting pose, minimum distance reached, and number of returns from the escape sphere.</td></tr>
<tr><td width="50%"><code>encounters.csv</code></td><td width="50%">Binding encounter poses for reacted trajectories: final position, orientation, and contact distances.</td></tr>
<tr><td width="50%"><code>near_misses.csv</code></td><td width="50%">Trajectories that approached the reaction surface but escaped.</td></tr>
<tr><td width="50%"><code>fpt_distribution.csv</code></td><td width="50%">First-passage times for reacted trajectories.</td></tr>
<tr><td width="50%"><code>pose_clusters.csv</code></td><td width="50%">Clustered binding poses from encounter geometries.</td></tr>
<tr><td width="50%"><code>paths.npz</code></td><td width="50%">Full trajectory coordinates sampled at configurable intervals.</td></tr>
<tr><td width="50%"><code>energetics.npz</code></td><td width="50%">Per-step energies and forces along trajectories.</td></tr>
<tr><td width="50%"><code>radial_density.csv</code></td><td width="50%">Radial probability density as a function of distance from the receptor.</td></tr>
<tr><td width="50%"><code>angular_map.npz</code></td><td width="50%">Angular occupancy map (theta, phi) on the b-surface.</td></tr>
<tr><td width="50%"><code>contact_frequency.csv</code></td><td width="50%">Per-pair contact frequencies for the reaction criterion atom pairs.</td></tr>
<tr><td width="50%"><code>milestone_flux.csv</code></td><td width="50%">Net flux across radial shells.</td></tr>
<tr><td width="50%"><code>transition_matrix.npz</code></td><td width="50%">Radial shell-to-shell transition counts.</td></tr>
<tr><td width="50%"><code>p_commit.npz</code></td><td width="50%">Commitment probabilities at each radial shell.</td></tr>
</tbody>
</table>

A timestamped log file (`pystarc_YYYYMMDD_HHMMSS.log`) is also written to `bd_sims/` containing the full simulation output.

## Notes
- This is a protein-protein complex with a large ligand (thrombomodulin), so force evaluation is automatically batched to fit GPU memory.
- The Brownian bridge diagnostic (`bb_effect.py`) is optional and only needed to validate the BB implementation.
