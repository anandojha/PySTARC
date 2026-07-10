# Thrombin-thrombomodulin (protein-protein complex)

## System

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th></tr></thead>
<tbody>
<tr><td>Receptor</td><td>Thrombin</td></tr>
<tr><td>Ligand</td><td>Thrombomodulin</td></tr>
<tr><td>b-surface</td><td>175.0 Å</td></tr>
<tr><td>Escape sphere</td><td>350.0 Å</td></tr>
<tr><td>Debye length</td><td>7.86 Å (150 mM NaCl)</td></tr>
<tr><td>Born desolvation</td><td>enabled</td></tr>
<tr><td>Hydrodynamic interactions</td><td>enabled</td></tr>
<tr><td>Overlap check</td><td>enabled</td></tr>
<tr><td>Trajectories</td><td>100,000 (quick demo, the published run used 5,000,000)</td></tr>
</tbody>
</table>

## Input files (provided)

<table width="100%">
<thead><tr><th align="left">File</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>receptor.pqr</code></td><td>Thrombin PQR file (pre-computed).</td></tr>
<tr><td><code>ligand.pqr</code></td><td>Thrombomodulin PQR file (pre-computed).</td></tr>
<tr><td><code>rxns.xml</code></td><td>Reaction criterion with binding-site atom pairs and cutoff distances.</td></tr>
<tr><td><code>input.xml</code></td><td>PySTARC input file with simulation parameters.</td></tr>
<tr><td><code>bb_effect.py</code></td><td>Brownian bridge A/B test: runs 4 seeds to verify BB genuinely improves P<sub>rxn</sub> (diagnostic script).</td></tr>
<tr><td><code>run.sh</code></td><td>Runs BD simulation in one command.</td></tr>
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
<thead><tr><th align="left">Output file</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>results.json</code></td><td>k<sub>on</sub>, P<sub>rxn</sub>, Wilson 95% CI, k<sub>b</sub>, D<sub>rel</sub>, wall time, and GPU info.</td></tr>
<tr><td><code>convergence.json</code></td><td>Convergence analysis: SE, relative SE, Wilson CI, convergence verdict, and trajectory estimates for target precision.</td></tr>
<tr><td><code>trajectories.csv</code></td><td>Per-trajectory record: number of steps, starting pose, minimum distance reached, and number of returns from the escape sphere.</td></tr>
<tr><td><code>encounters.csv</code></td><td>Binding encounter poses for reacted trajectories: final position, orientation, and contact distances.</td></tr>
<tr><td><code>near_misses.csv</code></td><td>Trajectories that approached the reaction surface but escaped.</td></tr>
<tr><td><code>fpt_distribution.csv</code></td><td>First-passage times for reacted trajectories.</td></tr>
<tr><td><code>pose_clusters.csv</code></td><td>Clustered binding poses from encounter geometries.</td></tr>
<tr><td><code>paths.npz</code></td><td>Full trajectory coordinates sampled at configurable intervals.</td></tr>
<tr><td><code>energetics.npz</code></td><td>Per-step energies and forces along trajectories.</td></tr>
<tr><td><code>radial_density.csv</code></td><td>Radial probability density as a function of distance from the receptor.</td></tr>
<tr><td><code>angular_map.npz</code></td><td>Angular occupancy map (theta, phi) on the b-surface.</td></tr>
<tr><td><code>contact_frequency.csv</code></td><td>Per-pair contact frequencies for the reaction criterion atom pairs.</td></tr>
<tr><td><code>milestone_flux.csv</code></td><td>Net flux across radial shells.</td></tr>
<tr><td><code>transition_matrix.npz</code></td><td>Radial shell-to-shell transition counts.</td></tr>
<tr><td><code>p_commit.npz</code></td><td>Commitment probabilities at each radial shell.</td></tr>
</tbody>
</table>

A timestamped log file (`pystarc_YYYYMMDD_HHMMSS.log`) is also written to `bd_sims/` containing the full simulation output.

## Notes
- This is a protein-protein complex with a large ligand (thrombomodulin), so force evaluation is automatically batched to fit GPU memory.
- The Brownian bridge diagnostic (`bb_effect.py`) is optional and only needed to validate the BB implementation.
