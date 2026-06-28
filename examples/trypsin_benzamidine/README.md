# Trypsin-benzamidine (protein-ligand complex)

## System

<table width="100%">
<thead><tr><th width="50%" align="left">Property</th><th width="50%" align="left">Value</th></tr></thead>
<tbody>
<tr><td>Receptor</td><td>Trypsin (3220 atoms, Q = +6 e)</td></tr>
<tr><td>Ligand</td><td>Benzamidine (18 atoms, Q = +1 e)</td></tr>
<tr><td>b-surface</td><td>45.0 Å</td></tr>
<tr><td>Escape sphere</td><td>90.0 Å</td></tr>
<tr><td>Debye length</td><td>7.86 Å (150 mM NaCl)</td></tr>
<tr><td>Contact mode</td><td>polar (N/O/S on both sides)</td></tr>
<tr><td>Contact cutoff</td><td>6.0 Å</td></tr>
<tr><td>Buffer</td><td>3.0 Å</td></tr>
<tr><td>Born desolvation</td><td>enabled</td></tr>
<tr><td>Hydrodynamic radii</td><td>receptor 22.5, ligand 5.0 Å</td></tr>
<tr><td>Trajectories</td><td>100,000</td></tr>
<tr><td>Experimental k<sub>on</sub></td><td>2.9 × 10<sup>7</sup> M<sup>-1</sup>s<sup>-1</sup></td></tr>
</tbody>
</table>

## Input files (provided)

<table width="100%">
<thead><tr><th width="50%" align="left">File</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>complex.pdb</code></td><td>Bound-state PDB containing trypsin, benzamidine, water, and ions.</td></tr>
<tr><td><code>complex.prmtop</code></td><td>AMBER topology file that provides partial charges, atom types, and connectivity for PQR generation.</td></tr>
<tr><td><code>setup.py</code></td><td>Automated setup script. Reads the PDB and topology, generates files for BD simulation.</td></tr>
<tr><td><code>run.sh</code></td><td>Runs setup and BD simulation in one command.</td></tr>
</tbody>
</table>

## What setup.py generates
Running `python setup.py` produces:

<table width="100%">
<thead><tr><th width="50%" align="left">Generated file</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>receptor.pqr</code></td><td>Receptor PQR file. Water and ions are stripped automatically.</td></tr>
<tr><td><code>ligand.pqr</code></td><td>Ligand PQR file.</td></tr>
<tr><td><code>rxns.xml</code></td><td>Reaction criterion file with atom pairs and cutoff distances identified automatically from the PDB.</td></tr>
<tr><td><code>input.xml</code></td><td>PySTARC input file with all simulation parameters.</td></tr>
</tbody>
</table>

## Setup and run
```bash
conda activate PySTARC
module load cuda
cd examples/trypsin_benzamidine
python setup.py
python ../../run_pystarc.py input.xml
```

Or in one command:
```bash
conda activate PySTARC
module load cuda
cd examples/trypsin_benzamidine
chmod +x run.sh
bash run.sh
```

## Output files
After the simulation completes, all results are written to `bd_sims/`.

<table width="100%">
<thead><tr><th width="50%" align="left">Output file</th><th width="50%" align="left">Description</th></tr></thead>
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
- PQR files are generated from the AMBER topology via `cpptraj` and `ambpdb` and water and ions are stripped automatically.
- Reaction contacts are identified automatically by `setup.py` from the bound-state PDB structure using polar contacts (N/O/S atoms on both receptor and ligand).
