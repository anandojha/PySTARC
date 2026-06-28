# Beta-cyclodextrin host-guest complexes

## Receptor-ligand complexes
All 7 receptor-ligand complexes share the same beta-cyclodextrin (BCD/MGO) receptor with different guest molecules.

<table width="100%">
<thead><tr><th align="left">Receptor-ligand complex</th><th align="left">Ligand</th></tr></thead>
<tbody>
<tr><td>BCD_1-propanol</td><td>1-propanol</td></tr>
<tr><td>BCD_1-butanol</td><td>1-butanol</td></tr>
<tr><td>BCD_tertbutanol</td><td>tert-butanol</td></tr>
<tr><td>BCD_methyl_butyrate</td><td>methyl butyrate</td></tr>
<tr><td>BCD_aspirin</td><td>aspirin</td></tr>
<tr><td>BCD_1-naphthylethanol</td><td>1-naphthylethanol</td></tr>
<tr><td>BCD_2-naphthylethanol</td><td>2-naphthylethanol</td></tr>
</tbody>
</table>

## Shared parameters 

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th></tr></thead>
<tbody>
<tr><td>Receptor</td><td>beta-cyclodextrin (MGO), 147 atoms</td></tr>
<tr><td>Ligand</td><td>APN</td></tr>
<tr><td>b-surface</td><td>30.0 Å</td></tr>
<tr><td>Escape sphere</td><td>60.0 Å</td></tr>
<tr><td>Debye length</td><td>7.86 Å (150 mM NaCl)</td></tr>
<tr><td>Contact mode</td><td>all (any heavy-atom contacts)</td></tr>
<tr><td>Contact cutoff</td><td>5.0 Å</td></tr>
<tr><td>Buffer</td><td>2.0 Å</td></tr>
<tr><td>Born desolvation</td><td>enabled</td></tr>
<tr><td>Trajectories</td><td>100,000 per complex</td></tr>
</tbody>
</table>

## Input files (provided)
Each `BCD_*/` directory contains the following files:

<table width="100%">
<thead><tr><th align="left">File</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>complex.pdb</code></td><td>Bound-state PDB containing receptor (MGO), ligand (APN), and water (WAT).</td></tr>
<tr><td><code>complex.parm7</code></td><td>AMBER topology file that provides partial charges, atom types, and connectivity for PQR generation.</td></tr>
<tr><td><code>setup.py</code></td><td>Automated setup script. Reads the PDB and topology, generates files for BD simulation.</td></tr>
</tbody>
</table>

The following scripts are in the `beta_cyclodextrin_guests/` directory:

<table width="100%">
<thead><tr><th align="left">File</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>run.sh</code></td><td>Runs setup and BD simulation for all 7 complexes sequentially, then compares rates against experiment.</td></tr>
<tr><td><code>compare_rates.py</code></td><td>Collects k<sub>on</sub> from all 7 complexes, compares against experimental values, computes Spearman rank correlation, and saves <code>summary.txt</code>.</td></tr>
</tbody>
</table>

## What setup.py generates
Running `python setup.py` produces:

<table width="100%">
<thead><tr><th align="left">Generated file</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>receptor.pqr</code></td><td>Receptor PQR file extracted from the topology. Contains atom positions, partial charges, and radii for the beta-cyclodextrin host.</td></tr>
<tr><td><code>ligand.pqr</code></td><td>Ligand PQR file extracted from the topology. Contains atom positions, partial charges, and radii for the guest molecule.</td></tr>
<tr><td><code>rxns.xml</code></td><td>Reaction criterion file that contains atom pairs and cutoff distances identified automatically from the bound-state PDB.</td></tr>
<tr><td><code>input.xml</code></td><td>PySTARC input file that contains all simulation parameters (b-surface, electrostatics, trajectories, GPU, and convergence).</td></tr>
</tbody>
</table>

## Setup and run (single complex)
```bash
conda activate PySTARC
module load cuda
cd examples/beta_cyclodextrin_guests/BCD_1-butanol
python setup.py
python ../../../run_pystarc.py input.xml
```

## Run all 7 complexes
```bash
conda activate PySTARC
module load cuda
cd examples/beta_cyclodextrin_guests
chmod +x run.sh
bash run.sh
```
`run.sh` performs the following for each of the 7 complexes: cleans any previous output files, runs `setup.py` to generate PQR files and input XMLs, then runs the BD simulation. After all simulations complete, on-rates are printed to terminal and saved to `summary.txt`. The comparison includes PySTARC k<sub>on</sub>, experimental k<sub>on</sub>, their ratio, and a Spearman rank correlation across all 7 complexes.

## Output files
After a simulation completes, all results are written to `bd_sims/` within each receptor-ligand complex directory.

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
- All 7 complexes use identical `setup.py` parameters (same receptor MGO, same ligand APN residue name).
- PQR files are generated from the AMBER topology via `cpptraj` and `ambpdb` and the water is stripped automatically.
- Reaction contacts are identified automatically by `setup.py` from the bound-state PDB structure.
- Charges come from the AMBER parm7 via `ambpdb -pqr`.
