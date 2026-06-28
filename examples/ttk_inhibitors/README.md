# TTK (MPS1) kinase-inhibitor complexes

## Receptor-ligand complexes
This example covers eight inhibitors of the TTK (MPS1) kinase domain. Each system is a separate crystal structure (the PDB accession is the directory name) paired with its co-crystallized inhibitor. BD models the diffusion-limited encounter of each inhibitor with the ATP-binding site of the kinase.

<table width="100%">
<thead><tr><th width="25%" align="left">Directory</th><th width="25%" align="left">PDB</th><th width="25%" align="left">Inhibitor</th><th width="25%" align="left">Exp k<sub>on</sub> (M<sup>-1</sup>s<sup>-1</sup>)</th></tr></thead>
<tbody>
<tr><td width="25%"><code>2X9E</code></td><td width="25%">2X9E</td><td width="25%">NMS-P715</td><td width="25%">6.4 × 10<sup>5</sup></td></tr>
<tr><td width="25%"><code>3GFW</code></td><td width="25%">3GFW</td><td width="25%">Mps1-IN-1</td><td width="25%">3.8 × 10<sup>5</sup></td></tr>
<tr><td width="25%"><code>3H9F</code></td><td width="25%">3H9F</td><td width="25%">Mps1-IN-2</td><td width="25%">1.2 × 10<sup>6</sup></td></tr>
<tr><td width="25%"><code>5LJJ</code></td><td width="25%">5LJJ</td><td width="25%">Reversine</td><td width="25%">2.1 × 10<sup>6</sup></td></tr>
<tr><td width="25%"><code>5N7V</code></td><td width="25%">5N7V</td><td width="25%">MPI-0479605</td><td width="25%">2.0 × 10<sup>6</sup></td></tr>
<tr><td width="25%"><code>5N84</code></td><td width="25%">5N84</td><td width="25%">Mps-BAY2b</td><td width="25%">2.6 × 10<sup>6</sup></td></tr>
<tr><td width="25%"><code>5N93</code></td><td width="25%">5N93</td><td width="25%">TC-Mps1-12</td><td width="25%">2.2 × 10<sup>7</sup></td></tr>
<tr><td width="25%"><code>5NAD</code></td><td width="25%">5NAD</td><td width="25%">BAY-1217389</td><td width="25%">3.8 × 10<sup>5</sup></td></tr>
</tbody>
</table>

Experimental k<sub>on</sub> values are from Uitdehaag et al. (2017). Verify the exact citation against the primary source before external use. See the Notes on `3GFW` below.

## Shared parameters

<table width="100%">
<thead><tr><th width="50%" align="left">Parameter</th><th width="50%" align="left">Value</th></tr></thead>
<tbody>
<tr><td width="50%">Receptor</td><td width="50%">TTK (MPS1) kinase domain</td></tr>
<tr><td width="50%">Ligand charge</td><td width="50%">neutral</td></tr>
<tr><td width="50%">b-surface</td><td width="50%">60.0 Å</td></tr>
<tr><td width="50%">Debye length</td><td width="50%">7.86 Å (150 mM, 0.15 M)</td></tr>
<tr><td width="50%">Reaction cutoff</td><td width="50%">4.5 Å (per-pair)</td></tr>
<tr><td width="50%">n_needed</td><td width="50%">3</td></tr>
<tr><td width="50%">Born desolvation</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Hydrodynamic interactions</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Overlap check</td><td width="50%">disabled</td></tr>
<tr><td width="50%">Trajectories</td><td width="50%">100,000 per complex</td></tr>
</tbody>
</table>

Unlike the host-guest examples, the reaction criterion here is defined explicitly per system: each `setup.py` lists the receptor residue / ligand atom pairs taken from the crystal pose (for example, for `2X9E`: GLU603 O - N15, GLY605 O - N18, GLY605 N - N15). With `n_needed=3`, three such contacts must be satisfied simultaneously for a binding event.

## Input files (provided)
Each system directory ships only the source structure and the setup script:

<table width="100%">
<thead><tr><th width="50%" align="left">File</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>&lt;PDB&gt;.pdb</code></td><td width="50%">Crystal structure (protein + co-crystallized inhibitor), named by PDB accession (e.g. <code>2X9E.pdb</code>).</td></tr>
<tr><td width="50%"><code>setup.py</code></td><td width="50%">Automated setup script. Extracts protein and ligand, parameterizes from scratch, and generates all BD input files.</td></tr>
</tbody>
</table>

The following script is in the `ttk_inhibitors/` directory:

<table width="100%">
<thead><tr><th width="50%" align="left">File</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>run.sh</code></td><td width="50%">Runs setup and BD simulation for all 8 complexes sequentially, then compares rates against experiment and saves <code>summary.txt</code>.</td></tr>
</tbody>
</table>

## What setup.py generates
No parameterized files are shipped. `setup.py` builds everything from scratch each run. If the source `<PDB>.pdb` is already present (as shipped) it is used directly. Otherwise setup downloads it from the RCSB PDB. Running `python setup.py` produces:

<table width="100%">
<thead><tr><th width="50%" align="left">Generated file</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>protein.prmtop</code>, <code>protein.rst7</code></td><td width="50%">Receptor AMBER topology and coordinates from tleap (ff14SB).</td></tr>
<tr><td width="50%"><code>ligand.prmtop</code>, <code>ligand.rst7</code></td><td width="50%">Ligand AMBER topology and coordinates from antechamber/tleap (GAFF2, AM1-BCC charges).</td></tr>
<tr><td width="50%"><code>receptor.pdb</code>, <code>ligand.pdb</code></td><td width="50%">Clean receptor and ligand PDBs.</td></tr>
<tr><td width="50%"><code>receptor.pqr</code></td><td width="50%">Receptor PQR file with atom positions, partial charges, and radii.</td></tr>
<tr><td width="50%"><code>ligand.pqr</code></td><td width="50%">Ligand PQR file with atom positions, partial charges, and radii.</td></tr>
<tr><td width="50%"><code>rxns.xml</code></td><td width="50%">Reaction criterion file with the per-system atom pairs and 4.5 Å cutoffs.</td></tr>
<tr><td width="50%"><code>input.xml</code></td><td width="50%">PySTARC input file with all simulation parameters (b-surface, electrostatics, trajectories, GPU, and convergence).</td></tr>
</tbody>
</table>

## Setup and run (single complex)
```bash
conda activate PySTARC
module load cuda
cd examples/ttk_inhibitors/2X9E
python setup.py
python ../../../run_pystarc.py input.xml
```

## Run all 8 complexes
```bash
conda activate PySTARC
module load cuda
cd examples/ttk_inhibitors
chmod +x run.sh
bash run.sh
```
`run.sh` performs the following for each of the 8 complexes: cleans any previous output files (keeping the source PDB), runs `setup.py` to parameterize and generate the PQR files and input XMLs, then runs the BD simulation. After all simulations complete, on-rates are printed to terminal and saved to `summary.txt`. The comparison includes PySTARC k<sub>on</sub>, experimental k<sub>on</sub>, their ratio, P<sub>rxn</sub>, and a Spearman rank correlation across the systems (with `3GFW` excluded, see Notes).

## Output files
After a simulation completes, all results are written to `bd_sims/` within each receptor-ligand complex directory.

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
- The source PDB is shipped with each system, so setup runs offline. If the PDB is removed, setup downloads it from `https://files.rcsb.org/download/<PDB>.pdb`, which requires network access.
- The reaction criterion is explicit per system (defined in each `setup.py`), not auto-discovered. Each pair uses a 4.5 Å cutoff and `n_needed=3`.
- Receptor parameters are ff14SB (tleap). Ligand charges are AM1-BCC (OpenEye) and ligand parameters are GAFF2 (antechamber/parmchk2). PQR files are generated via `cpptraj` and `ambpdb`.
- `3GFW` is included for completeness but its reaction criterion is too loose: the chosen contacts do not discriminate the bound pose, producing spurious reactions and an unreliable k<sub>on</sub>. It is excluded from the rank-correlation statistic and its reported rate should not be trusted without rebuilding the criterion.
