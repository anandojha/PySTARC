# HSP90-inhibitor complexes

## Receptor-ligand complexes
All 6 receptor-ligand complexes share the same HSP90 N-terminal domain receptor with different small-molecule inhibitors. The systems are a subset of the HSP90 inhibitor kinetics dataset of Kokh et al. (2018). The charged member of the original set is excluded so that all ligands modeled here are neutral. System indices (31, 37, ...) are dataset indices from that study.

<table width="100%">
<thead><tr><th width="50%" align="left">Receptor-ligand complex</th><th width="50%" align="left">Exp k<sub>on</sub> (M<sup>-1</sup>s<sup>-1</sup>)</th></tr></thead>
<tbody>
<tr><td width="50%"><code>31</code></td><td width="50%">1.0 × 10<sup>6</sup></td></tr>
<tr><td width="50%"><code>37</code></td><td width="50%">3.4 × 10<sup>5</sup></td></tr>
<tr><td width="50%"><code>43</code></td><td width="50%">8.4 × 10<sup>4</sup></td></tr>
<tr><td width="50%"><code>62</code></td><td width="50%">1.2 × 10<sup>5</sup></td></tr>
<tr><td width="50%"><code>65</code></td><td width="50%">2.1 × 10<sup>5</sup></td></tr>
<tr><td width="50%"><code>70</code></td><td width="50%">1.0 × 10<sup>4</sup></td></tr>
</tbody>
</table>

## Shared parameters

<table width="100%">
<thead><tr><th width="50%" align="left">Parameter</th><th width="50%" align="left">Value</th></tr></thead>
<tbody>
<tr><td width="50%">Receptor</td><td width="50%">HSP90 N-terminal domain</td></tr>
<tr><td width="50%">Ligand charge</td><td width="50%">neutral (auto-detected from formal charges)</td></tr>
<tr><td width="50%">b-surface</td><td width="50%">55.0 Å</td></tr>
<tr><td width="50%">Debye length</td><td width="50%">7.86 Å (150 mM, 0.15 M)</td></tr>
<tr><td width="50%">Contact mode</td><td width="50%">all (any heavy-atom contacts)</td></tr>
<tr><td width="50%">Contact cutoff</td><td width="50%">5.0 Å</td></tr>
<tr><td width="50%">Pairs / n_needed</td><td width="50%">up to 8 (one per residue) / 6</td></tr>
<tr><td width="50%">Born desolvation</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Hydrodynamic interactions</td><td width="50%">enabled</td></tr>
<tr><td width="50%">Overlap check</td><td width="50%">disabled</td></tr>
<tr><td width="50%">Trajectories</td><td width="50%">100,000 per complex</td></tr>
</tbody>
</table>

## Input files (provided)
Each system directory (`31/`, `37/`, ...) ships only the source structure and the setup script:

<table width="100%">
<thead><tr><th width="50%" align="left">File</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>complex.pdb</code></td><td width="50%">Bound-state PDB containing the receptor (HSP90) and the co-crystallized inhibitor.</td></tr>
<tr><td width="50%"><code>setup.py</code></td><td width="50%">Automated setup script. Reads the PDB, parameterizes from scratch, and generates all BD input files.</td></tr>
</tbody>
</table>

The following script is in the `hsp90_inhibitors/` directory:

<table width="100%">
<thead><tr><th width="50%" align="left">File</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>run.sh</code></td><td width="50%">Runs setup and BD simulation for all 6 complexes sequentially, then compares rates against experiment and saves <code>summary.txt</code>.</td></tr>
</tbody>
</table>

## What setup.py generates
Unlike the pre-generated examples, no parameterized files are shipped: `setup.py` builds everything from scratch each run. Running `python setup.py` produces:

<table width="100%">
<thead><tr><th width="50%" align="left">Generated file</th><th width="50%" align="left">Description</th></tr></thead>
<tbody>
<tr><td width="50%"><code>protein.prmtop</code>, <code>protein.rst7</code></td><td width="50%">Receptor AMBER topology and coordinates from tleap (ff14SB).</td></tr>
<tr><td width="50%"><code>ligand.prmtop</code>, <code>ligand.rst7</code></td><td width="50%">Ligand AMBER topology and coordinates from antechamber/tleap (GAFF2, AM1-BCC charges).</td></tr>
<tr><td width="50%"><code>receptor.pdb</code>, <code>ligand.pdb</code></td><td width="50%">Clean receptor and ligand PDBs from tleap.</td></tr>
<tr><td width="50%"><code>receptor.pqr</code></td><td width="50%">Receptor PQR file with atom positions, partial charges, and radii for the HSP90 host.</td></tr>
<tr><td width="50%"><code>ligand.pqr</code></td><td width="50%">Ligand PQR file with atom positions, partial charges, and radii for the inhibitor.</td></tr>
<tr><td width="50%"><code>rxns.xml</code></td><td width="50%">Reaction criterion file with atom pairs and cutoff distances identified automatically from the bound-state PDB.</td></tr>
<tr><td width="50%"><code>input.xml</code></td><td width="50%">PySTARC input file with all simulation parameters (b-surface, electrostatics, trajectories, GPU, and convergence).</td></tr>
</tbody>
</table>

## Setup and run (single complex)
```bash
conda activate PySTARC
module load cuda
cd examples/hsp90_inhibitors/31
python setup.py
python ../../../run_pystarc.py input.xml
```

## Run all 6 complexes
```bash
conda activate PySTARC
module load cuda
cd examples/hsp90_inhibitors
chmod +x run.sh
bash run.sh
```
`run.sh` performs the following for each of the 6 complexes: cleans any previous output files, runs `setup.py` to parameterize and generate the PQR files and input XMLs, then runs the BD simulation. After all simulations complete, on-rates are printed to terminal and saved to `summary.txt`. The comparison includes PySTARC k<sub>on</sub>, experimental k<sub>on</sub>, their ratio, P<sub>rxn</sub>, and a Spearman rank correlation across all 6 complexes.

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
- All 6 complexes use identical `setup.py` parameters, and only the source `complex.pdb` differs per system.
- Receptor charges and parameters are from ff14SB (tleap). Ligand charges are AM1-BCC (OpenEye) and ligand parameters are GAFF2 (antechamber/parmchk2).
- PQR files are generated from the AMBER topology via `cpptraj` and `ambpdb`.
- Reaction contacts are identified automatically by `setup.py` from the bound-state PDB structure (contact method, 5.0 Å cutoff, n_needed=6).
- All ligands modeled here are neutral. The charged member of the original Kokh et al. set is excluded.
