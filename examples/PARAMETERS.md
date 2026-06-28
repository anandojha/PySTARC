# Parameter selection

## Overview

For force field parameterization, receptor charges are assigned with the AMBER ff14SB force field, and small molecule ligand charges with GAFF2 + AM1-BCC. Electrostatic potential grids are generated with the Adaptive Poisson-Boltzmann Solver (APBS), and bimolecular association rate constants are computed within the Northrup-Allison-McCammon framework. All numerical values in this guide are taken from the on disk `input.xml`, `rxns.xml`, and `results.json` files for each example. Brownian dynamics trajectories begin on the b-surface, a sphere of radius b around the receptor, and the b-surface radius is the principal length scale that varies between systems.

> **Note on sources.** Where a reaction criterion *value* (cutoff, number of pairs, and contacts needed) appears below, it is read directly from the corresponding `rxns.xml`/`setup.py`.

---

## 1. Two charged spheres

Two uniformly charged spheres interact through a screened Coulomb potential. The ions carry charges Q<sub>rec</sub> = +1e and Q<sub>lig</sub> = −1e, each with a radius of 1.0 Å.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">10.0 Å</td><td width="33%">Combined maximum radii (1.0 + 1.0 = 2.0 Å) plus a buffer.</td></tr>
<tr><td width="34%">Hydrodynamic radii</td><td width="33%">0 / 0 (point charges)</td><td width="33%">No hydrodynamic radius is assigned as the spheres are treated as point particles for the analytical comparison.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.828 Å</td><td width="33%">Corresponds to ~150 mM ionic strength.</td></tr>
<tr><td width="34%">Protein / solvent dielectric</td><td width="33%">78.0 / 78.0</td><td width="33%">Uniform dielectric. Unlike the molecular complexes, no low dielectric interior is defined for the analytical test.</td></tr>
<tr><td width="34%">APBS fine grid</td><td width="33%">None</td><td width="33%">The screened Coulomb potential between the two point charges is evaluated analytically and no APBS grid is generated.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">129</td><td width="33%">Vestigial for the analytic potential.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">The adaptive timestep at r = 10 Å is ~10 ps, giving drift/noise ≈ 0.5.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">1,000,000</td><td width="33%">Sufficient for less than 1% relative standard error.</td></tr>
<tr><td width="34%">Reaction criterion</td><td width="33%">1 pair at 2.0 Å contact distance</td><td width="33%">The sum of radii defines exact contact.</td></tr>
<tr><td width="34%">Contacts needed</td><td width="33%">1</td><td width="33%">Single contact.</td></tr>
</tbody>
</table>

The analytical Smoluchowski rate is reproduced to within 0.1%, with k<sub>on</sub> = 1.56 × 10¹⁰ M⁻¹s⁻¹ versus the exact value of 1.56 × 10¹⁰ M⁻¹s⁻¹ (relative SE 0.1%).

---

## 2. Trypsin-benzamidine complex

Trypsin-benzamidine is a protein-ligand complex with well characterized experimental kinetics. Trypsin protein contains 3220 atoms with a net charge of +6e and a maximum radius of 28.4 Å. Benzamidine contains 18 atoms with a net charge of +1e and a maximum radius of 3.7 Å. Both molecules are positively charged, resulting in repulsive electrostatics.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">45.0 Å</td><td width="33%">Combined maximum radii (28.4 + 3.7 = 32.1 Å) plus a buffer.</td></tr>
<tr><td width="34%">Receptor hydrodynamic radius</td><td width="33%">22.5 Å</td><td width="33%">Stokes radius from molecular dimensions, approximately 0.80 × R<sub>max</sub> for a globular protein of this size.</td></tr>
<tr><td width="34%">Ligand hydrodynamic radius</td><td width="33%">5.0 Å</td><td width="33%">Stokes radius for a small planar organic molecule with 18 atoms.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.86 Å</td><td width="33%">Corresponds to 150 mM ionic strength.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">96 Å</td><td width="33%">Covers ±48 Å. At the b-surface, the outermost benzamidine atoms reach b + R<sub>max, lig</sub> = 48.7 Å, marginally beyond the grid edge.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">Yields a grid spacing of ~0.37 Å on the fine grid.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">The adaptive timestep at r = 45 Å is ~191 ps, giving drift/noise ≈ 3.4. This is acceptable for a small, rapidly diffusing ligand.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">10,000,000</td><td width="33%">The low reaction probability requires many trajectories for statistical convergence.</td></tr>
<tr><td width="34%">Contact cutoff (search)</td><td width="33%">6.0 Å</td><td width="33%">Maximum distance in the crystal structure used to identify binding contacts.</td></tr>
<tr><td width="34%">Buffer</td><td width="33%">3.0 Å</td><td width="33%">Added to the crystal distance to set the reaction cutoff, accounting for the rigid body approach.</td></tr>
<tr><td width="34%">Number of pairs</td><td width="33%">10</td><td width="33%">The top 10 closest polar contacts from the crystal structure.</td></tr>
<tr><td width="34%">Contacts needed</td><td width="33%">6</td><td width="33%">Six of the 10 pairs must be satisfied simultaneously.</td></tr>
<tr><td width="34%">Contact mode</td><td width="33%">Polar</td><td width="33%">Only N/O/S donor-acceptor pairs are considered, corresponding to hydrogen bonding contacts.</td></tr>
</tbody>
</table>

The setup script identifies the closest heavy atom contacts between the receptor and ligand in the crystal structure, filters for polar atoms (N, O, and S on both sides), retains the top 10 contacts with one per receptor residue, and sets each cutoff to the crystal distance. The resulting 10 pairs have cutoffs ranging from 6.0 to 8.5 Å. k<sub>on</sub> = 5.39 × 10⁷ M⁻¹s⁻¹ versus the experimental value of 2.9 × 10⁷ M⁻¹s⁻¹ (ratio 1.86×, relative SE 0.5%).

---

## 3. β-cyclodextrin host-guest complexes

Seven small molecule guests binding β-cyclodextrin form a host-guest benchmark in which the experimental association rates span an order of magnitude. β-cyclodextrin contains 147 atoms with zero net charge and a maximum radius of 8.6 Å. All seven guests are also neutral, with radii of 3 to 5 Å. Because every molecule is neutral, there is no electrostatic steering. The seven guests are 1-butanol, 1-propanol, tert-butanol, methyl butyrate, aspirin, 1-naphthylethanol, and 2-naphthylethanol.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">30.0 Å</td><td width="33%">Combined maximum radii (8.6 + 3 ≈ 11.6 Å) plus a buffer.</td></tr>
<tr><td width="34%">Hydrodynamic radii</td><td width="33%">Auto-computed</td><td width="33%">Determined from the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.86 Å</td><td width="33%">Corresponds to 150 mM ionic strength.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">96 Å</td><td width="33%">Covers ±48 Å. The sum b + R<sub>max, lig</sub> ≈ 33 Å is well within the grid.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">Yields a grid spacing of ~0.37 Å.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">The adaptive timestep at r = 30 Å is ~90 ps, giving drift/noise ≈ 1.5.</td></tr>
<tr><td width="34%">Overlap check</td><td width="33%">Enabled (true)</td><td width="33%">Retained for the host-guest complexes. The tight, apolar interface does not trigger the false rejections seen for hydrogen bonding protein criteria.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">2,000,000</td><td width="33%">Per host-guest complex.</td></tr>
<tr><td width="34%">Contact cutoff (search)</td><td width="33%">5.0 Å</td><td width="33%">A tight cutoff appropriate for the small host-guest complex.</td></tr>
<tr><td width="34%">Buffer</td><td width="33%">2.0 Å</td><td width="33%">Smaller than the protein complexes because the molecules are smaller and the contacts are tighter.</td></tr>
<tr><td width="34%">Number of pairs</td><td width="33%">7 to 8</td><td width="33%">Up to 8 contact pairs.</td></tr>
<tr><td width="34%">Contacts needed</td><td width="33%">4</td><td width="33%">Four pairs must be satisfied.</td></tr>
<tr><td width="34%">Contact mode</td><td width="33%">All heavy atoms</td><td width="33%">The host-guest interface has few polar atoms.</td></tr>
</tbody>
</table>

Each guest uses seven contact pairs, of which four must form for a reaction, at cutoffs of 5.0 to 6.5 Å taken from the crystal contacts. The O5 glycosidic oxygens of β-cyclodextrin (atoms 15, 57, and 99) recur as the receptor contact atoms across all of the guests, giving a consistent anchor.

<table width="100%">
<thead><tr><th width="20%" align="left">Guest</th><th width="20%" align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="20%" align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="20%" align="left">Ratio</th><th width="20%" align="left">rel SE</th></tr></thead>
<tbody>
<tr><td width="20%">1-butanol</td><td width="20%">2.8 × 10⁸</td><td width="20%">6.28 × 10⁸</td><td width="20%">2.24×</td><td width="20%">0.3%</td></tr>
<tr><td width="20%">1-propanol</td><td width="20%">5.1 × 10⁸</td><td width="20%">5.94 × 10⁸</td><td width="20%">1.17×</td><td width="20%">0.3%</td></tr>
<tr><td width="20%">tert-butanol</td><td width="20%">3.6 × 10⁸</td><td width="20%">6.30 × 10⁸</td><td width="20%">1.75×</td><td width="20%">0.3%</td></tr>
<tr><td width="20%">methyl butyrate</td><td width="20%">3.7 × 10⁸</td><td width="20%">4.53 × 10⁸</td><td width="20%">1.22×</td><td width="20%">0.4%</td></tr>
<tr><td width="20%">aspirin</td><td width="20%">7.2 × 10⁸</td><td width="20%">4.19 × 10⁸</td><td width="20%">0.58×</td><td width="20%">0.4%</td></tr>
<tr><td width="20%">1-naphthylethanol</td><td width="20%">4.7 × 10⁸</td><td width="20%">6.89 × 10⁷</td><td width="20%">0.15×</td><td width="20%">0.9%</td></tr>
<tr><td width="20%">2-naphthylethanol</td><td width="20%">2.9 × 10⁸</td><td width="20%">6.18 × 10⁸</td><td width="20%">2.13×</td><td width="20%">0.3%</td></tr>
</tbody>
</table>

---

## 4. Thrombin-thrombomodulin complex

This complex represents a strongly electrostatically steered protein-protein association. Thrombin contains 4727 atoms with a net charge of +3e and a maximum radius of 34.7 Å. Thrombomodulin EGF domains 4 to 6 contain 1650 atoms with a net charge of −15e and a maximum radius of 40.6 Å. Strong electrostatic complementarity drives fast association. Pre-computed PQR files with AMBER partial charges were used directly.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">85.0 Å</td><td width="33%">Combined maximum radii (34.7 + 40.6 = 75.3 Å) plus a buffer.</td></tr>
<tr><td width="34%">Receptor hydrodynamic radius</td><td width="33%">25.58 Å</td><td width="33%">Stokes radius from molecular dimensions, ~0.74 × R<sub>max</sub>.</td></tr>
<tr><td width="34%">Ligand hydrodynamic radius</td><td width="33%">21.88 Å</td><td width="33%">~0.54 × R<sub>max</sub>, with the lower ratio reflecting the elongated EGF domain fragment.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.86 Å</td><td width="33%">Corresponds to 150 mM ionic strength.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">192 Å</td><td width="33%">Covers ±96 Å. At b = 85, ligand atoms span 44 to 126 Å from the origin, and the grid encompasses the vast majority of encounter geometries.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">Yields a grid spacing of ~0.75 Å.</td></tr>
<tr><td width="34%">Base timestep</td><td width="33%">1.0 ps</td><td width="33%">Larger base step appropriate at the large b-surface.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">100 ps</td><td width="33%">Critical. Without the cap the adaptive timestep at r = 85 Å reaches ~1806 ps (drift/noise ≈ 5.4), producing ballistic trajectories that skip the electrostatic steering region. The cap restores drift/noise ≈ 2.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">1,000,000</td><td width="33%">-</td></tr>
<tr><td width="34%">Contacts needed</td><td width="33%">4</td><td width="33%">Four interfacial hydrogen bonding contacts must be satisfied simultaneously. The full pair list is in <code>rxns.xml</code>.</td></tr>
</tbody>
</table>

The experimental k<sub>on</sub> is 6.7 × 10⁶ M⁻¹s⁻¹ at physiological ionic strength, measured by surface plasmon resonance, and Debye-Hückel analysis of the ionic strength dependence confirms a nearly completely electrostatically steered interaction. The computed k<sub>on</sub> is 2.38 × 10⁶ M⁻¹s⁻¹ (ratio 0.36×, relative SE 6.2%).

---

## 5. Barnase-barstar (flexible chain Brownian dynamics)

Barnase-barstar is the classic electrostatically steered protein-protein association benchmark, included here as a sample for PySTARC's chain BD module, which extends the rigid body engine to internal conformational degrees of freedom. It pairs barnase (chain A of PDB 1BRS) and barstar (chain D), parameterized with AMBER ff14SB. Unlike the rigid body examples, the chain BD treatment retains internal flexibility rather than freezing each partner as a rigid body.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b / milestone radius</td><td width="33%">80 Å</td><td width="33%">Milestone radius for the chain BD run.</td></tr>
<tr><td width="34%">r<sub>escape</sub></td><td width="33%">160 Å</td><td width="33%">Outer absorbing radius for the chain BD run.</td></tr>
</tbody>
</table>

---

## 6. p38 MAPK - SB203580 complex

This is a kinase inhibitor complex in which the ligand is neutral and the receptor carries a large net negative charge. p38 MAPK alpha in the DFG-in conformation from PDB 1A9U (chain A, residues 4 to 354) contains 5658 atoms with a net charge of −9e and a maximum radius of 37.8 Å. SB203580 is a type I kinase inhibitor with 27 atoms and zero net charge. The receptor is parameterized with ff14SB and the ligand with GAFF2 + AM1-BCC via antechamber. The ligand binds in the ATP pocket at the hinge.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">60.0 Å</td><td width="33%">Combined maximum radii (37.8 + 7.8 = 45.6 Å) plus a buffer.</td></tr>
<tr><td width="34%">Hydrodynamic radii</td><td width="33%">Auto-computed</td><td width="33%">From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.86 Å</td><td width="33%">150 mM ionic strength.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">128 Å</td><td width="33%">Covers ±64 Å, and the multipole fallback handles outermost atom overshoot at the b-surface.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">~0.50 Å fine grid spacing.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">The adaptive timestep at r = 60 Å is moderate for a small, highly diffusive ligand.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">5,000,000</td><td width="33%">Production run.</td></tr>
</tbody>
</table>

Four crystal structure contacts (the hinge MET106 backbone to the pyridine N, the catalytic LYS50 NZ to an imidazole N, and the VAL102 O and THR103 N to the fluorine) at a uniform 7.0 Å cutoff, with n_needed = 3. k<sub>on</sub> = 2.86 × 10⁷ M⁻¹s⁻¹ versus the experimental value of 1.5 × 10⁷ M⁻¹s⁻¹ (ratio 1.91×, relative SE 1.1%).

---

## 7. Carbonic anhydrase sulfonamide inhibitors

Seven sulfonamide inhibitors binding three carbonic anhydrase isozymes (CA XIII, CA I, and CA II) form a multi target benchmark in which all ligands carry the same charge (−1e) and bind the same Zn-coordinating sulfonamide motif but differ in scaffold, size, and isozyme. All ligands are deprotonated sulfonamides (net charge −1e), and the intrinsic k<sub>on</sub> corrected for the protonation equilibrium is the correct BD comparison target. Five complexes use CA XIII (PDB 3CZV), one CA I (2NMX), one CA II (3HS4). The active site Zn²⁺ is included via `frcmod.ions234lm_126_tip3p`. Ligands are built from SMILES via rdkit, converted with obabel, and parameterized with antechamber (GAFF2 + AM1-BCC).

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">60.0 Å</td><td width="33%">Combined maximum radii (27 + 8 = 35 Å) plus a buffer.</td></tr>
<tr><td width="34%">Hydrodynamic radii</td><td width="33%">Auto-computed</td><td width="33%">From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">9.62 Å</td><td width="33%">Corresponds to 100 mM ionic strength.</td></tr>
<tr><td width="34%">Ion concentration</td><td width="33%">0.10 M</td><td width="33%">100 mM NaCl.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">128 Å</td><td width="33%">Covers ±64 Å, and the multipole fallback handles overshoot.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">~0.50 Å fine grid spacing.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">Moderate adaptive timestep for small, highly diffusive ligands.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">5,000,000</td><td width="33%">Per complex.</td></tr>
<tr><td width="34%">Reaction criterion</td><td width="33%">2 pairs (THR199 OG1 / GLU106 OE1 to sulfonamide N and amide N), cutoff 3.5 Å, n_needed = 2</td><td width="33%">Both Zn-coordinating / proton shuttle hydrogen bonds required.</td></tr>
</tbody>
</table>

<table width="100%">
<thead><tr><th width="17%" align="left">Complex</th><th width="17%" align="left">Isozyme</th><th width="17%" align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="17%" align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="16%" align="left">Ratio</th><th width="16%" align="left">rel SE</th></tr></thead>
<tbody>
<tr><td width="17%">CA I-VD12-69-1</td><td width="17%">CA I</td><td width="17%">2.7 × 10⁶</td><td width="17%">2.35 × 10⁶</td><td width="16%">0.87×</td><td width="16%">3.8%</td></tr>
<tr><td width="17%">CA XIII-AZM</td><td width="17%">CA XIII</td><td width="17%">1.5 × 10⁶</td><td width="17%">1.79 × 10⁶</td><td width="16%">1.20×</td><td width="16%">4.7%</td></tr>
<tr><td width="17%">CA XIII-VD11-26</td><td width="17%">CA XIII</td><td width="17%">1.5 × 10⁶</td><td width="17%">2.35 × 10⁶</td><td width="16%">1.57×</td><td width="16%">3.8%</td></tr>
<tr><td width="17%">CA XIII-VD12-69-1</td><td width="17%">CA XIII</td><td width="17%">2.5 × 10⁶</td><td width="17%">5.74 × 10⁶</td><td width="16%">2.30×</td><td width="16%">2.4%</td></tr>
<tr><td width="17%">CA XIII-VD11-25</td><td width="17%">CA XIII</td><td width="17%">4.6 × 10⁵</td><td width="17%">2.14 × 10⁶</td><td width="16%">4.66×</td><td width="16%">3.9%</td></tr>
<tr><td width="17%">CA XIII-VD12-09</td><td width="17%">CA XIII</td><td width="17%">3.3 × 10⁵</td><td width="17%">5.63 × 10⁶</td><td width="16%">17.06×</td><td width="16%">2.4%</td></tr>
<tr><td width="17%">CA II-VD11-4-2</td><td width="17%">CA II</td><td width="17%">1.8 × 10⁶</td><td width="17%">3.62 × 10⁵</td><td width="16%">0.20×</td><td width="16%">9.5%</td></tr>
</tbody>
</table>

---

## 8. TTK (MPS1) kinase inhibitors

Eight inhibitors of the mitotic kinase TTK/MPS1 form a single target series spanning roughly two orders of magnitude in experimental k<sub>on</sub>. The complexes are eight TTK co-crystal structures (PDB 2X9E, 3GFW, 3H9F, 5LJJ, 5N7V, 5N84, 5N93, and 5NAD), each parameterized with ff14SB for the receptor and GAFF2 + AM1-BCC for the ligand, with ligand charges assigned by the OpenEye toolkits (license required).

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">60.0 Å</td><td width="33%">Combined maximum radii of the receptor and ligand plus a buffer.</td></tr>
<tr><td width="34%">Hydrodynamic radii</td><td width="33%">Auto-computed</td><td width="33%">From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.86 Å</td><td width="33%">150 mM ionic strength.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">144 Å</td><td width="33%">Covers ±72 Å.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">~0.56 Å fine grid spacing.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">b &lt; 80 Å, so no cap is required.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">10,000,000</td><td width="33%">Per complex.</td></tr>
<tr><td width="34%">Reaction criterion</td><td width="33%">3 polar crystal contacts (GLU603 O, GLY605 O, and GLY605 N to ligand N9/N2/N3), uniform 4.5 Å cutoff, n_needed = 3</td><td width="33%">A single flat cutoff applied uniformly across all eight complexes.</td></tr>
</tbody>
</table>

<table width="100%">
<thead><tr><th width="17%" align="left">Complex</th><th width="17%" align="left">Inhibitor</th><th width="17%" align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="17%" align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="16%" align="left">Ratio</th><th width="16%" align="left">rel SE</th></tr></thead>
<tbody>
<tr><td width="17%">5LJJ</td><td width="17%">Reversine</td><td width="17%">2.08 × 10⁶</td><td width="17%">4.91 × 10⁶</td><td width="16%">2.36×</td><td width="16%">1.8%</td></tr>
<tr><td width="17%">2X9E</td><td width="17%">NMS-P715</td><td width="17%">6.41 × 10⁵</td><td width="17%">2.37 × 10⁶</td><td width="16%">3.70×</td><td width="16%">2.3%</td></tr>
<tr><td width="17%">5N84</td><td width="17%">Mps-BAY2b</td><td width="17%">2.60 × 10⁶</td><td width="17%">1.51 × 10⁶</td><td width="16%">0.58×</td><td width="16%">3.2%</td></tr>
<tr><td width="17%">3GFW</td><td width="17%">Mps1-IN-1</td><td width="17%">3.79 × 10⁵</td><td width="17%">9.32 × 10⁵</td><td width="16%">2.46×</td><td width="16%">4.0%</td></tr>
<tr><td width="17%">5N7V</td><td width="17%">MPI-0479605</td><td width="17%">1.96 × 10⁶</td><td width="17%">3.62 × 10⁶</td><td width="16%">1.85×</td><td width="16%">2.1%</td></tr>
<tr><td width="17%">5N93</td><td width="17%">TC-Mps1-12</td><td width="17%">2.16 × 10⁷</td><td width="17%">3.15 × 10⁶</td><td width="16%">0.15×</td><td width="16%">2.3%</td></tr>
<tr><td width="17%">5NAD</td><td width="17%">BAY-1217389</td><td width="17%">3.79 × 10⁵</td><td width="17%">1.21 × 10⁷</td><td width="16%">31.94×</td><td width="16%">1.1%</td></tr>
<tr><td width="17%">3H9F</td><td width="17%">Mps1-IN-2</td><td width="17%">1.19 × 10⁶</td><td width="17%">8.32 × 10⁶</td><td width="16%">6.99×</td><td width="16%">1.3%</td></tr>
</tbody>
</table>

---

## 9. HSP90 inhibitors

Six neutral HSP90 inhibitors form a single target series of uncharged ligands, isolating the diffusion limited encounter rate in the absence of electrostatic steering. The six complexes are HSP90 N-terminal domain co-crystal structures (31, 37, 43, 62, 65, and 70), each with a neutral inhibitor, the receptor parameterized with ff14SB and the ligands with GAFF2 + AM1-BCC (OpenEye toolkits, license required).

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">b-surface radius</td><td width="33%">55.0 Å</td><td width="33%">Combined maximum radii of the receptor and ligand plus a buffer.</td></tr>
<tr><td width="34%">Hydrodynamic radii</td><td width="33%">Auto-computed</td><td width="33%">From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td width="34%">Debye length</td><td width="33%">7.86 Å</td><td width="33%">150 mM ionic strength.</td></tr>
<tr><td width="34%">APBS fine grid length</td><td width="33%">144 Å</td><td width="33%">Covers ±72 Å.</td></tr>
<tr><td width="34%">APBS grid dimension</td><td width="33%">257</td><td width="33%">~0.56 Å fine grid spacing.</td></tr>
<tr><td width="34%">Max timestep cap</td><td width="33%">0 (no cap)</td><td width="33%">b &lt; 80 Å, so no cap is required.</td></tr>
<tr><td width="34%">Trajectories</td><td width="33%">10,000,000</td><td width="33%">Per complex.</td></tr>
<tr><td width="34%">Reaction criterion</td><td width="33%">8 pairs (contact method, SER37/ALA40/ASN36 CA anchors to ligand O2x/C15x/N1x), uniform 5.0 Å cutoff, n_needed = 6</td><td width="33%">A single flat 5.0 Å cutoff applied across all six complexes.</td></tr>
</tbody>
</table>

<table width="100%">
<thead><tr><th width="20%" align="left">Complex</th><th width="20%" align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="20%" align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th width="20%" align="left">Ratio</th><th width="20%" align="left">rel SE</th></tr></thead>
<tbody>
<tr><td width="20%">31</td><td width="20%">1.00 × 10⁶</td><td width="20%">2.09 × 10⁶</td><td width="20%">2.09×</td><td width="20%">3.0%</td></tr>
<tr><td width="20%">37</td><td width="20%">3.43 × 10⁵</td><td width="20%">2.70 × 10⁵</td><td width="20%">0.79×</td><td width="20%">7.3%</td></tr>
<tr><td width="20%">43</td><td width="20%">8.38 × 10⁴</td><td width="20%">6.37 × 10⁵</td><td width="20%">7.61×</td><td width="20%">4.8%</td></tr>
<tr><td width="20%">62</td><td width="20%">1.21 × 10⁵</td><td width="20%">4.10 × 10⁵</td><td width="20%">3.39×</td><td width="20%">5.9%</td></tr>
<tr><td width="20%">65</td><td width="20%">2.08 × 10⁵</td><td width="20%">1.08 × 10⁶</td><td width="20%">5.20×</td><td width="20%">3.7%</td></tr>
<tr><td width="20%">70</td><td width="20%">1.04 × 10⁴</td><td width="20%">6.10 × 10⁵</td><td width="20%">58.68×</td><td width="20%">5.0%</td></tr>
</tbody>
</table>

---

## Common parameters

Most complexes share the following. Exceptions are noted in the sections above.

<table width="100%">
<thead><tr><th width="34%" align="left">Parameter</th><th width="33%" align="left">Value</th><th width="33%" align="left">Rationale</th></tr></thead>
<tbody>
<tr><td width="34%">Protein dielectric</td><td width="33%">4.0</td><td width="33%">Standard for BD simulations. (Exception: the two sphere analytical test uses a uniform dielectric of 78.0.)</td></tr>
<tr><td width="34%">Solvent dielectric</td><td width="33%">78.0</td><td width="33%">Water at 298.15 K.</td></tr>
<tr><td width="34%">Solvent probe radius</td><td width="33%">1.4 Å</td><td width="33%">Water molecule radius.</td></tr>
<tr><td width="34%">Desolvation coupling</td><td width="33%">0.0795775</td><td width="33%">Equal to 1/(4π), the Born desolvation coupling constant.</td></tr>
<tr><td width="34%">Hydrodynamic interactions</td><td width="33%">Enabled</td><td width="33%">Included in the k<sub>b</sub> integral.</td></tr>
<tr><td width="34%">Overlap check</td><td width="33%">System-dependent</td><td width="33%"><strong>Disabled (false)</strong> for the protein-ligand and protein-protein hydrogen bonding criteria, where the atom pair overlap check would otherwise reject valid near contact binding poses, and <strong>enabled (true)</strong> for the β-cyclodextrin host-guest complexes.</td></tr>
<tr><td width="34%">Multipole fallback</td><td width="33%">Enabled</td><td width="33%">Monopole, dipole, and quadrupole Yukawa expansions are used beyond the APBS grid boundary.</td></tr>
<tr><td width="34%">Lennard-Jones forces</td><td width="33%">Disabled</td><td width="33%">Standard for rigid body BD.</td></tr>
<tr><td width="34%">Temperature</td><td width="33%">298.15 K</td><td width="33%">Room temperature, k<sub>B</sub>T = 1.0 in reduced units.</td></tr>
<tr><td width="34%">Base timestep</td><td width="33%">0.2 ps</td><td width="33%">Minimum core timestep near the receptor surface. (Exception: thrombin-thrombomodulin uses 1.0 ps.)</td></tr>
</tbody>
</table>

## Adaptive timestep cap

The variable timestep is Δt<sub>pair</sub> = f²r²/(2D) with f = 0.1, which keeps the RMS displacement per step below 10% of the intermolecular separation. A b-surface below 80 Å needs no cap, but for the larger protein-protein b-surfaces the timestep would exceed 1000 ps at the starting radius and skip the electrostatic steering region, so it is capped at a ceiling (typically 100 ps, or 0 for no cap).

<table width="100%">
<thead><tr><th width="20%" align="left">Complex</th><th width="20%" align="left">b (Å)</th><th width="20%" align="left">Δt at b (ps)</th><th width="20%" align="left">Drift/noise</th><th width="20%" align="left">Cap</th></tr></thead>
<tbody>
<tr><td width="20%">Charged spheres</td><td width="20%">10</td><td width="20%">10</td><td width="20%">0.5</td><td width="20%">No</td></tr>
<tr><td width="20%">β-cyclodextrin host-guest</td><td width="20%">30</td><td width="20%">90</td><td width="20%">1.5</td><td width="20%">No</td></tr>
<tr><td width="20%">Trypsin-benzamidine</td><td width="20%">45</td><td width="20%">191</td><td width="20%">3.4</td><td width="20%">No</td></tr>
<tr><td width="20%">HSP90 inhibitors</td><td width="20%">55</td><td width="20%">-</td><td width="20%">-</td><td width="20%">No (max_dt = 0)</td></tr>
<tr><td width="20%">p38 MAPK - SB203580</td><td width="20%">60</td><td width="20%">~360</td><td width="20%">~3.0</td><td width="20%">No</td></tr>
<tr><td width="20%">Carbonic anhydrase inhibitors</td><td width="20%">60</td><td width="20%">~360</td><td width="20%">~3.0</td><td width="20%">No</td></tr>
<tr><td width="20%">TTK inhibitors</td><td width="20%">60</td><td width="20%">-</td><td width="20%">-</td><td width="20%">No (max_dt = 0)</td></tr>
<tr><td width="20%">Thrombin-thrombomodulin</td><td width="20%">85</td><td width="20%">1806</td><td width="20%">5.4</td><td width="20%">Yes, 100 ps</td></tr>
</tbody>
</table>

The flexible chain BD example (§5) uses a distinct propagation scheme with its own timestep handling, and its parameters will be documented once the run is finalized.

---

## Summary

Across the protein-ligand, protein-protein, and host-guest complexes, the agreement between the computed and experimental association rate constants in log₁₀ space is summarized below.

<table width="100%">
<thead><tr><th width="50%" align="left">Metric</th><th width="50%" align="left">Value</th></tr></thead>
<tbody>
<tr><td width="50%">Pearson r</td><td width="50%">0.913 (r² = 0.833)</td></tr>
<tr><td width="50%">Spearman ρ</td><td width="50%">0.796</td></tr>
<tr><td width="50%">R² vs y = x</td><td width="50%">0.790</td></tr>
<tr><td width="50%">log₁₀ MAE</td><td width="50%">0.493 (mean fold error 3.1×)</td></tr>
<tr><td width="50%">log₁₀ RMSE</td><td width="50%">0.644</td></tr>
<tr><td width="50%">log₁₀ bias</td><td width="50%">0.277 (systematic 1.9× overprediction)</td></tr>
<tr><td width="50%">Converged</td><td width="50%">28 / 32</td></tr>
</tbody>
</table>

The systematic overprediction, largest for the slowest experimental binders, reflects the diffusion limited floor of rigid body BD, which cannot lower k<sub>on</sub> for binders whose rate is set by non-diffusional gating.
