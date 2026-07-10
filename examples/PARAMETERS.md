# Parameter selection

## Overview

For force field parameterization, receptor charges are assigned with the AMBER ff14SB force field, and small molecule ligand charges with GAFF2 + AM1-BCC. Electrostatic potential grids are generated with the Adaptive Poisson-Boltzmann Solver (APBS), and bimolecular association rate constants are computed within the Northrup-Allison-McCammon framework. All numerical values in this guide are taken from the on disk `input.xml`, `rxns.xml`, and `results.json` files for each example. Brownian dynamics trajectories begin on the b-surface, a sphere of radius b around the receptor, and the b-surface radius is the principal length scale that varies between systems.

> **Note on sources.** Where a reaction criterion *value* (cutoff, number of pairs, and contacts needed) appears below, it is read directly from the corresponding `rxns.xml`/`setup.py`.

---

## 1. Two charged spheres

Two uniformly charged spheres interact through a screened Coulomb potential. The ions carry charges Q<sub>rec</sub> = +1e and Q<sub>lig</sub> = −1e, each with a radius of 1.0 Å.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>10.0 Å</td><td>Combined maximum radii (1.0 + 1.0 = 2.0 Å) plus a buffer.</td></tr>
<tr><td>Hydrodynamic radii</td><td>0 / 0 (point charges)</td><td>No hydrodynamic radius is assigned as the spheres are treated as point particles for the analytical comparison.</td></tr>
<tr><td>Debye length</td><td>7.828 Å</td><td>Corresponds to ~150 mM ionic strength.</td></tr>
<tr><td>Protein / solvent dielectric</td><td>78.0 / 78.0</td><td>Uniform dielectric. Unlike the molecular complexes, no low dielectric interior is defined for the analytical test.</td></tr>
<tr><td>APBS fine grid</td><td>None</td><td>The screened Coulomb potential between the two point charges is evaluated analytically and no APBS grid is generated.</td></tr>
<tr><td>APBS grid dimension</td><td>129</td><td>Vestigial for the analytic potential.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>The adaptive timestep at r = 10 Å is ~10 ps, giving drift/noise ≈ 0.5.</td></tr>
<tr><td>Trajectories</td><td>1,000,000</td><td>Sufficient for less than 1% relative standard error.</td></tr>
<tr><td>Reaction criterion</td><td>1 pair at 2.0 Å contact distance</td><td>The sum of radii defines exact contact.</td></tr>
<tr><td>Contacts needed</td><td>1</td><td>Single contact.</td></tr>
</tbody>
</table>

The analytical Smoluchowski rate is reproduced to within 0.1%, with k<sub>on</sub> = 1.56 × 10¹⁰ M⁻¹s⁻¹ versus the exact value of 1.56 × 10¹⁰ M⁻¹s⁻¹ (relative SE 0.1%).

---

## 2. Trypsin-benzamidine complex

Trypsin-benzamidine is a protein-ligand complex with well characterized experimental kinetics. Trypsin protein contains 3220 atoms with a net charge of +6e and a maximum radius of 28.4 Å. Benzamidine contains 18 atoms with a net charge of +1e and a maximum radius of 3.7 Å. Both molecules are positively charged, resulting in repulsive electrostatics.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>45.0 Å</td><td>Combined maximum radii (28.4 + 3.7 = 32.1 Å) plus a buffer.</td></tr>
<tr><td>Receptor hydrodynamic radius</td><td>22.5 Å</td><td>Stokes radius from molecular dimensions, approximately 0.80 × R<sub>max</sub> for a globular protein of this size.</td></tr>
<tr><td>Ligand hydrodynamic radius</td><td>5.0 Å</td><td>Stokes radius for a small planar organic molecule with 18 atoms.</td></tr>
<tr><td>Debye length</td><td>7.86 Å</td><td>Corresponds to 150 mM ionic strength.</td></tr>
<tr><td>APBS fine grid length</td><td>96 Å</td><td>Covers ±48 Å. At the b-surface, the outermost benzamidine atoms reach b + R<sub>max, lig</sub> = 48.7 Å, marginally beyond the grid edge.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>Yields a grid spacing of ~0.37 Å on the fine grid.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>The adaptive timestep at r = 45 Å is ~191 ps, giving drift/noise ≈ 3.4. This is acceptable for a small, rapidly diffusing ligand.</td></tr>
<tr><td>Trajectories</td><td>10,000,000</td><td>The low reaction probability requires many trajectories for statistical convergence.</td></tr>
<tr><td>Contact cutoff (search)</td><td>6.0 Å</td><td>Maximum distance in the crystal structure used to identify binding contacts.</td></tr>
<tr><td>Buffer</td><td>3.0 Å</td><td>Added to the crystal distance to set the reaction cutoff, accounting for the rigid body approach.</td></tr>
<tr><td>Number of pairs</td><td>10</td><td>The top 10 closest polar contacts from the crystal structure.</td></tr>
<tr><td>Contacts needed</td><td>6</td><td>Six of the 10 pairs must be satisfied simultaneously.</td></tr>
<tr><td>Contact mode</td><td>Polar</td><td>Only N/O/S donor-acceptor pairs are considered, corresponding to hydrogen bonding contacts.</td></tr>
</tbody>
</table>

The setup script identifies the closest heavy atom contacts between the receptor and ligand in the crystal structure, filters for polar atoms (N, O, and S on both sides), retains the top 10 contacts with one per receptor residue, and sets each cutoff to the crystal distance. The resulting 10 pairs have cutoffs ranging from 6.0 to 8.5 Å. k<sub>on</sub> = 5.39 × 10⁷ M⁻¹s⁻¹ versus the experimental value of 2.9 × 10⁷ M⁻¹s⁻¹ (ratio 1.86×, relative SE 0.5%).

---

## 3. β-cyclodextrin host-guest complexes

Seven small molecule guests binding β-cyclodextrin form a host-guest benchmark in which the experimental association rates span an order of magnitude. β-cyclodextrin contains 147 atoms with zero net charge and a maximum radius of 8.6 Å. All seven guests are also neutral, with radii of 3 to 5 Å. Because every molecule is neutral, there is no electrostatic steering. The seven guests are 1-butanol, 1-propanol, tert-butanol, methyl butyrate, aspirin, 1-naphthylethanol, and 2-naphthylethanol.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>30.0 Å</td><td>Combined maximum radii (8.6 + 3 ≈ 11.6 Å) plus a buffer.</td></tr>
<tr><td>Hydrodynamic radii</td><td>Auto-computed</td><td>Determined from the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td>Debye length</td><td>7.86 Å</td><td>Corresponds to 150 mM ionic strength.</td></tr>
<tr><td>APBS fine grid length</td><td>96 Å</td><td>Covers ±48 Å. The sum b + R<sub>max, lig</sub> ≈ 33 Å is well within the grid.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>Yields a grid spacing of ~0.37 Å.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>The adaptive timestep at r = 30 Å is ~90 ps, giving drift/noise ≈ 1.5.</td></tr>
<tr><td>Overlap check</td><td>Enabled (true)</td><td>Retained for the host-guest complexes. The tight, apolar interface does not trigger the false rejections seen for hydrogen bonding protein criteria.</td></tr>
<tr><td>Trajectories</td><td>2,000,000</td><td>Per host-guest complex.</td></tr>
<tr><td>Contact cutoff (search)</td><td>5.0 Å</td><td>A tight cutoff appropriate for the small host-guest complex.</td></tr>
<tr><td>Buffer</td><td>2.0 Å</td><td>Smaller than the protein complexes because the molecules are smaller and the contacts are tighter.</td></tr>
<tr><td>Number of pairs</td><td>7 to 8</td><td>Up to 8 contact pairs.</td></tr>
<tr><td>Contacts needed</td><td>4</td><td>Four pairs must be satisfied.</td></tr>
<tr><td>Contact mode</td><td>All heavy atoms</td><td>The host-guest interface has few polar atoms.</td></tr>
</tbody>
</table>

Each guest uses seven contact pairs, of which four must form for a reaction, at cutoffs of 5.0 to 6.5 Å taken from the crystal contacts. The O5 glycosidic oxygens of β-cyclodextrin (atoms 15, 57, and 99) recur as the receptor contact atoms across all of the guests, giving a consistent anchor.

<table width="100%">
<thead><tr><th align="left">Guest</th><th align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">Ratio</th><th align="left">rel SE</th></tr></thead>
<tbody>
<tr><td>1-butanol</td><td>2.8 × 10⁸</td><td>6.28 × 10⁸</td><td>2.24×</td><td>0.3%</td></tr>
<tr><td>1-propanol</td><td>5.1 × 10⁸</td><td>5.94 × 10⁸</td><td>1.17×</td><td>0.3%</td></tr>
<tr><td>tert-butanol</td><td>3.6 × 10⁸</td><td>6.30 × 10⁸</td><td>1.75×</td><td>0.3%</td></tr>
<tr><td>methyl butyrate</td><td>3.7 × 10⁸</td><td>4.53 × 10⁸</td><td>1.22×</td><td>0.4%</td></tr>
<tr><td>aspirin</td><td>7.2 × 10⁸</td><td>4.19 × 10⁸</td><td>0.58×</td><td>0.4%</td></tr>
<tr><td>1-naphthylethanol</td><td>4.7 × 10⁸</td><td>6.89 × 10⁷</td><td>0.15×</td><td>0.9%</td></tr>
<tr><td>2-naphthylethanol</td><td>2.9 × 10⁸</td><td>6.18 × 10⁸</td><td>2.13×</td><td>0.3%</td></tr>
</tbody>
</table>

---

## 4. Thrombin-thrombomodulin complex

This complex represents a strongly electrostatically steered protein-protein association. Thrombin contains 4727 atoms with a net charge of +3e and a maximum radius of 34.7 Å. Thrombomodulin EGF domains 4 to 6 contain 1650 atoms with a net charge of −15e and a maximum radius of 40.6 Å. Strong electrostatic complementarity drives fast association. Pre-computed PQR files with AMBER partial charges were used directly.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>85.0 Å</td><td>Combined maximum radii (34.7 + 40.6 = 75.3 Å) plus a buffer.</td></tr>
<tr><td>Receptor hydrodynamic radius</td><td>25.58 Å</td><td>Stokes radius from molecular dimensions, ~0.74 × R<sub>max</sub>.</td></tr>
<tr><td>Ligand hydrodynamic radius</td><td>21.88 Å</td><td>~0.54 × R<sub>max</sub>, with the lower ratio reflecting the elongated EGF domain fragment.</td></tr>
<tr><td>Debye length</td><td>7.86 Å</td><td>Corresponds to 150 mM ionic strength.</td></tr>
<tr><td>APBS fine grid length</td><td>192 Å</td><td>Covers ±96 Å. At b = 85, ligand atoms span 44 to 126 Å from the origin, and the grid encompasses the vast majority of encounter geometries.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>Yields a grid spacing of ~0.75 Å.</td></tr>
<tr><td>Base timestep</td><td>1.0 ps</td><td>Larger base step appropriate at the large b-surface.</td></tr>
<tr><td>Max timestep cap</td><td>100 ps</td><td>Critical. Without the cap the adaptive timestep at r = 85 Å reaches ~1806 ps (drift/noise ≈ 5.4), producing ballistic trajectories that skip the electrostatic steering region. The cap restores drift/noise ≈ 2.</td></tr>
<tr><td>Trajectories</td><td>5,000,000</td><td>Production run.</td></tr>
<tr><td>Contacts needed</td><td>4</td><td>Four interfacial hydrogen bonding contacts must be satisfied simultaneously. The full pair list is in <code>rxns.xml</code>.</td></tr>
</tbody>
</table>

The experimental k<sub>on</sub> is 6.7 × 10⁶ M⁻¹s⁻¹ at physiological ionic strength, measured by surface plasmon resonance, and Debye-Hückel analysis of the ionic strength dependence confirms a nearly completely electrostatically steered interaction. The computed k<sub>on</sub> is 2.42 × 10⁶ M⁻¹s⁻¹ (ratio 0.36×, relative SE 2.8%).

---

## 5. Barnase-barstar (flexible chain Brownian dynamics)

Barnase-barstar is the classic electrostatically steered protein-protein association benchmark, included here as a sample for PySTARC's chain BD module, which extends the rigid body engine to internal conformational degrees of freedom. It pairs barnase (chain A of PDB 1BRS) and barstar (chain D), parameterized with AMBER ff14SB. Unlike the rigid body examples, the chain BD treatment retains internal flexibility rather than freezing each partner as a rigid body.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b / milestone radius</td><td>80 Å</td><td>Milestone radius for the chain BD run.</td></tr>
<tr><td>r<sub>escape</sub></td><td>160 Å</td><td>Outer absorbing radius for the chain BD run.</td></tr>
</tbody>
</table>

---

## 6. p38 MAPK - SB203580 complex

This is a kinase inhibitor complex in which the ligand is neutral and the receptor carries a large net negative charge. p38 MAPK alpha in the DFG-in conformation from PDB 1A9U (chain A, residues 4 to 354) contains 5658 atoms with a net charge of −9e and a maximum radius of 37.8 Å. SB203580 is a type I kinase inhibitor with 27 atoms and zero net charge. The receptor is parameterized with ff14SB and the ligand with GAFF2 + AM1-BCC via antechamber. The ligand binds in the ATP pocket at the hinge.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>60.0 Å</td><td>Combined maximum radii (37.8 + 7.8 = 45.6 Å) plus a buffer.</td></tr>
<tr><td>Hydrodynamic radii</td><td>Auto-computed</td><td>From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td>Debye length</td><td>7.86 Å</td><td>150 mM ionic strength.</td></tr>
<tr><td>APBS fine grid length</td><td>128 Å</td><td>Covers ±64 Å, and the multipole fallback handles outermost atom overshoot at the b-surface.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>~0.50 Å fine grid spacing.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>The adaptive timestep at r = 60 Å is moderate for a small, highly diffusive ligand.</td></tr>
<tr><td>Trajectories</td><td>5,000,000</td><td>Production run.</td></tr>
</tbody>
</table>

Four crystal structure contacts (the hinge MET106 backbone to the pyridine N, the catalytic LYS50 NZ to an imidazole N, and the VAL102 O and THR103 N to the fluorine) at a uniform 7.0 Å cutoff, with n_needed = 3. k<sub>on</sub> = 2.88 × 10⁷ M⁻¹s⁻¹ versus the experimental value of 1.5 × 10⁷ M⁻¹s⁻¹ (ratio 1.92×, relative SE 1.1%).

---

## 7. Carbonic anhydrase sulfonamide inhibitors

Seven sulfonamide inhibitors binding three carbonic anhydrase isozymes (CA XIII, CA I, and CA II) form a multi target benchmark in which all ligands carry the same charge (−1e) and bind the same Zn-coordinating sulfonamide motif but differ in scaffold, size, and isozyme. All ligands are deprotonated sulfonamides (net charge −1e), and the intrinsic k<sub>on</sub> corrected for the protonation equilibrium is the correct BD comparison target. Five complexes use CA XIII (PDB 3CZV), one CA I (2NMX), one CA II (3HS4). The active site Zn²⁺ is included via `frcmod.ions234lm_126_tip3p`. Ligands are built from SMILES via rdkit, converted with obabel, and parameterized with antechamber (GAFF2 + AM1-BCC).

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>60.0 Å</td><td>Combined maximum radii (27 + 8 = 35 Å) plus a buffer.</td></tr>
<tr><td>Hydrodynamic radii</td><td>Auto-computed</td><td>From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td>Debye length</td><td>9.62 Å</td><td>Corresponds to 100 mM ionic strength.</td></tr>
<tr><td>Ion concentration</td><td>0.10 M</td><td>100 mM NaCl.</td></tr>
<tr><td>APBS fine grid length</td><td>128 Å</td><td>Covers ±64 Å, and the multipole fallback handles overshoot.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>~0.50 Å fine grid spacing.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>Moderate adaptive timestep for small, highly diffusive ligands.</td></tr>
<tr><td>Trajectories</td><td>10,000,000</td><td>Per complex.</td></tr>
<tr><td>Reaction criterion</td><td>2 pairs (THR199 OG1 / GLU106 OE1 to sulfonamide N and amide N), cutoff 3.5 Å, n_needed = 2</td><td>Both Zn-coordinating / proton shuttle hydrogen bonds required.</td></tr>
</tbody>
</table>

<table width="100%">
<thead><tr><th align="left">Complex</th><th align="left">Isozyme</th><th align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">Ratio</th><th align="left">rel SE</th></tr></thead>
<tbody>
<tr><td>CA I-VD12-69-1</td><td>CA I</td><td>2.7 × 10⁶</td><td>2.57 × 10⁶</td><td>0.95×</td><td>2.6%</td></tr>
<tr><td>CA XIII-AZM</td><td>CA XIII</td><td>1.5 × 10⁶</td><td>1.87 × 10⁶</td><td>1.25×</td><td>3.3%</td></tr>
<tr><td>CA XIII-VD11-26</td><td>CA XIII</td><td>1.5 × 10⁶</td><td>2.50 × 10⁶</td><td>1.66×</td><td>2.6%</td></tr>
<tr><td>CA XIII-VD12-69-1</td><td>CA XIII</td><td>2.5 × 10⁶</td><td>5.30 × 10⁶</td><td>2.12×</td><td>1.8%</td></tr>
<tr><td>CA XIII-VD11-25</td><td>CA XIII</td><td>4.6 × 10⁵</td><td>1.63 × 10⁶</td><td>3.54×</td><td>3.2%</td></tr>
<tr><td>CA XIII-VD12-09</td><td>CA XIII</td><td>3.3 × 10⁵</td><td>6.16 × 10⁶</td><td>18.67×</td><td>1.6%</td></tr>
<tr><td>CA II-VD11-4-2</td><td>CA II</td><td>1.8 × 10⁶</td><td>3.36 × 10⁵</td><td>0.19×</td><td>7.0%</td></tr>
</tbody>
</table>

---

## 8. TTK (MPS1) kinase inhibitors

Eight inhibitors of the mitotic kinase TTK/MPS1 form a single target series spanning roughly two orders of magnitude in experimental k<sub>on</sub>. The complexes are eight TTK co-crystal structures (PDB 2X9E, 3GFW, 3H9F, 5LJJ, 5N7V, 5N84, 5N93, and 5NAD), each parameterized with ff14SB for the receptor and GAFF2 + AM1-BCC for the ligand, with ligand charges assigned by the OpenEye toolkits (license required).

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>60.0 Å</td><td>Combined maximum radii of the receptor and ligand plus a buffer.</td></tr>
<tr><td>Hydrodynamic radii</td><td>Auto-computed</td><td>From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td>Debye length</td><td>7.86 Å</td><td>150 mM ionic strength.</td></tr>
<tr><td>APBS fine grid length</td><td>144 Å</td><td>Covers ±72 Å.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>~0.56 Å fine grid spacing.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>b &lt; 80 Å, so no cap is required.</td></tr>
<tr><td>Trajectories</td><td>10,000,000</td><td>Per complex.</td></tr>
<tr><td>Reaction criterion</td><td>3 polar crystal contacts (GLU603 O, GLY605 O, and GLY605 N to ligand N9/N2/N3), uniform 4.5 Å cutoff, n_needed = 3</td><td>A single flat cutoff applied uniformly across all eight complexes.</td></tr>
</tbody>
</table>

<table width="100%">
<thead><tr><th align="left">Complex</th><th align="left">Inhibitor</th><th align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">Ratio</th><th align="left">rel SE</th></tr></thead>
<tbody>
<tr><td>5LJJ</td><td>Reversine</td><td>2.08 × 10⁶</td><td>4.87 × 10⁶</td><td>2.34×</td><td>1.8%</td></tr>
<tr><td>2X9E</td><td>NMS-P715</td><td>6.41 × 10⁵</td><td>2.18 × 10⁶</td><td>3.39×</td><td>2.4%</td></tr>
<tr><td>5N84</td><td>Mps-BAY2b</td><td>2.60 × 10⁶</td><td>1.51 × 10⁶</td><td>0.58×</td><td>3.2%</td></tr>
<tr><td>3GFW</td><td>Mps1-IN-1</td><td>3.79 × 10⁵</td><td>9.10 × 10⁵</td><td>2.40×</td><td>4.0%</td></tr>
<tr><td>5N7V</td><td>MPI-0479605</td><td>1.96 × 10⁶</td><td>3.68 × 10⁶</td><td>1.88×</td><td>2.1%</td></tr>
<tr><td>5N93</td><td>TC-Mps1-12</td><td>2.16 × 10⁷</td><td>3.15 × 10⁶</td><td>0.15×</td><td>2.3%</td></tr>
<tr><td>5NAD</td><td>BAY-1217389</td><td>3.79 × 10⁵</td><td>1.21 × 10⁷</td><td>31.94×</td><td>1.1%</td></tr>
<tr><td>3H9F</td><td>Mps1-IN-2</td><td>1.19 × 10⁶</td><td>8.28 × 10⁶</td><td>6.96×</td><td>1.4%</td></tr>
</tbody>
</table>

---

## 9. HSP90 inhibitors

Six neutral HSP90 inhibitors form a single target series of uncharged ligands, isolating the diffusion limited encounter rate in the absence of electrostatic steering. The six complexes are HSP90 N-terminal domain systems named by inhibitor scaffold (resorcinol, indazole-5LNZ, indazole-5OCI, quinazoline-6EI5, quinazoline, aminopyridine; Kokh et al. compounds 31, 37, 43, 62, 65, 70), each with a neutral inhibitor, the receptor parameterized with ff14SB and the ligands with GAFF2 + AM1-BCC (OpenEye toolkits, license required).

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>b-surface radius</td><td>55.0 Å</td><td>Combined maximum radii of the receptor and ligand plus a buffer.</td></tr>
<tr><td>Hydrodynamic radii</td><td>Auto-computed</td><td>From the PQR files via Monte Carlo surface integration.</td></tr>
<tr><td>Debye length</td><td>7.86 Å</td><td>150 mM ionic strength.</td></tr>
<tr><td>APBS fine grid length</td><td>144 Å</td><td>Covers ±72 Å.</td></tr>
<tr><td>APBS grid dimension</td><td>257</td><td>~0.56 Å fine grid spacing.</td></tr>
<tr><td>Max timestep cap</td><td>0 (no cap)</td><td>b &lt; 80 Å, so no cap is required.</td></tr>
<tr><td>Trajectories</td><td>20,000,000</td><td>Per complex.</td></tr>
<tr><td>Reaction criterion</td><td>8 pairs (contact method, SER37/ALA40/ASN36 CA anchors to ligand O2x/C15x/N1x), uniform 5.0 Å cutoff, n_needed = 6</td><td>A single flat 5.0 Å cutoff applied across all six complexes.</td></tr>
</tbody>
</table>

<table width="100%">
<thead><tr><th align="left">Complex</th><th align="left">Exp k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">PySTARC k<sub>on</sub> (M⁻¹s⁻¹)</th><th align="left">Ratio</th><th align="left">rel SE</th></tr></thead>
<tbody>
<tr><td>HSP90-resorcinol</td><td>1.00 × 10⁶</td><td>1.97 × 10⁶</td><td>1.97×</td><td>2.2%</td></tr>
<tr><td>HSP90-indazole-5LNZ</td><td>3.43 × 10⁵</td><td>2.85 × 10⁵</td><td>0.83×</td><td>5.0%</td></tr>
<tr><td>HSP90-indazole-5OCI</td><td>8.38 × 10⁴</td><td>6.31 × 10⁵</td><td>7.53×</td><td>3.4%</td></tr>
<tr><td>HSP90-quinazoline-6EI5</td><td>1.21 × 10⁵</td><td>3.75 × 10⁵</td><td>3.10×</td><td>4.4%</td></tr>
<tr><td>HSP90-quinazoline</td><td>2.08 × 10⁵</td><td>1.09 × 10⁶</td><td>5.24×</td><td>2.6%</td></tr>
<tr><td>HSP90-aminopyridine</td><td>1.04 × 10⁴</td><td>6.67 × 10⁵</td><td>64.14×</td><td>3.4%</td></tr>
</tbody>
</table>

---

## Common parameters

Most complexes share the following. Exceptions are noted in the sections above.

<table width="100%">
<thead><tr><th align="left">Parameter</th><th align="left">Value</th><th align="left">Rationale</th></tr></thead>
<tbody>
<tr><td>Protein dielectric</td><td>4.0</td><td>Standard for BD simulations. (Exception: the two sphere analytical test uses a uniform dielectric of 78.0.)</td></tr>
<tr><td>Solvent dielectric</td><td>78.0</td><td>Water at 298.15 K.</td></tr>
<tr><td>Solvent probe radius</td><td>1.4 Å</td><td>Water molecule radius.</td></tr>
<tr><td>Desolvation coupling</td><td>0.0795775</td><td>Equal to 1/(4π), the Born desolvation coupling constant.</td></tr>
<tr><td>Hydrodynamic interactions</td><td>Enabled</td><td>Included in the k<sub>b</sub> integral.</td></tr>
<tr><td>Overlap check</td><td>System-dependent</td><td><strong>Disabled (false)</strong> for the protein-ligand and protein-protein hydrogen bonding criteria, where the atom pair overlap check would otherwise reject valid near contact binding poses, and <strong>enabled (true)</strong> for the β-cyclodextrin host-guest complexes.</td></tr>
<tr><td>Multipole fallback</td><td>Enabled</td><td>Monopole, dipole, and quadrupole Yukawa expansions are used beyond the APBS grid boundary.</td></tr>
<tr><td>Lennard-Jones forces</td><td>Disabled</td><td>Standard for rigid body BD.</td></tr>
<tr><td>Temperature</td><td>298.15 K</td><td>Room temperature, k<sub>B</sub>T = 1.0 in reduced units.</td></tr>
<tr><td>Base timestep</td><td>0.2 ps</td><td>Minimum core timestep near the receptor surface. (Exception: thrombin-thrombomodulin uses 1.0 ps.)</td></tr>
</tbody>
</table>

## Adaptive timestep cap

The variable timestep is Δt<sub>pair</sub> = f²r²/(2D) with f = 0.1, which keeps the RMS displacement per step below 10% of the intermolecular separation. A b-surface below 80 Å needs no cap, but for the larger protein-protein b-surfaces the timestep would exceed 1000 ps at the starting radius and skip the electrostatic steering region, so it is capped at a ceiling (typically 100 ps, or 0 for no cap).

<table width="100%">
<thead><tr><th align="left">Complex</th><th align="left">b (Å)</th><th align="left">Δt at b (ps)</th><th align="left">Drift/noise</th><th align="left">Cap</th></tr></thead>
<tbody>
<tr><td>Charged spheres</td><td>10</td><td>10</td><td>0.5</td><td>No</td></tr>
<tr><td>β-cyclodextrin host-guest</td><td>30</td><td>90</td><td>1.5</td><td>No</td></tr>
<tr><td>Trypsin-benzamidine</td><td>45</td><td>191</td><td>3.4</td><td>No</td></tr>
<tr><td>HSP90 inhibitors</td><td>55</td><td>-</td><td>-</td><td>No (max_dt = 0)</td></tr>
<tr><td>p38 MAPK - SB203580</td><td>60</td><td>~360</td><td>~3.0</td><td>No</td></tr>
<tr><td>Carbonic anhydrase inhibitors</td><td>60</td><td>~360</td><td>~3.0</td><td>No</td></tr>
<tr><td>TTK inhibitors</td><td>60</td><td>-</td><td>-</td><td>No (max_dt = 0)</td></tr>
<tr><td>Thrombin-thrombomodulin</td><td>85</td><td>1806</td><td>5.4</td><td>Yes, 100 ps</td></tr>
</tbody>
</table>

The flexible chain BD example (§5) uses a distinct propagation scheme with its own timestep handling, and its parameters will be documented once the run is finalized.

---

## Summary

Across the protein-ligand, protein-protein, and host-guest complexes, the agreement between the computed and experimental association rate constants in log₁₀ space is summarized below.

<table width="100%">
<thead><tr><th align="left">Metric</th><th align="left">Value</th></tr></thead>
<tbody>
<tr><td>Pearson r</td><td>0.912 (r² = 0.832)</td></tr>
<tr><td>Spearman ρ</td><td>0.795</td></tr>
<tr><td>R² vs y = x</td><td>0.790</td></tr>
<tr><td>log₁₀ MAE</td><td>0.488 (mean fold error 3.1×)</td></tr>
<tr><td>log₁₀ RMSE</td><td>0.644</td></tr>
<tr><td>log₁₀ bias</td><td>0.274 (systematic 1.9× overprediction)</td></tr>
<tr><td>Converged</td><td>31 / 32</td></tr>
</tbody>
</table>

The systematic overprediction, largest for the slowest experimental binders, reflects the diffusion limited floor of rigid body BD, which cannot lower k<sub>on</sub> for binders whose rate is set by non-diffusional gating.
