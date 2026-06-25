# Parameter selection

## Overview

All complexes use the AMBER ff14SB force field for receptor charge assignment (GAFF2 + AM1-BCC for small molecule ligands), APBS for electrostatic potential grids, and the Northrup-Allison-McCammon framework for computing bimolecular association rate constants. All numerical values in this guide are taken from the on-disk `input.xml`, `rxns.xml`, and `results.json` files for each example.

> **Note on sources.** Where a reaction-criterion *value* (cutoff, number of pairs, contacts needed) appears below, it is read directly from the corresponding `rxns.xml`/`setup.py`.

---

## 1. Two charged spheres

**Purpose.** This complex validates the BD engine against the exact analytical Smoluchowski solution for two uniformly charged spheres interacting via a screened Coulombic potential.

**System.** Two spherical ions with charges Q<sub>rec</sub> = +1e and Q<sub>lig</sub> = −1e, each with radius 1.0 Å. This is a purely analytical test of the BD propagator.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b-surface radius | 10.0 Å | Five times the contact distance of 2.0 Å. The system is small and has no molecular extent. |
| Hydrodynamic radii | 0 / 0 (point charges) | No hydrodynamic radius is assigned as the spheres are treated as point particles for the analytical comparison. |
| Debye length | 7.828 Å | Corresponds to ~150 mM ionic strength. |
| Protein / solvent dielectric | 78.0 / 78.0 | Uniform dielectric. Unlike the molecular systems, no low-dielectric interior is defined for the analytical test. |
| APBS fine grid | None | The screened Coulomb potential between the two point charges is evaluated analytically and no APBS grid is generated. |
| APBS grid dimension | 129 | Vestigial for the analytic potential. |
| Max timestep cap | 0 (no cap) | The adaptive timestep at r = 10 Å is ~10 ps, giving drift/noise ≈ 0.5. |
| Trajectories | 1,000,000 | Sufficient for less than 1% relative standard error. |
| Reaction criterion | 1 pair at 2.0 Å contact distance | The sum of radii defines exact contact. |
| Contacts needed | 1 | Single contact. |

**Result.** PySTARC reproduces the analytical Smoluchowski rate to within 0.1%: k<sub>on</sub> = 1.56 × 10¹⁰ M⁻¹s⁻¹ versus the exact value of 1.56 × 10¹⁰ M⁻¹s⁻¹ (relative SE 0.1%).

---

## 2. Trypsin-benzamidine complex

**Purpose.** The trypsin-benzamidine complex validates PySTARC for a protein and a small molecule complex with well-characterized experimental kinetics.

**System.** Trypsin protein contains 3220 atoms with a net charge of +6e and a maximum radius of 28.4 Å. Benzamidine contains 18 atoms with a net charge of +1e and a maximum radius of 3.7 Å. Both molecules are positively charged, resulting in repulsive electrostatics.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b-surface radius | 45.0 Å | The sum of the maximum molecular radii (28.4 + 3.7 = 32.1 Å) plus clearance. |
| Receptor hydrodynamic radius | 22.5 Å | Stokes radius from molecular dimensions, approximately 0.80 × R<sub>max</sub> for a globular protein of this size. |
| Ligand hydrodynamic radius | 5.0 Å | Stokes radius for a small planar organic molecule with 18 atoms. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 96 Å | Covers ±48 Å. At the b-surface, the outermost benzamidine atoms reach b + R<sub>max, lig</sub> = 48.7 Å, marginally beyond the grid edge. |
| APBS grid dimension | 257 | Yields a grid spacing of ~0.37 Å on the fine grid. |
| Max timestep cap | 0 (no cap) | The adaptive timestep at r = 45 Å is ~191 ps, giving drift/noise ≈ 3.4. This is acceptable for a small, rapidly diffusing ligand. |
| Trajectories | 10,000,000 | The low reaction probability requires many trajectories for statistical convergence. |
| Contact cutoff (search) | 6.0 Å | Maximum distance in the crystal structure used to identify binding contacts. |
| Buffer | 3.0 Å | Added to the crystal distance to set the reaction cutoff, accounting for the rigid-body approach. |
| Number of pairs | 10 | The top 10 closest polar contacts from the crystal structure. |
| Contacts needed | 6 | Six of the 10 pairs must be satisfied simultaneously. |
| Contact mode | Polar | Only N/O/S donor-acceptor pairs are considered, corresponding to hydrogen bonding contacts. |

**Reaction criterion construction.** The setup script identifies the closest heavy-atom contacts between the receptor and ligand in the crystal structure, filters for polar atoms (N, O, S on both sides), retains the top 10 contacts with one per receptor residue, and sets each cutoff to the crystal distance. The resulting 10 pairs have cutoffs ranging from 6.0 to 8.5 Å.

**Result.** k<sub>on</sub> = 5.39 × 10⁷ M⁻¹s⁻¹ versus the experimental value of 2.9 × 10⁷ M⁻¹s⁻¹ (ratio 1.86×, relative SE 0.5%).

---

## 3. β-cyclodextrin host-guest complexes

**Purpose.** Seven small molecule guests binding β-cyclodextrin test PySTARC on a host-guest benchmark where experimental association rates span an order of magnitude.

**System.** β-cyclodextrin contains 147 atoms with zero net charge and a maximum radius of 8.6 Å. All seven guest molecules also carry a net charge of zero and have radii ranging from 3 to 5 Å. Because all molecules are electrically neutral, no electrostatic steering occurs. The seven guests are 1-butanol, 1-propanol, tert-butanol, methyl butyrate, aspirin, 1-naphthylethanol, and 2-naphthylethanol.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b-surface radius | 30.0 Å | The sum of molecular radii (8.6 + 3 ≈ 11.6 Å) plus clearance. |
| Hydrodynamic radii | Auto-computed | Determined from the PQR files via Monte Carlo surface integration. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 96 Å | Covers ±48 Å. The sum b + R<sub>max, lig</sub> ≈ 33 Å is well within the grid. |
| APBS grid dimension | 257 | Yields a grid spacing of ~0.37 Å. |
| Max timestep cap | 0 (no cap) | The adaptive timestep at r = 30 Å is ~90 ps, giving drift/noise ≈ 1.5. |
| Overlap check | Enabled (true) | Retained for the host-guest complexes. The tight, apolar interface does not trigger the false rejections seen for H-bonding protein criteria. |
| Trajectories | 2,000,000 | Per host-guest complex. |
| Contact cutoff (search) | 5.0 Å | A tight cutoff appropriate for the small host-guest complex. |
| Buffer | 2.0 Å | Smaller than the protein complexes because the molecules are smaller and the contacts are tighter. |
| Number of pairs | 7–8 | Up to 8 contact pairs. |
| Contacts needed | 4 | Four pairs must be satisfied. |
| Contact mode | All heavy atoms | The host-guest interface has few polar atoms. |

**Per-guest reaction criteria (from rxns.xml).** All use 7 pairs with n_needed = 4. Cutoffs: 1-butanol 5.0/6.0/6.5 Å; 1-propanol 5.5/6.0/6.5 Å; tert-butanol 5.5/6.0 Å; methyl butyrate 5.5/6.0/6.5 Å; aspirin 5.0/5.5/6.0 Å; 1-naphthylethanol 5.0/5.5/6.0 Å; 2-naphthylethanol 5.5/6.0 Å. The O5 glycosidic oxygens of β-cyclodextrin (atoms 15, 57, 99) recur as receptor contact atoms across all complexes, providing a consistent anchor.

**Results.**

| Guest | Exp k<sub>on</sub> (M⁻¹s⁻¹) | PySTARC k<sub>on</sub> (M⁻¹s⁻¹) | Ratio | rel SE |
|-------|------------------------------|-----------------------------------|-------|--------|
| 1-butanol | 2.8 × 10⁸ | 6.28 × 10⁸ | 2.24× | 0.3% |
| 1-propanol | 5.1 × 10⁸ | 5.94 × 10⁸ | 1.17× | 0.3% |
| tert-butanol | 3.6 × 10⁸ | 6.30 × 10⁸ | 1.75× | 0.3% |
| methyl butyrate | 3.7 × 10⁸ | 4.53 × 10⁸ | 1.22× | 0.4% |
| aspirin | 7.2 × 10⁸ | 4.19 × 10⁸ | 0.58× | 0.4% |
| 1-naphthylethanol | 4.7 × 10⁸ | 6.89 × 10⁷ | 0.15× | 0.9% |
| 2-naphthylethanol | 2.9 × 10⁸ | 6.18 × 10⁸ | 2.13× | 0.3% |

---

## 4. Thrombin-thrombomodulin complex

**Purpose.** This complex represents a strongly electrostatically steered protein-protein association.

**System.** Thrombin contains 4727 atoms with a net charge of +3e and a maximum radius of 34.7 Å. Thrombomodulin EGF domains 4–6 contain 1650 atoms with a net charge of −15e and a maximum radius of 40.6 Å. Strong electrostatic complementarity drives fast association. Pre-computed PQR files with AMBER partial charges were used directly.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 85.0 Å | The sum of maximum molecular radii (34.7 + 40.6 = 75.3 Å) plus ~10 Å of clearance. |
| Receptor hydrodynamic radius | 25.58 Å | Stokes radius from molecular dimensions, ~0.74 × R<sub>max</sub>. |
| Ligand hydrodynamic radius | 21.88 Å | ~0.54 × R<sub>max</sub>; the lower ratio reflects the elongated EGF-domain fragment. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 192 Å | Covers ±96 Å. At b = 85, ligand atoms span 44–126 Å from the origin; the grid encompasses the vast majority of encounter geometries. |
| APBS grid dimension | 257 | Yields a grid spacing of ~0.75 Å. |
| Base timestep | 1.0 ps | Larger base step appropriate at the large b surface. |
| Max timestep cap | 100 ps | Critical. Without the cap the adaptive timestep at r = 85 Å reaches ~1806 ps (drift/noise ≈ 5.4), producing ballistic trajectories that skip the electrostatic steering region. The cap restores drift/noise ≈ 2. |
| Trajectories | 1,000,000 | — |
| Contacts needed | 4 | Four interfacial hydrogen-bonding contacts must be satisfied simultaneously. (The full pair list is in `rxns.xml`; it was not re-parsed into report.txt — confirm pair count/cutoffs against the file.) |

**Experimental target.** k<sub>on</sub> = 6.7 × 10⁶ M⁻¹s⁻¹ at physiological ionic strength, measured by surface plasmon resonance (Baerga-Ortiz et al., 2000). Debye-Hückel analysis of the ionic-strength dependence confirms a nearly completely electrostatically steered interaction.

**Result.** k<sub>on</sub> = 2.38 × 10⁶ M⁻¹s⁻¹ (ratio 0.36×, relative SE 6.2%).

---

## 5. Barnase-barstar (flexible chain Brownian dynamics)

**Status.** This example is under active validation. The flexible-chain barnase-barstar run is the primary test case for PySTARC's chain BD module, which extends the rigid-body engine to internal conformational degrees of freedom. **A converged k<sub>on</sub> is not yet on disk; the production run (job 6482856) was not complete at the time of writing, so the parameter table and result below are pending finalization and must not be cited until populated from `examples/barnase_barstar_chainbd/input.xml` and `bd_sims/results.json`.**

**Purpose.** Barnase-barstar is the classic electrostatically steered protein-protein association benchmark with extensive BD literature, making it the natural system on which to validate the chain BD propagator against both rigid-body PySTARC and experiment.

**System.** Barnase (chain A of PDB 1BRS) and barstar (chain D), parameterized with AMBER ff14SB. Unlike the rigid-body examples, the chain BD treatment retains internal flexibility rather than freezing each partner as a rigid body.

**Setup finding (confirmed).** During chain BD setup, a milestone/b-sphere radius of 60 Å produced placement-degenerate reactions at t = 0 (partners reacting on the initial placement before any dynamics). A radius of 80 Å was confirmed clean via a short verification run. The active production run accordingly uses b = 80 Å with r<sub>escape</sub> = 160 Å.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b / milestone radius | 80 Å | Confirmed clean; b = 60 Å gave placement-degenerate t = 0 reactions. (Value from the production run; confirm the shipped example's `input.xml`.) |
| r<sub>escape</sub> | 160 Å | Outer absorbing radius for the chain BD run. (Production value; confirm in the example.) |
| Trajectories / steps | *pending* | To be taken from the finalized run. |
| Chain timestep, constraints, COFFDROP / solvent terms | *pending* | The chain BD propagator uses a distinct integration scheme; document from the example once finalized. |

**Result.** *Pending — k<sub>on</sub> to be populated from `bd_sims/results.json` when job 6482856 completes.*

**Experimental references (for the eventual comparison).** Wild-type k<sub>on</sub> has been reported as 6.0 × 10⁸ M⁻¹s⁻¹ (Schreiber and Fersht, 1993) and 2.86 × 10⁸ M⁻¹s⁻¹ (Frembgen-Kesner and Elcock, 2010), both at 50 mM ionic strength; the basal rate without electrostatics is 5.8 × 10⁶ M⁻¹s⁻¹ (Northrup and Erickson, 1992).

> The earlier rigid-body barnase-barstar section (wild-type and R59A mutant) has been removed: it documented a rigid-body example that is no longer shipped (only `barnase_barstar_chainbd/` exists on disk), and that work is superseded by the chain BD validation here. If the rigid-body WT/R59A results should be retained as historical record, they can be reinstated on request.

---

## 6. p38 MAPK - SB203580 complex

**Purpose.** This complex validates PySTARC on a kinase-inhibitor system where the ligand is electrically neutral and the receptor carries a large net negative charge.

**System.** p38 MAPK alpha in the DFG-in conformation from PDB 1A9U (chain A, residues 4–354) contains 5658 atoms with a net charge of −9e and a maximum radius of 37.8 Å. SB203580 is a type I kinase inhibitor with 27 atoms and zero net charge. The receptor is parameterized with ff14SB and the ligand with GAFF2 + AM1-BCC via antechamber. The ligand binds in the ATP pocket at the hinge.
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 60.0 Å | The sum of maximum radii (37.8 + 7.8 = 45.6 Å) plus ~14 Å of clearance. |
| Hydrodynamic radii | Auto-computed | From the PQR files via Monte Carlo surface integration. |
| Debye length | 7.86 Å | 150 mM ionic strength. |
| APBS fine grid length | 128 Å | Covers ±64 Å; the multipole fallback handles outermost-atom overshoot at the b surface. |
| APBS grid dimension | 257 | ~0.50 Å fine-grid spacing. |
| Max timestep cap | 0 (no cap) | The adaptive timestep at r = 60 Å is moderate for a small, highly diffusive ligand. |
| Trajectories | 5,000,000 | Production run (earlier validation used 100,000). |

**Reaction criterion (from rxns.xml).** Four crystal-structure contacts (hinge MET106 backbone–pyridine N; catalytic LYS50 NZ–imidazole N; VAL102 O and THR103 N–fluorine) at a uniform 7.0 Å cutoff, with **n_needed = 3** (raised from the earlier 2).
**Result.** k<sub>on</sub> = 2.86 × 10⁷ M⁻¹s⁻¹ versus the experimental value of 1.5 × 10⁷ M⁻¹s⁻¹ (ratio 1.91×, relative SE 1.1%).

---

## 7. Carbonic anhydrase sulfonamide inhibitors

**Purpose.** Seven sulfonamide inhibitors binding three carbonic anhydrase isozymes (CA XIII, CA I, CA II) test PySTARC on a multi-target protein-ligand benchmark where all ligands carry the same charge (−1e) and bind the same Zn-coordinating sulfonamide motif but differ in scaffold, size, and isozyme.

**System.** All ligands are deprotonated sulfonamides (net charge −1e); the intrinsic k<sub>on</sub> corrected for the protonation equilibrium is the correct BD comparison target. Five systems use CA XIII (PDB 3CZV), one CA I (2NMX), one CA II (3HS4). The active-site Zn²⁺ is included via `frcmod.ions234lm_126_tip3p`. Ligands are built from SMILES via rdkit, converted with obabel, and parameterized with antechamber (GAFF2 + AM1-BCC).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 60.0 Å | Receptor R<sub>max</sub> ≈ 27 Å plus ligand 4–8 Å plus ~25 Å clearance. |
| Hydrodynamic radii | Auto-computed | From the PQR files via Monte Carlo surface integration. |
| Debye length | 9.62 Å | Corresponds to 100 mM ionic strength, matching the SPR conditions of Linkuviene et al. (2018). |
| Ion concentration | 0.10 M | 100 mM NaCl. |
| APBS fine grid length | 128 Å | Covers ±64 Å; multipole fallback handles overshoot. |
| APBS grid dimension | 257 | ~0.50 Å fine-grid spacing. |
| Max timestep cap | 0 (no cap) | Moderate adaptive timestep for small, highly diffusive ligands. |
| Trajectories | 5,000,000 | Per system. |
| Reaction criterion | 2 pairs (THR199 OG1 / GLU106 OE1 to sulfonamide N and amide N), cutoff 3.5 Å, n_needed = 2 | Both Zn-coordinating / proton-shuttle hydrogen bonds required. |

**Results.**

| System | Isozyme | Exp k<sub>on</sub> (M⁻¹s⁻¹) | PySTARC k<sub>on</sub> (M⁻¹s⁻¹) | Ratio | rel SE |
|--------|---------|------------------------------|-----------------------------------|-------|--------|
| CA I-VD12-69-1 | CA I | 2.7 × 10⁶ | 2.35 × 10⁶ | 0.87× | 3.8% |
| CA XIII-AZM | CA XIII | 1.5 × 10⁶ | 1.79 × 10⁶ | 1.20× | 4.7% |
| CA XIII-VD11-26 | CA XIII | 1.5 × 10⁶ | 2.35 × 10⁶ | 1.57× | 3.8% |
| CA XIII-VD12-69-1 | CA XIII | 2.5 × 10⁶ | 5.74 × 10⁶ | 2.30× | 2.4% |
| CA XIII-VD11-25 | CA XIII | 4.6 × 10⁵ | 2.14 × 10⁶ | 4.66× | 3.9% |
| CA XIII-VD12-09 | CA XIII | 3.3 × 10⁵ | 5.63 × 10⁶ | 17.06× | 2.4% |
| CA II-VD11-4-2 | CA II | 1.8 × 10⁶ | 3.62 × 10⁵ | 0.20× | 9.5% |

---

## 8. TTK (MPS1) kinase inhibitors

**Purpose.** Eight inhibitors of the mitotic kinase TTK/MPS1 test PySTARC across a single-target series spanning roughly two orders of magnitude in experimental k<sub>on</sub>.

**System.** Eight TTK co-crystal structures (PDB 2X9E, 3GFW, 3H9F, 5LJJ, 5N7V, 5N84, 5N93, 5NAD), each parameterized with ff14SB (receptor) and GAFF2 + AM1-BCC (ligand). Ligand charge assignment uses the OpenEye toolkits (license required).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 60.0 Å | Globular kinase plus small-molecule inhibitor with clearance. |
| Hydrodynamic radii | Auto-computed | From the PQR files via Monte Carlo surface integration. |
| Debye length | 7.86 Å | 150 mM ionic strength. |
| APBS fine grid length | 144 Å | Covers ±72 Å. |
| APBS grid dimension | 257 | ~0.56 Å fine-grid spacing. |
| Max timestep cap | 0 (no cap) | b < 80 Å; no cap required. |
| Trajectories | 10,000,000 | Per system. |
| Reaction criterion | 3 polar crystal contacts (GLU603 O, GLY605 O, GLY605 N to ligand N9/N2/N3), uniform 4.5 Å cutoff, n_needed = 3 | A single flat cutoff applied uniformly across all eight systems. |

**Results.**

| System | Inhibitor | Exp k<sub>on</sub> (M⁻¹s⁻¹) | PySTARC k<sub>on</sub> (M⁻¹s⁻¹) | Ratio | rel SE |
|--------|-----------|------------------------------|-----------------------------------|-------|--------|
| 5LJJ | Reversine | 2.08 × 10⁶ | 4.91 × 10⁶ | 2.36× | 1.8% |
| 2X9E | NMS-P715 | 6.41 × 10⁵ | 2.37 × 10⁶ | 3.70× | 2.3% |
| 5N84 | Mps-BAY2b | 2.60 × 10⁶ | 1.51 × 10⁶ | 0.58× | 3.2% |
| 3GFW | Mps1-IN-1 | 3.79 × 10⁵ | 9.32 × 10⁵ | 2.46× | 4.0% |
| 5N7V | MPI-0479605 | 1.96 × 10⁶ | 3.62 × 10⁶ | 1.85× | 2.1% |
| 5N93 | TC-Mps1-12 | 2.16 × 10⁷ | 3.15 × 10⁶ | 0.15× | 2.3% |
| 5NAD | BAY-1217389 | 3.79 × 10⁵ | 1.21 × 10⁷ | 31.94× | 1.1% |
| 3H9F | Mps1-IN-2 | 1.19 × 10⁶ | 8.32 × 10⁶ | 6.99× | 1.3% |

---

## 9. HSP90 inhibitors

**Purpose.** Six neutral HSP90 inhibitors test PySTARC on a single-target series of uncharged ligands, isolating the diffusion-limited encounter rate in the absence of electrostatic steering.

**System.** Six HSP90 N-terminal domain co-complexes (systems 31, 37, 43, 62, 65, 70), each with a neutral inhibitor. Receptor parameterized with ff14SB; ligands with GAFF2 + AM1-BCC (OpenEye toolkits, license required).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 55.0 Å | Globular HSP90 N-domain plus small inhibitor with clearance. |
| Hydrodynamic radii | Auto-computed | From the PQR files via Monte Carlo surface integration. |
| Debye length | 7.86 Å | 150 mM ionic strength. |
| APBS fine grid length | 144 Å | Covers ±72 Å. |
| APBS grid dimension | 257 | ~0.56 Å fine-grid spacing. |
| Max timestep cap | 0 (no cap) | b < 80 Å; no cap required. |
| Trajectories | 10,000,000 | Per system. |
| Reaction criterion | 8 pairs (contact method; SER37/ALA40/ASN36 CA anchors to ligand O2x/C15x/N1x), uniform 5.0 Å cutoff, n_needed = 6 | A single flat 5.0 Å cutoff applied across all six systems. |

**Results.**

| System | Exp k<sub>on</sub> (M⁻¹s⁻¹) | PySTARC k<sub>on</sub> (M⁻¹s⁻¹) | Ratio | rel SE |
|--------|------------------------------|-----------------------------------|-------|--------|
| 31 | 1.00 × 10⁶ | 2.09 × 10⁶ | 2.09× | 3.0% |
| 37 | 3.43 × 10⁵ | 2.70 × 10⁵ | 0.79× | 7.3% |
| 43 | 8.38 × 10⁴ | 6.37 × 10⁵ | 7.61× | 4.8% |
| 62 | 1.21 × 10⁵ | 4.10 × 10⁵ | 3.39× | 5.9% |
| 65 | 2.08 × 10⁵ | 1.08 × 10⁶ | 5.20× | 3.7% |
| 70 | 1.04 × 10⁴ | 6.10 × 10⁵ | 58.68× | 5.0% |

---

## Common parameters

Most complexes share the following. Exceptions are noted in the per-system sections above.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Protein dielectric | 4.0 | Standard for BD simulations. (Exception: the two-sphere analytical test uses a uniform dielectric of 78.0.) |
| Solvent dielectric | 78.0 | Water at 298.15 K. |
| Solvent probe radius | 1.4 Å | Water molecule radius. |
| Desolvation coupling | 0.0795775 | Equal to 1/(4π), the Born desolvation coupling constant. |
| Hydrodynamic interactions | Enabled | Included in the k<sub>b</sub> integral. |
| Overlap check | System-dependent | **Disabled (false)** for the protein-ligand and protein-protein hydrogen-bonding criteria, where the atom-pair overlap check would otherwise reject valid near-contact binding poses; **enabled (true)** for the β-cyclodextrin host-guest systems. |
| Multipole fallback | Enabled | Monopole, dipole, and quadrupole Yukawa expansions are used beyond the APBS grid boundary. |
| Lennard-Jones forces | Disabled | Standard for rigid-body BD. |
| Temperature | 298.15 K | Room temperature, k<sub>B</sub>T = 1.0 in reduced units. |
| Base timestep | 0.2 ps | Minimum core timestep near the receptor surface. (Exception: thrombin-thrombomodulin uses 1.0 ps.) |

## Adaptive timestep cap

The variable-timestep scheme uses Δt<sub>pair</sub> = f²r²/(2D), with f = 0.1, keeping the RMS displacement per step below 10% of the intermolecular separation. For complexes where the b surface is below 80 Å, this yields moderate timesteps with drift/noise below ~4, and no cap is required. For the larger protein-protein b surfaces, the adaptive timestep exceeds 1000 ps at the starting radius, producing deterministic trajectories that skip the electrostatic steering region; the max timestep parameter caps it at a user-specified ceiling (typically 100 ps), restoring proper Brownian sampling. When max timestep is 0 (default), no cap is applied.

| Complex | b (Å) | Δt at b (ps) | Drift/noise | Cap |
|---------|-------|--------------|-------------|-----|
| Charged spheres | 10 | 10 | 0.5 | No |
| β-cyclodextrin host-guest | 30 | 90 | 1.5 | No |
| Trypsin-benzamidine | 45 | 191 | 3.4 | No |
| HSP90 inhibitors | 55 | — | — | No (max_dt = 0) |
| p38 MAPK - SB203580 | 60 | ~360 | ~3.0 | No |
| Carbonic anhydrase inhibitors | 60 | ~360 | ~3.0 | No |
| TTK inhibitors | 60 | — | — | No (max_dt = 0) |
| Thrombin-thrombomodulin | 85 | 1806 | 5.4 | Yes, 100 ps |

The flexible chain BD example (§5) uses a distinct propagation scheme with its own timestep handling; parameters will be documented there once the run is finalized.

---

## Summary

Across the 32 rigid-body systems above, agreement with experiment in log₁₀ space is:

| Metric | Value |
|--------|-------|
| Pearson r | 0.913 (r² = 0.833) |
| Spearman ρ | 0.796 |
| R² vs y = x | 0.790 |
| log₁₀ MAE | 0.493 (mean fold-error 3.1×) |
| log₁₀ RMSE | 0.644 |
| log₁₀ bias | 0.277 (systematic 1.9× over-prediction) |
| Converged | 28 / 32 |

The positive bias and the pattern of overprediction for the slow experimental binders are consistent with a rigid body systematic floor in BD simulations. It computes the diffusion-limited encounter rate and cannot lower k<sub>on</sub> for binders whose experimental rate is set by non-diffusional gating.
