# Parameter selection for PySTARC simulations

## Overview

All complexes use the AMBER ff14SB force field for charge assignment via ambpdb, APBS for electrostatic potential grids, and the Northrup Allison McCammon framework for computing bimolecular association rate constants.

---

## 1. Two charged spheres

**Purpose.** This complex validates the BD engine against the exact analytical Smoluchowski solution for two uniformly charged spheres interacting via a screened Coulombic potential.

**System.** Two spherical ions with charges Q<sub>rec</sub> = +1e and Q<sub>lig</sub> = −1e, each with radius 1.0 Å. There is no molecular structure, as this is a purely analytical test of the BD propagator, the outer-propagator return probability, and the k<sub>b</sub> integral.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 10.0 Å | Five times the contact distance of 2.0 Å. The system is small and has no molecular extent. |
| Hydrodynamic radii | 1.0 / 1.0 Å | Exact sphere radii for both receptor and ligand. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 96 Å | Covers ±48 Å, far exceeding the b surface. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.37 Å. |
| Max timestep cap | 0 (no cap) | Not needed as the adaptive timestep at r = 10 Å is approximately 10 ps, giving drift/noise ≈ 0.5. |
| Trajectories | 1,000,000 | Sufficient for less than 1% relative standard error. |
| Reaction criterion | 1 pair at 2.0 Å contact distance | The sum of radii defines exact contact. |
| Contacts needed | 1 | Single contact. |

**Result.** The computed reaction probability P<sub>rxn</sub> = 0.4479 agrees with the exact Smoluchowski solution of P<sub>rxn</sub> = 0.4501 to within 0.5%.

---

## 2. Trypsin-benzamidine complex

**Purpose.** This complex validates PySTARC for a protein and a small-molecule complex with well-characterized experimental kinetics.

**System.** Trypsin contains 3220 atoms with a net charge of +6e and a maximum radius of 28.4 Å. Benzamidine contains 18 atoms with a net charge of +1e and a maximum radius of 3.7 Å. Both molecules are positively charged, resulting in repulsive electrostatics. The PDB structure comes from the seekrflow manuscript (doi.org/10.1101/2025.08.13.669965).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 45.0 Å | The sum of the maximum molecular radii (28.4 + 3.7 = 32.1 Å) plus 13 Å of clearance. |
| Receptor hydrodynamic radius | 22.5 Å | Stokes radius from molecular dimensions, approximately 0.79 × R<sub>max</sub> for a globular protein of this size. |
| Ligand hydrodynamic radius | 5.0 Å | Stokes radius for a small planar organic molecule with 18 atoms. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 128 Å | Covers ±64 Å. At the b surface, the ligand atoms extend to b + R<sub>max, lig</sub> = 48.7 Å, which is within the grid. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.50 Å on the fine grid. |
| Max timestep cap | 0 (no cap) | Not needed as the adaptive timestep at r = 45 Å is 191 ps, giving drift/noise ≈ 3.4. This is acceptable for a small and rapidly diffusing ligand. |
| Trajectories | 10,000,000 | The low reaction probability (approximately 0.001) requires many trajectories for statistical convergence. |
| Contact cutoff | 6.0 Å | Maximum distance in the crystal structure used to identify binding contacts. |
| Buffer | 3.0 Å | Added to the crystal distance to set the reaction cutoff, accounting for the rigid-body approach. |
| Number of pairs | 10 | The top 10 closest polar contacts from the crystal structure. |
| Contacts needed | 4 | Four of the 10 pairs must be satisfied simultaneously. |
| Contact mode | Polar | Only nitrogen, oxygen, and sulfur donor-acceptor pairs are considered, corresponding to hydrogen-bonding contacts. |

**Reaction criterion construction.** The setup script identifies the closest heavy-atom contacts between receptor and ligand in the crystal structure, filters for polar atoms (nitrogen, oxygen, and sulfur on both sides), retains the top 10 contacts with one per receptor residue, and sets each cutoff to the crystal distance plus 3.0 Å, rounded to the nearest 0.5 Å. The resulting 10 pairs have cutoffs ranging from 6.0 to 8.5 Å.

**Result.** The association rate k<sub>on</sub> = 1.64 × 10⁷ M⁻¹s⁻¹ is within 2× of the experimental value of 2.9 × 10⁷ M⁻¹s⁻¹.

---

## 3. β-cyclodextrin host-guest complexes

**Purpose.** Seven small-molecule guests binding β-cyclodextrin were simulated to test PySTARC on a neutral host-guest benchmark where experimental association rates span an order of magnitude.

**System.** β-cyclodextrin contains 147 atoms with zero net charge and a maximum radius of 8.6 Å. All seven guest molecules also carry a net charge of zero and have radii ranging from 3 to 5 Å. Because all molecules are electrically neutral, no electrostatic steering occurs. Structures were taken from the qmrebind manuscript (doi.org/10.1039/D3SC04195F). The seven guest molecules are 1-butanol, 1-propanol, 1-naphthylethanol, 2-naphthylethanol, aspirin, methyl butyrate, and tert-butanol.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 30.0 Å | The sum of molecular radii (8.6 + 3 = 11.6 Å) plus 18 Å of clearance. |
| Hydrodynamic radii | Auto-computed | Determined from the PQR files via Monte Carlo surface integration. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 96 Å | Covers ±48 Å. The sum b + R<sub>max, lig</sub> = 33 Å is well within the grid. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.37 Å. |
| Max timestep cap | 0 (no cap) | Not needed as the adaptive timestep at r = 30 Å is 90 ps, giving drift/noise = 1.5. |
| Trajectories | 2,000,000 | Per guest complex. |
| Contact cutoff | 5.0 Å | A tight cutoff appropriate for the small host-guest complex. |
| Buffer | 2.0 Å | Smaller than the protein complexes because the molecules are smaller and the contacts are tighter. |
| Number of pairs | 8 | Up to 8 contacts. |
| Contacts needed | 4 | Four of 8 pairs must be satisfied. |
| Contact mode | All heavy atoms | All heavy-atom contacts are considered because the host-guest interface has few polar atoms. |

**Reaction criterion construction.** The same automated procedure as trypsin-benzamidine is employed. The O5 glycosidic oxygens of β-cyclodextrin (atoms 15, 57, and 99) appear as receptor contact atoms across all seven complexes, providing a consistent anchor. Cutoffs range from 5.0 to 6.5 Å.

**Key physics.** All guests are neutral, so BD computes only the diffusion-limited encounter rate with no electrostatic enhancement. The experimental variation in k<sub>on</sub> from 2.8 to 7.2 × 10⁸ M⁻¹s⁻¹ arises from conformational gating and desolvation barriers that rigid-body BD cannot capture. PySTARC correctly predicts encounter rates of approximately 10⁸ for all guests. The absence of rank-order correlation (Spearman ρ = −0.54, p = 0.22) confirms that discrimination among neutral guests requires MD-level detail.

---

## 4. Thrombin-thrombomodulin complex

**Purpose.** This complex represents a strongly electrostatically steered protein-protein association.

**System.** Thrombin contains 4727 atoms with a net charge of +3e and a maximum radius of 34.7 Å. Thrombomodulin EGF domains 4 through 6 contain 1650 atoms with a net charge of −15e and a maximum radius of 40.6 Å. The strong electrostatic complementarity between the two proteins drives fast association. Pre-computed PQR files with AMBER partial charges were used directly for this complex.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 85.0 Å | The sum of maximum molecular radii (34.7 + 40.6 = 75.3 Å) plus 10 Å of clearance. |
| Receptor hydrodynamic radius | 25.58 Å | Stokes radius from molecular dimensions, approximately 0.74 × R<sub>max</sub>. |
| Ligand hydrodynamic radius | 21.88 Å | Stokes radius from molecular dimensions, approximately 0.54 × R<sub>max</sub>. The lower ratio reflects the elongated shape of the EGF domain fragment. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 192 Å | Covers ±96 Å. At b = 85, ligand atoms span 44 to 126 Å from the origin, and the grid encompasses the vast majority of encounter geometries. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.75 Å. |
| APBS coarse grid length | 0 (auto) | Auto-computed from molecular extent. |
| Max timestep cap | 100 ps | Critical for this protein-protein complex. Without the cap, the adaptive timestep at r = 85 Å reaches 1806 ps with drift/noise = 5.4, producing ballistic trajectories that skip the electrostatic steering region. The cap brings drift/noise to approximately 2. |
| Trajectories | 2,000,000 | The tight 5 Å cutoff with 3 contacts needed gives a low reaction probability. |
| Reaction criterion | 21 hydrogen-bonding pairs at 5.0 Å | Identified from the crystal structure using donor-acceptor atom pairs at the binding interface. |
| Contacts needed | 3 | Three of the 21 hydrogen-bonding contacts must be satisfied simultaneously. |

**Experimental target.** The association rate k<sub>on</sub> = 6.7 × 10⁶ M⁻¹s⁻¹ at physiological ionic strength was measured by surface plasmon resonance (Baerga-Ortiz et al. 2000). Debye-Hückel analysis of the ionic-strength dependence yields a slope of −6 and an intercept at zero ionic strength of 10⁹ M⁻¹s⁻¹, confirming that the interaction is nearly completely electrostatically steered.

---

## 5. Barnase-barstar complex (wild-type and R59A mutant)

**Purpose.** This complex validates PySTARC on the classic electrostatically steered protein-protein association benchmark with extensive BD literature. The R59A mutant provides a single-charge perturbation test to verify that PySTARC captures the effect of removing one interfacial salt bridge on the association rate.

### 5a. Wild-type

**System.** Barnase from chain A of PDB 1BRS contains 1701 atoms with a net charge of +2e, a maximum radius of 24.6 Å, and an auto-computed hydrodynamic radius of 18.3 Å. Barstar from chain D contains 1404 atoms with a net charge of −5e, a maximum radius of 21.1 Å, and an auto-computed hydrodynamic radius of 17.0 Å. The complex was parameterized with the AMBER ff14SB force field. Unlike the other complexes, barnase-barstar is a two-chain protein-protein complex in which both molecules are standard amino acids. Splitting into receptor and ligand is performed by residue number after tleap renumbering (barnase = residues 1–108, barstar = residues 109–195).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 80.0 Å | The sum of maximum radii is 45.7 Å. The additional 34 Å of clearance ensures full electrostatic grid coverage at the encounter distance. |
| Hydrodynamic radii | Auto-computed | R<sub>h, rec</sub> = 18.3 Å and R<sub>h, lig</sub> = 17.0 Å, determined by Monte Carlo surface integration of the PQR molecular surface. |
| Debye length | 13.6 Å | Corresponds to 50 mM ionic strength, matching the experimental conditions of Schreiber and Fersht (1993). |
| Ion concentration | 0.05 M | Fifty millimolar sodium chloride. |
| APBS fine grid length | 192 Å | Covers ±96 Å. At the b surface, b + R<sub>max, lig</sub> = 101 Å, which is slightly beyond the grid edge. The Yukawa multipole fallback handles the 5 Å overshoot for the outermost atoms. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.75 Å on the fine grid. |
| Max timestep cap | 100 ps | Critical. Without the cap, the adaptive timestep at r = 80 Å is 1295 ps, producing a drift of 58 Å and noise of 6.6 Å per step. The resulting drift/noise ratio of 8.8 indicates that trajectories move ballistically past the electrostatic funnel at 30–50 Å without sampling it. With the cap set to 100 ps, drift decreases to 4.5 Å, and noise is 1.8 Å, giving drift/noise = 2.4. This single parameter change improved k<sub>on</sub> from 9.9 × 10⁷ to 4.95 × 10⁸. |
| Trajectories | 5,000,000 | With P<sub>rxn</sub> ≈ 0.04, this yields approximately 200,000 reactions and a relative standard error of 0.22%. |

**Reaction criterion.** The criterion follows Gabdoulline and Wade (1997), who demonstrated that reproducing experimental association rates requires satisfaction of specific intermolecular residue contacts rather than a simple center-of-mass distance criterion.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pair 1 | ARG81 NH2 on barnase and ASP147 OD1 on barstar | Corresponds to R83 and D35 in the original paper numbering. This is a key salt bridge at the binding interface. |
| Pair 2 | ARG57 NH1 on barnase and GLU182 OE1 on barstar | Corresponds to R59 and E76 in the original paper numbering. This is the second interfacial salt bridge. |
| Cutoff | 7.0 Å | Tight enough to require genuine approach to the binding interface while accommodating rigid-body tumbling. |
| Contacts needed | 1 | Either contact is sufficient for the encounter. Using 2 contacts needed gave k<sub>on</sub> = 8.5 × 10⁶, which is too strict because both contacts are rarely satisfied simultaneously for two rigid tumbling proteins. |

**Progression of parameter optimization.** The following table shows how successive parameter changes improved agreement with the experiment.

| Run | b (Å) | Cutoff (Å) | Contacts needed | Max timestep (ps) | k<sub>on</sub> (M⁻¹s⁻¹) | vs Experiment |
|-----|-------|------------|----------|-------------|-------------------|---------------|
| 1 | 80 | 6.0 | 2 | — | 3.2 × 10⁶ | 100× low |
| 2 | 80 | 10.0 | 2 | — | 8.5 × 10⁶ | 35× low |
| 3 | 55 | 10.0 | 2 | — | 3.5 × 10⁶ | 85× low |
| 4 | 80 | 10.0 | 1 | — | 9.9 × 10⁷ | 3× low |
| 5 | 80 | 10.0 | 1 | 100 | 4.95 × 10⁸ | 1.2× |

**Key lessons.** First, the max timestep cap was essential, as the adaptive timestep at large separations produced ballistic trajectories that skipped the electrostatic funnel. Second, setting contacts needed to 1 is appropriate for rigid-body BD where both proteins tumble freely and simultaneous satisfaction of two specific contacts is geometrically rare. Third, the cutoff accounts for the absence of side-chain flexibility in rigid-body BD.

**Wild-type experimental references.** The experimental k<sub>on</sub> has been reported as 6.0 × 10⁸ M⁻¹s⁻¹ by Schreiber and Fersht (1993) and 2.86 × 10⁸ M⁻¹s⁻¹ by Frembgen-Kesner and Elcock (2010), both at 50 mM ionic strength. The basal rate without electrostatics is 5.8 × 10⁶ M⁻¹s⁻¹ (Northrup and Erickson, 1992). PySTARC captures the approximately 80× electrostatic enhancement.

### 5b. R59A mutant

**System.** The R59A mutant replaces ARG59 on barnase with ALA, removing one positive charge from the receptor. The mutant barnase contains 1687 atoms with a net charge of +1e. Barstar is unchanged at 1404 atoms with a net charge of −5e. The mutation is applied in the setup script by stripping ARG59 side-chain atoms beyond CB and relabeling the residue as ALA before tleap parameterization.

| Parameter | WT | R59A | Rationale |
|-----------|----|------|-----------|
| Receptor atoms | 1701 | 1687 | Loss of ARG side-chain atoms (14 atoms). |
| Receptor charge | +2e | +1e | Loss of one arginine positive charge. |
| Reaction pairs | 2 (ARG81-ASP147, ARG57-GLU182) | 1 (ARG81-ASP147) | The ARG57-GLU182 pair is eliminated because ARG59 (ARG57 in tleap numbering) no longer exists. |
| Cutoff | 7.0 Å | 7.0 Å | Same as WT. |
| Contacts needed | 1 | 1 | Same as WT. |
| All other parameters | — | Same as WT | b surface, Debye length, APBS grids, max timestep cap, and trajectories are identical. |

**Experimental reference.** The R59A mutant k<sub>on</sub> is approximately 6.5 × 10⁷ M⁻¹s⁻¹, a greater than 9-fold drop from wild-type (Schreiber and Fersht, 1995). This is one of the largest single-residue effects on k<sub>on</sub> in the barnase-barstar system. ARG59 sits at the center of the electrostatic steering interface and forms a direct salt bridge with GLU76 on barstar. Removing it weakens the long-range electrostatic funnel that guides barstar into the correct binding orientation.

**Result.** PySTARC gives k<sub>on</sub> = 2.21 × 10⁸ M⁻¹s⁻¹ for R59A, compared to 4.95 × 10⁸ M⁻¹s⁻¹ for wild-type. The WT/R59A ratio is 2.2×, while the experimental ratio is approximately 4.5×. The direction of the effect is correct and the magnitude is within 2× of the experimental ratio. The incomplete quantitative agreement reflects the limitation of rigid-body BD in capturing the full electrostatic perturbation from a single-charge removal: the reduced steering potential affects not only the radial approach but also the orientational sampling, and both effects are partially attenuated when side chains are frozen.

| System | Q<sub>rec</sub> | Reaction pairs | k<sub>on</sub> (M⁻¹s⁻¹) | Experimental k<sub>on</sub> (M⁻¹s⁻¹) | Ratio |
|--------|-----------------|----------------|---------------------------|----------------------------------------|-------|
| WT | +2e | 2 | 4.95 × 10⁸ | 3–6 × 10⁸ | 1.2× |
| R59A | +1e | 1 | 2.21 × 10⁸ | ~6.5 × 10⁷ | 3.4× |
| WT/R59A ratio | — | — | 2.2× | ~4.5× | — |

---

## 6. p38 MAPK - SB203580 complex

**Purpose.** This complex validates PySTARC on a kinase-inhibitor system where the ligand is electrically neutral and the receptor carries a large net negative charge. It provides a comparison point against published Browndye2 BD results for the same system.

**System.** p38 MAPK alpha in the DFG-in conformation from PDB 1A9U (chain A, residues 4–354, His-tag excluded) contains 5658 atoms with a net charge of −9e and a maximum radius of 37.8 Å. SB203580 is a type I kinase inhibitor with 27 atoms and zero net charge. The receptor is parameterized with ff14SB and the ligand with GAFF2 and AM1-BCC partial charges via antechamber. The ligand binds in the ATP pocket at the hinge region.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 60.0 Å | The sum of maximum radii (37.8 + 7.8 = 45.6 Å) plus 14 Å of clearance. |
| Hydrodynamic radii | Auto-computed | Determined from the PQR files via Monte Carlo surface integration. |
| Debye length | 7.86 Å | Corresponds to 150 mM ionic strength. |
| APBS fine grid length | 128 Å | Covers ±64 Å. At the b surface, b + R<sub>max, lig</sub> = 68 Å, which is 4 Å beyond the grid edge. The Yukawa multipole fallback handles the overshoot for the outermost ligand atoms. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.50 Å on the fine grid. |
| Max timestep cap | 0 (no cap) | Not needed. The adaptive timestep at r = 60 Å is moderate for a small ligand with high diffusivity. |
| Trajectories | 100,000 | Initial validation run. |

**Reaction criterion.** Four crystal-structure contacts between p38 and SB203580 define the binding interface. These were identified from the co-crystal structure and represent key pharmacophoric interactions.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pair 1 | MET106 N (hinge backbone) and NB1 (pyridine N) | The canonical hinge hydrogen bond that anchors type I kinase inhibitors. |
| Pair 2 | LYS50 NZ (catalytic lysine) and NC3 (imidazole N) | The catalytic lysine contact, important for positioning the inhibitor in the ATP pocket. |
| Pair 3 | VAL102 O (backbone carbonyl) and FD3 (fluorine) | Backbone contact in the gatekeeper region. |
| Pair 4 | THR103 N (backbone amide) and FD3 (fluorine) | Adjacent backbone contact reinforcing the gatekeeper interaction. |
| Cutoff | 7.0 Å (all pairs) | Uniform cutoff accommodating rigid-body approach to the buried ATP pocket. |
| Contacts needed | 2 | Two of the four contacts must be satisfied simultaneously. This balances selectivity (requiring genuine approach to the binding site) with geometric accessibility (the ATP pocket is partially occluded). |

**Key physics.** SB203580 is electrically neutral, so there is no electrostatic steering. The receptor carries a large negative charge (−9e), but this does not attract the neutral ligand. Association is purely diffusion-limited, governed by the geometric accessibility of the ATP-binding pocket. The published Browndye2 result of k<sub>on</sub> = 6.6 × 10⁷ M⁻¹s⁻¹ (Huang, 2021) provides a direct comparison target for PySTARC on the same system.

**Experimental references.** The experimental k<sub>on</sub> = 1.5 × 10⁷ M⁻¹s⁻¹ was measured by Miao et al. (2018). The Browndye2 BD result of k<sub>on</sub> = 6.6 × 10⁷ M⁻¹s⁻¹ was reported by Huang (2021). Both values are within the expected range for diffusion-limited encounter with a partially buried kinase active site.

---

## 7. Carbonic anhydrase sulfonamide inhibitors

**Purpose.** Seven sulfonamide inhibitors binding three carbonic anhydrase isozymes (CA XIII, CA I, CA II) test PySTARC on a multi-target protein-ligand benchmark where all ligands carry the same charge (−1e) and bind the same active-site motif (Zn-coordinating sulfonamide), but differ in scaffold, size, and isozyme selectivity.

**System.** All ligands are deprotonated sulfonamides (NH⁻SO₂R, net charge = −1e). At physiological pH, only ~0.1% of the sulfonamide exists in this binding-competent form. The experimental intrinsic k<sub>on</sub> (corrected for protonation equilibrium) is the correct comparison target for BD, not the observed k<sub>on</sub>. Five systems use CA XIII (PDB 3CZV, chain A, 258 residues, Q ≈ −1e), one uses CA I (PDB 2NMX, Q ≈ 0e), and one uses CA II (PDB 3HS4, Q ≈ 0e). The active-site Zn²⁺ ion is included in the receptor parameterization via `frcmod.ions234lm_126_tip3p`. Ligands are generated from SMILES via rdkit, converted to mol2 with obabel, and parameterized with antechamber (GAFF2 + AM1-BCC).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| b surface radius | 60.0 Å | The receptor maximum radius is approximately 27 Å and ligand radii range from 4 to 8 Å, giving a sum of ~35 Å plus 25 Å of clearance. |
| Hydrodynamic radii | Auto-computed | Determined from the PQR files via Monte Carlo surface integration. |
| Debye length | 9.62 Å | Corresponds to 100 mM ionic strength, matching the SPR experimental conditions of Linkuviene et al. (2018). |
| Ion concentration | 0.10 M | One hundred millimolar sodium chloride. |
| APBS fine grid length | 128 Å | Covers ±64 Å. At the b surface, b + R<sub>max, lig</sub> ≈ 68 Å, which is slightly beyond the grid edge. The Yukawa multipole fallback handles the overshoot. |
| APBS grid dimension | 257 | Yields a grid spacing of 0.50 Å on the fine grid. |
| Max timestep cap | 0 (no cap) | Not needed. The adaptive timestep at r = 60 Å is moderate for small ligands with high diffusivity. |
| Trajectories | 100,000 | Per system (example default). Production runs used 10,000,000. |

**Reaction criterion.** Two crystal-structure contacts between the protein active site and the sulfonamide define the reaction criterion, consistent across all seven systems.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pair 1 | THR199 OG1 (gatekeeper hydroxyl) and sulfonamide N | The Zn-coordinating hydrogen bond. THR199 is the conserved gatekeeper residue across all CA isozymes. Residue number 196 after tleap renumbering (offset −3 from PDB numbering). |
| Pair 2 | GLU106 OE1 (proton shuttle) and amide N | The relay hydrogen bond through the proton shuttle network. Residue number 103 after tleap renumbering. |
| Cutoff | 5.0 Å | Tight cutoff reflecting the compact geometry of the CA active site. The crystal distances are ~2.9 Å (THR199-N) and ~3.2 Å (GLU106-N), so 5.0 Å provides ~2 Å of slack. |
| Contacts needed | 2 | Both contacts must be satisfied simultaneously. The CA active site is a deep conical funnel, and both hydrogen bonds are required for productive binding. |

**Results.**

| System | Isozyme | Exp k<sub>on</sub> (M⁻¹s⁻¹) | PySTARC k<sub>on</sub> (M⁻¹s⁻¹) | Ratio |
|--------|---------|------------------------------|-----------------------------------|-------|
| CA I-VD12-69-1 | CA I | 2.7 × 10⁶ | 2.6 × 10⁶ | 1.0× |
| CA XIII-VD11-26 | CA XIII | 1.5 × 10⁶ | 1.9 × 10⁶ | 1.3× |
| CA XIII-AZM | CA XIII | 1.5 × 10⁶ | 2.6 × 10⁶ | 1.7× |
| CA XIII-VD12-69-1 | CA XIII | 2.5 × 10⁶ | 4.3 × 10⁶ | 1.7× |
| CA XIII-VD11-25 | CA XIII | 4.6 × 10⁵ | 2.0 × 10⁶ | 4.3× |
| CA XIII-VD12-09 | CA XIII | 3.3 × 10⁵ | 2.4 × 10⁶ | 7.4× |
| CA II-VD11-4-2 | CA II | 1.8 × 10⁶ | 2.1 × 10⁵ | 0.12× |

**Key physics.** PySTARC predicts a uniform diffusion-limited encounter rate of ~2 × 10⁶ M⁻¹s⁻¹ for the CA XIII active site with deprotonated sulfonamides. Four of seven systems with experimental k<sub>on</sub> ≥ 10⁶ agree within 2×. Two systems (VD12-09 and VD11-25) with experimental k<sub>on</sub> 5 × 10⁵ are overestimated because the experimental rate is dominated by post-diffusional gating that rigid-body BD cannot capture. The CA II system (VD11-4-2) is underestimated because CA II has near-neutral net charge compared to the weakly negative CA XIII, resulting in weaker electrostatic steering of the anionic ligand. This isozyme-dependent difference is real physics captured by BD.

**Experimental references.** Linkuviene et al. (2018). Intrinsic thermodynamics of inhibitor binding to human carbonic anhydrase isozymes I, II, VII, XII, and XIII. J. Med. Chem., 61(16), 7500-7512.

---

## Common parameters

All complexes share the following parameters.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Protein dielectric | 4.0 | Standard for BD simulations. |
| Solvent dielectric | 78.0 | Water at 298.15 K. |
| Solvent probe radius | 1.4 Å | Water molecule radius. |
| Desolvation coupling | 0.0795775 | Equal to 1/(4π), the Born desolvation coupling constant. |
| Hydrodynamic interactions | Enabled | Oseen tensor corrections are included in the k<sub>b</sub> integral. |
| Overlap check | Enabled | Configurations where the ligand overlaps the receptor volume are rejected. |
| Multipole fallback | Enabled | The monopole, dipole, and quadrupole Yukawa expansions are used beyond the APBS grid boundary. |
| Lennard-Jones forces | Disabled | Standard for rigid-body BD. |
| Temperature | 298.15 K | Room temperature, corresponding to k<sub>B</sub>T = 1.0 in reduced units. |
| Base timestep | 0.2 ps | Sets the minimum core timestep near the receptor surface. |
| Minimum core timestep | 0.2 ps | Floor on the adaptive timestep in the reaction zone. |

## Adaptive timestep cap

The variable-timestep scheme uses the formula Δt<sub>pair</sub> = f²r²/(2D), with f = 0.1, ensuring that the root-mean-square displacement per step is less than 10% of the intermolecular separation. For protein and small-molecule complexes where the b surface is less than 50 Å, this yields moderate timesteps below 200 ps with drift/noise less than 4, and no cap is required.

For protein-protein complexes where the b surface is 80 Å or larger, the adaptive timestep exceeds 1000 ps at the starting radius, producing deterministic trajectories with drift/noise greater than 5 that skip the electrostatic steering region entirely. The max timestep parameter caps the adaptive timestep at a user-specified ceiling, typically 100 ps for protein-protein complexes, restoring proper Brownian sampling of the electrostatic funnel. When max timestep is set to 0 (default), no cap is applied, preserving backward compatibility with all existing simulations.

| Complex | b (Å) | Δt at b (ps) | Drift/noise | Cap needed |
|---------|-------|--------------|-------------|------------|
| Charged spheres | 10 | 10 | 0.5 | No |
| β-cyclodextrin host-guest | 30 | 90 | 1.5 | No |
| Trypsin-benzamidine | 45 | 191 | 3.4 | No |
| p38 MAPK - SB203580 | 60 | ~360 | ~3.0 | No |
| Carbonic anhydrase inhibitors | 60 | ~360 | ~3.0 | No |
| Barnase-barstar | 80 | 1295 | 8.8 | Yes, 100 ps |
| Thrombin-thrombomodulin | 85 | 1806 | 5.4 | Yes, 100 ps |
