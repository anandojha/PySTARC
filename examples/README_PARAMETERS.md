# PySTARC Benchmark Systems: Parameter Selection Guide

## Overview

This document describes the simulation parameters for each PySTARC benchmark system and the physical reasoning behind their selection. All systems use the AMBER ff14SB force field for charge assignment (via `ambpdb -pqr`), APBS for electrostatic potential grids, and the Northrup-Allison-McCammon (NAM) framework for computing bimolecular association rate constants.

---

## 1. Two Charged Spheres (Analytical Validation)

**Purpose**: Validate the BD engine against the exact analytical Smoluchowski solution for two uniformly charged spheres with screened Coulombic interaction.

**System**: Two spherical ions, Q_rec = +1e, Q_lig = -1e, radii = 1.0 A each. No molecular structure — purely analytical test of the BD propagator, outer-propagator return probability, and k_b integral.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bd_milestone_radius` | 10.0 A | 5x contact distance (2.0 A). Small system, no molecular extent. |
| `r_hydro_rec` / `r_hydro_lig` | 1.0 / 1.0 A | Exact sphere radii. |
| `debye_length` | 7.86 A | 150 mM ionic strength. |
| `apbs_fglen` | 96 A | Covers +/-48 A, far exceeds b-surface. |
| `apbs_dime` | 257 | High resolution (0.37 A spacing). |
| `max_dt` | 0 | No cap needed. dt_pair at r=10 is small (~10 ps). |
| `n_trajectories` | 1,000,000 | Sufficient for <1% relative SE. |
| Reaction criterion | 1 pair, contact distance 2.0 A | Sum of radii = exact contact. |
| `n_needed` | 1 | Single contact. |

**Result**: P_rxn = 0.4479 vs exact 0.4501 (0.5% error).

---

## 2. Trypsin-Benzamidine (Protein-Small Molecule)

**Purpose**: Validate PySTARC against Browndye2 for a protein-small molecule system with well-characterized experimental kinetics.

**System**: Trypsin (3220 atoms, Q = +6e, R_max = 28.4 A) + benzamidine (18 atoms, Q = +1e, R_max = 3.7 A). Both positively charged — repulsive electrostatics. PDB structure from SEEKR2 benchmark (Votapka et al. 2022, JCTC).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bd_milestone_radius` | 45.0 A | rec_maxR (28.4) + lig_maxR (3.7) + 13 A clearance. Matches BD2 setup. |
| `r_hydro_rec` | 22.5 A | Matched to BD2 reference value (Lane Votapka's BD2 inputs). |
| `r_hydro_lig` | 5.0 A | Matched to BD2 reference value. |
| `debye_length` | 7.86 A | 150 mM ionic strength. |
| `apbs_fglen` | 128 A | Covers +/-64 A. At b=45 with lig_maxR=3.7, b+lig = 49 A < 64 A. Updated from 96 to provide margin. |
| `apbs_dime` | 257 | 0.50 A spacing on fine grid. |
| `max_dt` | 0 | No cap needed. dt_pair at r=45 = 191 ps, drift/noise ~ 3.4. Acceptable for small ligand. |
| `n_trajectories` | 10,000,000 | Low P_rxn (~0.001) requires many trajectories. |
| `CONTACT_CUTOFF` | 6.0 A | Maximum distance in crystal structure to identify binding contacts. |
| `BUFFER` | 3.0 A | Added to crystal distance for reaction cutoff (accounts for rigid-body approach). |
| `N_PAIRS` | 10 | Top 10 closest polar contacts from crystal structure. |
| `N_NEEDED` | 4 | 4 of 10 pairs must be satisfied simultaneously. Moderately strict. |
| `CONTACT_MODE` | polar | Only N/O/S donor-acceptor pairs (hydrogen-bonding contacts). |

**Reaction criterion construction**: The `setup.py` script identifies the closest heavy-atom contacts between receptor and ligand in the crystal structure, filters for polar (N/O/S) atoms, keeps the top 10 contacts (one per receptor residue), and sets cutoff = crystal_distance + 3.0 A (rounded to nearest 0.5 A). Resulting cutoffs range from 6.0 to 8.5 A.

**Result**: k_b = 25.632 vs BD2's 25.620 (0.05% match). k_on = 1.64 x 10^7 vs experimental 2.9 x 10^7 (within 2x).

---

## 3. Beta-Cyclodextrin Host-Guest (7 Systems)

**Purpose**: Test PySTARC on a host-guest benchmark with multiple ligands binding the same receptor, where experimental k_on values span an order of magnitude.

**System**: Beta-cyclodextrin (147 atoms, Q = 0, R_max = 8.6 A) + 7 small-molecule guests (12-30 atoms, Q = 0, R_max = 3-5 A). All molecules are electrically neutral — no electrostatic steering. Structures from SEEKR2 benchmark (Lane Votapka).

**Guests**: 1-butanol, 1-propanol, 1-naphthylethanol, 2-naphthylethanol, aspirin, methyl butyrate, tert-butanol.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bd_milestone_radius` | 30.0 A | rec_maxR (8.6) + lig_maxR (~3) + 18 A clearance. |
| `r_hydro_rec` / `r_hydro_lig` | 0 / 0 (auto) | Auto-computed from PQR via MC surface integration. |
| `debye_length` | 7.86 A | 150 mM ionic strength. |
| `apbs_fglen` | 96 A | Covers +/-48 A. b+lig = 33 A, well within grid. |
| `apbs_dime` | 257 | 0.37 A spacing. |
| `max_dt` | 0 | No cap needed. dt_pair at r=30 = 90 ps, drift/noise = 1.5. |
| `n_trajectories` | 2,000,000 | Per guest system. |
| `CONTACT_CUTOFF` | 5.0 A | Tight cutoff for small host-guest complex. |
| `BUFFER` | 2.0 A | Smaller buffer than protein systems (smaller molecules, tighter contacts). |
| `N_PAIRS` | 8 | Up to 8 contacts. |
| `N_NEEDED` | 4 | 4 of 8 pairs must be satisfied. |
| `CONTACT_MODE` | all | All heavy-atom contacts (host-guest has few polar atoms). |

**Reaction criterion construction**: Same automated procedure as trypsin. The O5 glycosidic oxygens of BCD (atoms 15, 57, 99) appear as receptor contact atoms across all 7 systems, providing a consistent anchor. Cutoffs are 5.0-6.5 A.

**Key physics**: All guests are neutral, so BD computes only the diffusion-limited encounter rate with no electrostatic enhancement. Experimental k_on variation (2.8-7.2 x 10^8) arises from conformational gating and desolvation barriers that BD cannot capture. PySTARC correctly predicts encounter rates of ~10^8 for all guests.

**Result**: Spearman rho = -0.54 (no rank-order correlation), confirming that k_on discrimination requires MD-level detail for neutral ligands in a small cavity.

---

## 4. Thrombin-Thrombomodulin (Protein-Protein, from BD2 Tutorial)

**Purpose**: Reproduce the Browndye2 tutorial benchmark for a strongly electrostatically steered protein-protein association.

**System**: Thrombin (4727 atoms, Q = +3e, R_max = 34.7 A) + thrombomodulin EGF domains 4-5-6 (1650 atoms, Q = -15e, R_max = 40.6 A). Strong electrostatic complementarity drives fast association. PQR files from Gary Huber's BD2 tutorial.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bd_milestone_radius` | 85.0 A | rec_maxR (34.7) + lig_maxR (40.6) + 10 A clearance. Original BD2 tutorial used 175 A (unnecessarily large). |
| `r_hydro_rec` | 25.58 A | From BD2 tutorial (pre-computed). |
| `r_hydro_lig` | 21.88 A | From BD2 tutorial (pre-computed). |
| `debye_length` | 7.86 A | 150 mM ionic strength. |
| `apbs_fglen` | 192 A | Covers +/-96 A. At b=85, lig atoms span 44-126 A from origin. |
| `apbs_dime` | 257 | 0.75 A spacing. |
| `apbs_cglen` | 0 (auto) | BD2 tutorial used 1000 A (unnecessary). Auto-compute is sufficient. |
| `max_dt` | 100 ps | Critical for protein-protein. Without cap, dt_pair at r=85 = 1806 ps, drift/noise = 5.4. Cap brings drift/noise to ~2. |
| `n_trajectories` | 2,000,000 | Tight cutoff (5 A) with n_needed=3 gives low P_rxn. |
| Reaction criterion | 21 H-bond pairs, 5.0 A cutoff | From BD2's `make_rxn_file` with correct 5 A distance (not tutorial's loose 15 A). |
| `n_needed` | 3 | 3 of 21 hydrogen-bonding contacts must be satisfied. From BD2 paper (Huber & Kim, 2010). |

**Parameter corrections from BD2 tutorial**: Gary Huber explicitly states in the tutorial: "the actual reaction rate is much smaller than what is calculated in this tutorial. I have increased the reaction distance from 5 Angstroms to 15 Angstroms so you can actually obtain a 'rate' from only 1000 trajectories." We restored the correct 5 A cutoff and increased trajectory count accordingly.

**Experimental target**: k_on = 6.7 x 10^6 M^-1 s^-1 at physiological ionic strength (Baerga-Ortiz et al. 2000). Debye-Huckel analysis shows slope = -6 and intercept at zero ionic strength of 10^9 M^-1 s^-1, confirming the interaction is nearly completely electrostatically steered.

---

## 5. Barnase-Barstar (Protein-Protein Benchmark)

**Purpose**: Validate PySTARC on the classic electrostatically steered protein-protein association system with extensive BD literature.

**System**: Barnase (1700 atoms, Q = +2e, R_max = 24.6 A, R_hydro = 18.3 A) + barstar (1403 atoms, Q = -5e, R_max = 21.1 A, R_hydro = 17.0 A). PDB 1BRS, chains A (barnase) + D (barstar), parameterized with ff14SB.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bd_milestone_radius` | 80.0 A | rec_maxR (24.6) + lig_maxR (21.1) + 34 A clearance. Large enough for full electrostatic grid coverage. |
| `r_hydro_rec` / `r_hydro_lig` | 0 / 0 (auto) | Auto-computed: rec = 18.3 A, lig = 17.0 A. |
| `debye_length` | 13.6 A | 50 mM ionic strength (experimental condition of Schreiber & Fersht). |
| `ion_concentration` | 0.05 M | 50 mM NaCl. |
| `apbs_fglen` | 192 A | Covers +/-96 A. At b=80, b+lig_maxR = 101 A, slightly beyond grid edge. Yukawa fallback handles the 5 A overshoot for outermost atoms. |
| `apbs_dime` | 257 | 0.75 A spacing on fine grid. |
| `max_dt` | 100 ps | Critical. Without cap: dt_pair at r=80 = 1295 ps, drift = 58 A, noise = 6.6 A, drift/noise = 8.8. Trajectories fly ballistically past the electrostatic funnel. With max_dt=100: drift = 4.5 A, noise = 1.8 A, drift/noise = 2.4. Electrostatic steering functions properly. This single parameter change improved k_on from 9.9 x 10^7 to 4.85 x 10^8. |
| `n_trajectories` | 1,000,000 | P_rxn ~ 0.04 gives 40,000 reactions, relative SE = 0.5%. |

**Reaction criterion**: Based on Gabdoulline & Wade (1997), who showed that reproducing experimental rates requires satisfaction of intermolecular residue contacts, not simple RMS distance.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pair 1 | ARG81 NH2 (rec atom 1228) <-> ASP147 OD1 (lig atom 633) | R83-D35 in paper numbering. Key salt bridge at binding interface. |
| Pair 2 | ARG57 NH1 (rec atom 848) <-> GLU182 OE1 (lig atom 1212) | R59-E76 in paper numbering. Second interfacial salt bridge. |
| Cutoff | 10.0 A | Loose enough for rigid-body BD (no side-chain flexibility). |
| `n_needed` | 1 | Either contact is sufficient for encounter. Using n_needed=2 with 10 A cutoff gave k_on = 8.5 x 10^6 (too strict — both contacts rarely satisfied simultaneously for rigid tumbling proteins). |

**Progression of parameter optimization**:

| Run | b (A) | Cutoff (A) | n_needed | max_dt (ps) | k_on (M^-1 s^-1) | vs Experiment |
|-----|-------|------------|----------|-------------|-------------------|---------------|
| 1 | 80 | 6.0 | 2 | none | 3.2 x 10^6 | 100x low |
| 2 | 80 | 10.0 | 2 | none | 8.5 x 10^6 | 35x low |
| 3 | 55 | 10.0 | 2 | none | 3.5 x 10^6 | 85x low |
| 4 | 80 | 10.0 | 1 | none | 9.9 x 10^7 | 3x low |
| 5 | 80 | 10.0 | 1 | 100 | **4.85 x 10^8** | **within 1.2x** |

**Key lessons**: (1) The `max_dt` cap was essential — without it, the adaptive timestep at large separations produced ballistic trajectories that skipped the electrostatic funnel. (2) `n_needed=1` is appropriate for rigid-body BD where both proteins tumble freely and simultaneous satisfaction of two specific contacts is geometrically rare. (3) The 10 A cutoff accounts for the lack of side-chain flexibility in rigid-body BD.

**Experimental references**: k_on = 6.0 x 10^8 (Schreiber & Fersht, 1993), 2.86 x 10^8 (Frembgen-Kesner & Elcock, 2010), both at 50 mM ionic strength. Basal rate without electrostatics: 5.8 x 10^6 (Northrup & Erickson, 1992). PySTARC captures the ~80x electrostatic enhancement.

---

## Common Parameters Across All Systems

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `pdie` | 4.0 | Protein interior dielectric constant (standard for BD). |
| `sdie` | 78.0 | Solvent (water) dielectric constant at 298.15 K. |
| `srad` | 1.4 A | Solvent probe radius (water). |
| `desolvation_alpha` | 0.0795775 | Born desolvation coupling constant (= 1/(4*pi)). |
| `hydrodynamic_interactions` | true | Include Oseen tensor HI corrections in k_b integral. |
| `overlap_check` | true | Reject configurations where ligand overlaps receptor volume. |
| `multipole_fallback` | true | Use monopole+dipole+quadrupole Yukawa expansion beyond APBS grid. |
| `lj_forces` | false | No Lennard-Jones forces (standard for rigid-body BD). |
| `temperature` | 298.15 K | Room temperature (kT = 1.0 in reduced units). |
| `dt` | 0.2 ps | Base timestep (sets minimum_core_dt near receptor surface). |
| `minimum_core_dt` | 0.2 ps | Floor on adaptive timestep near reaction zone. |

## The `max_dt` Parameter

The adaptive timestep formula `dt_pair = 0.005 * r^2 / D` ensures the mean displacement per step is <10% of the inter-molecular separation. However, for protein-protein systems with large b-surfaces (b > 60 A), this produces timesteps of thousands of picoseconds at the starting radius, causing deterministic (drift >> noise) trajectories that skip the electrostatic steering region.

The `max_dt` parameter caps the adaptive timestep. Setting `max_dt = 100 ps` ensures drift/noise remains ~2, allowing proper Brownian sampling of the electrostatic funnel. For protein-small molecule systems (b < 50 A), the adaptive dt is already moderate and no cap is needed (`max_dt = 0`).

| System type | b (A) | dt_pair at b (ps) | drift/noise | max_dt needed? |
|-------------|-------|-------------------|-------------|----------------|
| Charged spheres | 10 | ~10 | 0.5 | No |
| BCD host-guest | 30 | 90 | 1.5 | No |
| Trypsin-benzamidine | 45 | 191 | 3.4 | No (small ligand) |
| Thrombin-TM | 85 | 1806 | 5.4 | Yes (100 ps) |
| Barnase-barstar | 80 | 1295 | 8.8 | Yes (100 ps) |
