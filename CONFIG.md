# PySTARC configuration

```
flow      config.xml, python setup.py, receptor.pqr + ligand.pqr + rxns.xml + input.xml, python run_pystarc.py input.xml, bd_sims/results.json
scope     config.xml keys of the rigid route, input.xml keys outside config.xml, chain route keys
```

## New complex from a PDB of the bound complex

| step | command | writes |
|---|---|---|
| install, once | `bash install_PySTARC.sh` in the PySTARC checkout on a Linux node with an NVIDIA GPU and a driver for CUDA 12 or newer (exit before any change otherwise), then `conda activate PySTARC` | conda environment PySTARC (AmberTools 24.8, APBS, RDKit, OpenBabel, CuPy), editable install of the checkout with pystarc/templates, earlier PySTARC environment removed first, new environment removed after a failed CuPy kernel test |
| folder | `mkdir examples/<name>` then `cd examples/<name>` | empty folder |
| drivers | `cp <example folder>/{setup.py,run.sh,submit_SLURM_single_GPU.sh,submit_SLURM_multi_GPUs.sh} .` with `<example folder>` any rigid example folder holding config.xml (setup.py identical in every rigid example) | setup.py, run.sh, SLURM scripts |
| structure | `cp /path/to/bound_complex.pdb complex.pdb` | complex.pdb (PDB requirements block) |
| config | `python -m pystarc.pipeline.build_inputs --template > config.xml` | config.xml with every section and blank tags (structure with ligand_source only, criterion with mode, pair_cutoff and n_needed only) |
| edit | config.xml per the section tables | config.xml |
| build | `python setup.py` | receptor.pqr, ligand.pqr, rxns.xml, input.xml, build_record.json |
| run, 1 GPU | `bash run.sh` | bd_sims (deleted first, setup.py rerun) |
| run, cluster, 1 GPU | `sbatch submit_SLURM_single_GPU.sh` | bd_sims |
| run, cluster, 4 GPUs | `sbatch submit_SLURM_multi_GPUs.sh` | bd_sims/bd_1 to bd_sims/bd_4, pooled files in bd_sims |
| result | `bd_sims/results.json` | k_on, k_on_low, k_on_high (per molar per second), run summary at the end of `bd_sims/pystarc_<timestamp>.log` |

```
build folder         work_dir, default the folder of config.xml, python setup.py path/to/config.xml for another config.xml
setup.py printout    atom count and net charge of each PQR (atom count and source file only for pqr), composition verified, pair count, geometry line with OK or WARNING
geometry WARNING     b + largest ligand atom distance from the ligand centroid ≥ apbs_fglen / 2, no stop
run folder           folder of input.xml (run.sh and the SLURM scripts change into the folder)
outside checkout     export PYSTARC_ROOT=<checkout> for run.sh and the SLURM scripts
input.xml edits      lost on run.sh and on the SLURM scripts (setup.py rerun), python <checkout>/run_pystarc.py input.xml instead
SLURM deletions      bd_sims, plus input.xml, rxns.xml, receptor.pqr, ligand.pqr, receptor.pdb, ligand.pdb, protein.prmtop, protein.rst7, ligand.prmtop, ligand.rst7, *.cache and _full.rst7 before setup.py (folders with setup.py), user files of the same names included
build deletions      build intermediates after a successful build, ligand scratch at the end of the ligand parameterization, prmtop scratch at the end of the prmtop step (files block), user files of the same names included
build abort          abort inside the source step = every file of the step kept, later abort (composition check of ligand.pqr, criterion, missing run key) = receptor.pqr and ligand.pqr written, rxns.xml too for a missing run key, build intermediates kept, ligand and prmtop scratch already deleted
```

PDB requirements of `complex_pdb`

```
protein               ATOM records, residue names of leaprc.protein.ff14SB, one model
protein hydrogens     discarded, rebuilt by tleap from the residue names
protonation           residue names HID, HIE, HIP, ASH, GLH, LYN, CYX, plain HIS read as HIE
termini               built by tleap, ACE and NME as HETATM records dropped
protein altlocs       not filtered
ligand                one HETATM residue, one copy, first altloc kept
dropped               HOH, WAT, NA, CL, MG, ZN, ACE, NME, Na+, Cl-, K+, CAL and every other het group (cofactors, metals)
ligand chemistry 1    residue name = CCD code (or ligand_ccd), CCD atom names, every heavy atom of the component except leaving atoms, component CIF from the build folder or files.rcsb.org (deleted after each successful build)
ligand chemistry 2    residue name UNK, UNL, LIG, DRG, INH or MOL, ligand_ccd blank, every ligand hydrogen in the PDB, ligand_net_charge
tools                 tleap, antechamber, sqm, parmchk2, cpptraj, ambpdb and obabel on PATH, RDKit in Python
```

Example structure, names and criterion blocks (`complex_pdb`, ligand residue LIG with every hydrogen in complex.pdb)

```xml
<structure>
  <ligand_source>complex_pdb</ligand_source>
  <complex_pdb>complex.pdb</complex_pdb>
  <ligand_net_charge>0</ligand_net_charge>
</structure>
<names>
  <ligand_formula>C8 H10 N2 O</ligand_formula>
  <receptor_resname>REC</receptor_resname>
  <ligand_resname>LIG</ligand_resname>
</names>
<criterion>
  <mode>site</mode>
  <margin>3.0</margin>
</criterion>
```

## Inputs per ligand_source

| ligand_source | provide | build | charges and radii | network |
|---|---|---|---|---|
| `pqr` | receptor_pqr_file, ligand_pqr_file (bound pose for contacts and site) | copy to receptor.pqr and ligand.pqr minus GHO and PKT atoms | as given | none |
| `prmtop` | prmtop with receptor and ligand (solvent and ions allowed), complex_pdb with every topology atom in topology order | cpptraj, ambpdb with `-pqr`, split by residue name | topology charges and topology radius set, no parameterization | none |
| `complex_pdb` | complex_pdb per the PDB requirements, optional ligand_ccd, ligand_net_charge, ligand_protonate, ligand_deprotonate | tleap ff14SB protein, RDKit bond orders and hydrogens, obabel, antechamber AM1 BCC charges with GAFF2 types, parmchk2, tleap gaff2, cpptraj, ambpdb | ff14SB protein charges, AM1 BCC ligand charges, tleap default PBRadii set (none set by the build) | component CIF from files.rcsb.org with a CCD code |
| `rcsb` | pdb_id, ligand_resid, optional chain, rec_resid_min, ligand_net_charge, ligand_ccd, ligand_protonate, ligand_deprotonate | ATOM records of the chain from rec_resid_min as deposited, ligand HETATM residue with the ligand CONECT records, every other HETATM record dropped, then the `complex_pdb` steps with a component definition | as `complex_pdb` | entry without `<pdb_id>.pdb` in the build folder, component CIF on every build |
| `rcsb_smiles` | pdb_id, ligand_resid, smiles, optional chain, keep_hetatm, ligand_net_charge | ATOM records of the chain minus OXT, keep_hetatm groups, ligand from the SMILES (one ETKDG conformer, seed of the simulation section, MMFF), tleap with ion parameters, obabel, antechamber AM1 BCC charges with GAFF2 types, parmchk2, tleap gaff2, cpptraj, ambpdb | as `complex_pdb` | entry without `<pdb_id>.pdb` in the build folder |

```
PDB only              complex_pdb, or rcsb with an entry code, or rcsb_smiles with an entry code and a SMILES
prmtop with a PDB     reuse of an existing topology (receptor outside ff14SB, other force field, protonation fixed elsewhere, solvated topology), PDB for coordinates only
radii                 prmtop (radius set of the topology), complex_pdb, rcsb and rcsb_smiles (tleap default PBRadii set), pqr (radii of the files)
protonation           no pKa prediction, ligand from the component definition ± ligand_protonate and ligand_deprotonate, from the PDB hydrogens (generic residue name) or from the SMILES, protein from the residue names
```

## config.xml rules

```
section order         structure, names, criterion, geometry, electrostatics, physics, simulation, compute, outputs (order of --template)
criterion keys        inside <criterion> only
other keys            any section outside <criterion>
key placement         every key inside a section
blank tag             absent
unknown key           inside a section, build abort naming the unused keys
required              every key of geometry to outputs except soft_repulsion, minimum_core_reaction_dt, full_paths, energetics
run keys              copied as text into input.xml, types checked at run start, receptor dime rule at the APBS step (dime rule block)
range checks          run start abort for n_trajectories < 1, debye_length ≤ 0 or more than 1 % off λ, sdie ≤ 0, ion_concentration < 0, r_escape outside the r_escape row, desolvation_alpha < 0, gpu_force_batch < 0, convergence_tol or confidence_interval outside (0, 1), save_interval < 0, apbs_ligand_dime outside the dime rule, soft_repulsion true with lj_forces true
flags at run start    true, false, yes, no, 1, 0 in any case
input.xml only keys   refused in config.xml, set by an edit of input.xml after setup.py
```

## structure

| key | accepts | default | does |
|---|---|---|---|
| `ligand_source` | `pqr`, `prmtop`, `complex_pdb`, `rcsb`, `rcsb_smiles` (case sensitive) | none, required | origin of receptor and ligand coordinates, charges and radii (inputs per ligand_source), build abort for a blank or other entry and for any structure key outside the chosen source |
| `receptor_pqr_file` | path of an existing receptor PQR, relative to the build folder (work_dir, default the config.xml folder) or absolute, not the receptor.pqr of the build folder | none, required with `pqr`, refused otherwise | copy to receptor.pqr minus GHO and PKT atoms, charges and radii unchanged, fixed PDB columns for atom name, residue name and residue number, x y z charge radius as the last five numbers |
| `ligand_pqr_file` | path of an existing ligand PQR, relative to the build folder (work_dir, default the config.xml folder) or absolute, not the ligand.pqr of the build folder | none, required with `pqr`, refused otherwise | copy to ligand.pqr minus GHO and PKT atoms, bound pose for contacts and site |
| `prmtop` | path of an AMBER topology (.prmtop or .parm7) with receptor and ligand, solvent and ions allowed, relative to the build folder (work_dir, default the config.xml folder) or absolute | none, required with `prmtop`, refused otherwise | charges and radii from the topology, residues named ligand_resname to ligand.pqr, residues HOH, WAT, NA, CL, MG, ZN, ACE, NME, Na+, Cl-, K+, CAL dropped, every other residue to receptor.pqr (other ions and cofactors included) |
| `complex_pdb` | path of a PDB file, relative to the build folder (work_dir, default the config.xml folder) or absolute | none, required with `prmtop` and `complex_pdb`, refused otherwise | `prmtop` (coordinates of every topology atom in topology order, solvent included, read by cpptraj in order), `complex_pdb` (bound complex per the PDB requirements) |
| `pdb_id` | RCSB entry code, file name as written | none, required with `rcsb` and `rcsb_smiles`, refused otherwise | `<pdb_id>.pdb` of the build folder, else a download from files.rcsb.org in PDB format, kept after the build |
| `ligand_resid` | residue name of the ligand (3 characters at most), not a residue number | none, required with `rcsb` and `rcsb_smiles`, refused otherwise | `rcsb` (ligand HETATM residue, copy in chain else first copy, first altloc, CONECT records kept, component code with ligand_ccd blank, residue name in ligand.pqr), `rcsb_smiles` (residue name in ligand.pqr only, deposited copy left out) |
| `smiles` | SMILES read by RDKit, protonation and charge as written | none, required with `rcsb_smiles`, refused otherwise | ligand chemistry, hydrogens added, one ETKDG conformer with randomSeed = seed, MMFF optimization, AM1 BCC charges on the embedded conformer, ligand outside the pocket (RDKit frame), antechamber atom names in ligand.pqr, mode contacts with picked pairs or `auto` ligand atoms not meaningful (no bound pose), named pairs, centroid or site instead |
| `chain` | one chain character, blank = all chains | blank | `rcsb` (ATOM records of the chain, ligand copy in the chain), `rcsb_smiles` (ATOM and keep_hetatm records of the chain, pocket_ligand search), `complex_pdb` (accepted, no effect), refused with `prmtop` and `pqr`, match on PDB column 22 |
| `rec_resid_min` | integer residue number as deposited | none, no trim | `rcsb` only, ATOM records below the number dropped, offset of the receptor_numbering pdb map, build abort for a non integer |
| `ligand_net_charge` | integer, `-1`, `0`, `1` and so on, `1.0` refused | blank = sum of the component formal charges after ligand_protonate and ligand_deprotonate (`complex_pdb`, `rcsb`), SMILES formal charge (`rcsb_smiles`) | ligand net charge of antechamber `-nc`, build abort when different from the component charge, required without a component definition (bond orders from the geometry at the given charge), refused with `prmtop` and `pqr` |
| `ligand_ccd` | Chemical Component Dictionary code | blank = ligand_resid (`rcsb`), PDB residue name of the ligand unless UNK, UNL, LIG, DRG, INH or MOL (`complex_pdb`) | `complex_pdb` and `rcsb` only, `<code>.cif` from the build folder or files.rcsb.org (deleted after a successful build), bond orders SING, DOUB, TRIP, formal charges, elements and leaving atoms of the component, structure atom names = component atom names, ligand choice among several het groups (`complex_pdb`) |
| `ligand_protonate` | `<atom name="N1"/>` children, `<item>N1</item>` children or one name as text | none | `complex_pdb` and `rcsb` with a component definition only, one extra proton on each named N, O or S atom (formal charge +1), hydrogen placed by RDKit |
| `ligand_deprotonate` | same three forms | none | `complex_pdb` and `rcsb` with a component definition only, one proton less on each named N, O or S atom (formal charge −1), build abort for a name in both keys |
| `keep_hetatm` | `<item>ZN</item>` children, `<atom name="ZN"/>` children or one name as text | none, every HETATM record dropped | `rcsb_smiles` only, het groups of the chain kept with the receptor, tleap parameters for ions only (frcmod.ions234lm_126_tip3p) |
| `work_dir` | path, relative to the current folder | folder of config.xml | build folder of every structure and setup file, any section, absent from `--template`, no effect on the work_dir of input.xml (bd_sims) |

## names

| key | accepts | default | does |
|---|---|---|---|
| `ligand_formula` | element symbols in standard case (`Cl`, `Br`) with counts, spaces optional (`C8 H10 N2 O` or `C8H10N2O`), every hydrogen | none, required | composition check of ligand.pqr after the build (abort with the per element difference, receptor.pqr and ligand.pqr already written), second check before antechamber (`complex_pdb`, `rcsb`), protonation state of the built ligand |
| `receptor_resname` | any text, not blank | none, required | label in input.xml, no effect on a run |
| `ligand_resname` | residue name, 3 characters at most | none, required | `prmtop` (residue name of the ligand in the topology, case sensitive), `complex_pdb` (ligand choice among several het groups, residue name in ligand.pqr), `rcsb` and `rcsb_smiles` (replaced by ligand_resid), `pqr` (no effect), label in input.xml |

## criterion

| key | accepts | default | does |
|---|---|---|---|
| `mode` | `contacts`, `centroid`, `site` | none, required | pair rule (reaction criterion modes), build abort for any other entry (former `auto`, `explicit`, `center` included) and for a criterion key outside the mode |
| `receptor_numbering` | `internal`, `pdb` | `internal` | numbering of the resid of receptor targets, `internal` (receptor.pqr numbers, from 1 after tleap and ambpdb, as written for `pqr`), `pdb` (deposited numbers through the entry map, `rcsb` only, chain set for entries with several chains), named pairs and site anchors only |
| `search_cutoff` | number > 0 (angstroms) | none, required for picked pairs | heavy atom pairs of the bound pose with distance < search_cutoff, `contacts` only |
| `n_pairs` | `1` with n_needed 1, even number with n_needed = n_pairs / 2 | none, required for picked pairs | pairs kept, closest first, first pair per receptor residue number, build abort for fewer pairs than n_pairs |
| `atom_filter` | `all`, `polar`, `nonpolar`, `any_polar`, `receptor_polar` | none, required for picked pairs and for an auto ligand side, refused otherwise | element rule, `all` (any heavy atoms), `polar` (N, O or S on both sides), `nonpolar` (C on both sides), `any_polar` (N, O or S on either side), `receptor_polar` (N, O or S on the receptor side), hydrogens never |
| `excluded` | `<item>NAME</item>` children, `<atom name="NAME"/>` children or one name as text | empty | atom names skipped on both sides of the picked pair search |
| `receptor` | `<target resid="N" atom="NAME" resname="RES"/>` children, one per pair, one in site | none, required for named pairs, optional in site | receptor atom of each named pair or the receptor anchor of site, exact match of resid, atom and resname in receptor.pqr |
| `ligand` | `<atom name="NAME"/>` or `<item>NAME</item>` children one to one with the targets, one name as text for one target, `auto` as text, `<atom name="auto"/>` or `<item>auto</item>` per pair, one named atom in site | none, required with receptor | ligand atom of each pair or the ligand anchor of site, `auto` = closest heavy ligand atom under atom_filter in the bound pose, name unique in ligand.pqr |
| `pair_cutoff` | number > 0 (angstroms) | none, required in contacts and centroid, site = pair_cutoff or margin | reaction distance of every pair, pair satisfied at distance < cutoff |
| `margin` | number ≥ 0 (angstroms) | none, site only | cutoff = closest accessible approach + margin, rounded to 0.25 (angstroms), stored in build_record.json |
| `pocket_ligand` | residue name of a deposited ligand of the entry | none, required for site without anchors with `rcsb_smiles`, refused otherwise | PKT site atom at the centroid of the pocket_ligand records of the chain (altloc blank or A) |
| `pocket_ligand_resid` | integer residue number | none | copy of pocket_ligand for a chain with several copies, build abort without |
| `n_needed` | `1` with one pair, half the pair count with an even pair count | none, required in contacts, refused in centroid and site | satisfied pairs required at one time for a reaction, `1` written for centroid and site, `-1` refused |
| `state_machine_reactions` (input.xml only) | `true`, `false` | `false` | `true` = every rxns.xml reaction with state_before and state_after labels (one pair per reaction), `false` = one reaction as one criterion (rxns.xml of setup.py), refused in config.xml |

## geometry

| key | accepts | default | does |
|---|---|---|---|
| `bd_milestone_radius` | number > 0 (angstroms) | none, required (input.xml default 30.0) | radius b of the start sphere around the receptor centroid, uniform start positions and orientations on the sphere, escape sphere at 2 b |
| `r_hydro_rec` | number (angstroms), `0` | none, required (input.xml default 0) | receptor hydrodynamic radius, `0` = Monte Carlo radius (hydrodynamic radii block), number > 0 = fixed radius |
| `r_hydro_lig` | number (angstroms), `0` | none, required (input.xml default 0) | ligand hydrodynamic radius, rule of r_hydro_rec |
| `r_escape` (input.xml only) | `0` or number > 1.1 b (angstroms) | `0` = 2 b | escape sphere radius of rigid runs, refused in config.xml |

## electrostatics

| key | accepts | default | does |
|---|---|---|---|
| `debye_length` | number > 0 (angstroms), `inf` for no salt | none, required (input.xml without the tag = λ from ion_concentration, sdie and temperature in rigid runs, 7.858 (angstroms) in chain runs) | Debye length λ of the far field, the k_b integral and the desolvation grids (κ = 1/λ), run start abort outside 1 % of λ = 10¹⁰ √(ε₀ sdie k_B T / (2000 N_A e² c)), λ = 7.8295 (angstroms) at 0.15 (molar) and 9.5891 (angstroms) at 0.10 (molar) for 298.15 (kelvin) and sdie 78, no effect on APBS |
| `ion_concentration` | number ≥ 0 (molar) | none, required (input.xml default 0.150) | monovalent salt of every APBS electrostatic run (ions +1 and −1), `0` = no ions and λ = inf |
| `ion_radius_pos` | number (angstroms) | none, required (input.xml default 0.95) | cation radius of the APBS ion line |
| `ion_radius_neg` | number (angstroms) | none, required (input.xml default 1.81) | anion radius of the APBS ion line |
| `pdie` | number | none, required (input.xml default 4.0) | solute dielectric of APBS and of the desolvation grids, pdie = sdie for no desolvation |
| `sdie` | number > 0 | none, required (input.xml default 78.0) | solvent dielectric of APBS, the desolvation grids, λ, the far field and the k_b integral |
| `srad` | number ≥ 0 (angstroms) | none, required (input.xml default 1.5) | APBS probe radius (srfm smol), derived grid edges, inflation of every atom radius in the hydrodynamic radius (`0` = none) |
| `apbs_fglen` | number (angstroms), `0` | none, required (input.xml default 0) | edge of the fine receptor grid receptor1.dx, `0` = derived edge (electrostatic grids) |
| `apbs_dime` | integer d ≥ 5 with d − 1 divisible by 4 | none, required (input.xml default 129) | points per axis of both receptor electrostatic levels and both receptor desolvation grids (receptor0_born.dx widened above apbs_dime when needed), rule check at the APBS step, no rounding |
| `apbs_cglen` | number (angstroms) above the fine edge, `0` | none, required (input.xml default 0) | edge of the coarse receptor grid receptor0.dx, `0` = 2 × fine edge, abort at the APBS step for an edge ≤ fine edge |
| `apbs_coarse_dime` | `0` or an integer under the apbs_dime rule | none, required (input.xml default 0) | points per axis of receptor0.dx, `0` = apbs_dime |
| `apbs_fine_dime` | `0` or an integer under the apbs_dime rule | none, required (input.xml default 0) | points per axis of receptor1.dx, `0` = apbs_dime |
| `apbs_ligand_dime` (input.xml only) | integer under the apbs_dime rule | 129 | points per axis of every ligand grid, ligand0_born.dx widened above apbs_ligand_dime when needed, rule check at run start, refused in config.xml |

## physics

| key | accepts | default | does |
|---|---|---|---|
| `soft_repulsion` | `true`, `false` | `false` (optional) | soft wall V = kT (a/r)¹² with a = 0.9345 (r_i + r_j) between receptor and ligand atoms at r < 2a (PQR radii, hydrogens included), in place of overlap_check, wall stiffness limit on the time step, GPU only (run abort with gpu false), refused with lj_forces true |
| `overlap_check` | `true`, `false` | none, required (input.xml default true with soft_repulsion named, run start abort with neither tag named) | GPU rejection of a move with a ligand atom within 0.5 (r_i + r_j) of a receptor atom (clock still advanced), surface limit on the time step, no effect with soft_repulsion true, no excluded volume with both false, CPU model hard sphere test at r_i + r_j in every case |
| `lj_forces` | `true`, `false` | none, required (input.xml default false) | GPU WCA repulsion between atoms of radius > 0, σ = r_i + r_j, ε = 0.1 (kilocalories per mole), range r < 2^(1/6) σ, refused with soft_repulsion true, run abort without overlap_check true, no effect on the CPU model |
| `hydrodynamic_interactions` | `true`, `false` | none, required (input.xml default true) | `true` = Rotne Prager Yamakawa relative diffusion D(r) with the drift dD/dr Δt in every step and in the k_b integral, `false` = constant D_rec + D_lig, independent of r_hydro_rec and r_hydro_lig |
| `desolvation_alpha` | number ≥ 0 | none, required (input.xml default 1.0) | multiplier of the Born desolvation energy α q² φ_born of every charge, `1.0` = base model, `0` = desolvation off |
| `enable_born2_torque` | `true`, `false` | none, required (input.xml default true) | GPU desolvation of the receptor charges in the ligand Born field, reaction force and torque on the ligand, `false` = neither, CPU model on in every case |
| `multipole_fallback` | `true`, `false` | none, required (input.xml default true) | GPU receptor field outside every electrostatic grid as screened monopole, dipole and quadrupole, multipole term of the k_b integral for a charged ligand, `false` = screened monopole only (no field for a neutral receptor), CPU model without field outside the grids |
| `brownian_bridge` (input.xml only) | `true`, `false` | `true` | reaction crossing test between steps, probability exp(−x₀ x₁ / (D Δt)) per pair outside the pair cutoff at both step ends, `false` = test at step ends only, refused in config.xml |

## simulation

| key | accepts | default | does |
|---|---|---|---|
| `n_trajectories` | integer ≥ 1 | none, required (input.xml default 10000) | trajectories, all on the GPU at once, relative standard error of P_rxn √((1 − P)/(N P)), one share per bd_i folder in multi GPU runs |
| `max_steps` | integer | none, required (input.xml default 1000000) | step limit per trajectory, trajectory at the limit = not reacted, bounds k_on_bound_low and k_on_bound_high in results.json, pilot check `python -m pystarc.analysis.step_limit bd_sims/trajectories.csv` |
| `dt` | number (picoseconds) | none, required (input.xml default 0.2) | outer step of chain runs, no effect on rigid runs |
| `minimum_core_dt` | number ≥ 0 (picoseconds), `0` | none, required (input.xml default 0) | GPU floor of the adaptive step, applied after every other step rule and after max_dt, `0` = no floor |
| `minimum_core_reaction_dt` | number ≥ 0 (picoseconds), `0` | `0` (optional) | GPU floor of the reaction surface refinement in state_machine_reactions runs only (none in setup.py input.xml), CPU model step floor, `0` = 0.05 (picoseconds), and floor / 4 within 1.5 × the smallest reaction distance |
| `max_dt` | number ≥ 0 (picoseconds), `0` | none, required (input.xml default 0) | GPU ceiling of the adaptive step, before the minimum_core_dt floor, `0` = no ceiling |
| `temperature` | number (kelvin) | none, required (input.xml default 298.15) | kT of APBS, λ, diffusion coefficients, far field and k_b integral, viscosity fixed at 0.890 (millipascal seconds), warning away from 298.15, chain runs at 298.15 only |
| `seed` | integer ≥ 0 | none, required (input.xml default 1523) | random seed, GPU draws keyed by seed, trajectory and step (independent of gpu_force_batch and of a resume), CPU trajectory i with seed + i, multi GPU shard i with seed + 11111111 i, conformer seed of `rcsb_smiles` (ligand geometry and charges) |
| `convergence_check` | `true`, `false` | none, required (input.xml default true) | rigid runs, bd_sims/convergence.json after the run (P_rxn, standard error, Wilson interval, converged flag, N for 10, 5 and 1 % relative error) |
| `convergence_interval` | integer (percent of trajectories) | none, required (input.xml default 10) | GPU live P_rxn and k_on line at every convergence_interval percent of finished trajectories, `0` = none, progress line every 10 (seconds) in every case |
| `convergence_tol` | number in (0, 1) | none, required (input.xml default 0.05) | relative standard error threshold of the converged flag of convergence.json (single run and multi GPU pool), convergence_tol copied into results.json, no run stop |
| `checkpoint_interval` | integer, `0` | none, required (input.xml default 0) | GPU checkpoint bd_sims/checkpoint.npz after each checkpoint_interval finished trajectories, resume with `python run_pystarc.py input.xml` and unchanged settings and inputs, file removed after success, `0` = off, lost on run.sh and the SLURM scripts (bd_sims deleted) |
| `confidence_interval` (input.xml only) | number in (0, 1) | 0.95 | Wilson confidence level of P_rxn and k_on, refused in config.xml |

## compute

| key | accepts | default | does |
|---|---|---|---|
| `gpu` | `true`, `false` | none, required (input.xml default true) | `true` = GPU batch simulator (CuPy with a working kernel, run abort before setup otherwise), `false` = CPU model with n_threads processes and the hard sphere test, soft_repulsion true refused, chain runs on CPUs in every case |
| `gpu_force_batch` | integer ≥ 0, `0` | none, required (input.xml default 0) | trajectories per GPU force evaluation, `0` = max(1000, 4 × 1024³ / (150 × ligand atoms)), capped at the running count, no effect on results |
| `n_threads` | integer | none, required (input.xml default 24) | worker processes of the CPU model, min(n_threads, n_trajectories, CPU count), `1` or less = serial, no effect on GPU runs |

## outputs

| key | accepts | default | does |
|---|---|---|---|
| `save_interval` | integer ≥ 0 | none, required (input.xml default 100) | GPU steps between paths.npz and energetics.npz records of every running trajectory, `0` = start and last pose in paths.npz and no energetics.npz |
| `full_paths` | `true`, `false` | `true` (optional) | paths.npz, columns traj_id, step, x, y, z, q0, q1, q2, q3 |
| `energetics` | `true`, `false` | `true` (optional) | energetics.npz, columns traj_id, step, fx, fy, fz (kT per angstrom), dt (picoseconds), no energies in rigid runs |
| file switches (input.xml only) | `results_json`, `trajectories_csv`, `encounters_csv`, `near_misses_csv`, `radial_density`, `angular_map`, `bd_time_to_reaction` (alias `fpt_distribution`), `contact_frequency`, `milestone_flux`, `transition_matrix`, `pose_clusters`, each `true` or `false` | `true` each | one switch per bd_sims file inside the outputs block of input.xml, refused in config.xml |

## Other input.xml only keys

| key | accepts | default | does |
|---|---|---|---|
| `pdb` | path of a PDB of the complex, relative to the current folder | none | PDB route of run_pystarc.py for a blank receptor_pqr or ligand_pqr (ligand residue ligand_resname, antechamber AM1 BCC charges, tleap with protein_ff and ligand_ff), run start abort for a named missing file, no effect with the input.xml of setup.py |
| `ligand_charge` | integer | `0` | antechamber `-nc` of the PDB route |
| `protein_ff` | suffix of a leaprc.protein file of AmberTools | `ff14SB` | tleap `source leaprc.protein.<protein_ff>` of the PDB route |
| `ligand_ff` | `gaff`, `gaff2` | `gaff` | tleap `source leaprc.<ligand_ff>` of the PDB route |
| `ghost_atoms` | `auto`, or lines `receptor index,ligand index,cutoff` with 0 based indices and the cutoff (angstroms) | `auto` | reaction criterion for a blank rxns_xml, lines = pairs all required at one time, `auto` = GHO to GHO pair at bd_milestone_radius_inner, no effect with the input.xml of setup.py |
| `bd_milestone_radius_inner` | number (angstroms) in (0, bd_milestone_radius) | `0` | GHO to GHO reaction distance for ghost_atoms `auto` with a blank rxns_xml, abort at the reaction setup for a number outside (0, bd_milestone_radius) in the same case, no effect with the input.xml of setup.py |
| `relative_viscosity` | `1` | `1` | run start abort for any other number, viscosity fixed at 0.890 (millipascal seconds) |

## chain

```
route                 flexible chain BD on CPUs, separate from build_inputs
folder                examples/<name> with config.xml, setup.py, make_grids.py, submit_shards.sh, combine_shards.py
config.xml            sections structure, names, criterion, run, every key required, read as text, no defaults
setup.py              specific to the chain example, output chain.json, reaction_pairs.json and input.xml with a chain block, receptor_pqr centered in place
commands              python setup.py, python make_grids.py, ./submit_shards.sh, python combine_shards.py
```

| key | accepts | default | does |
|---|---|---|---|
| `source_pdb` | path of a local PDB file of the complex, relative to the current folder | none, required | receptor residue check, bead chain positions and interface contacts (heavy atoms of ATOM records), no download |
| `receptor_chain_id` | one chain character | none, required | receptor chain of source_pdb, residues matched one to one with receptor_pqr (tleap variants of HIS, CYS, ASP, GLU, LYS accepted) |
| `ligand_chain_id` | one chain character | none, required | chain turned into the COFFDROP bead chain |
| `receptor_pqr` | path of an existing receptor PQR | none, required | receptor charges and radii, centered in place by setup.py, receptor_pqr of input.xml |
| `target_grid_dx` | path of an existing DX file, or a path ending in `<stem>1.dx` or `<stem>0.dx` for a grid built when absent | none, required | receptor electrostatic grid of the beads, built by make_grids.py or at run start when absent, one grid per run (no field outside) |
| `born_grid_dx` | path of an existing DX file, or `<stem>0_born.dx` or `<stem>1_born.dx` next to target_grid_dx for a grid built when absent | none, required | receptor desolvation grid of the beads, target_grid_dx required |
| `chain_name` | text | none, required | name of the bead chain |
| `chain_json` | output path | none, required | bead chain of setup.py (topology and bead positions in the source_pdb frame, centered at load by the run) |
| `reaction_pairs_json` | output path | none, required | pairs [receptor atom index, bead index, distance] of setup.py, 0 based indices |
| `input_xml` | output path | none, required | input.xml of setup.py |
| `work_dir` | folder name | none, required | work_dir of input.xml, removed by setup.py when empty |
| `interface_cutoff_a` | number (angstroms) | none, required | heavy atom distance of the interface contacts between the two chains (distance < cutoff) |
| `reaction_distance_a` | number (angstroms) | none, required | cutoff of every reaction pair |
| `min_reaction_pairs` | integer | none, required | setup.py abort below the pair count |
| `reaction_n_needed` | integer from 1 to the pair count | none, required (chain block default 3) | satisfied pairs required at one time, `-1` refused |
| `dt` | number (picoseconds) | none, required (input.xml default 0.2) | outer step, chain_steps_per_outer × dt_chain = dt (run start abort otherwise), 0.05 (picoseconds) with any pair distance < 1.5 × pair cutoff |
| `dt_chain` | number > 0 (picoseconds) | none, required (chain block default 0.05) | inner step of the bead coordinates |
| `chain_steps_per_outer` | integer ≥ 1 | none, required (chain block default 4) | inner steps per outer step |
| `n_equilibration_steps` | integer ≥ 0 | none, required (chain block default 0) | inner steps before each trajectory without receptor forces |
| `r_escape` | number ≥ 0 (angstroms) | none, required (chain block default 0 = 2 b) | escape radius without the outer propagator |
| `auto_diffusion` | `true`, `false`, any case | none, required (chain block default false) | `true` = Rotne Prager Yamakawa diffusion tensors from the beads, `false` = d_trans and d_rot, setup.py test `true` in any case, D_trans and D_rot in the chain block for any other text, `yes` or `1` = true flag with D_trans and D_rot (run start abort for d_trans or d_rot > 0) |
| `d_trans` | number > 0 (square angstroms per picosecond) | none, required | D_trans of the chain block with auto_diffusion false |
| `d_rot` | number > 0 (square radians per picosecond) | none, required | D_rot of the chain block with auto_diffusion false |
| `use_soft_repulsion` | `true`, `false` | none, required (chain block default false) | WCA repulsion between beads and receptor atoms and between bead pairs without a COFFDROP table |
| `soft_repulsion_eps` | number ≥ 0 (kT) | none, required (chain block default 1.0) | WCA ε |
| `n_workers` | integer ≥ 1 | none, required (chain block default 1) | worker processes per run, results independent of n_workers |
| `n_shards` | integer ≥ 1 | none, read by submit_shards.sh only | independent shard runs, seed of shard i = 11111111 i + 1, n_trajectories per shard |
| shared run keys | bd_milestone_radius, n_trajectories, max_steps, temperature, seed, desolvation_alpha, save_interval, apbs_fglen, apbs_dime | none, required | rigid table meaning, temperature 298.15 (kelvin) only, debye_length 7.858 (angstroms) in input.xml without the tag |
| ignored run keys | gpu, convergence_check, convergence_interval, convergence_tol | none, required by setup.py | no effect on chain runs, run start abort for convergence_tol outside (0, 1) |
| chain block only (input.xml edit) | diffusion_update_interval (10), use_hard_sphere (true), elastic_network_cutoff (0), elastic_network_k (0), use_self_born (false), gb_eps_in (1.0), gb_eps_out (78.5), gb_obc_alpha (1.0), gb_obc_beta (0.8), gb_obc_gamma (4.85), coffdrop_active (false) | defaults in parentheses | tensor rebuild interval, hard sphere rejection, elastic network, generalized Born of the chain, COFFDROP table check |

## Electrostatic grids

| grid | file in bd_sims | edge (angstroms) | points per axis | method | read by |
|---|---|---|---|---|---|
| receptor coarse | receptor0.dx | apbs_cglen, `0` = 2 × fine edge | apbs_coarse_dime, `0` = apbs_dime | APBS, Debye Huckel boundary (bcfl sdh) | both force engines |
| receptor fine | receptor1.dx | apbs_fglen, `0` = derived edge | apbs_fine_dime, `0` = apbs_dime | APBS, boundary from the coarse potential (bcfl map, usemap pot 1) | both force engines |
| receptor desolvation | receptor0_born.dx, receptor1_born.dx | derived edges, coarse grid widened to hold the 15 (angstroms) cutoff | apbs_dime, coarse grid widened to any odd count | analytic Born image field, no APBS | both force engines |
| ligand desolvation | ligand0_born.dx, ligand1_born.dx | derived edges, coarse grid widened to hold the 15 (angstroms) cutoff | apbs_ligand_dime, coarse grid widened to any odd count | analytic Born image field, no APBS | enable_born2_torque (GPU), CPU model in every case |

```
derived fine edge     max((L + 2 (r_max + srad)) × 1.05, 4 (r_max + srad) × 1.025)
L                     longest edge of the box of atom centers
r_max                 largest atom radius
ghost atoms           GHO and PKT left out of edges and centers
center                mean atom position, the origin after the run centers each PQR
spacing               edge / (points − 1)
coarse edge check     coarse edge ≤ fine edge, abort at the APBS step
edge ratio            cglen / fglen ≤ 4 (APBS limit)
debye_length          no effect on the edges
APBS options          mg-manual, lpbe, srfm smol, chgm spl2, sdens 10.0, pdie, sdie, srad, temperature, ion lines
evaluation            finest grid with the atom inside the usable box of the grid, usable box 3 spacings inside the faces of an APBS grid and 1 spacing inside the faces of a desolvation grid (GPU)
outside the grids     GPU far field (multipole_fallback), CPU model without field
reuse                 bd_sims/receptor_grids.key and ligand_grids.key (SHA1 of the centered PQR and every grid setting), reuse on a match
```

Dime rule

```
APBS rule             d = c × 2^(l+1) + 1, c ≥ 1, l ≥ 1
PySTARC check         d ≥ 5 and d − 1 divisible by 4 (same set, d = 4k + 1, k ≥ 1)
valid d               5, 9, 13, 17, 21, 25 and every 4k + 1 above
invalid d             4, 100, 200, 256, 300, 400, 500
receptor dimes        apbs_dime, apbs_coarse_dime, apbs_fine_dime, ValueError at the APBS step of run_pystarc.py
ligand dime           apbs_ligand_dime, ValueError at run start
rounding              none
APBS recommendation   c × 32 + 1 above 65 (97, 129, 161, 193, 225, 257, 289, 321, 353, 385, 417, 449, 481, 513, 545, 577 and every further step of 32)
```

## Reaction criterion modes

| mode | keys | pairs in rxns.xml | n_needed | coordinates read |
|---|---|---|---|---|
| contacts, picked pairs | search_cutoff, n_pairs, atom_filter, excluded, pair_cutoff, n_needed | n_pairs heavy atom pairs of the bound pose with distance < search_cutoff, closest first, first pair per receptor residue number | n_needed | bound pose |
| contacts, named pairs | receptor, ligand, receptor_numbering, atom_filter (auto only), pair_cutoff, n_needed | each target with the named ligand atom, or with the closest allowed heavy ligand atom for auto | n_needed | bound pose for auto |
| centroid | pair_cutoff | receptor GHO with ligand GHO, both appended by the run at the molecule centroids (positions n_rec + 1 and n_lig + 1) | 1 | none |
| site with anchors | receptor (one target), ligand (one atom), receptor_numbering, pair_cutoff or margin | receptor anchor with ligand anchor | 1 | bound pose for margin |
| site without anchors | pair_cutoff or margin, pocket_ligand and pocket_ligand_resid (`rcsb_smiles`) | PKT ghost atom (charge 0, radius 0) appended to receptor.pqr at the bound ligand centroid (deposited pocket_ligand centroid for `rcsb_smiles`), with the ligand GHO | 1 | bound pose for PKT |

```
pair test             distance < cutoff
reaction              satisfied pairs ≥ n_needed at one time, crossing test between steps (brownian_bridge)
pair count rule       1 pair with n_needed 1, 2k pairs with n_needed k
named and picked      never together in one config.xml
margin cutoff         closest accessible approach + margin, rounded to 0.25 (angstroms)
closest approach      smallest distance of the ligand anchor or ligand centroid from the receptor anchor or PKT over rigid ligand poses with soft wall energy ≤ 2 (kT)
scan                  2,000,000 poses in a ball of radius 16 (angstroms), 8,000,000 poses in a ball 1.5 (angstroms) wider than the first result, 2048 descents from the bound pose for a result below 8 (angstroms), none for rcsb_smiles, seed 20261108, GPU with CuPy, NumPy otherwise
rxns.xml              one reaction (association, start to end), one based atom positions in receptor.pqr and ligand.pqr, GHO counted
former modes          auto = contacts with picked pairs, explicit = contacts with named pairs, center = centroid, all three refused
```

## Hydrodynamic radii

```
switch                hydrodynamic_interactions (physics), independent of the radii
r_hydro 0             Monte Carlo radius from bd_sims/receptor.pqr or bd_sims/ligand.pqr
r_hydro > 0           fixed radius, Monte Carlo radius still computed and cached
software              PySTARC only, pystarc/hydrodynamics/mc_hydro_radius.py (Hansen, J. Chem. Phys. 121, 9111, 2004)
atoms                 every atom except PKT (GHO at the centroid kept), radius + srad
voxel grid            spacing max(0.02, min(1.0, B / 100)) (angstroms), B = longest edge of the atom box with radius + srad, padding 3 (angstroms)
surface               surface cubes of the voxel grid, area weights A_k at points x_k
sampling              1,000,000 random pairs of surface points, seed 1111111
radius                a = Σ A_i A_j / Σ (A_i A_j / |x_i − x_j|)
failure               largest atom distance from the centroid plus atom radius, with a warning
cache                 bd_sims/<pqr>.r_hydro_s<spacing>_p<srad>_n1000000_<hash>.cache
log                   geometry block, checklist line Hydro radii with (MC) or (XML)
diffusion             D_t = kT / (6 π η a), D_r = kT / (8 π η a³), η = 0.890 (millipascal seconds)
fixed radii           one run with 0, printed radii copied into r_hydro_rec and r_hydro_lig
```

## Run folder files

```
examples/<name>/
├── config.xml                              input
├── complex.pdb                             input of complex_pdb and prmtop
├── <pdb_id>.pdb                            entry of rcsb and rcsb_smiles, kept
├── receptor.pqr                            setup.py, receptor charges and radii, PKT appended for site without anchors
├── ligand.pqr                              setup.py, ligand charges and radii
├── rxns.xml                                setup.py, reaction criterion
├── input.xml                               setup.py, run settings, work_dir bd_sims
├── build_record.json                       setup.py, mode, ligand_source, pairs, n_needed, site details, md5 of receptor.pqr, ligand.pqr, rxns.xml, input.xml
└── bd_sims/                                run_pystarc.py
    ├── pystarc_<YYYYmmdd_HHMMSS>.log       log and run summary
    ├── receptor.pqr, ligand.pqr            centered copies with a GHO atom at the origin
    ├── receptor0.dx, receptor1.dx          receptor electrostatic grids
    ├── receptor0_born.dx, receptor1_born.dx receptor desolvation grids
    ├── ligand0.dx, ligand1.dx              ligand electrostatic grids
    ├── ligand0_born.dx, ligand1_born.dx    ligand desolvation grids
    ├── receptor_grids.key, ligand_grids.key grid fingerprints
    ├── *.pqr.r_hydro_*.cache               hydrodynamic radii
    ├── results.json                        k_on, k_on_low, k_on_high, P_rxn with interval, k_b, D_rel, counts, bounds
    ├── convergence.json                    convergence_check true
    ├── trajectories.csv                    one row per trajectory
    ├── encounters.csv                      pose of each reaction
    ├── near_misses.csv                     closest approach of escaped trajectories
    ├── paths.npz                           full_paths true
    ├── energetics.npz                      energetics true, save_interval > 0
    ├── radial_density.csv                  steps and time per radial shell
    ├── angular_map.npz                     steps and time per angular bin
    ├── bd_time_to_reaction.csv             time to reaction of reacted trajectories
    ├── contact_frequency.csv               steps and time inside the cutoff per pair
    ├── milestone_flux.csv                  crossings of 11 shells from b to the escape sphere
    ├── transition_matrix.npz               radial bin transitions
    ├── pose_clusters.csv                   rotation angle bins of the encounter poses, more than 5 encounters
    └── checkpoint.npz                      checkpoint_interval > 0, removed after success
```

```
build intermediates   deleted after a successful build, kept after an abort
                      protein.pdb, ligand_raw.pdb, tleap_protein.in, tleap_protein.log, tleap_ligand.in, tleap_ligand.log, leap.log, protein.prmtop, protein.rst7, protein_leap.pdb, ligand.prmtop, ligand.rst7, ligand_leap.pdb, ligand.mol2, ligand.frcmod, <code>.cif
ligand scratch        deleted at the end of the ligand parameterization (complex_pdb, rcsb, rcsb_smiles), kept after an abort inside the parameterization
                      ligand_h.sdf, ligand_raw.sdf, ligand_obabel.mol2, sqm.in, sqm.out, sqm.pdb, ANTECHAMBER_*.AC, ANTECHAMBER_*.AC0, ATOMTYPE.INF, _protein.rst7, _ligand.rst7
prmtop scratch        deleted at the end of the prmtop step, kept after an abort inside the step
                      _full.rst7
multi GPU             bd_sims/split.json, bd_sims/bd_1 to bd_sims/bd_N (seed + 11111111 i, share of n_trajectories, linked grids and caches, own PQR and result files), pooled results.json, convergence.json, CSV and NPZ files in bd_sims (combine_data.py)
chain runs            results.json (chain schema), trajectories.csv, near_misses.csv, paths.npz, radial_density.csv, angular_map.npz, milestone_flux.csv, energetics.npz, encounters.csv, fpt_distribution.csv, contact_frequency.csv, input.xml copy and log in work_dir
```
