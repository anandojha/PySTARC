# PySTARC tutorial

This tutorial computes the association rate constant k<sub>on</sub> of trypsin and benzamidine with rigid body Brownian dynamics on an NVIDIA GPU. The input files are in `examples/trypsin_benzamidine`. Steps 3 to 11 fill the sections of `config.xml` one by one and explain every key with the accepted entries. A new complex follows the same steps with a PDB file of the bound complex (step 3). The flexible chain model is described in `examples/barnase_barstar`.

## 1. Install PySTARC

PySTARC runs on a Linux node with an NVIDIA GPU and a driver for CUDA 12 or newer.

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install.sh
conda activate PySTARC
```

`install.sh` creates the conda environment PySTARC with AmberTools 24.8, APBS, RDKit, OpenBabel and CuPy. An existing environment named PySTARC is replaced. On a host without an NVIDIA GPU or with an older driver the script stops before any change.

## 2. Make a run folder

```bash
mkdir examples/my_trypsin
cd examples/my_trypsin
cp ../trypsin_benzamidine/{complex.pdb,complex.prmtop,setup.py,run.sh,submit_SLURM_single_GPU.sh,submit_SLURM_multi_GPUs.sh} .
python -m pystarc.pipeline.build_inputs --template > config.xml
```

The last command writes a blank `config.xml` with the nine sections structure, names, criterion, geometry, electrostatics, physics, simulation, compute and outputs. `run.sh` and the SLURM scripts find PySTARC in a parent folder of the run folder. A run folder outside the PySTARC folder needs `export PYSTARC_ROOT=/path/to/PySTARC`.

| file | content |
|---|---|
| `complex.pdb` | trypsin and benzamidine in the bound pose with water and ions |
| `complex.prmtop` | AMBER topology of the atoms in `complex.pdb` |
| `config.xml` | settings of the simulation |
| `setup.py` | builds the simulation inputs from `config.xml` (same file in every rigid body example with a `config.xml`) |
| `run.sh` | runs `setup.py` and the simulation on one GPU |
| `submit_SLURM_single_GPU.sh` | the same on one GPU of a SLURM cluster |
| `submit_SLURM_multi_GPUs.sh` | the same on four GPUs of a SLURM cluster |

## 3. Receptor and ligand structures (`structure`)

`ligand_source` selects the origin of the receptor and ligand coordinates, charges and radii.

| `ligand_source` | input | charges and radii |
|---|---|---|
| `pqr` | receptor and ligand PQR files | from the PQR files |
| `prmtop` | AMBER topology and a PDB file with the coordinates of every topology atom in topology order | from the topology |
| `complex_pdb` | PDB file of the bound complex | ff14SB charges for the protein, AM1 BCC charges and GAFF2 types for the ligand, default tleap radii |
| `rcsb` | RCSB entry code and the residue name of the ligand | as for `complex_pdb` |
| `rcsb_smiles` | RCSB entry code, the residue name of the ligand and a SMILES of the ligand | as for `complex_pdb`, ligand built from the SMILES away from the pocket |

The trypsin benzamidine example reuses an existing AMBER topology.

```xml
<structure>
  <ligand_source>prmtop</ligand_source>
  <prmtop>complex.prmtop</prmtop>
  <complex_pdb>complex.pdb</complex_pdb>
</structure>
```

`setup.py` reads `complex.pdb` with the topology and writes the topology charges and radii into `receptor.pqr` and `ligand.pqr`. The residues named `ligand_resname` (step 4) form the ligand. Water, ions and caps (HOH, WAT, Na+, Cl-, K+, NA, CL, MG, ZN, CAL, ACE, NME) are dropped. Every other residue is written to `receptor.pqr`. The prmtop route keeps the charges and radii of an existing setup of the complex, for example a molecular dynamics setup.

### Start from a PDB file alone

A PDB file of the bound complex is sufficient with `ligand_source` `complex_pdb`. The example `examples/hsp90_inhibitors/HSP90-aminopyridine` starts from a PDB file alone.

```xml
<structure>
  <ligand_source>complex_pdb</ligand_source>
  <complex_pdb>complex.pdb</complex_pdb>
  <ligand_ccd>YKJ</ligand_ccd>
</structure>
```

`setup.py` builds the protein with tleap and ff14SB. The bond orders and formal charges of the ligand come from the Chemical Component Dictionary (CCD) entry `ligand_ccd`, read from `<code>.cif` in the run folder or downloaded from files.rcsb.org. RDKit adds the ligand hydrogens and antechamber assigns AM1 BCC charges and GAFF2 types.

| records in the PDB file | requirement |
|---|---|
| protein | ATOM records with AMBER residue names, HID, HIE or HIP for histidine and CYX for a disulfide (plain HIS is read as HIE) |
| ligand with a CCD entry | one HETATM residue with the CCD atom names and every heavy atom of the CCD entry except the leaving atoms, named with the CCD code or with the code in `ligand_ccd` |
| ligand without a CCD entry | one HETATM residue named UNK, UNL, LIG, DRG, INH or MOL with every hydrogen, `ligand_ccd` blank and `ligand_net_charge` set |
| everything else | water, ions, cofactors and other HETATM groups dropped, protein hydrogens rebuilt by tleap |

### Other keys of `structure`

| key | used with | meaning |
|---|---|---|
| `receptor_pqr_file`, `ligand_pqr_file` | `pqr` | PQR files of the receptor and the ligand in the bound pose, other than the `receptor.pqr` and `ligand.pqr` of the run folder |
| `prmtop` | `prmtop` | AMBER topology (.prmtop or .parm7) |
| `complex_pdb` | `prmtop`, `complex_pdb` | PDB file of the complex |
| `pdb_id` | `rcsb`, `rcsb_smiles` | RCSB entry code, read from `<pdb_id>.pdb` in the run folder or downloaded |
| `ligand_resid` | `rcsb`, `rcsb_smiles` | residue name of the ligand, read from the entry with `rcsb` and given to the ligand built from the SMILES with `rcsb_smiles`, not a residue number |
| `smiles` | `rcsb_smiles` | SMILES of the ligand in the protonation state of the simulation |
| `chain` | `rcsb`, `rcsb_smiles` | one chain identifier, blank for every chain |
| `rec_resid_min` | `rcsb` | lowest receptor residue number kept |
| `ligand_net_charge` | `complex_pdb`, `rcsb`, `rcsb_smiles` | integer net charge of the ligand, blank for the charge of the CCD entry or the SMILES |
| `ligand_ccd` | `complex_pdb`, `rcsb` | CCD code of the ligand, blank for the ligand residue name |
| `ligand_protonate`, `ligand_deprotonate` | `complex_pdb`, `rcsb` | ligand N, O or S atom names with one proton added or removed |
| `keep_hetatm` | `rcsb_smiles` | metal ions kept with the receptor, for example `ZN` |
| `work_dir` | every source | folder for the files written by `setup.py`, default the folder of `config.xml` |

`setup.py` stops with an error for a blank or unknown `ligand_source` and for a key of another source.

## 4. Names (`names`)

```xml
<names>
  <ligand_formula>C7 H9 N2</ligand_formula>
  <receptor_resname>TRP</receptor_resname>
  <ligand_resname>BEN</ligand_resname>
</names>
```

C7 H9 N2 is benzamidinium, the protonated form of benzamidine with net charge +1. `setup.py` compares the formula with `ligand.pqr` and stops on a mismatch. A wrong protonation state of the ligand therefore stops the build. For the example `setup.py` prints `ligand.pqr: 18 atoms, Q=+1.0 e` and `composition verified: C7 H9 N2`.

| key | meaning |
|---|---|
| `ligand_formula` | element symbols with counts and every hydrogen, as `C7 H9 N2` or `C7H9N2` (two letter elements as `Cl` and `Br`) |
| `receptor_resname` | label of the receptor in `input.xml`, no effect on the simulation |
| `ligand_resname` | residue name of the ligand, 3 characters at most |

`ligand_resname` acts per `ligand_source`. With `prmtop` the residues named `ligand_resname` in the topology form the ligand (case sensitive). With `complex_pdb` the name picks the ligand among several HETATM groups and names the ligand residue in `ligand.pqr`. With `rcsb` and `rcsb_smiles` the residue name comes from `ligand_resid`. With `pqr` the name is a label.

## 5. Reaction criterion (`criterion`)

The reaction criterion defines the bound state. `setup.py` writes the criterion into `rxns.xml` as receptor and ligand atom pairs with a reaction distance. A trajectory reacts when at least `n_needed` pairs are closer than the reaction distance at one time. `mode` selects the pairs.

| `mode` | pairs | former mode |
|---|---|---|
| `contacts` | receptor and ligand atoms picked from the contacts of the bound pose, or named in `config.xml` | `auto` (picked), `explicit` (named) |
| `centroid` | receptor centroid with ligand centroid | `center` |
| `site` | a point of the binding site with the ligand centroid, or one named receptor atom with one named ligand atom | none |

`setup.py` stops with an error for the former mode names.

### Pairs picked from the bound pose (former `auto`)

The trypsin benzamidine example picks six pairs.

```xml
<criterion>
  <mode>contacts</mode>
  <search_cutoff>6.0</search_cutoff>
  <n_pairs>6</n_pairs>
  <atom_filter>polar</atom_filter>
  <pair_cutoff>7.0</pair_cutoff>
  <n_needed>3</n_needed>
</criterion>
```

`setup.py` lists every pair of a receptor heavy atom and a ligand heavy atom closer than `search_cutoff` in the bound pose. `atom_filter` `polar` keeps the pairs with N, O or S on both sides. The `n_pairs` closest pairs enter `rxns.xml`, with at most one pair per receptor residue. A trajectory reacts when `n_needed` of the six pairs are closer than `pair_cutoff` at one time, here 3 pairs within 7.0 Å.

| receptor atom in `receptor.pqr` | ligand atom | distance in the bound pose (Å) |
|---|---|---|
| TRP 193 O | N2 | 2.82 |
| SER 192 O | N2 | 2.86 |
| CYX 173 O | N1 | 3.17 |
| VAL 205 O | N2 | 3.34 |
| GLN 174 N | N1 | 4.24 |
| GLY 194 N | N2 | 4.31 |

### Pairs named in `config.xml` (former `explicit`)

```xml
<criterion>
  <mode>contacts</mode>
  <receptor>
    <target resid="193" atom="O" resname="TRP"/>
    <target resid="173" atom="O" resname="CYX"/>
  </receptor>
  <ligand>
    <atom name="N2"/>
    <atom name="N1"/>
  </ligand>
  <pair_cutoff>7.0</pair_cutoff>
  <n_needed>1</n_needed>
</criterion>
```

Each `<target>` names one receptor atom by residue number, atom name and residue name in `receptor.pqr`. The ligand atoms follow in the order of the targets. `<ligand>auto</ligand>` with an `atom_filter` pairs each target with the closest allowed heavy ligand atom of the bound pose. `receptor_numbering` `internal` (default) takes the residue numbers of `receptor.pqr`. With `prmtop`, `complex_pdb`, `rcsb` and `rcsb_smiles` the residues are numbered from 1 in topology order. With `pqr` the numbers of the given file stay. With `ligand_source` `rcsb` the residue numbers of the RCSB entry are accepted with `receptor_numbering` `pdb`. A first run of `setup.py` with mode `centroid` writes `receptor.pqr` for a look at the numbering.

Picked and named pairs never mix in one `config.xml`. One pair needs `n_needed` 1. An even number of pairs 2k needs `n_needed` k.

### Distance of the centroids (`centroid`)

```xml
<criterion>
  <mode>centroid</mode>
  <pair_cutoff>8.0</pair_cutoff>
</criterion>
```

The simulation places a ghost atom GHO without charge or radius at the centroid of each molecule. A trajectory reacts when the two centroids are closer than `pair_cutoff`. The beta cyclodextrin examples use mode `centroid`.

### Distance to the binding site (`site`)

```xml
<criterion>
  <mode>site</mode>
  <margin>3.0</margin>
</criterion>
```

`setup.py` adds a ghost atom PKT without charge or radius to `receptor.pqr` at the centroid of the bound ligand. A trajectory reacts when the ligand centroid is closer to PKT than the cutoff. The cutoff is `pair_cutoff`, or the closest accessible approach plus `margin` rounded to 0.25 Å. The closest accessible approach is the smallest distance of the ligand centroid from PKT among rigid ligand poses with a soft wall energy of at most 2 kT against the receptor. `setup.py` finds the closest accessible approach by a scan of ligand poses and writes the cutoff into `build_record.json`. The HSP90, carbonic anhydrase, p38 and TTK examples use mode `site` with `margin` 3.0.

One `<target>` in `receptor` and one `<atom>` in `ligand` replace PKT and the ligand centroid by two named atoms. With `rcsb_smiles` the ligand has no bound pose, and PKT sits at the centroid of the deposited ligand `pocket_ligand`.

### Keys of `criterion`

| key | mode | meaning |
|---|---|---|
| `mode` | every mode | `contacts`, `centroid` or `site` |
| `search_cutoff` | `contacts`, picked | search distance in the bound pose (Å) |
| `n_pairs` | `contacts`, picked | pairs kept, closest first, one per receptor residue |
| `atom_filter` | `contacts`, picked or `auto` | `all` (any heavy atoms), `polar` (N, O or S on both sides), `nonpolar` (C on both sides), `any_polar` (N, O or S on at least one side), `receptor_polar` (N, O or S on the receptor side) |
| `excluded` | `contacts`, picked | atom names left out of the search, as `<item>NAME</item>` children |
| `receptor` | `contacts`, named, and `site` | receptor atoms as `<target resid="..." atom="..." resname="..."/>` |
| `ligand` | `contacts`, named, and `site` | ligand atoms as `<atom name="..."/>` in the order of the targets, or `auto` in `contacts` |
| `receptor_numbering` | named atoms | `internal` (default, numbers of `receptor.pqr`) or `pdb` (numbers of the RCSB entry with `rcsb`) |
| `pair_cutoff` | every mode | reaction distance of every pair (Å), in `site` either `pair_cutoff` or `margin` |
| `n_needed` | `contacts` | pairs closer than `pair_cutoff` at one time |
| `margin` | `site` | distance added to the closest accessible approach (Å) |
| `pocket_ligand` | `site` with `rcsb_smiles` | residue name of the deposited ligand at the site |
| `pocket_ligand_resid` | `site` with `rcsb_smiles` | residue number of `pocket_ligand` for a chain with several copies |

## 6. Start sphere and hydrodynamic radii (`geometry`)

Every key of steps 6 to 11 is required in `config.xml`, except `soft_repulsion` (default `false`), `minimum_core_reaction_dt` (default 0), `full_paths` (default `true`) and `energetics` (default `true`). `setup.py` stops with an error for a missing key.

```xml
<geometry>
  <bd_milestone_radius>45.0</bd_milestone_radius>
  <r_hydro_rec>22.3728</r_hydro_rec>
  <r_hydro_lig>4.9867</r_hydro_lig>
</geometry>
```

Each trajectory starts at a random position on the sphere of radius b = `bd_milestone_radius` (Å) around the receptor centroid, with a random orientation of the ligand. A trajectory that reaches the radius 2b goes back to the start sphere with the computed probability of a return from 2b to b. The remaining trajectories at 2b escape.

`r_hydro_rec` and `r_hydro_lig` set the hydrodynamic radii a of the receptor and the ligand (Å). The radii fix the diffusion coefficients D<sub>t</sub> = k<sub>B</sub>T/(6πηa) and D<sub>r</sub> = k<sub>B</sub>T/(8πηa³) with the viscosity η = 0.890 mPa s of water at 25 °C. PySTARC computes a by the Monte Carlo method of Hansen (J. Chem. Phys. 121, 9111, 2004) from the PQR file, with every atom radius enlarged by `srad`. The calculation runs inside PySTARC (`pystarc/hydrodynamics/mc_hydro_radius.py`) and needs no other software. A cache file next to each PQR file in `bd_sims` stores the Monte Carlo radius for the next run. An entry of 0 uses the Monte Carlo radius. A positive entry replaces the Monte Carlo radius. The run log then prints both values. The trypsin benzamidine example fixes the radii at 22.3728 Å and 4.9867 Å, computed with BrownDye 2 for a comparison of PySTARC with BrownDye 2. An entry of 0 gives 22.30 Å and 4.90 Å for the same PQR files. Hydrodynamic interactions stay on in both cases. `hydrodynamic_interactions` (step 8) alone switches the hydrodynamic interactions on and off.

## 7. Salt, dielectrics and electrostatic grids (`electrostatics`)

```xml
<electrostatics>
  <debye_length>7.85751</debye_length>
  <ion_concentration>0.15</ion_concentration>
  <ion_radius_pos>0.95</ion_radius_pos>
  <ion_radius_neg>1.81</ion_radius_neg>
  <pdie>4.0</pdie>
  <sdie>78.0</sdie>
  <srad>1.4</srad>
  <apbs_fglen>96</apbs_fglen>
  <apbs_dime>257</apbs_dime>
  <apbs_cglen>0</apbs_cglen>
  <apbs_coarse_dime>0</apbs_coarse_dime>
  <apbs_fine_dime>0</apbs_fine_dime>
</electrostatics>
```

| key | meaning |
|---|---|
| `ion_concentration` | concentration of a monovalent salt in every APBS calculation (M), 0 for no salt |
| `ion_radius_pos`, `ion_radius_neg` | radii of the cation and the anion in APBS (Å) |
| `debye_length` | Debye length λ (Å) of the screened field outside the grids, of the k<sub>b</sub> integral and of the desolvation grids, `inf` for no salt |
| `pdie` | dielectric constant of the solute |
| `sdie` | dielectric constant of the solvent |
| `srad` | solvent probe radius of the APBS surface and of the hydrodynamic radius (Å) |

The run stops when `debye_length` differs by more than 1% from the Debye length of `ion_concentration`, `sdie` and `temperature`, for example 7.83 Å at 0.15 M, 78 and 298.15 K.

### Two grids for the receptor potential

APBS solves the linearized Poisson Boltzmann equation of the receptor on two nested cubic grids centered on the receptor.

| grid | file in `bd_sims` | edge (Å) | points per axis | boundary condition |
|---|---|---|---|---|
| coarse | `receptor0.dx` | `apbs_cglen`, 0 for twice the fine edge | `apbs_coarse_dime`, 0 for `apbs_dime` | Debye Hückel potential of one sphere with the charge, dipole and quadrupole of the receptor (APBS `bcfl sdh`) |
| fine | `receptor1.dx` | `apbs_fglen`, 0 for the derived edge | `apbs_fine_dime`, 0 for `apbs_dime` | potential of the coarse grid (APBS `bcfl map`) |

The derived fine edge is max[(L + 2(r<sub>max</sub> + srad)) × 1.05, 4(r<sub>max</sub> + srad) × 1.025], with L the longest extent of the atom centers and r<sub>max</sub> the largest atom radius. `apbs_cglen` must exceed `apbs_fglen`. `apbs_dime` also sets the points per axis of the receptor desolvation grids.

During the simulation the force on each ligand atom comes from the finest grid that holds the atom at least three grid spacings inside the faces. Outside both grids the receptor field is a screened multipole expansion (`multipole_fallback`, step 8). For trypsin the fine grid spans 96 Å with 257 points (spacing 0.375 Å), and the coarse grid spans 192 Å (spacing 0.75 Å). `setup.py` prints a warning when b plus the ligand radius reaches past the fine grid (b + ligand radius ≥ `apbs_fglen`/2). Trypsin gives the warning (45 Å + 3.7 Å > 48 Å), and the ligand atoms outside the fine grid read the coarse grid.

### Grid points per axis

The multigrid solver of APBS needs d = c·2<sup>l+1</sup> + 1 points per axis, with integers c ≥ 1 and l ≥ 1 for l multigrid levels. APBS keeps every d = 4k + 1 up to 65, for example 5, 9, 13, 33 and 65. Above 65 APBS keeps only d = 32c + 1, for example 97, 129, 161, 193, 225 and 257. APBS lowers any other d above 65 to the next smaller 32c + 1, for example 101 to 97. PySTARC checks `apbs_dime`, `apbs_coarse_dime` and `apbs_fine_dime` before APBS runs and stops for any other number, without rounding.

## 8. Forces and excluded volume (`physics`)

```xml
<physics>
  <soft_repulsion>true</soft_repulsion>
  <overlap_check>false</overlap_check>
  <lj_forces>false</lj_forces>
  <hydrodynamic_interactions>true</hydrodynamic_interactions>
  <desolvation_alpha>1.0</desolvation_alpha>
  <enable_born2_torque>true</enable_born2_torque>
  <multipole_fallback>true</multipole_fallback>
</physics>
```

Every flag accepts `true`, `false`, `yes`, `no`, `1` and `0` with any capitalization. Steps 8 and 9 describe the GPU engine. The CPU model of step 10 differs in several keys.

| key | entries | effect |
|---|---|---|
| `soft_repulsion` | `true` or `false`, default `false` | soft wall in place of the overlap check |
| `overlap_check` | `true` or `false` | rejection of moves into the receptor |
| `lj_forces` | `true` or `false` | Weeks Chandler Andersen (WCA) repulsion |
| `hydrodynamic_interactions` | `true` or `false` | Rotne Prager Yamakawa D(r) or constant D |
| `desolvation_alpha` | number ≥ 0 | factor on the Born desolvation energy |
| `enable_born2_torque` | `true` or `false` | desolvation of the receptor charges in the ligand Born field |
| `multipole_fallback` | `true` or `false` | receptor multipole field outside the grids |

### Excluded volume

The soft wall adds the energy V = kT (a/r)<sup>12</sup> with a = 0.9345 (r<sub>i</sub> + r<sub>j</sub>) for every receptor atom i and ligand atom j closer than 2a. The radii come from the PQR files. Atoms of radius 0 take no part. The overlap check rejects a move bringing a ligand atom closer than 0.5 (r<sub>i</sub> + r<sub>j</sub>) to a receptor atom. The trajectory keeps the old pose and the clock still advances. The WCA repulsion is the force of the potential 4ε[(σ/r)<sup>12</sup> − (σ/r)<sup>6</sup>] below r = 2<sup>1/6</sup>σ, with σ = r<sub>i</sub> + r<sub>j</sub> and ε = 0.1 kcal/mol. The soft wall and the overlap check also shorten the time step near the receptor (step 9).

| `soft_repulsion` | `overlap_check` | `lj_forces` | excluded volume |
|---|---|---|---|
| `true` | `true` or `false` | `false` | soft wall |
| `true` | `true` or `false` | `true` | error before the first step |
| `false` | `true` | `false` | overlap check |
| `false` | `true` | `true` | overlap check and WCA repulsion |
| `false` | `false` | `false` | none |
| `false` | `false` | `true` | error before the first step |

### Diffusion, desolvation and the far field

With `hydrodynamic_interactions` true the relative diffusion coefficient D(r) of two spheres with the radii of step 6 follows the Rotne Prager Yamakawa tensor. D(r) falls as the molecules approach. Each step adds the drift (dD/dr)Δt and the k<sub>b</sub> integral also uses D(r). With `false` the relative diffusion coefficient stays at D<sub>rec</sub> + D<sub>lig</sub>.

The Born desolvation energy of a charge q in the Born field φ<sub>Born</sub> of the other molecule is α q<sup>2</sup> φ<sub>Born</sub> with α = `desolvation_alpha`. The value 1.0 is the base model. The value 0 switches desolvation off. A negative value stops the run at start. With α above 0 the desolvation acts on the ligand charges in the receptor Born field. `enable_born2_torque` true also adds the receptor charges in the ligand Born field as a force and a torque on the ligand.

With `multipole_fallback` true the receptor field outside every electrostatic grid comes from a screened expansion of the receptor charges with monopole, dipole and quadrupole terms. The k<sub>b</sub> integral then also holds the orientation average of the squared dipole and quadrupole terms for a charged ligand. With `false` the field outside the grids comes from the net receptor charge alone. A neutral receptor then has no field outside the grids.

The trypsin example uses the soft wall, hydrodynamic interactions, the base desolvation in both directions and the multipole field. `overlap_check` false has no effect with the soft wall on.

## 9. Trajectories, time step and convergence (`simulation`)

```xml
<simulation>
  <n_trajectories>10000000</n_trajectories>
  <max_steps>1000000</max_steps>
  <dt>0.2</dt>
  <minimum_core_dt>0.2</minimum_core_dt>
  <minimum_core_reaction_dt>0.05</minimum_core_reaction_dt>
  <max_dt>0</max_dt>
  <temperature>298.15</temperature>
  <seed>20261107</seed>
  <convergence_check>true</convergence_check>
  <convergence_interval>10</convergence_interval>
  <convergence_tol>0.05</convergence_tol>
  <checkpoint_interval>0</checkpoint_interval>
</simulation>
```

| key | entries | effect |
|---|---|---|
| `n_trajectories` | integer ≥ 1 | trajectories started on the b sphere |
| `max_steps` | integer ≥ 1 | step limit of each trajectory |
| `dt` | number (ps) | outer step of the flexible chain model only |
| `minimum_core_dt` | number ≥ 0 (ps), 0 for none | floor of the time step |
| `minimum_core_reaction_dt` | number ≥ 0 (ps), default 0 | floor of the time step near a reaction surface in state machine runs |
| `max_dt` | number ≥ 0 (ps), 0 for none | ceiling of the time step |
| `temperature` | number (K) | kT of APBS, the diffusion coefficients and the forces |
| `seed` | integer ≥ 0 | seed of the random numbers |
| `convergence_check` | `true` or `false` | `bd_sims/convergence.json` after the run |
| `convergence_interval` | integer (percent), 0 for none | log lines with the running P<sub>rxn</sub> and k<sub>on</sub> |
| `convergence_tol` | number between 0 and 1 | relative standard error for the converged flag |
| `checkpoint_interval` | integer, 0 for none | finished trajectories between checkpoints |

The trypsin run starts 10,000,000 trajectories with a step limit of 1,000,000. `submit_SLURM_multi_GPUs.sh` splits the run into four runs of 2,500,000 trajectories with the seeds `seed` + 11111111 i for i = 1 to 4.

### Time step

Rigid body runs use an adaptive time step and ignore `dt`. At every step the GPU engine sets the step of each trajectory from the center distance and the net charges. Beyond 1.1b the distance to the start and escape spheres also limits the step. The soft wall or the overlap check shortens the step near the receptor surface. `max_dt` then caps the step. `minimum_core_dt` comes last and raises the step to the floor over every other rule. The trypsin run has the floor 0.2 ps and no ceiling. No trypsin step is shorter than 0.2 ps, even at the receptor surface.

`minimum_core_reaction_dt` acts only in state machine runs, with `state_machine_reactions` true in `input.xml` (default false). In a state machine run the step shrinks near a reachable reaction surface but not below `minimum_core_reaction_dt`. The trypsin run keeps `state_machine_reactions` false. The trypsin value 0.05 then has no effect.

### Step limit

Each trajectory runs to a reaction, an escape or `max_steps` steps. A trajectory at the step limit counts as not reacted. `results.json` gives the count `n_max_steps` and the bounds `k_on_bound_low` and `k_on_bound_high`. `k_on_bound_low` counts every trajectory at the limit as not reacted and `k_on_bound_high` as reacted. `python -m pystarc.analysis.step_limit bd_sims/trajectories.csv` reads a pilot run and proposes a `max_steps` with a change of k<sub>on</sub> of at most 1% from the trajectories at the limit.

### Convergence

After N trajectories the relative standard error of P<sub>rxn</sub> is √((1 − P<sub>rxn</sub>)/(N P<sub>rxn</sub>)). With `convergence_check` true the run writes `bd_sims/convergence.json` with P<sub>rxn</sub>, k<sub>on</sub>, the standard errors, the 95% Wilson intervals, the converged flag and the N for a relative error of 10, 5 and 1%. The flag is true for a relative standard error below `convergence_tol`. The trypsin value 0.05 flags convergence below a relative standard error of 5%.

`convergence_interval` prints the running P<sub>rxn</sub> and k<sub>on</sub> each time the count of finished trajectories passes a multiple of `convergence_interval` percent of `n_trajectories`. A progress line also appears every 10 s once trajectories finish. None of the convergence keys stops a run. The run always takes every trajectory to a reaction, an escape or the step limit.

### Checkpoints

With `checkpoint_interval` above 0 the run writes `bd_sims/checkpoint.npz` each time `checkpoint_interval` more trajectories have finished. A stopped run resumes with `python /path/to/PySTARC/run_pystarc.py input.xml` in the run folder. `run.sh` and the SLURM scripts delete `bd_sims` and start over. The resumed run needs the same input files and the same keys. Of the `config.xml` keys only `gpu_force_batch`, `n_threads`, `convergence_interval`, the `outputs` keys and a positive `checkpoint_interval` may change. A checkpoint of another run stops the run with an error. With `save_interval` above 0 the files `paths.npz` and `energetics.npz` of a resumed run hold the records after the resume only. A finished run deletes the checkpoint. The trypsin run writes no checkpoint.

### Temperature and seed

`temperature` sets kT in APBS, the desolvation grids, the diffusion coefficients and the forces. The viscosity stays at 0.890 mPa s of water at 25 °C. The run prints a warning for a temperature other than 298.15 K. A new `temperature` needs a matching `debye_length` (step 7).

The random numbers of a trajectory depend only on `seed`, the trajectory number and the step count. `gpu_force_batch` and a resume leave the random numbers unchanged. With `rcsb_smiles` the seed also fixes the ligand conformer of `setup.py` (step 3).

## 10. Hardware (`compute`)

```xml
<compute>
  <gpu>true</gpu>
  <gpu_force_batch>0</gpu_force_batch>
  <n_threads>32</n_threads>
</compute>
```

| key | entries | effect |
|---|---|---|
| `gpu` | `true` or `false` | GPU engine for `true`, CPU model for `false` |
| `gpu_force_batch` | integer ≥ 0, 0 for automatic | trajectories per GPU force evaluation |
| `n_threads` | integer | worker processes of the CPU model |

With `gpu` true the GPU engine runs every trajectory at once on one GPU through CuPy. The run stops before the first setup step when CuPy cannot run a kernel.

`gpu_force_batch` splits the force evaluation of the running trajectories into batches and bounds the GPU memory. The value 0 picks max(1000, 4 × 1024<sup>3</sup>/(150 × ligand atoms)) trajectories per batch, capped at the count of running trajectories. The 18 ligand atoms and GHO in `bd_sims/ligand.pqr` give 1,507,006 trajectories per batch. The batch size has no effect on the random numbers.

### CPU model

With `gpu` false the CPU model runs the trajectories in `n_threads` worker processes, at most one per CPU core and one per trajectory. A value of 1 or less runs one process. The CPU model differs from the GPU engine of steps 8 and 9 in the keys below.

| key | CPU model |
|---|---|
| `soft_repulsion` | `true` stops the run before the first step |
| `overlap_check`, `lj_forces` | no effect, a move with a ligand atom closer than r<sub>i</sub> + r<sub>j</sub> to a receptor atom is drawn again up to five times and the old pose stays after five failures |
| `enable_born2_torque` | no effect, both desolvation directions in every run |
| `multipole_fallback` | no effect, no receptor field outside the grids |
| `minimum_core_dt`, `max_dt` | no effect |
| `minimum_core_reaction_dt` | floor of the time step (0.05 ps for 0), a quarter of the floor at a center distance below 1.5 times the smallest reaction distance |
| `seed` | seed `seed` + i for trajectory i |
| `convergence_interval`, `checkpoint_interval` | no effect |
| `save_interval`, `full_paths`, `energetics` | no effect |

The CPU model prints P<sub>rxn</sub> and k<sub>on</sub> in the log and writes `bd_sims/convergence.json`. The CPU model writes no `results.json`, `trajectories.csv`, `paths.npz` or `energetics.npz`.

The trypsin example runs on the GPU. `n_threads` 32 has no effect on the trypsin run.

## 11. Saved paths (`outputs`)

```xml
<outputs>
  <save_interval>100</save_interval>
  <full_paths>true</full_paths>
  <energetics>true</energetics>
</outputs>
```

| key | entries | effect |
|---|---|---|
| `save_interval` | integer ≥ 0 | steps between records of every running trajectory |
| `full_paths` | `true` or `false`, default `true` | `bd_sims/paths.npz` |
| `energetics` | `true` or `false`, default `true` | `bd_sims/energetics.npz` |

Every `save_interval` steps the GPU run adds one record per running trajectory to `paths.npz` and `energetics.npz`.

| file | columns |
|---|---|
| `paths.npz` | traj_id, step, x, y, z, q0, q1, q2, q3, with the ligand center (Å) and the orientation quaternion |
| `energetics.npz` | traj_id, step, fx, fy, fz, dt, with the force on the ligand (kT/Å) and the time step (ps) |

`energetics.npz` holds no energies. With `save_interval` 0 the file `paths.npz` holds the start pose and the last pose of every trajectory. The run then writes no `energetics.npz`. The trypsin run records every 100 steps.

## 12. Build the simulation inputs

```bash
python setup.py
```

`setup.py` reads `config.xml` and writes five files into the run folder, or into `work_dir` (step 3). A blank key of steps 3 to 5, a key of another source or mode and an unknown key stop the build before any file is written. `setup.py` then builds `receptor.pqr` and `ligand.pqr` (step 3), compares `ligand.pqr` with `ligand_formula` (step 4) and picks the reaction pairs (step 5). A missing key of steps 6 to 11 stops the build before `input.xml`. For the trypsin benzamidine example `setup.py` prints five lines.

```
  receptor.pqr: 3220 atoms, Q=+6.0 e
  ligand.pqr: 18 atoms, Q=+1.0 e
  composition verified: C7 H9 N2
  6 pairs, n_needed 3
  rmax=28.4 lmax=3.7 b=45 fglen=96  WARNING, b-surface outside the grid
```

| line | content |
|---|---|
| `receptor.pqr`, `ligand.pqr` | atom count and net charge Q (e) of each PQR file |
| `composition verified` | `ligand.pqr` matches `ligand_formula` |
| `6 pairs, n_needed 3` | pairs in `rxns.xml` and `n_needed` |
| `rmax`, `lmax` | largest distance of a receptor atom and of a ligand atom from the centroid of the molecule (Å) |
| `b`, `fglen` | `bd_milestone_radius` and the edge of the fine grid (Å) |

The last line ends with OK for b + lmax < fglen/2 and with WARNING for b + lmax ≥ fglen/2 (step 7). The warning does not stop the build. For `apbs_fglen` 0 `setup.py` derives the fine edge as the run does (step 7). Other structure sources and mode `site` print other lines.

| file | content |
|---|---|
| `receptor.pqr` | coordinates, charges and radii of the receptor atoms |
| `ligand.pqr` | coordinates, charges and radii of the ligand atoms |
| `rxns.xml` | one reaction with the atom pairs, the reaction distances and `n_needed` |
| `input.xml` | settings of the run with the keys of steps 6 to 11 and the three file names above |
| `build_record.json` | criterion, pairs and MD5 checksums of the four files above |

A complete build deletes the intermediate files of the structure step, for example `protein.pdb`, `protein.prmtop`, `ligand.mol2` and `leap.log`. User files with the same names are deleted too. A rerun of `setup.py` repeats the whole build and overwrites the five files. An edit of `input.xml` is then lost. `setup.py` never changes `bd_sims`.

## 13. Run the simulation

| command | GPUs | actions |
|---|---|---|
| `bash run.sh` | one GPU of the current node | deletes `bd_sims`, runs `setup.py` and `run_pystarc.py input.xml` |
| `sbatch submit_SLURM_single_GPU.sh` | one GPU of a SLURM node | deletes `bd_sims` and files of the last build, runs `setup.py` and `run_pystarc.py input.xml` |
| `sbatch submit_SLURM_multi_GPUs.sh` | four GPUs of a SLURM node | deletes `bd_sims` and files of the last build, runs `setup.py`, splits the trajectories over four GPUs and pools the results |

Submit the SLURM scripts from the run folder. The `#SBATCH` lines need the partition, memory and time limit of the local cluster. Every script deletes `bd_sims` with every earlier result and checkpoint. The SLURM scripts also delete intermediate files such as `protein.prmtop` from the run folder, user files with the same names included.

The four GPU script builds the grids once in `bd_sims` and starts four runs in `bd_sims/bd_1` to `bd_sims/bd_4`. Each run takes a quarter of `n_trajectories` with a seed of its own (step 9). `pystarc/multi_GPU/combine_data.py` then pools the counts of the finished runs into `bd_sims/results.json` and `bd_sims/convergence.json`. A run without `results.json` stays out of the pool.

### Start without the scripts

A run from a checkpoint or with an edited `input.xml` starts `run_pystarc.py` directly in the run folder and keeps `bd_sims`.

```bash
python /path/to/PySTARC/run_pystarc.py input.xml
```

The run resumes from `bd_sims/checkpoint.npz` when present (step 9). For an edited `input.xml` run `python setup.py` first and edit `input.xml` before the run. The run reuses the grids and hydrodynamic radii of `bd_sims` for unchanged settings and PQR files. A four GPU run resumes with the same command in each `bd_sims/bd_i`. `python /path/to/PySTARC/pystarc/multi_GPU/combine_data.py` in the run folder then pools the four runs.

## 14. Results and output files

`run_pystarc.py` writes every output into `bd_sims`. Each start writes a log `bd_sims/pystarc_<date>_<time>.log` with the printout of the run. The log ends with the run summary and, with `convergence_check` `true`, the convergence analysis. An error during the run ends the log with the error message. An invalid entry of `input.xml` stops the run before the log starts. A four GPU run keeps the logs in `bd_sims/bd_1` to `bd_sims/bd_4`. The expected results of the trypsin benzamidine example are in `examples/trypsin_benzamidine/README.md`.

### results.json

`bd_sims/results.json` holds k<sub>on</sub> and the counts of the run. After a four GPU run `bd_sims/results.json` holds the pool of the four runs.

| field | content |
|---|---|
| `k_on` | association rate constant 6.022×10<sup>8</sup> k<sub>b</sub> P<sub>rxn</sub> (M⁻¹ s⁻¹) |
| `k_on_low`, `k_on_high` | Wilson interval of k<sub>on</sub> at `confidence_level` (M⁻¹ s⁻¹) |
| `P_rxn` | reaction probability, `n_reacted` over `n_trajectories` |
| `P_rxn_low`, `P_rxn_high` | Wilson interval of P<sub>rxn</sub> at `confidence_level` |
| `P_rxn_bound_low`, `P_rxn_bound_high` | P<sub>rxn</sub> with the trajectories at `max_steps` counted as not reacted and as reacted |
| `k_on_bound_low`, `k_on_bound_high` | k<sub>on</sub> with the trajectories at `max_steps` counted as not reacted and as reacted (M⁻¹ s⁻¹) |
| `k_b` | rate constant of diffusion from infinite separation to the start sphere (Å³/ps) |
| `D_rel` | relative translational diffusion coefficient D<sub>rec</sub> + D<sub>lig</sub> at infinite separation (Å²/ps) |
| `n_trajectories` | trajectories of the run |
| `n_reacted`, `n_escaped`, `n_max_steps` | trajectories ended by reaction, by escape and at `max_steps` |
| `confidence_level` | `confidence_interval` of `input.xml`, default 0.95 (section Keys of input.xml only) |
| `r_start`, `r_escape` | radius b of the start sphere and radius of the escape sphere (Å) |
| `log10_k_on` | base 10 logarithm of `k_on`, written for `k_on` above 0 |
| `convergence_tol` | `convergence_tol` of step 9 |
| `wall_time_sec`, `steps_per_sec`, `total_steps` | wall time of the BD simulation (s), BD steps per second and BD steps of all trajectories |
| `contact_total_steps` | steps counted for `contact_frequency.csv` |
| `k_on_units`, `k_b_units`, `D_rel_units` | unit labels |
| `completed_reactions` | firing count of each reaction with `state_machine_reactions` `true` |

The factor 6.022×10<sup>8</sup> converts Å³/ps into M⁻¹ s⁻¹. A trajectory at `max_steps` counts as not reacted in P<sub>rxn</sub>. The low bounds therefore repeat P<sub>rxn</sub> and k<sub>on</sub>. The high bounds count every trajectory at `max_steps` as reacted (step 9). The Wilson intervals use z = 1.96 at the confidence level 0.95. The pooled file of a four GPU run has no `total_steps` and no `contact_total_steps`. The pooled file gives `wall_time_sec` of the slowest run and `aggregate_gpu_sec`, the summed wall time of the pooled runs (s).

### convergence.json

`run_pystarc.py` writes `bd_sims/convergence.json` with `convergence_check` `true` (step 9). `combine_data.py` writes the pooled `convergence.json` of a four GPU run.

| field | content |
|---|---|
| `N`, `n_reacted`, `n_escaped`, `n_max_steps` | trajectory counts, no `n_max_steps` in the pooled file |
| `P_rxn`, `SE` | reaction probability and standard error √(P<sub>rxn</sub>(1 − P<sub>rxn</sub>)/N) |
| `relative_SE`, `relative_SE_pct` | SE/P<sub>rxn</sub> as a fraction and in percent |
| `k_on`, `SE_kon` | k<sub>on</sub> and standard error of k<sub>on</sub> (M⁻¹ s⁻¹) |
| `wilson_CI`, `wilson_CI_P` | Wilson interval of k<sub>on</sub> (M⁻¹ s⁻¹) and of P<sub>rxn</sub> |
| `converged` | `true` for 0 < P<sub>rxn</sub> < 1 and `relative_SE` below `convergence_tol` |
| `confidence_level`, `tol`, `tol_pct` | confidence level and `convergence_tol` as a fraction and in percent |
| `N_needed` | trajectories (1 − P<sub>rxn</sub>)/(P<sub>rxn</sub> r²) for a relative SE r of 10%, 5% and 1% |

### Files in bd_sims

| file | content |
|---|---|
| `pystarc_<date>_<time>.log` | printout of one start of `run_pystarc.py` |
| `receptor.pqr`, `ligand.pqr` | copies centered on the atom centroid, with a GHO atom at the centroid |
| `receptor0.dx`, `receptor1.dx` | coarse and fine electrostatic grids of the receptor (step 7) |
| `receptor0_born.dx`, `receptor1_born.dx` | coarse and fine desolvation grids of the receptor |
| `ligand0.dx`, `ligand1.dx`, `ligand0_born.dx`, `ligand1_born.dx` | electrostatic and desolvation grids of the ligand |
| `receptor_grids.key`, `ligand_grids.key` | fingerprints of the PQR file and the grid settings for the reuse of the grids |
| `receptor.pqr.r_hydro_*.cache`, `ligand.pqr.r_hydro_*.cache` | Monte Carlo hydrodynamic radius of each molecule (step 6) |
| `results.json` | rate constant and counts |
| `convergence.json` | statistics of P<sub>rxn</sub> |
| `trajectories.csv` | outcome, step count, start pose and closest approach of each trajectory |
| `encounters.csv` | ligand position and orientation at each reaction |
| `near_misses.csv` | ligand position and orientation at the closest approach of each escaped trajectory |
| `bd_time_to_reaction.csv` | BD time inside the escape sphere of each reacted trajectory |
| `contact_frequency.csv` | steps and time of each reaction pair inside the pair cutoff |
| `pose_clusters.csv` | rotation angle of each reaction pose in up to 8 bins, written for more than 5 reactions |
| `radial_density.csv` | steps and time per radial shell |
| `angular_map.npz` | steps and time per angular bin |
| `milestone_flux.csv` | outward and inward crossings of 11 shells from b to the escape sphere |
| `transition_matrix.npz` | moves between 50 radial bins |
| `paths.npz`, `energetics.npz` | records every `save_interval` steps (step 11) |
| `checkpoint.npz` | state of a run with `checkpoint_interval` above 0, deleted at the end |
| `split.json`, `bd_1` to `bd_4` | split of a four GPU run |

A four GPU run writes each run into `bd_sims/bd_1` to `bd_sims/bd_4` with a separate `input.xml`, log, `results.json` and output files. `bd_sims` holds the grids, `split.json` and the pooled `results.json`, `convergence.json` and output files.


## Keys of input.xml only

`config.xml` refuses the keys below, and `setup.py` stops with an error naming the key. Set the keys in `input.xml` after `setup.py` and start `run_pystarc.py` directly (step 13). `run.sh` and the SLURM scripts run `setup.py` again and overwrite `input.xml`. Top level keys go into `<pystarc>` and the file switches into `<outputs>`. The lines below set an escape sphere of 100 Å, a confidence level of 0.99 and no `near_misses.csv`.

```xml
  <r_escape>100.0</r_escape>
  <confidence_interval>0.99</confidence_interval>
  <outputs>
    <save_interval>100</save_interval>
    <full_paths>true</full_paths>
    <energetics>true</energetics>
    <near_misses_csv>false</near_misses_csv>
  </outputs>
```

| key | entries | effect |
|---|---|---|
| `r_escape` | `0` for 2b, or a radius above 1.1 b (Å), default `0` | radius of the escape sphere, where a trajectory returns to the start sphere or escapes |
| `apbs_ligand_dime` | 4k + 1 with k ≥ 1, default 129 | grid points per axis of the ligand grids |
| `brownian_bridge` | `true` or `false`, default `true` | `true` also detects reactions between two steps, `false` tests the end of each step only |
| `confidence_interval` | number between 0 and 1, default 0.95 | confidence level of the Wilson intervals |
| `state_machine_reactions` | `true` or `false`, default `false` | `true` fires each one pair reaction of `rxns.xml` from `state_before` to `state_after`, `false` reads one reaction with every pair |
| `relative_viscosity` | `1` only, default `1` | viscosity relative to water at 25 °C |
| `trajectories_csv`, `encounters_csv`, `near_misses_csv`, `radial_density`, `angular_map`, `bd_time_to_reaction`, `contact_frequency`, `milestone_flux`, `transition_matrix`, `pose_clusters` | `true` or `false`, default `true` | `false` skips the file of the same name in `bd_sims` |
| `results_json` | `true` or `false`, default `true` | `false` skips `results.json`, sets k<sub>on</sub> to 0 in `convergence.json` and stops the pooling of a four GPU run |
| `receptor_pqr`, `ligand_pqr` | PQR file, default blank | receptor and ligand of the run, blank for the PDB route |
| `rxns_xml` | reaction file, default blank | reaction criterion, blank for `ghost_atoms` |
| `ghost_atoms` | `auto` or lines `i,j,d`, default `auto` | reaction criterion for a blank `rxns_xml` |
| `bd_milestone_radius_inner` | distance between 0 and b (Å), default 0 | reaction distance of the GHO pair for `ghost_atoms` `auto` |
| `pdb` | PDB file, default blank | complex of the PDB route |
| `ligand_charge` | integer, default 0 | net charge of the ligand in the PDB route |
| `protein_ff` | `leaprc.protein` suffix of AmberTools, default `ff14SB` | protein force field of the PDB route |
| `ligand_ff` | `gaff` or `gaff2`, default `gaff` | ligand force field of the PDB route |

`setup.py` writes `receptor.pqr`, `ligand.pqr`, `rxns.xml` and `bd_sims` into `receptor_pqr`, `ligand_pqr`, `rxns_xml` and `work_dir`. The `work_dir` of `input.xml` is the output folder of the run, default `bd_sims`, and differs from the `work_dir` of `config.xml` (step 3). `fpt_distribution` is an alias of `bd_time_to_reaction`.

`ghost_atoms` lines name receptor atom i and ligand atom j, numbered from 0 in the PQR files of `bd_sims`, and the reaction distance d (Å). A trajectory reacts when every pair is closer than the pair distance d at one time. `ghost_atoms` `auto` pairs the two GHO atoms at the distance `bd_milestone_radius_inner`.

The PDB route builds `receptor.pqr` and `ligand.pqr` in `bd_sims` from `pdb` when `receptor_pqr` or `ligand_pqr` is blank. antechamber assigns AM1 BCC charges to the residue `ligand_resname` with the net charge `ligand_charge`, and tleap builds the complex with `protein_ff` and `ligand_ff`. The `input.xml` of `setup.py` names both PQR files and `rxns.xml`. `pdb`, `ligand_charge`, `protein_ff`, `ligand_ff`, `ghost_atoms` and `bd_milestone_radius_inner` therefore have no effect with the `input.xml` of `setup.py`. A named `pdb` file must exist.

The `rxns.xml` of `setup.py` holds one reaction with every pair. A GPU run with `state_machine_reactions` `true` needs one pair per reaction, for example from mode `centroid` or `site`. A `<chain>` block selects the flexible chain model of `examples/barnase_barstar`. An unknown tag stops the run with an error naming the tag. Retired tags and tags read only by `prepare_bd_surface`, for example `stall_check` and `parm7`, give a warning and have no effect.

## 15. Flexible chain model (barnase and barstar)

The flexible chain model treats the ligand as a chain of coarse grained COFFDROP beads with internal motion and the receptor as a rigid body. The chain model runs on CPUs and needs no GPU. The example `examples/barnase_barstar` has a separate `config.xml` with the four sections structure, names, criterion and run.

```xml
<config>
  <structure>
    <source_pdb>1BRS.pdb</source_pdb>
    <receptor_chain_id>A</receptor_chain_id>
    <ligand_chain_id>D</ligand_chain_id>
    <receptor_pqr>barnase.pqr</receptor_pqr>
    <target_grid_dx>apbs_output/barnase1.dx</target_grid_dx>
    <born_grid_dx>apbs_output/barnase0_born.dx</born_grid_dx>
  </structure>
  <names>
    <chain_name>barstar</chain_name>
    <chain_json>chain.json</chain_json>
    <reaction_pairs_json>reaction_pairs.json</reaction_pairs_json>
    <input_xml>input.xml</input_xml>
    <work_dir>bd_sims</work_dir>
  </names>
  <criterion>
    <interface_cutoff_a>6.0</interface_cutoff_a>
    <reaction_distance_a>8.0</reaction_distance_a>
    <min_reaction_pairs>5</min_reaction_pairs>
    <reaction_n_needed>3</reaction_n_needed>
  </criterion>
  <run>
    <bd_milestone_radius>60.0</bd_milestone_radius>
    <n_trajectories>200</n_trajectories>
    <max_steps>500000</max_steps>
    <dt>0.2</dt>
    <dt_chain>0.025</dt_chain>
    <chain_steps_per_outer>8</chain_steps_per_outer>
    <n_equilibration_steps>0</n_equilibration_steps>
    <r_escape>160.0</r_escape>
    <temperature>298.15</temperature>
    <seed>1</seed>
    <gpu>false</gpu>
    <desolvation_alpha>1.0</desolvation_alpha>
    <save_interval>10</save_interval>
    <convergence_check>false</convergence_check>
    <convergence_interval>10</convergence_interval>
    <convergence_tol>0.05</convergence_tol>
    <auto_diffusion>true</auto_diffusion>
    <d_trans>0.015</d_trans>
    <d_rot>0.00005</d_rot>
    <use_soft_repulsion>true</use_soft_repulsion>
    <soft_repulsion_eps>1.0</soft_repulsion_eps>
    <n_workers>96</n_workers>
    <n_shards>25</n_shards>
    <apbs_fglen>140.0</apbs_fglen>
    <apbs_dime>449</apbs_dime>
  </run>
</config>
```

Four commands run the example in order.

```bash
cd examples/barnase_barstar
python setup.py
python make_grids.py
./submit_shards.sh
python combine_shards.py
```

`setup.py` moves the centroid of `barnase.pqr` to the origin in place. `setup.py` builds barstar, chain D of `1BRS.pdb`, as 230 COFFDROP beads for 87 residues. `chain.json` holds the beads, the bonded terms and the bead pairs of the COFFDROP pair tables. `chain.json` names the four COFFDROP data files in `pystarc/coffdrop_data` by path and checksum. The heavy atom contacts of chains A and D closer than `interface_cutoff_a` turn into the 568 receptor atom and bead pairs of `reaction_pairs.json`. `setup.py` then writes `input.xml` with a `<chain>` block.

`make_grids.py` runs APBS on `barnase.pqr` and writes the potential grids `barnase0.dx` (coarse) and `barnase1.dx` (fine) into `apbs_output`. `make_grids.py` also writes the Born desolvation grids `barnase0_born.dx` and `barnase1_born.dx`. `README.md` submits `make_grids.py` as a SLURM job with 128 GB of memory.

`submit_shards.sh` makes the `n_shards` folders `shards/shard_01` to `shards/shard_25` with links to `chain.json`, `reaction_pairs.json`, `barnase.pqr` and `apbs_output`. Each shard folder gets a copy of `input.xml` with the seed 11111111 i + 1 for shard i. One SLURM job per shard runs `run_pystarc.py input.xml` in the shard folder on 96 CPUs. The run writes `results.json`, `trajectories.csv`, the saved paths and a log into `bd_sims` of the shard. The partition, CPUs, memory and time limit of the jobs are set in `submit_shards.sh`.

`combine_shards.py` reads the `results.json` of every finished shard and prints the counts per shard. `combine_shards.py` then prints the pooled k<sub>on</sub> = k<sub>b</sub> P<sub>rxn</sub> with a Wilson 95% interval. P<sub>rxn</sub> = n<sub>reacted</sub>/(n<sub>reacted</sub> + n<sub>escaped</sub>) with the counts summed over the shards. The trajectories at `max_steps` count as not reacted in each `results.json` and stay out of the pooled P<sub>rxn</sub>.

Each trajectory starts with the chain centroid on the sphere of radius b = `bd_milestone_radius` around the receptor centroid in a random orientation. Each outer step of `dt` moves the chain as a rigid body. `chain_steps_per_outer` internal steps of `dt_chain` move the beads within the chain during the outer step. Beyond 1.1 b an outer propagator returns the chain to the b sphere or ends the trajectory as escaped.

`setup.py` requires every key except `n_shards` and ignores any other key. `setup.py` writes no salt, dielectric or hydrodynamic key into `input.xml`. `make_grids.py` and the chain run take the parser values `ion_concentration` 0.15 M, `ion_radius_pos` 0.95 Å, `ion_radius_neg` 1.81 Å, `debye_length` 7.858 Å, `pdie` 4, `sdie` 78, `srad` 1.5 Å and `r_hydro_rec` 0 (steps 6 and 7). `apbs_cglen`, `apbs_coarse_dime` and `apbs_fine_dime` stay at 0 (step 7). `hydrodynamic_interactions`, `brownian_bridge` and `use_hard_sphere` (rejection of bead and receptor atom overlaps) are `true`.

| key | entries | effect |
|---|---|---|
| `source_pdb` | PDB file of the bound complex | chain sequence, bead positions and interface contacts |
| `receptor_chain_id` | one chain identifier of `source_pdb` | receptor chain, checked residue by residue against `receptor_pqr` |
| `ligand_chain_id` | one chain identifier of `source_pdb` | chain built as the bead model |
| `receptor_pqr` | PQR file of the receptor chain without chain identifiers | charges and radii of the rigid receptor, moved to the origin in place by `setup.py` |
| `target_grid_dx` | `apbs_output/<name>1.dx`, blank for no field | APBS potential on the bead charges, no field outside the grid |
| `born_grid_dx` | `apbs_output/<name>0_born.dx` with `target_grid_dx` set, blank for none | Born desolvation energy of the bead charges, scaled by `desolvation_alpha` |
| `chain_name` | any text | label in `chain.json` and `results.json`, no effect on the simulation |
| `chain_json`, `reaction_pairs_json`, `input_xml` | file names | bead model, reaction pairs and run input written by `setup.py` |
| `work_dir` | folder name | output folder of the run in each shard |
| `interface_cutoff_a` | distance (Å) | heavy atom contacts of the two chains in `source_pdb` turned into reaction pairs |
| `reaction_distance_a` | distance (Å) | reaction distance of every pair |
| `min_reaction_pairs` | integer | `setup.py` stops with fewer pairs |
| `reaction_n_needed` | integer from 1 to the number of pairs | pairs closer than `reaction_distance_a` at one time for a reaction |
| `bd_milestone_radius` | radius b (Å) | start sphere of the chain centroid, outer propagator beyond 1.1 b |
| `n_trajectories`, `max_steps`, `desolvation_alpha`, `apbs_fglen`, `apbs_dime` | as in steps 7 to 9 | same as in steps 7 to 9, `max_steps` counts outer steps |
| `dt` | `chain_steps_per_outer` × `dt_chain` (ps) | outer step, 0.05 ps while a pair is closer than 1.5 × `reaction_distance_a` |
| `dt_chain` | time > 0 (ps) | internal step of the beads |
| `chain_steps_per_outer` | integer ≥ 1 | internal steps per outer step of `dt` |
| `n_equilibration_steps` | integer ≥ 0 | internal steps without receptor forces before each trajectory, 0 for none |
| `r_escape` | 0 for 2b or a radius (Å) | escape radius only when the outer propagator fails to build |
| `temperature` | 298.15 (K) | an entry more than 0.01 K away stops the chain run |
| `seed` | integer ≥ 0 | written to `input.xml`, replaced by 11111111 i + 1 in shard i |
| `gpu` | `true` or `false` | `true` runs the active-slot GPU chain simulator (`GPUChainBDSimulator`), `false` runs the CPU worker simulator (`ChainBDSimulator`) |
| `convergence_check`, `convergence_interval`, `convergence_tol` | as in steps 9 and 10 | live progress milestones on GPU and post-run `convergence.json` report |
| `flexible_partner` | `ligand` (default) or `receptor` | inside `<chain>`, selects which binding partner is modeled as the flexible COFFDROP chain while the other is the centered rigid PQR target (`receptor_pqr` or `ligand_pqr`) |
| `coarse_grid_dx`, `far_field` | optional DX path, `true` or `false` | inside `<chain>`, adds a coarse outer APBS grid and screened Yukawa/multipole far-field continuation on the GPU |
| `save_interval` | integer ≥ 0 | outer steps between saved positions, energies and contacts, 0 for none |
| `auto_diffusion` | `true` or `false` | `true` for chain diffusion tensors from the beads rebuilt every 10 outer steps, `false` for `d_trans` and `d_rot` |
| `d_trans`, `d_rot` | > 0 (Å<sup>2</sup>/ps), > 0 (rad<sup>2</sup>/ps) | chain diffusion coefficients with `auto_diffusion` `false`, ignored with `true` |
| `use_soft_repulsion` | `true` or `false` | repulsion between beads and receptor atoms and between bead pairs with no bond, angle or COFFDROP table |
| `soft_repulsion_eps` | energy ≥ 0 (kT) | ε of 4ε[(σ/r)<sup>12</sup> − (σ/r)<sup>6</sup>] + ε for r < 2<sup>1/6</sup>σ, σ the sum of the radii |
| `n_workers` | integer ≥ 1 | worker processes per shard, same results for any count |
| `n_shards` | integer ≥ 1 | SLURM jobs of `n_trajectories` each, read by `submit_shards.sh` only |

`setup.py` and `submit_shards.sh` find PySTARC in a parent folder of the example folder. `make_grids.py`, `submit_shards.sh` and `combine_shards.py` use the example names `input.xml`, `chain.json`, `reaction_pairs.json`, `barnase.pqr`, `apbs_output` and `bd_sims`.
