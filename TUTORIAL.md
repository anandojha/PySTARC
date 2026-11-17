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

`r_hydro_rec` and `r_hydro_lig` set the hydrodynamic radii a of the receptor and the ligand (Å). The radii fix the diffusion coefficients D<sub>t</sub> = k<sub>B</sub>T/(6πηa) and D<sub>r</sub> = k<sub>B</sub>T/(8πηa³) with the viscosity η = 0.890 mPa s of water at 25 °C. An entry of 0 lets PySTARC compute a by the Monte Carlo method of Hansen (J. Chem. Phys. 121, 9111, 2004) from the PQR file, with every atom radius enlarged by `srad`. The calculation runs inside PySTARC (`pystarc/hydrodynamics/mc_hydro_radius.py`) and needs no other software. The run log prints both radii, and a cache file next to each PQR file in `bd_sims` stores the radius. A positive entry fixes the radius. The trypsin benzamidine example fixes the radii at 22.3728 Å and 4.9867 Å, computed with BrownDye 2 for a comparison of PySTARC with BrownDye 2. An entry of 0 gives 22.30 Å and 4.90 Å for the same PQR files. Hydrodynamic interactions stay on in both cases. `hydrodynamic_interactions` (step 8) alone switches the hydrodynamic interactions on and off.

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
