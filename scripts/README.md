# Standalone Pre-Configuration Scripts (`scripts/`)

`scripts/` holds two standalone command-line tools for preparing starting structures and reaction criteria for any molecular complex before filling `config.xml`, together with per-complex preparation folders for all 33 benchmark complexes across the 9 families in `examples/`. Neither script is imported or invoked by PySTARC at runtime.

```
scripts/
├── prepare_pdb.py                   Prepares receptor + ligand .pdb or .pqr files and prints <structure>
├── find_reaction_criteria.py        Diagnoses pocket geometry, verifies rigid clearance, and prints <criterion> (or reaction_pairs.json)
├── verify_criteria.py               Audits all 32 rigid prepare.sh scripts against examples/*/config.xml (0 mismatches)
├── prepare_all.sh                   Runs every complex prepare.sh across all 33 benchmark complexes
├── README.md                        Documentation, input inventory, and workflow commands
├── two_charged_spheres/             prepare.sh (analytical charged spheres, zero raw files needed)
├── trypsin_benzamidine/             complex.prmtop, complex.pdb, prepare.sh
├── beta_cyclodextrin_guests/        7 guest folders (complex.parm7, complex.pdb, prepare.sh) + prepare_all.sh
├── thrombin_thrombomodulin/         t.pqr, m.pqr, prepare.sh
├── p38_mapk_sb203580/               1A9U.pdb, prepare.sh
├── carbonic_anhydrase_inhibitors/   7 complex folders (3CZV/2NMX/3HS4.pdb, prepare.sh) + prepare_all.sh
├── hsp90_inhibitors/                6 complex folders (2YKJ/5LNZ/6EI5.pdb or complex.pdb, prepare.sh) + prepare_all.sh
├── ttk_inhibitors/                  8 complex folders (2X9E..5NAD.pdb, prepare.sh) + prepare_all.sh
└── barnase_barstar/                 1BRS.pdb, prepare.sh
```

---

## 1. Raw Inputs Needed and Handoff to `config.xml`

Each complex folder under `scripts/` stores the minimal raw starting inputs required for that complex plus a `prepare.sh` wrapper calling `prepare_pdb.py` and `find_reaction_criteria.py`.

| Family (count) | Minimal raw inputs in `scripts/<complex>/` | Preparation workflow | Outputs copied to `examples/<complex>/` |
|---|---|---|---|
| `two_charged_spheres` (1) | None (charge and radius parameters `1.0:1.0,-1.0:1.0`) | `prepare_pdb.py --sphere-pair` + `find_reaction_criteria.py --mode centroid` | `receptor.pqr`, `ligand.pqr`, `<structure>` + `<criterion>` |
| `trypsin_benzamidine` (1) | AMBER `complex.prmtop` + `complex.pdb` | `prepare_pdb.py --prmtop --ligand BEN` + `find_reaction_criteria.py --atom-filter polar` | `receptor.pqr`, `ligand.pqr`, `<structure>` + `<criterion>` |
| `beta_cyclodextrin_guests` (7) | AMBER `complex.parm7` + `complex.pdb` | `prepare_pdb.py --prmtop --ligand APN` + `find_reaction_criteria.py --mode centroid` | `receptor.pqr`, `ligand.pqr`, `<structure>` + `<criterion>` |
| `thrombin_thrombomodulin` (1) | Pre-parameterized BrownDye2 `t.pqr` + `m.pqr` | `prepare_pdb.py --receptor-pqr --ligand-pqr` + `find_reaction_criteria.py --atom-filter polar` | `receptor.pqr`, `ligand.pqr`, `<structure>` + `<criterion>` |
| `p38_mapk_sb203580` (1) | Raw crystal `1A9U.pdb` (or `--pdb-id 1A9U`) | `prepare_pdb.py --chain A --resid-min 4 --ligand SB2` + `find_reaction_criteria.py --receptor-targets` | `receptor.pdb`, `ligand.pdb`, `<structure>` + `<criterion>` |
| `carbonic_anhydrase_inhibitors` (7) | Raw crystal `3CZV.pdb`, `2NMX.pdb`, or `3HS4.pdb` (+ ligand SMILES for non-crystal VD compounds) | `prepare_pdb.py --keep-hetatm ZN --metal-his` (`--ligand AZM --deprotonate N1` or `--smiles` + `--pocket-ligand`) + `find_reaction_criteria.py --receptor-targets` | `receptor.pdb`, `ligand.pdb`, `<structure>` + `<criterion>` |
| `hsp90_inhibitors` (6) | Raw crystal `2YKJ.pdb`, `5LNZ.pdb`, `6EI5.pdb` (3 complexes) or equilibrated `complex.pdb` (3 complexes) | `prepare_pdb.py --ligand` (or `--split-complex --ligand UNK`) + `find_reaction_criteria.py --receptor-targets` | `receptor.pdb`, `ligand.pdb`, `<structure>` + `<criterion>` |
| `ttk_inhibitors` (8) | Raw crystal `2X9E.pdb`..`5NAD.pdb` (plus `--protonate N32` on `2X9E`, `--mutate 604:CYS` on `5LJJ`) | `prepare_pdb.py --chain A --resid-min 515/516 --ligand <CCD>` + `find_reaction_criteria.py --receptor-targets` | `receptor.pdb`, `ligand.pdb`, `<structure>` + `<criterion>` |
| `barnase_barstar` (1) | Raw crystal `1BRS.pdb` | `prepare_pdb.py --chain A --ligand-chain D --model chain` + `find_reaction_criteria.py --chain-pairs-out` | `receptor.pdb`, `ligand.pdb`, `reaction_pairs.json`, `<structure>` + `<criterion>` |

### Running Per-Complex Preparation

Run a single complex, a whole family, or all 33 complexes from inside the `PySTARC` conda environment:

```bash
# Prepare one complex into scripts/trypsin_benzamidine/prepared/
bash scripts/trypsin_benzamidine/prepare.sh

# Prepare one complex directly into a custom run folder
bash scripts/trypsin_benzamidine/prepare.sh /path/to/my_run_dir

# Prepare an entire family
bash scripts/ttk_inhibitors/prepare_all.sh

# Prepare all 33 benchmark complexes
bash scripts/prepare_all.sh
```

Every `prepare.sh` writes into `prepared/` (or the custom output directory passed as the first argument):
1. `receptor.pdb` + `ligand.pdb` (or `receptor.pqr` + `ligand.pqr`)
2. `reaction_pairs.json` (for flexible-chain `barnase_barstar`)
3. `config_snippet.xml` holding the exact `<structure>` and `<criterion>` blocks ready to paste into `config.xml`.

Both `prepare_pdb.py` and `find_reaction_criteria.py` write diagnostic summaries to `stderr` and clean XML to `stdout`.

---

## 2. `prepare_pdb.py` Solute Structure Preparation

Produces matching `receptor.pdb` + `ligand.pdb` (or `receptor.pqr` + `ligand.pqr`) files for single-chain, multi-chain, AMBER `prmtop`, pre-parameterized PQR, or analytical sphere complexes, assigns protonation states and explicit 3D ligand hydrogens, and prints the `<structure>` block (including integer `<ligand_net_charge>`) to `stdout`.

### Supported Inputs and Options

| Option | Purpose |
|---|---|
| `--pdb <PATH>` / `--pdb-id <ID>` | Read a local PDB file or download `<ID>.pdb` from RCSB |
| `--chain <IDS>` | Comma-separated receptor chain IDs (for example `A` or `H,L`), inserting `TER` between chains (blank keeps all chains) |
| `--ligand-chain <IDS>` | Comma-separated protein or peptide ligand chain IDs extracted into `--ligand-out` |
| `--resid-min`, `--resid-max` | Receptor residue number window to retain |
| `--altloc`, `--ligand-altloc` | Preferred alternate conformer indicator (default `A`) |
| `--mse-to-met` | Convert crystallographic selenomethionine (`HETATM MSE`) to `ATOM MET` |
| `--mutate <SPECS>` | Point mutations (`RESID:RESNAME` or `CHAIN:RESID:RESNAME`) with `tleap` sidechain heavy-atom completion when installed |
| `--hid`, `--hie`, `--hip`, `--metal-his` | Explicit histidine tautomers (`RESID` or `CHAIN:RESID`), or automatic `HID`/`HIE` on metal-coordinating histidines |
| `--cyx`, `--auto-cyx` | Explicit disulfide cysteines (`CYX`), or automatic flagging of `SG-SG <= 2.2 Å` |
| `--keep-hetatm <NAMES>` | Retain structural metal ions or cofactors (`ZN`, `MG`, `MN`, `FE`, `CU`, `CA`) with the receptor |
| `--ligand <RESNAME>` | Extract crystal or docked `HETATM` ligand by 3-letter residue name (`--ligand-resid` selects one copy, `--ccd`/`--cif` selects CCD bond orders) |
| `--ligand-file <PATH>` | Read an external 3D or docked ligand from `.sdf`, `.mol`, `.mol2`, or `.pdb` |
| `--ligand-smiles <SMILES>` | Assign bond orders and formal charges onto a non-CCD docked ligand from a SMILES template |
| `--ligand-charge <INT>` | Assign bond orders on a non-CCD `.pdb` ligand with explicit hydrogens, or set `<ligand_net_charge>` for `--split-complex` |
| `--protonate`, `--deprotonate` | Comma-separated ligand heavy-atom names gaining or losing one proton |
| `--smiles <SMILES>` | Embed and MMFF-optimize a 3D ligand conformer from SMILES (optional `--pocket-ligand` places its centroid in the pocket) |
| `--formula <FORMULA>` | Optional element-count check on the prepared ligand (for example `"C16 H22 F3 N2 O5 S2"`) |
| `--split-complex <PATH>` | Split an already-prepared complex `.pdb` into `receptor.pdb` + `ligand.pdb`, stripping bulk solvent/ions (`HOH`, `WAT`, `NA`, `CL`, `K`) |
| `--prmtop <PATH> --pdb <PATH>` | Export `receptor.pqr` + `ligand.pqr` from an AMBER topology (`--ligand <RESNAME>` or `--ligand-residues <START-END>`) |
| `--receptor-pqr <PATH> --ligand-pqr <PATH>` | Validate and export pre-parameterized receptor + ligand `.pqr` files and print `<structure>` |
| `--sphere-pair <Q1:R1,Q2:R2>` | Write analytical single-atom charged-sphere `receptor.pqr` + `ligand.pqr` (for example `1.0:1.0,-1.0:1.0`) |
| `--receptor-out`, `--ligand-out` | Output file paths (default `receptor.pdb`/`ligand.pdb`, or `.pqr` for PQR workflows) |

### Examples

```bash
# 1. Crystal protein-ligand complex from RCSB (downloads 1A9U.pdb and SB2.cif)
python scripts/prepare_pdb.py \
  --pdb-id 1A9U --chain A --resid-min 4 --ligand SB2 \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 2. Metalloprotein with Zn2+, coordinating His tautomers, and deprotonated ligand atom N1
python scripts/prepare_pdb.py \
  --pdb-id 3CZV --chain A --keep-hetatm ZN --metal-his \
  --ligand AZM --ligand-resid 263 --deprotonate N1 \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 3. Multi-chain receptor (chains H,L) + docked ligand from an external .sdf or .mol2 file
python scripts/prepare_pdb.py \
  --pdb antibody_raw.pdb --chain H,L --auto-cyx \
  --ligand-file docked_pose.sdf --ligand-resname LIG \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 4. Pre-equilibrated solvated complex PDB split into receptor.pdb + ligand.pdb
python scripts/prepare_pdb.py \
  --split-complex complex.pdb --ligand UNK --ligand-charge 0 \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 5. Protein-protein complex (rigid-body or flexible COFFDROP chain)
python scripts/prepare_pdb.py \
  --pdb 1BRS.pdb --chain A --ligand-chain D --model chain \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 6. Export receptor.pqr + ligand.pqr from an AMBER prmtop
python scripts/prepare_pdb.py \
  --prmtop complex.prmtop --pdb complex.pdb --ligand BEN \
  --receptor-out receptor.pqr --ligand-out ligand.pqr

# 7. Validate and export pre-parameterized PQR files
python scripts/prepare_pdb.py \
  --receptor-pqr t.pqr --ligand-pqr m.pqr \
  --receptor-out receptor.pqr --ligand-out ligand.pqr

# 8. Generate analytical charged-sphere PQR files
python scripts/prepare_pdb.py \
  --sphere-pair 1.0:1.0,-1.0:1.0 \
  --receptor-out receptor.pqr --ligand-out ligand.pqr
```

---

## 3. `find_reaction_criteria.py` Reaction Criterion Selection

Reads prepared `receptor` and `ligand` files (`.pdb + .pdb` or `.pqr + .pqr`), prints geometric diagnostics, contact distances, or centroid geometry to `stderr`, and prints a ready-to-paste `<criterion>` XML block to `stdout` (or writes `reaction_pairs.json` for `model = chain`).

### Default Three-Tier Structural Protocol (Zero Kinetic Fitting)

When configuring a brand-new complex without a measured $k_{\mathrm{on}}$, run `find_reaction_criteria.py --diagnose` to classify the complex from 3D coordinates alone into one of three structural tiers (with `n_needed = 3` fixed across all `3`-of-`6` contact criteria):

| Tier | Geometric Condition | Default Criterion Rule | Benchmark Complexes (`32` rigid) |
|---|---|---|---|
| **Tier 1 (`centroid`)** | `N_rec_heavy < 120` (symmetric sphere pair or macrocyclic host cavity) | `mode = centroid` (`2.5 Å` analytical sphere contact; `8.0 Å` host cavity capture) | `two_charged_spheres` (`1`), `beta_cyclodextrin_guests` (`7`) |
| **Tier 2 (`open-site polar`)** | Solid-angle pocket enclosure `f_enclosed < 0.70` or compact ligand `N_lig_heavy <= 15` | Automatic `3`-of-`6` shortest bound polar contacts (`N, O, S`, one per receptor residue, `search_cutoff = 6.0 Å`) at **`7.0 Å`** | `thrombin_thrombomodulin` (`1`), `trypsin_benzamidine` (`1`), `ca13_azm` (`1`) |
| **Tier 3 (`pocket-rim`)** | Deep enclosed pocket (`f_enclosed >= 0.70`) and bulky ligand (`N_lig_heavy > 15`) | `3`-of-`6` conserved solvent-exposed pocket-mouth rim contacts (`Cβ` mouth ring paired with the conserved pharmacophore anchor for single-scaffold pockets, or outer-mouth sidechain tips paired automatically via `--atom-filter all` with nearest bound heavy atoms across multi-scaffold series) | `CA-VD` (`6`, `7.0 Å`), `HSP90` (`6`, `7.5 Å`), `p38 MAPK` (`1`, `8.0 Å`), `TTK` (`8`, `9.0 Å`) |

For Tier 3 pockets, `pair_cutoff` starts at the universal default **`7.0 Å`** and steps up in `1.0 Å` increments (`8.0 Å`, `9.0 Å`) to the smallest whole integer that clears rigid-body steric exclusion (`--check-accessibility`) across every complex in the structural family. Across all `32` rigid complexes, running `python scripts/verify_criteria.py` confirms that every `scripts/*/prepare.sh` reproduces the exact `<criterion>` block in `examples/*/config.xml` (`0` mismatches). Flexible-chain `barnase_barstar` (`model = chain`) constructs `reaction_pairs.json` independently via `--chain-pairs-out`.

### Supported Options

| Option | Purpose |
|---|---|
| `--receptor`, `--ligand` | Prepared receptor and ligand `.pdb` or `.pqr` paths |
| `--mode contacts\|centroid` | Reaction criterion mode (default `contacts`) |
| `--atom-filter <KIND>` | `polar` (`N, O, S`), `hetero` (any non-C heavy atom), `all`, `nonpolar`, `any_polar`, or `receptor_polar` |
| `--elements <REC:LIG>` | Custom element filter overriding `--atom-filter` (for example `ZN,O,N:N,O`) |
| `--search-cutoff <Å>` | Bound-pose contact search cutoff in Å (default `6.0`) |
| `--n-pairs <INT>` | Maximum contact pairs retained (default `6`, trimmed to even count `2k` with `n_needed = k`, or `1`) |
| `--allow-same-residue` | Permit multiple contact pairs on the same receptor residue |
| `--exclude <NAMES>` | Comma-separated atom names excluded from contact picking |
| `--receptor-targets <SPECS>` | Explicit pocket-rim atoms (`RESID:ATOM:RESNAME` or `CHAIN:RESID:ATOM:RESNAME`) |
| `--ligand-atoms <SPECS>` | Ligand atom names (or `RESID:ATOM:RESNAME`) paired with `--receptor-targets` (omit to auto-pick closest partner) |
| `--pair-cutoff <Å>` | Reaction distance written into `<pair_cutoff>` (default `7.0`) |
| `--n-needed <INT>` | Override `<n_needed>` explicitly |
| `--diagnose` | Print solute heavy-atom counts, solid-angle pocket enclosure `f_enclosed`, and recommended structural tier |
| `--suggest-rim` | Rank solvent-exposed pocket-mouth rim residues by outward exit-vector projection and azimuthal spread |
| `--rim-atom <CB\|tip>` | Atom type evaluated by `--suggest-rim` (default `CB`) |
| `--check-accessibility` | Fast NumPy rigid-body steric clearance audit across `7.0`, `8.0`, and `9.0 Å` |
| `--chain-pairs-out <JSON>` | Write COFFDROP `[rec_pqr_idx, bead_idx, pair_cutoff]` JSON for `model = chain` |

### Examples

```bash
# 0. Diagnose a new complex from 3D coordinates and suggest pocket-rim targets if enclosed
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb --diagnose --suggest-rim

# 1. Tier 2 default: pick 6 shortest bound polar contacts (1 per receptor residue) at 7.0 A
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --atom-filter polar --search-cutoff 6.0 --n-pairs 6 --pair-cutoff 7.0 \
  --check-accessibility

# 2. Tier 3 multi-scaffold pocket: conserved outer-mouth targets paired automatically with nearest heavy atoms
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --receptor-targets 51:ND2:ASN,54:OD1:ASP,58:NZ:LYS,98:SD:MET,102:OD1:ASP,107:CG:LEU \
  --atom-filter all --pair-cutoff 7.5 --check-accessibility

# 3. Tier 3 conserved-pharmacophore pocket: mouth CB ring paired with the conserved anchor atom
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --receptor-targets 529:CB:LYS,531:CB:ILE,606:CB:ASN,611:CB:SER,651:CB:ALA,663:CB:ILE \
  --ligand-atoms N1 --pair-cutoff 9.0 --check-accessibility

# 4. Tier 1 centroid criterion (writes <n_needed>0</n_needed>)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pqr --ligand ligand.pqr \
  --mode centroid --pair-cutoff 8.0

# 5. Flexible-chain COFFDROP reaction_pairs.json (model = chain)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --search-cutoff 6.0 --pair-cutoff 8.0 --n-needed 3 \
  --chain-pairs-out reaction_pairs.json

# 6. Audit all 32 rigid benchmark scripts against examples/*/config.xml
python scripts/verify_criteria.py
```
