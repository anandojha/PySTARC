# Standalone Pre-Configuration Scripts (`scripts/`)

`scripts/` contains two optional standalone command-line tools for preparing solute structure files and selecting reaction criteria for any molecular complex before filling `config.xml`. Neither script is imported or invoked by PySTARC at runtime.

```
scripts/
├── prepare_pdb.py             Prepares separate receptor and ligand .pdb or .pqr files and prints <structure>
├── find_reaction_criteria.py  Inspects prepared structures and prints <criterion> (or reaction_pairs.json)
└── README.md                  Documentation and usage examples for both scripts
```

---

## 1. `prepare_pdb.py` — Solute Structure Preparation

Produces matching `receptor.pdb` + `ligand.pdb` (or `receptor.pqr` + `ligand.pqr`) files for single-chain or multi-chain complexes, assigns protonation states and explicit 3D ligand hydrogens, and prints the `<structure>` block (including integer `<ligand_net_charge>`) for `config.xml`.

### Supported inputs and options

| Option | Purpose |
|---|---|
| `--pdb <PATH>` / `--pdb-id <ID>` | Read a local PDB file or download `<ID>.pdb` from RCSB |
| `--chain <IDS>` | Comma-separated receptor chain IDs (e.g. `A` or `H,L`; inserts `TER` between chains; blank keeps all chains) |
| `--ligand-chain <IDS>` | Comma-separated protein or peptide ligand chain IDs extracted into `--ligand-out` |
| `--resid-min`, `--resid-max` | Receptor residue number window to retain |
| `--altloc`, `--ligand-altloc` | Preferred alternate conformer indicator (default `A`) |
| `--mse-to-met` | Convert crystallographic selenomethionine (`HETATM MSE`) to `ATOM MET` |
| `--mutate <SPECS>` | Point mutations (`RESID:RESNAME` or `CHAIN:RESID:RESNAME`); truncates sidechain so `tleap` rebuilds the target residue |
| `--hid`, `--hie`, `--hip`, `--metal-his` | Explicit histidine tautomers (`RESID` or `CHAIN:RESID`), or automatic `HID`/`HIE` on metal-coordinating histidines |
| `--cyx`, `--auto-cyx` | Explicit disulfide cysteines (`CYX`), or automatic detection of `SG-SG <= 2.2 Å` |
| `--keep-hetatm <NAMES>` | Retain structural metal ions or cofactors (`ZN`, `MG`, `MN`, `FE`, `CU`, `CA`, etc.) with the receptor |
| `--ligand <RESNAME>` | Extract crystal or docked `HETATM` ligand by 3-letter residue name (`--ligand-resid` selects one copy; `--ccd` / `--cif` sets CCD bond orders) |
| `--ligand-file <PATH>` | Read an external 3D or docked ligand from `.sdf`, `.mol`, `.mol2`, or `.pdb` |
| `--ligand-smiles <SMILES>` | Assign bond orders and formal charges onto a non-CCD docked ligand from a SMILES template |
| `--ligand-charge <INT>` | Assign bond orders on a non-CCD `.pdb` ligand with explicit hydrogens from its formal net charge |
| `--protonate`, `--deprotonate` | Comma-separated ligand heavy-atom names gaining or losing one proton |
| `--smiles <SMILES>` | Embed and MMFF-optimize a 3D ligand conformer from SMILES (optional `--pocket-ligand` places its centroid in the pocket) |
| `--formula <FORMULA>` | Optional element-count check on the prepared ligand (e.g. `"C16 H22 F3 N2 O5 S2"`) |
| `--split-complex <PATH>` | Split an already-prepared complex `.pdb` into `receptor.pdb` + `ligand.pdb` (supports `--ligand` or `--ligand-chain`) |
| `--prmtop <PATH> --pdb <PATH>` | Export `receptor.pqr` + `ligand.pqr` from an AMBER topology (`--ligand <RESNAME>` or `--ligand-residues <START-END>`) |
| `--receptor-out`, `--ligand-out` | Output file paths (default `receptor.pdb` / `ligand.pdb`, or `.pqr` with `--prmtop`) |

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

# 4. Docked non-CCD ligand inside a complex PDB with SMILES bond-order template
python scripts/prepare_pdb.py \
  --pdb docked_complex.pdb --chain A \
  --ligand LIG --ligand-smiles "NC(=[NH2+])c1ccccc1" \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 5. Protein-protein complex (rigid-body or flexible COFFDROP chain)
python scripts/prepare_pdb.py \
  --pdb 1BRS.pdb --chain A --ligand-chain D --model chain \
  --receptor-out receptor.pdb --ligand-out ligand.pdb

# 6. Export receptor.pqr + ligand.pqr from an AMBER prmtop
python scripts/prepare_pdb.py \
  --prmtop complex.prmtop --pdb complex.pdb --ligand BEN \
  --receptor-out receptor.pqr --ligand-out ligand.pqr
```

---

## 2. `find_reaction_criteria.py` — Reaction Criterion Selection

Reads prepared `receptor` and `ligand` files (`.pdb + .pdb` or `.pqr + .pqr`), prints candidate contact distances or centroid geometry to `stderr`, and prints a ready-to-paste `<criterion>` XML block to `stdout` (or writes `reaction_pairs.json` for `model = chain`).

### Supported options

| Option | Purpose |
|---|---|
| `--receptor`, `--ligand` | Prepared receptor and ligand `.pdb` or `.pqr` paths |
| `--mode contacts\|centroid` | Reaction criterion mode (default `contacts`) |
| `--atom-filter <KIND>` | `polar` (`N, O, S`), `hetero` (any non-C heavy atom), `all`, `nonpolar`, `any_polar`, or `receptor_polar` |
| `--elements <REC:LIG>` | Custom element filter overriding `--atom-filter` (e.g. `ZN,O,N:N,O`) |
| `--search-cutoff <Å>` | Bound-pose contact search cutoff in Å (default `6.0`) |
| `--n-pairs <INT>` | Maximum contact pairs kept (default `6`; trims to even count `2k` with `n_needed = k`, or `1`) |
| `--allow-same-residue` | Permit multiple contact pairs on the same receptor residue |
| `--exclude <NAMES>` | Comma-separated atom names excluded from contact picking |
| `--receptor-targets <SPECS>` | Explicit pocket atoms (`RESID:ATOM:RESNAME` or `CHAIN:RESID:ATOM:RESNAME`) |
| `--ligand-atoms <SPECS>` | Ligand atom names (or `RESID:ATOM:RESNAME`) paired with `--receptor-targets`; omit to auto-pick closest bound-pose partner |
| `--pair-cutoff <Å>` | Reaction distance written into `<pair_cutoff>` (default `7.0`) |
| `--n-needed <INT>` | Override `<n_needed>` explicitly |
| `--chain-pairs-out <JSON>` | Write COFFDROP `[rec_pqr_idx, bead_idx, pair_cutoff]` JSON for `model = chain` |

### Examples

```bash
# 1. Pick up to 6 polar contact pairs from a bound pose (1 per receptor residue)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --atom-filter polar --search-cutoff 6.0 --n-pairs 6 --pair-cutoff 7.0

# 2. Custom element pairing (for example metal + polar receptor atoms against ligand N/O)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --elements ZN,N,O:N,O --search-cutoff 5.0 --n-pairs 4 --pair-cutoff 7.0

# 3. Pair named active-site residues with a specific ligand atom (or omit --ligand-atoms to auto-pick)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --receptor-targets 199:OG1:THR,94:NE2:HID,96:NE2:HID,119:ND1:HIE \
  --ligand-atoms N1 --pair-cutoff 7.0

# 4. Centroid criterion (writes <n_needed>0</n_needed>)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pqr --ligand ligand.pqr \
  --mode centroid --pair-cutoff 8.0

# 5. Flexible-chain COFFDROP reaction_pairs.json (model = chain)
python scripts/find_reaction_criteria.py \
  --receptor receptor.pdb --ligand ligand.pdb \
  --search-cutoff 6.0 --pair-cutoff 8.0 --n-needed 3 \
  --chain-pairs-out reaction_pairs.json
```
