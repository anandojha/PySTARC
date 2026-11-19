# Standalone Pre-Configuration Scripts (`scripts/`)

`scripts/` contains two optional standalone command-line tools for preparing receptor and ligand structure files and choosing reaction criteria **before** filling `config.xml`. Neither script is imported or called by `pystarc`, `run_pystarc.py`, or `config.xml`.

```
scripts/
├── prepare_pdb.py             Prepares separate receptor and ligand .pdb or .pqr files and prints <structure>
└── find_reaction_criteria.py  Inspects prepared structures and prints <criterion> (or reaction_pairs.json)
```

---

## 1. `prepare_pdb.py`

Prepares matching `receptor.pdb` + `ligand.pdb` (or `receptor.pqr` + `ligand.pqr`) files and prints the `<structure>` XML block (including integer `<ligand_net_charge>`) for `config.xml`.

### Common workflows

1. **Crystal protein-ligand PDB (with RCSB CCD bond orders and protonation):**
   ```bash
   python scripts/prepare_pdb.py \
     --pdb-id 1A9U --chain A --resid-min 4 \
     --ligand SB2 \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

2. **Crystal metalloprotein with metal-coordinating histidine tautomers (`--metal-his`), retained cofactor (`--keep-hetatm`), and ligand deprotonation (`--deprotonate`):**
   ```bash
   python scripts/prepare_pdb.py \
     --pdb-id 3CZV --chain A --keep-hetatm ZN --metal-his \
     --ligand AZM --ligand-resid 263 --deprotonate N1 \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

3. **Docked non-CCD ligand PDB (bond orders assigned from `--ligand-smiles` or `--ligand-charge`):**
   ```bash
   python scripts/prepare_pdb.py \
     --pdb docked.pdb --chain A --auto-cyx \
     --ligand LIG --ligand-smiles "NC(=[NH2+])c1ccccc1" \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

4. **Revert engineered crystal point mutations (`--mutate` truncates the sidechain for `tleap` rebuild):**
   ```bash
   python scripts/prepare_pdb.py \
     --pdb-id 5LJJ --chain A --mutate 604:CYS \
     --ligand 79K \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

5. **Protein-protein complex (rigid-body or flexible COFFDROP chain):**
   ```bash
   python scripts/prepare_pdb.py \
     --pdb 1BRS.pdb --chain A --ligand-chain D --model chain \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

6. **Receptor PDB plus 3D ligand embedded from SMILES:**
   ```bash
   python scripts/prepare_pdb.py \
     --pdb-id 3CZV --chain A --keep-hetatm ZN --metal-his \
     --smiles "CC(C)(C)NC1=C(C(F)=C(F)C(=C1F)S([NH-])(=O)=O)S(=O)(=O)CCC1=CC=CC=C1" \
     --pocket-ligand AZM --pocket-ligand-resid 263 \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

7. **Split an already-prepared combined complex PDB:**
   ```bash
   python scripts/prepare_pdb.py \
     --split-complex complex.pdb \
     --receptor-out receptor.pdb --ligand-out ligand.pdb
   ```

8. **Export `receptor.pqr` + `ligand.pqr` from an AMBER `.prmtop` + coordinate `.pdb`:**
   ```bash
   python scripts/prepare_pdb.py \
     --prmtop complex.prmtop --pdb complex.pdb --ligand BEN \
     --receptor-out receptor.pqr --ligand-out ligand.pqr
   ```

Run `python scripts/prepare_pdb.py --help` for the full list of options (`--hid`, `--hie`, `--hip`, `--cyx`, `--auto-cyx`, `--altloc`, `--formula`).

---

## 2. `find_reaction_criteria.py`

Reads prepared `receptor` and `ligand` files, prints candidate distances to `stderr`, and prints a ready-to-paste `<criterion>` block to `stdout`.

### Common workflows

1. **Bound-pose contact pairs (`polar` or `all` heavy atoms, at most one pair per receptor residue):**
   ```bash
   python scripts/find_reaction_criteria.py \
     --receptor receptor.pdb --ligand ligand.pdb \
     --atom-filter polar --search-cutoff 6.0 --n-pairs 6 --pair-cutoff 7.0
   ```

2. **Named active-site receptor atoms paired with a specific ligand atom (or closest bound-pose ligand atom when `--ligand-atoms` is omitted):**
   ```bash
   python scripts/find_reaction_criteria.py \
     --receptor receptor.pdb --ligand ligand.pdb \
     --receptor-targets 199:OG1:THR,94:NE2:HID,96:NE2:HID,119:ND1:HIE \
     --ligand-atoms N1 --pair-cutoff 7.0
   ```

3. **Centroid reaction criterion (`mode = centroid`):**
   ```bash
   python scripts/find_reaction_criteria.py \
     --receptor receptor.pqr --ligand ligand.pqr \
     --mode centroid --pair-cutoff 8.0
   ```

4. **Flexible-chain COFFDROP `reaction_pairs.json` (`model = chain`):**
   ```bash
   python scripts/find_reaction_criteria.py \
     --receptor receptor.pdb --ligand ligand.pdb \
     --search-cutoff 6.0 --pair-cutoff 8.0 --n-needed 3 \
     --chain-pairs-out reaction_pairs.json
   ```
