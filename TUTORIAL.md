# PySTARC tutorial

This tutorial computes the association rate constant k<sub>on</sub> of trypsin and benzamidine with rigid body Brownian dynamics on an NVIDIA GPU. The input files are in `examples/trypsin_benzamidine`. Steps 3 to 11 fill the sections of `config.xml` one by one and explain every key with the accepted entries. A new complex follows the same steps with a PDB file of the bound complex (step 3). The flexible chain model is described in `examples/barnase_barstar`.

## 1. Install PySTARC

PySTARC runs on a Linux node with an NVIDIA GPU and a driver for CUDA 12 or newer.

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install_PySTARC.sh
conda activate PySTARC
```

`install_PySTARC.sh` creates the conda environment PySTARC with AmberTools 24.8, APBS, RDKit, OpenBabel and CuPy. An existing environment named PySTARC is replaced. On a host without an NVIDIA GPU or with an older driver the script stops before any change.

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
| `setup.py` | builds the simulation inputs from `config.xml` (same file in every rigid body example) |
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
| ligand with a CCD entry | one HETATM residue with the CCD atom names and every heavy atom of the CCD entry, named with the CCD code or with the code in `ligand_ccd` |
| ligand without a CCD entry | one HETATM residue named UNK, UNL, LIG, DRG, INH or MOL with every hydrogen, `ligand_ccd` blank and `ligand_net_charge` set |
| everything else | water, ions, cofactors and other HETATM groups dropped, protein hydrogens rebuilt by tleap |

### Other keys of `structure`

| key | used with | meaning |
|---|---|---|
| `receptor_pqr_file`, `ligand_pqr_file` | `pqr` | PQR files of the receptor and the ligand in the bound pose, named other than `receptor.pqr` and `ligand.pqr` |
| `prmtop` | `prmtop` | AMBER topology (.prmtop or .parm7) |
| `complex_pdb` | `prmtop`, `complex_pdb` | PDB file of the complex |
| `pdb_id` | `rcsb`, `rcsb_smiles` | RCSB entry code, read from `<pdb_id>.pdb` in the run folder or downloaded |
| `ligand_resid` | `rcsb`, `rcsb_smiles` | residue name of the ligand in the entry, not a residue number |
| `smiles` | `rcsb_smiles` | SMILES of the ligand in the protonation state of the simulation |
| `chain` | `rcsb`, `rcsb_smiles` | one chain identifier, blank for every chain |
| `rec_resid_min` | `rcsb` | lowest receptor residue number kept |
| `ligand_net_charge` | `complex_pdb`, `rcsb`, `rcsb_smiles` | integer net charge of the ligand, blank for the charge of the CCD entry or the SMILES |
| `ligand_ccd` | `complex_pdb`, `rcsb` | CCD code of the ligand, blank for the ligand residue name |
| `ligand_protonate`, `ligand_deprotonate` | `complex_pdb`, `rcsb` | ligand N, O or S atom names with one proton added or removed |
| `keep_hetatm` | `rcsb_smiles` | metal ions kept with the receptor, for example `ZN` |
| `work_dir` | every source | folder for the files written by `setup.py`, default the folder of `config.xml` |

`setup.py` stops with an error for a blank or unknown `ligand_source` and for a key of another source.
