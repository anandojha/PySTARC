# PySTARC examples

Seven validation examples of increasing complexity:

| Example                              | System                                  | Type                  |
|--------------------------------------|-----------------------------------------|-----------------------|
| `two_charged_spheres/`               | Two oppositely charged spheres          | Analytical validation |
| `trypsin_benzamidine/`               | Trypsin-benzamidine                     | Protein-ligand        |
| `beta_cyclodextrin_guests/`          | 7 BCD host-guest complexes              | Host-guest            |
| `thrombin_thrombomodulin/`           | Thrombin-thrombomodulin                 | Protein-protein       |
| `barnase_barstar/`                   | Barnase-barstar (WT + R59A mutant)      | Protein-protein       |
| `p38_SB2/`                           | p38 MAPK / SB203580                     | Protein-ligand        |
| `carbonic_anhydrase_inhibitors/`     | 7 CA sulfonamide inhibitors (3 isozymes)| Protein-ligand        |

Each example directory contains its own `README.md` with system parameters, input files, run instructions, and output file descriptions. See [`PARAMETERS.md`](PARAMETERS.md) for a detailed parameter selection guide covering all benchmark complexes.

## Directory structure

```
examples/
├── README.md                           This file
├── PARAMETERS.md                       Parameter selection guide for all benchmarks
│
├── two_charged_spheres/                Analytical validation (exact Smoluchowski solution)
│   ├── README.md
│   ├── receptor.pqr                    Single-atom receptor (+1 e)
│   ├── ligand.pqr                      Single-atom ligand (-1 e)
│   ├── rxns.xml                        Reaction criterion (contact at 2.0 A)
│   ├── input.xml                       Simulation parameters
│   ├── analytical.py                   Exact solution comparison script
│   ├── convergence.py                  Multi-seed convergence test
│   └── run.sh                          Run simulation + verification
│
├── trypsin_benzamidine/                Protein-ligand (charged ligand, surface pocket)
│   ├── README.md
│   ├── complex.pdb                     Bound-state PDB
│   ├── complex.prmtop                  AMBER topology
│   ├── setup.py                        Generates PQR, rxns.xml, input.xml
│   └── run.sh                          Run setup + simulation
│
├── beta_cyclodextrin_guests/           Host-guest (7 neutral guests, same receptor)
│   ├── README.md
│   ├── run.sh                          Run all 7 complexes sequentially
│   ├── compare_rates.py                Collect and compare rates across all guests
│   ├── BCD_1-propanol/
│   │   ├── complex.pdb
│   │   ├── complex.parm7
│   │   └── setup.py
│   ├── BCD_1-butanol/
│   ├── BCD_tertbutanol/
│   ├── BCD_methyl_butyrate/
│   ├── BCD_aspirin/
│   ├── BCD_1-naphthylethanol/
│   └── BCD_2-naphthylethanol/
│
├── thrombin_thrombomodulin/            Protein-protein (electrostatically steered)
│   ├── README.md
│   ├── receptor.pqr                    Thrombin PQR (pre-computed)
│   ├── ligand.pqr                      Thrombomodulin PQR (pre-computed)
│   ├── rxns.xml                        Reaction criterion (21 pairs)
│   ├── input.xml                       Simulation parameters
│   ├── bb_effect.py                    Brownian bridge diagnostic script
│   └── run.sh                          Run simulation + BB diagnostic
│
├── barnase_barstar/                    Protein-protein (classic BD benchmark)
│   ├── README.md
│   ├── WT/                             Wild-type (Q_rec = +2 e, 2 reaction pairs)
│   │   ├── 1BRS.pdb
│   │   ├── complex.pdb
│   │   ├── complex.prmtop
│   │   ├── complex.rst7
│   │   ├── complex.pqr
│   │   ├── receptor.pqr
│   │   ├── ligand.pqr
│   │   ├── rxns.xml
│   │   ├── input.xml
│   │   ├── setup.py
│   │   └── run.sh
│   └── R59A/                           R59A mutant (Q_rec = +1 e, 1 reaction pair)
│       ├── 1BRS.pdb
│       ├── complex.pdb
│       ├── complex.prmtop
│       ├── complex.rst7
│       ├── complex.pqr
│       ├── receptor.pqr
│       ├── ligand.pqr
│       ├── rxns.xml
│       ├── input.xml
│       ├── setup.py
│       └── run.sh
│
├── p38_SB2/                            Protein-ligand (neutral kinase inhibitor)
│   ├── README.md
│   ├── setup.py                        Downloads PDB, parameterizes with antechamber
│   └── run.sh                          Run setup + simulation
│
└── carbonic_anhydrase_inhibitors/      Protein-ligand (7 sulfonamides, 3 CA isozymes)
    ├── README.md
    ├── ca13_azm/                       CA XIII + acetazolamide (PDB 3CZV)
    │   ├── setup.py
    │   ├── run.sh
    │   ├── *.pdb, *.pqr, *.prmtop, *.rst7, rxns.xml, input.xml
    ├── ca13_vd1125/                    CA XIII + VD11-25 (PDB 3CZV)
    ├── ca13_vd1126/                    CA XIII + VD11-26 (PDB 3CZV)
    ├── ca13_vd1209/                    CA XIII + VD12-09 (PDB 3CZV)
    ├── ca13_vd1269/                    CA XIII + VD12-69-1 (PDB 3CZV)
    ├── ca1_vd1269/                     CA I + VD12-69-1 (PDB 2NMX)
    └── ca2_vd1142/                     CA II + VD11-4-2 (PDB 3HS4)
```

## Quick start

All examples follow the same pattern:

```bash
conda activate PySTARC
module load cuda
cd examples/<example_name>
bash run.sh
```

Results are written to `bd_sims/` within each example directory. The primary output is `bd_sims/results.json` containing k<sub>on</sub>, P<sub>rxn</sub>, confidence intervals, and run statistics.
