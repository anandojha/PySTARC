# PySTARC examples

Ten validation examples of increasing complexity:

| Example                               | System                                   | Type                       |
|---------------------------------------|------------------------------------------|----------------------------|
| `two_charged_spheres/`                | Two oppositely charged spheres           | Analytical validation      |
| `trypsin_benzamidine/`                | Trypsin-benzamidine                      | Protein-ligand             |
| `beta_cyclodextrin_guests/`           | 7 BCD host-guest complexes               | Host-guest                 |
| `thrombin_thrombomodulin/`            | Thrombin-thrombomodulin                  | Protein-protein            |
| `barnase_barstar_chainbd/`            | Barnase-barstar (flexible chain)         | Chain BD / protein-protein |
| `p38_mapk_sb203580/`                  | p38 MAPK / SB203580                      | Protein-ligand             |
| `carbonic_anhydrase_inhibitors/`      | 7 CA sulfonamide inhibitors (3 isozymes) | Protein-ligand             |
| `hsp90_inhibitors/`                   | 6 HSP90 inhibitors                       | Protein-ligand             |
| `ttk_inhibitors/`                     | 8 TTK (MPS1) kinase inhibitors           | Protein-ligand             |
| `trypsin_benzamidine_multi_GPUs/`     | Trypsin-benzamidine (SLURM, 1 and 4 GPUs)| Cluster / multi-GPU demo   |

Each example directory contains its own `README.md` with system parameters, input files, run instructions, and output file descriptions. See [`PARAMETERS.md`](PARAMETERS.md) for a detailed parameter selection guide covering all benchmark complexes.

## Directory structure

```
examples/
├── README.md                           This file
├── PARAMETERS.md                       Parameter selection guide for all benchmarks
│
├── two_charged_spheres/                Analytical validation (exact Smoluchowski solution)
│   ├── README.md
│   ├── receptor.pqr                    Single-atom receptor (+1e)
│   ├── ligand.pqr                      Single-atom ligand (−1e)
│   ├── rxns.xml                        Reaction criterion (contact at 2.0 Å)
│   ├── input.xml                       Simulation parameters
│   ├── analytical.py                   Exact solution comparison script
│   ├── convergence.py                  Multi-seed convergence test
│   └── run.sh                          Run simulation + verification
│
├── trypsin_benzamidine/                Protein-ligand (charged ligand, surface pocket)
│   ├── README.md
│   ├── complex.pdb                     Bound-state PDB
│   ├── complex.prmtop                  AMBER topology
│   ├── receptor.pqr                    Pre-generated trypsin PQR
│   ├── ligand.pqr                      Pre-generated benzamidine PQR
│   ├── rxns.xml                        Reaction criterion
│   ├── input.xml                       Simulation parameters
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
│   │   ├── receptor.pqr
│   │   ├── ligand.pqr
│   │   ├── rxns.xml
│   │   ├── input.xml
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
├── barnase_barstar_chainbd/            Flexible chain BD (protein-protein, under active validation)
│
├── p38_mapk_sb203580/                  Protein-ligand (neutral kinase inhibitor)
│   ├── README.md
│   ├── 1A9U.pdb                        Crystal structure
│   ├── receptor.pdb, receptor.pqr      Clean receptor + PQR
│   ├── ligand.pdb, ligand.pqr          Clean ligand + PQR
│   ├── protein.prmtop, protein.rst7    Receptor AMBER topology + coordinates
│   ├── ligand.prmtop, ligand.rst7      Ligand AMBER topology + coordinates
│   ├── rxns.xml                        Reaction criterion
│   ├── input.xml                       Simulation parameters
│   ├── setup.py                        Regenerates all files from the PDB
│   └── run.sh                          Run setup + simulation
│
├── carbonic_anhydrase_inhibitors/      Protein-ligand (7 sulfonamides, 3 CA isozymes)
│   ├── README.md
│   ├── ca13_azm/                       CA XIII + acetazolamide (PDB 3CZV)
│   │   ├── 3CZV.pdb
│   │   ├── setup.py
│   │   ├── run.sh
│   │   └── *.pdb, *.pqr, *.prmtop, *.rst7, rxns.xml, input.xml
│   ├── ca13_vd1125/                    CA XIII + VD11-25 (PDB 3CZV)
│   ├── ca13_vd1126/                    CA XIII + VD11-26 (PDB 3CZV)
│   ├── ca13_vd1209/                    CA XIII + VD12-09 (PDB 3CZV)
│   ├── ca13_vd1269/                    CA XIII + VD12-69-1 (PDB 3CZV)
│   ├── ca1_vd1269/                     CA I + VD12-69-1 (PDB 2NMX)
│   └── ca2_vd1142/                     CA II + VD11-4-2 (PDB 3HS4)
│
├── hsp90_inhibitors/                   Protein-ligand (6 neutral HSP90 inhibitors)
│   ├── README.md
│   ├── run.sh                          Run all 6 complexes + compare to experiment
│   ├── 31/                             HSP90 + inhibitor 31
│   │   ├── complex.pdb                 Source structure (the single input)
│   │   └── setup.py                    Regenerates all files from the PDB
│   ├── 37/
│   ├── 43/
│   ├── 62/
│   ├── 65/
│   └── 70/
│
├── ttk_inhibitors/                     Protein-ligand (8 TTK/MPS1 kinase inhibitors)
│   ├── README.md
│   ├── run.sh                          Run all 8 complexes + compare to experiment
│   ├── 2X9E/                           TTK + NMS-P715
│   │   ├── 2X9E.pdb                    Source structure (named by PDB accession)
│   │   └── setup.py                    Regenerates all files from the PDB
│   ├── 3GFW/                           TTK + Mps1-IN-1
│   ├── 3H9F/                           TTK + Mps1-IN-2
│   ├── 5LJJ/                           TTK + Reversine
│   ├── 5N7V/                           TTK + MPI-0479605
│   ├── 5N84/                           TTK + Mps-BAY2b
│   ├── 5N93/                           TTK + TC-Mps1-12
│   └── 5NAD/                           TTK + BAY-1217389
│
└── trypsin_benzamidine_multi_GPUs/     Cluster SLURM demo (single-GPU and multi-GPU)
    ├── README.md
    ├── complex.pdb                     Bound-state PDB
    ├── complex.prmtop                  AMBER topology
    ├── setup.py                        Generates PQR, rxns.xml, input.xml
    ├── receptor.pqr                    Pre-generated trypsin PQR
    ├── ligand.pqr                      Pre-generated benzamidine PQR
    ├── rxns.xml                        Reaction criterion
    ├── input.xml                       Simulation parameters
    ├── submit_SLURM_single_GPU.sh      SLURM: 1 GPU × 10M trajectories
    └── submit_SLURM_multi_GPUs.sh      SLURM: 4 GPUs × 2.5M trajectories, auto-combine
```

## Quick start

All examples follow the same pattern for interactive runs:

```bash
conda activate PySTARC
module load cuda
cd examples/<example_name>
bash run.sh
```

For the SLURM cluster example:

```bash
cd examples/trypsin_benzamidine_multi_GPUs
sbatch submit_SLURM_single_GPU.sh       # or submit_SLURM_multi_GPUs.sh
```

Results are written to `bd_sims/` within each example directory. The primary output is `bd_sims/results.json` containing k<sub>on</sub>, P<sub>rxn</sub>, confidence intervals, and run statistics. Multi-GPU runs additionally produce `bd_sims/combined_results.json` pooling results across all GPUs.
