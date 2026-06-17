# COFFDROP Chain BD: User Guide

The `pystarc/simulation/coffdrop_chain.py` module builds and
simulates flexible peptide chains under the COFFDROP coarse-grained
force field. This document describes the user-facing API and design.

## Quick start

```python
from pystarc.simulation.coffdrop_chain import (
    chain_from_sequence,
    ChainState,
    compute_chain_forces,
)
import numpy as np

# Build a chain from sequence (single-letter codes)
chain = chain_from_sequence("ARWGL")
# Or: chain_from_sequence("ALA-ARG-TRP-GLY-LEU")

# Place atoms (here at relaxed geometry)
positions = np.zeros((chain.n_atoms, 3))
# ... fill positions ...

# Compute forces using full COFFDROP physics
state = ChainState.from_template(chain, positions)
compute_chain_forces(state)
# state.forces is an (n_atoms, 3) array in kBT/A units
```

See `examples/coffdrop_chain_intro.py` for a complete working example.

## What's covered

The chain builder constructs a `ChainCommon` with:

- **Backbone bonds**: CA-CA between consecutive residues (eq length 3.8 A)
- **Sidechain bonds**: per-residue linear chains (CA-CB, CB-CG, CB-NG, CB-OG, CB-SB, CG-CD)
- **Backbone angles**: CA(r)-CA(r+1)-CA(r+2) for each consecutive triple
- **Sidechain-backbone angles**: SC1(r)-CA(r)-CA(r+/-1) where SC1 is the first sidechain bead
- **Intra-residue angles**: CA-SC1-SC2 and SC1-SC2-SC3 (TRP only)
- **Backbone torsions**: CA(r)-CA(r+1)-CA(r+2)-CA(r+3)
- **Sidechain torsions** (incoming, outgoing): CA-CA-CA-CB and CB-CA-CA-CA
- **Cross-residue sidechain torsions**: CB(r)-CA(r)-CA(r+1)-CB(r+1)
- **Sidechain-extending torsions**: SC2-SC1-CA-CA forward and backward
- **Non-bonded pair forces**: COFFDROP tabulated pair potentials

Each force type uses tabulated potentials from COFFDROP's parameter
files (coffdrop.xml, map.xml, connectivity.xml, charges.xml) when
the lookup matches; otherwise falls back to harmonic (bonds) or
zero contribution.

## Supported residues

All 20 standard amino acids:
- 1 bead (CA only): GLY
- 2 beads (CA + 1 sidechain): ALA, CYS (uses SB), PRO, SER, THR, VAL
- 3 beads (CA + CB + second sidechain): ARG, LYS, HIS, HIP (NG); ASN, GLN, ILE, LEU, MET, PHE, TYR (CG); ASP, GLU (OG)
- 4 beads (CA + CB + CG + CD): TRP

CYS uses SB instead of CB as its first sidechain bead. The lookup
machinery is generalized to handle this and any future residue
naming variations.

## Limitations

The current implementation does **not** include:
- Cap residues (ACE, NME) - peptides are assumed uncapped at both ends
- N-terminal/C-terminal cap atoms (CN, CC, NN, OC, HC) - so terminal modifications are not represented
- 47 less-common dihedral types involving caps and termini

These limitations affect peptides with N-acetylation or C-amidation
modifications. For uncapped peptides (the most common case in BD
simulations of intrinsically disordered regions and peptide-protein
binding), the full force field is reachable.

## API reference

### `chain_from_sequence(sequence, coffdrop_dir=..., name=None, sidechains=True, k_spring=100.0)`

Build a `ChainCommon` from a sequence string.

- `sequence`: single-letter ("ARWGL") or 3-letter ("ALA-ARG-TRP-GLY-LEU" or "ALA ARG TRP GLY LEU")
- `coffdrop_dir`: path to directory with COFFDROP XML files (default: `pystarc/coffdrop_data`)
- `name`: optional chain name; auto-generated from sequence if not given
- `sidechains`: if True (default), build with sidechain beads; if False, CA-only backbone
- `k_spring`: harmonic spring constant for bonds (kBT/A^2)

Returns: `ChainCommon` ready for `compute_chain_forces`.

### `compute_chain_forces(state, kT=1.0, soft_repulsion=False, soft_repulsion_eps=1.0)`

Fill `state.forces` from all force contributions.

- Bond forces (harmonic; tabulated when type_idx assigned)
- Angle forces (tabulated COFFDROP angle potentials)
- Torsion forces (tabulated COFFDROP dihedral potentials, all 6 types)
- Pair forces (tabulated COFFDROP pair potentials, vectorized via `deriv_array`)

### `build_chain_common_from_coffdrop(residues, params, name=None, k_spring=100.0)`

Lower-level: build a CA-only backbone chain from a list of 3-letter
residue codes and pre-loaded COFFDROPParams.

### `build_chain_common_with_sidechains_from_coffdrop(residues, params, name=None, k_spring=100.0)`

Lower-level: build a chain with sidechain beads. Same args as above.

## Physics validation

The chain force pipeline has been validated for:
- **Translational invariance**: forces unchanged under uniform position shifts
- **Rotational equivariance**: forces rotate with positions under SO(3)
  (strongest physical-correctness check)
- **Newton's 3rd law**: net force is zero across all atom configurations
- **Reversal symmetry**: homopolymer forces respect chain reversal
- **Stable integration**: 100-step Euler integration of 10-residue
  heteropolymer doesn't blow up

See the `TestCOFFDROPTabulatedForces` class in `tests/test_pystarc.py` for the
force-machinery, sidechain-topology, and validation tests.

## Performance

Pair force evaluation is vectorized via `TabulatedPotential.deriv_array`
(grouped batch spline calls per type_idx). Approximate timing for a
single `compute_chain_forces` call:

| Chain length | Atoms | Pairs | ms/call |
|--------------|-------|-------|---------|
| 5 residues   | 10    | 25    | ~0.4    |
| 10 residues  | 20    | 145   | ~0.9    |
| 20 residues  | 40    | 685   | ~2.0    |
| 40 residues  | 80    | 2965  | ~4.4    |

For larger systems or long trajectories, further optimization
(numba JIT or C extension) would help; the current implementation
prioritizes correctness over peak performance.

