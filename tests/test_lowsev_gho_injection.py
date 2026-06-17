"""
Low-severity robustness test for pystarc.pipeline.gho_injection.

This covers the dead-code cleanup in inject_gho_from_manual, where an
unreachable else (centroid1) was removed from a branch whose condition already
guarantees that the molecule 1 index is in range. The test confirms the healthy
path still resolves GHO positions to the exact molecule positions relative to
the hydrodynamic centre, so the cleanup did not change behavior.
"""

import numpy as np

from pystarc.pipeline.gho_injection import inject_gho_from_manual


def test_inject_gho_from_manual_resolves_mol1_and_mol2_positions():
    mol1_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    mol2_positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [11.0, 12.0, 13.0],
        ]
    )
    mol1_hydro_cen = np.array([1.0, 1.0, 1.0])
    mol2_hydro_cen = np.array([2.0, 2.0, 2.0])

    # Global index 1 belongs to molecule 1 (n1 = 3). Global index 3 maps to
    # local index 0 of molecule 2.
    spec = "1,0,17.0\n3,1,10.0"

    mol1_ghos, mol2_ghos = inject_gho_from_manual(
        spec,
        mol1_positions,
        mol2_positions,
        mol1_hydro_cen,
        mol2_hydro_cen,
    )

    assert len(mol1_ghos) == 1
    assert len(mol2_ghos) == 1

    # Molecule 1 atom: position is mol1_positions[1] relative to its hydro centre.
    assert mol1_ghos[0].atom_index == 0
    np.testing.assert_array_equal(
        mol1_ghos[0].pos_rel, mol1_positions[1] - mol1_hydro_cen
    )

    # Molecule 2 atom: global index 3 - n1 = local index 0.
    assert mol2_ghos[0].atom_index == 1
    np.testing.assert_array_equal(
        mol2_ghos[0].pos_rel, mol2_positions[0] - mol2_hydro_cen
    )


def test_inject_gho_from_manual_first_atom_index_zero():
    # Boundary check: global_atom_idx == 0 must still land in the molecule 1
    # branch and index mol1_positions[0] exactly, with no fallback.
    mol1_positions = np.array([[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]])
    mol2_positions = np.array([[0.0, 0.0, 0.0]])
    mol1_hydro_cen = np.zeros(3)
    mol2_hydro_cen = np.zeros(3)

    mol1_ghos, mol2_ghos = inject_gho_from_manual(
        "0,5,12.0",
        mol1_positions,
        mol2_positions,
        mol1_hydro_cen,
        mol2_hydro_cen,
    )

    assert len(mol1_ghos) == 1
    assert len(mol2_ghos) == 0
    assert mol1_ghos[0].atom_index == 5
    np.testing.assert_array_equal(mol1_ghos[0].pos_rel, mol1_positions[0])
