"""Regression tests for chain constraint endpoint resolution and the removal
of the unused tabulated dihedral force evaluator in coffdrop_chain.

Constraint endpoints may be given either as bare integer indices or as
AtomRef objects (ChainAtomRef). These tests confirm that constraints defined
with AtomRef endpoints resolve to the same chain-atom indices as the bonded
path and that the constraint machinery (violations, SHAKE, Newton, hybrid,
and the Jacobian) evaluates without error, producing results identical to the
equivalent raw-integer definitions.
"""

import math

import numpy as np
import pytest

from pystarc.simulation.coffdrop_chain import (
    ChainAtom,
    ChainAtomRef,
    ChainCommon,
    ChainState,
    CoplanarConstraint,
    LengthConstraint,
    _build_constraint_jacobian,
    _chain_idx,
    _coplanar_violation,
    compute_constraint_violations,
    satisfy_constraints,
    satisfy_constraints_hybrid,
    satisfy_constraints_newton,
)


def _atoms(n):
    return [ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(n)]


# ---------------------------------------------------------------------------
# _chain_idx resolves both AtomRef and raw integer endpoints.
# ---------------------------------------------------------------------------


def test_chain_idx_resolves_chain_atom_ref():
    assert _chain_idx(ChainAtomRef(0)) == 0
    assert _chain_idx(ChainAtomRef(7)) == 7


def test_chain_idx_passes_through_raw_int():
    assert _chain_idx(3) == 3
    assert _chain_idx(np.int64(5)) == 5


# ---------------------------------------------------------------------------
# Length constraint with AtomRef endpoints resolves and evaluates.
# ---------------------------------------------------------------------------


def test_length_constraint_atomref_violation_evaluates():
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    phi = compute_constraint_violations(state)
    assert phi.shape == (1,)
    assert phi[0] == pytest.approx(0.7, abs=1e-12)


def test_length_constraint_atomref_matches_raw_int():
    positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)

    common_ref = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    common_int = ChainCommon(
        name="len_int",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(0, 1, 5.0)],
    )
    phi_ref = compute_constraint_violations(
        ChainState.from_template(common_ref, positions.copy())
    )
    phi_int = compute_constraint_violations(
        ChainState.from_template(common_int, positions.copy())
    )
    assert np.allclose(phi_ref, phi_int, atol=1e-14)


def test_length_constraint_atomref_shake_converges():
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [7.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints(state, tol=1e-10)
    r = float(np.linalg.norm(state.positions[0] - state.positions[1]))
    assert r == pytest.approx(5.0, abs=1e-9)


def test_length_constraint_atomref_newton_converges():
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [7.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints_newton(state, tol=1e-10)
    r = float(np.linalg.norm(state.positions[0] - state.positions[1]))
    assert r == pytest.approx(5.0, abs=1e-8)


def test_length_constraint_atomref_jacobian_builds():
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    J = _build_constraint_jacobian(state)
    assert J.shape == (1, 6)
    # The atom-a row is the unit vector (r_a - r_b)/|r_a - r_b|; with b at +x
    # this points along -x. The atom-b row is its negation.
    assert np.allclose(J[0, 0:3], [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(J[0, 3:6], [1.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Coplanar constraint with AtomRef endpoints resolves and evaluates.
# ---------------------------------------------------------------------------


def test_coplanar_constraint_atomref_violation_evaluates():
    common = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    # Atom a sits one unit above the z=0 plane defined by b, c, d.
    positions = np.array(
        [[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float
    )
    state = ChainState.from_template(common, positions)
    phi = compute_constraint_violations(state)
    assert phi.shape == (1,)
    assert abs(phi[0]) == pytest.approx(1.0, abs=1e-9)


def test_coplanar_constraint_atomref_matches_raw_int():
    positions = np.array(
        [[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float
    )
    common_ref = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    common_int = ChainCommon(
        name="cop_int",
        atoms=_atoms(4),
        coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
    )
    phi_ref = compute_constraint_violations(
        ChainState.from_template(common_ref, positions.copy())
    )
    phi_int = compute_constraint_violations(
        ChainState.from_template(common_int, positions.copy())
    )
    assert np.allclose(phi_ref, phi_int, atol=1e-14)


def test_coplanar_violation_helper_atomref():
    common = ChainCommon(name="cop_ref", atoms=_atoms(4))
    positions = np.array(
        [[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float
    )
    state = ChainState.from_template(common, positions)
    c = CoplanarConstraint(
        ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
    )
    assert abs(_coplanar_violation(state, c)) == pytest.approx(1.0, abs=1e-9)


def test_coplanar_constraint_atomref_shake_projects_onto_plane():
    common = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    positions = np.array(
        [[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float
    )
    state = ChainState.from_template(common, positions)
    satisfy_constraints(state, tol=1e-10)
    phi = compute_constraint_violations(state)
    assert abs(phi[0]) < 1e-9


def test_coplanar_constraint_atomref_jacobian_builds():
    common = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    positions = np.array(
        [[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float
    )
    state = ChainState.from_template(common, positions)
    J = _build_constraint_jacobian(state)
    assert J.shape == (1, 12)
    # The analytic atom-a row is the plane normal; here the plane is z=0.
    assert np.allclose(J[0, 0:3], [0.0, 0.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Mixed and raw-int endpoints keep working (healthy path unchanged).
# ---------------------------------------------------------------------------


def test_mixed_atomref_and_raw_int_endpoints_resolve():
    common = ChainCommon(
        name="mixed",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), 1, 5.0)],
    )
    positions = np.array([[0, 0, 0], [5.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    phi = compute_constraint_violations(state)
    assert abs(phi[0]) < 1e-12


def test_hybrid_solver_with_atomref_endpoints():
    common = ChainCommon(
        name="hybrid_ref",
        atoms=_atoms(3),
        length_constraints=[
            LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 3.0),
            LengthConstraint(ChainAtomRef(1), ChainAtomRef(2), 3.0),
        ],
    )
    positions = np.array([[0, 0, 0], [4.0, 0, 0], [9.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints_hybrid(state, tol=1e-9)
    phi = compute_constraint_violations(state)
    assert float(np.max(np.abs(phi))) < 1e-8


# ---------------------------------------------------------------------------
# The unused tabulated dihedral force evaluator has been removed.
# ---------------------------------------------------------------------------


def test_coffdrop_force_evaluator_removed():
    import pystarc.simulation.coffdrop_chain as mod

    assert not hasattr(mod, "COFFDROPForceEvaluator")
