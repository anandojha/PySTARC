"""Low-severity robustness test for satisfy_constraints_newton.

When max_iter == 0 the Newton loop body never runs. The final RuntimeError
after the loop formats new_violation, which used to be unbound in that case
and raised NameError. After initializing new_violation before the loop, the
intended RuntimeError is raised instead. This test exercises that path with
no GPU or external binaries required.
"""

import numpy as np
import pytest

from pystarc.simulation.coffdrop_chain import (
    ChainAtom,
    ChainAtomRef,
    ChainCommon,
    ChainState,
    LengthConstraint,
    satisfy_constraints_newton,
)


def _make_constrained_state() -> ChainState:
    """Two beads with one length constraint, deliberately violated.

    The constraint guard at the top of satisfy_constraints_newton returns 0
    only when there are no constraints. Providing one length constraint forces
    the solver into its iteration loop logic. The target length differs from
    the actual distance so the violation is nonzero.
    """
    atoms = [ChainAtom(radius=2.0, charge=0.0), ChainAtom(radius=2.0, charge=0.0)]
    common = ChainCommon(
        name="pair",
        atoms=atoms,
        length_constraints=[
            LengthConstraint(a=ChainAtomRef(0), b=ChainAtomRef(1), length=1.0)
        ],
    )
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    return ChainState.from_template(common, positions)


def test_newton_max_iter_zero_raises_runtimeerror():
    state = _make_constrained_state()
    # With max_iter == 0 the loop never runs and the solver must raise the
    # intended RuntimeError (not a NameError from an unbound new_violation).
    with pytest.raises(RuntimeError):
        satisfy_constraints_newton(state, tol=1e-8, max_iter=0)


def test_newton_no_constraints_returns_zero():
    # Sanity check that the healthy no-constraint early return is intact and
    # does not depend on the new_violation initialization.
    atoms = [ChainAtom(radius=2.0, charge=0.0), ChainAtom(radius=2.0, charge=0.0)]
    common = ChainCommon(name="free", atoms=atoms)
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    state = ChainState.from_template(common, positions)
    assert satisfy_constraints_newton(state, max_iter=0) == 0
