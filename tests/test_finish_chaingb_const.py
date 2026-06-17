"""Tests for the chain Generalized Born Coulomb and dielectric constants.

These tests recompute COULOMB_K_KBT_A from the documented first-principles
expression at the stated temperature and assert that the module literal matches
to about four significant figures. They also confirm that the water dielectric is
single-sourced through WATER_DIELECTRIC and that this value is the default eps_out
on the chain Generalized Born energy and force routines, so the constant cannot
drift between nearby literals such as 78.5 and 78.54.

This is chain Generalized Born code only and is not part of the rigid-body NAM
benchmark, so changing this constant does not affect the benchmark path.
"""

import inspect

from pystarc.forces import chain_gb


def _recompute_coulomb_k_kbt_a() -> float:
    """Recompute k_e e^2 / kBT in angstrom from the documented expression.

    The expression is COULOMB_K_KBT_A = k_e e^2 * 1e10 / (kB T) with k_e the Coulomb
    constant in N m^2 / C^2, e the elementary charge in C, kB the Boltzmann constant
    in J/K, and T = 300.15 K. The factor 1e10 converts meters to angstrom.
    """
    k_e = 8.9875517873681764e9  # Coulomb constant, N m^2 / C^2
    e = 1.602176634e-19  # elementary charge, C
    kB = 1.380649e-23  # Boltzmann constant, J/K
    T = 300.15  # K
    return k_e * e * e * 1e10 / (kB * T)


def test_coulomb_constant_matches_first_principles_value():
    """The module literal matches the recomputed value to about 4 significant figures."""
    expected = _recompute_coulomb_k_kbt_a()
    literal = chain_gb.COULOMB_K_KBT_A

    # The recomputed value is about 556.72 A. Match to four significant figures, that
    # is a relative tolerance of roughly 5e-5.
    rel_err = abs(literal - expected) / expected
    assert rel_err < 5e-5, (
        f"COULOMB_K_KBT_A literal {literal} disagrees with the recomputed value "
        f"{expected} by relative error {rel_err}"
    )


def test_coulomb_constant_is_not_the_stale_literal():
    """The literal is no longer the stale 556.86 value flagged in the audit."""
    assert abs(chain_gb.COULOMB_K_KBT_A - 556.86) > 0.1


def test_water_dielectric_is_single_sourced():
    """WATER_DIELECTRIC is defined and is the single consistent water dielectric."""
    assert hasattr(chain_gb, "WATER_DIELECTRIC")
    assert chain_gb.WATER_DIELECTRIC == 78.5


def test_eps_out_defaults_use_water_dielectric():
    """Every chain GB routine with an eps_out argument defaults to WATER_DIELECTRIC.

    This guards against the dielectric drifting between nearby literals across the
    energy and force routines.
    """
    routines = [
        chain_gb.gb_self_born_energy,
        chain_gb.gb_offdiagonal_energy,
        chain_gb.chain_self_born_diagonal_force,
        chain_gb.chain_offdiagonal_gb_force,
        chain_gb.chain_full_gb_force,
    ]
    for fn in routines:
        sig = inspect.signature(fn)
        assert "eps_out" in sig.parameters, f"{fn.__name__} has no eps_out parameter"
        default = sig.parameters["eps_out"].default
        assert default == chain_gb.WATER_DIELECTRIC, (
            f"{fn.__name__} eps_out default {default} is not WATER_DIELECTRIC "
            f"{chain_gb.WATER_DIELECTRIC}"
        )
