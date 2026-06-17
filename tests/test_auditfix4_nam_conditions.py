"""NAMParameters must drive the outer-propagator physical conditions."""
import math

from pystarc.structures.molecules import Molecule, Atom
from pystarc.pathways.reaction_interface import (
    ContactPair,
    ReactionCriteria,
    ReactionInterface,
    PathwaySet,
)
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.simulation.nam_simulator import NAMParameters, NAMSimulator, zero_force


def _sim(**param_kwargs):
    mol1 = Molecule(name="m1")
    mol1.atoms = [Atom(x=0, y=0, z=0, charge=1.0, radius=2.0)]
    mol2 = Molecule(name="m2")
    mol2.atoms = [Atom(x=0, y=0, z=0, charge=-1.0, radius=2.0)]
    mob = MobilityTensor.from_radii(20.0, 20.0)
    ps = PathwaySet([ReactionInterface("rxn", ReactionCriteria(pairs=[ContactPair(0, 0, 4.0)]))])
    params = NAMParameters(n_trajectories=1, r_start=50.0, seed=1, **param_kwargs)
    return NAMSimulator(mol1, mol2, mob, ps, params, zero_force)


def test_default_conditions_match_the_previous_constants():
    # The defaults must reproduce the values that used to be hard-coded, so the
    # default behavior is unchanged.
    op = _sim()._outer_prop
    assert op is not None
    assert op.kT == 0.5961
    assert op.debye_len == 8.0
    assert math.isclose(op.viscosity, 1.002e-3 * 1e-4 / 1e-12)


def test_configured_conditions_propagate_to_the_outer_propagator():
    op = _sim(debye_length=4.0, temperature_kT=0.55)._outer_prop
    assert op is not None
    assert op.debye_len == 4.0
    assert op.kT == 0.55


def test_dielectric_scales_the_screened_coulomb_prefactor():
    # V_factor is inversely proportional to the dielectric, so halving the
    # dielectric must double it.
    op_hi = _sim(dielectric=78.54)._outer_prop
    op_lo = _sim(dielectric=39.27)._outer_prop
    assert op_hi is not None and op_lo is not None
    assert abs(op_lo.V_factor / op_hi.V_factor - 2.0) < 1e-9
