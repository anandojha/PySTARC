"""Reaction rules of the GPU constructor and the gate of Born direction 2. The
tests need a GPU."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1).sum()
except Exception:
    pytest.skip("no usable CUDA device", allow_module_level=True)

from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator
from pystarc.simulation.nam_simulator import NAMParameters
from pystarc.structures.molecules import Atom, ContactPair, Molecule, ReactionCriteria


def _mol(n):
    return Molecule(atoms=[Atom(index=i, x=float(i), charge=0.1, radius=1.5) for i in range(n)])


def _rxn(name, pairs, n_needed=-1, sb=None, sa=None, p=1.0):
    crit = ReactionCriteria(name=name, pairs=[ContactPair(i, j, 5.0) for i, j in pairs],
                            n_needed=n_needed, state_before=sb, state_after=sa)
    return ReactionInterface(name=name, criteria=crit, probability=p, state_before=sb, state_after=sa)


def _sim(*reactions, first_state=None):
    return GPUBatchSimulator(_mol(3), _mol(2), MobilityTensor.from_radii(10.0, 2.0),
                             PathwaySet(list(reactions), first_state=first_state), NAMParameters(), None)


@pytest.mark.parametrize("nn,need", [(-1, 2), (1, 1), (2, 2)])
def test_n_needed_inside_the_range_is_kept(nn, need):
    assert _sim(_rxn("a", [(0, 0), (1, 1)], nn))._rxn_n_needed == need


@pytest.mark.parametrize("nn", [0, 3, -2])
def test_n_needed_outside_the_range_raises(nn):
    with pytest.raises(ValueError, match="n_needed"):
        _sim(_rxn("a", [(0, 0), (1, 1)], nn))


def test_state_machine_reaction_needs_its_one_pair():
    _sim(_rxn("a", [(0, 0)], 1, "s", "t"), first_state="s")
    with pytest.raises(ValueError, match="n_needed"):
        _sim(_rxn("a", [(0, 0)], 2, "s", "t"), first_state="s")


def test_unlabelled_reactions_and_probabilities_are_refused():
    with pytest.raises(ValueError, match="without state labels"):
        _sim(_rxn("a", [(0, 0)]), _rxn("b", [(1, 1)]))
    with pytest.raises(NotImplementedError):
        _sim(_rxn("a", [(0, 0)], p=0.5))


def _grid():
    return DXGrid(origin=np.full(3, -8.0), delta=np.eye(3), data=np.zeros((17, 17, 17)))


@pytest.mark.parametrize("flag,alpha,on", [(True, 1.0, True), (False, 1.0, False), (True, 0.0, False)])
def test_born_direction_2_is_set_up_only_when_it_runs(capsys, flag, alpha, on):
    e = GPUBatchForceEngine([_grid()], [_grid()], alpha=alpha, lig_born_grids=[_grid()],
                            rec_positions=np.zeros((2, 3)), rec_charges=np.ones(2),
                            enable_born2_torque=flag)
    out = capsys.readouterr().out
    assert bool(e._lig_born_grids_gpu) is on and e._enable_born2_torque is flag
    assert ("Born both-directions" in out) is on and ("Born direction 2 off" in out) is not on
