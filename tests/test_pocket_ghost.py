"""The pocket ghost atom PKT is left out of the centring and the grid sizing, like GHO."""
import numpy as np

from pystarc.pipeline.prepare_bd_surface import PQRAtom, centroid, compute_grid_params
from pystarc.pipeline.run_apbs import _grid_key, _read_pqr_atoms, _strip_ghost_pqr_bytes


def _atoms():
    real = [PQRAtom(serial=i + 1, name="CA", resname="ALA", resid=i + 1, x=float(x), y=0.0, z=0.0,
                    charge=0.1, radius=1.7, record="ATOM") for i, x in enumerate((0.0, 2.0, 4.0))]
    pkt = PQRAtom(serial=4, name="PKT", resname="PKT", resid=4, x=30.0, y=30.0, z=30.0,
                  charge=0.0, radius=0.0, record="HETATM")
    gho = PQRAtom(serial=5, name="GHO", resname="GHO", resid=5, x=-30.0, y=0.0, z=0.0,
                  charge=0.0, radius=0.0, record="HETATM")
    return real, real + [pkt, gho]


def test_centroid_leaves_out_the_ghost_atoms():
    real, every = _atoms()
    assert np.allclose(centroid(every), [2.0, 0.0, 0.0])


def test_apbs_grid_atoms_leave_out_the_ghost_atoms(tmp_path):
    real, every = _atoms()
    p_real = tmp_path / "real.pqr"
    p_all = tmp_path / "all.pqr"
    p_real.write_text("".join(a.to_pqr_line() for a in real) + "END\n")
    p_all.write_text("".join(a.to_pqr_line() for a in every) + "END\n")
    assert [round(a[0], 3) for a in _read_pqr_atoms(p_all)] == [0.0, 2.0, 4.0]
    stripped = _strip_ghost_pqr_bytes(p_all).decode("utf-8")
    assert "PKT" not in stripped and "GHO" not in stripped
    assert _grid_key(p_all, {"dime": 129}) == _grid_key(p_real, {"dime": 129})
