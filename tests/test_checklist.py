"""The run log checklist says what ran."""

from types import SimpleNamespace as NS

from pystarc.pipeline.pipeline import _checklist


def _run(alpha=1.0, lig_born=True, yukawa=True, multipole=True, bridge=True, hi=True,
         soft=True, overlap=False, lj=False, r_rec=22.0, r_lig=0.0):
    cfg = NS(desolvation_alpha=alpha, r_hydro_rec=r_rec, r_hydro_lig=r_lig, temperature=298.15,
             minimum_core_dt=0.2)
    params = NS(hydrodynamic_interactions=hi, _soft_repulsion=soft, _overlap_check=overlap,
                use_brownian_bridge=bridge)
    geom = NS(receptor=NS(hydrodynamic_r=22.0), ligand=NS(hydrodynamic_r=4.98765))
    e = NS(_elec_grids_gpu=[1, 2], _born_grids_gpu=[1, 2], _lig_born_grids_gpu=[1, 2] if lig_born else [],
           _has_yukawa=yukawa, _multipole=object() if multipole else None, _use_lj=lj)
    return "\n".join(_checklist(cfg, params, geom, 64.6763, e, None, None))


def test_born_line_follows_what_ran():
    assert "Born both directions            (alpha=1.00000000)" in _run()
    assert "Born on the ligand atoms only" in _run(lig_born=False)
    assert "Born desolvation off" in _run(alpha=0.0, lig_born=False)


def test_bridge_far_field_and_wall_lines():
    assert "Brownian bridge at reaction surface" in _run()
    assert "Brownian bridge off (reaction test at the end of each step)" in _run(bridge=False)
    assert "Multipole far-field" in _run() and "Monopole far-field" in _run(multipole=False)
    assert "No far-field force beyond the grids" in _run(yukawa=False, multipole=False)
    assert "APBS grids finest first (2 elec, 2 Born)" in _run()
    assert "Overlap check (overlapping moves are rejected)" in _run(soft=False, overlap=True)
    assert "elastic" not in _run(soft=False, overlap=True) and "Fine-only" not in _run()


def test_hydro_radii_show_their_own_source_and_hi():
    text = _run()
    assert "receptor 22.0000 Å (XML), ligand 4.9877 Å (MC)" in text
    assert "HI in k_b integral             (enabled)" in text
    assert "Isotropic D(r) with HI" in text and "Isotropic D_rel" in _run(hi=False)
    assert "k_b Romberg integral           (k_b=64.6763 ų/ps)" in text
