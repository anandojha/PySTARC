"""pystarc.pipeline.build_inputs: the criterion modes contacts, centroid and site,
the anchor atoms of mode site, the pair-count rule, the 0.25 A rounding of the
site margin, the rxns.xml layout, the deposited ligand centroid, and the closest
approach scan of the ligand centroid and of an anchor atom on two spheres, a
dumbbell and the zinc site of carbonic anhydrase XIII."""
import os

import numpy as np
import pytest

from pystarc.pipeline import build_inputs as bi

# revision_prep/rerun_inputs/trypsin_benzamidine/rxns_3of6_7A.xml
TRYPSIN_RXNS = '<top>\n  <first_state> start </first_state>\n  <reactions>\n    <reaction>\n      <name> association </name>\n      <state_before> start </state_before>\n      <state_after> end </state_after>\n      <criterion>\n        <molecules>\n          <molecule0> rec rec </molecule0>\n          <molecule1> lig lig </molecule1>\n        </molecules>\n        <n_needed> 3 </n_needed>\n        <pair>\n          <atoms> 2767 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2743 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2497 8 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2924 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2498 8 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2768 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n      </criterion>\n    </reaction>\n  </reactions>\n</top>\n'
TRYPSIN_PAIRS = [(2766, 8), (2742, 8), (2496, 7), (2923, 8), (2497, 7), (2767, 8)]


@pytest.mark.parametrize("x,want", [(9.24, 9.25), (9.12, 9.0), (9.13, 9.25), (5.631, 5.75), (9.97, 10.0),
                                    (12.42, 12.5), (9.125, 9.25), (9.0, 9.0), (0.1, 0.0), (0.13, 0.25)])
def test_round_quarter(x, want):
    assert bi.round_quarter(x) == want


@pytest.mark.parametrize("n,need,ok", [(1, 1, True), (6, 3, True), (4, 2, True), (2, 1, True), (3, 3, False),
                                       (8, 6, False), (4, 3, False), (1, 0, False), (2, 2, False), (5, 2, False)])
def test_pair_count_rule(n, need, ok):
    assert bi.pair_count_ok(n, need) is ok


def test_modes():
    assert sorted(bi.MODES) == ["centroid", "contacts", "site"]


@pytest.mark.parametrize("c,src", [
    (dict(mode="site", pair_cutoff=9.0), "rcsb"),
    (dict(mode="site", margin=3.0), "complex_pdb"),
    (dict(mode="site", margin=3.0, pocket_ligand="AZM", pocket_ligand_resid=263), "rcsb_smiles"),
    (dict(mode="site", receptor=[(259, "ZN", "ZN")], ligand=[("N3",)], margin=3.0), "rcsb_smiles"),
    (dict(mode="site", receptor=[(109, "N", "MET")], ligand="NB1", receptor_numbering="pdb", pair_cutoff=6.0), "rcsb"),
    (dict(mode="centroid", pair_cutoff=8.0), "prmtop"),
    (dict(mode="contacts", search_cutoff=6.0, n_pairs=6, atom_filter="polar", pair_cutoff=7.0, n_needed=3), "prmtop"),
    (dict(mode="contacts", search_cutoff=5.0, n_pairs=1, atom_filter="all", excluded=["O1"], pair_cutoff=7.0, n_needed=1), "rcsb"),
    (dict(mode="contacts", receptor=[(1, "N", "ALA"), (2, "O", "GLY")], ligand="auto", atom_filter="polar",
          pair_cutoff=7.0, n_needed=1), "rcsb"),
    (dict(mode="contacts", receptor=[(1, "N", "ALA")], ligand=["N1"], receptor_numbering="pdb", pair_cutoff=7.0,
          n_needed=1), "rcsb"),
])
def test_valid_blocks(c, src):
    assert bi.check_criterion(c, src) == c["mode"]


@pytest.mark.parametrize("c,src,words", [
    (dict(mode="zinc", pair_cutoff=12.4), "rcsb_smiles", "must be one of"),
    (dict(mode="pocket", pair_cutoff=9.0), "rcsb", "must be one of"),
    (dict(mode="center", pair_cutoff=8.0), "prmtop", "must be one of"),
    (dict(mode="explicit", receptor=[(1, "N", "A")], ligand=["N1"], pair_cutoff=7.0, n_needed=1), "rcsb", "must be one of"),
    (dict(mode="site", pair_cutoff=9.0, margin=3.0), "rcsb", "exactly one"),
    (dict(mode="site"), "rcsb", "exactly one"),
    (dict(mode="site", margin=3.0), "rcsb_smiles", "needs pocket_ligand"),
    (dict(mode="site", margin=3.0, pocket_ligand="AZM"), "rcsb", "applies only"),
    (dict(mode="site", margin=-1.0), "rcsb", "0 or more"),
    (dict(mode="site", pair_cutoff=9.0, n_needed=1), "rcsb", "does not use"),
    (dict(mode="site", receptor=[(259, "ZN", "ZN")], margin=3.0), "rcsb_smiles", "one receptor target and one ligand"),
    (dict(mode="site", receptor=[(1, "N", "A"), (2, "N", "B")], ligand=["N1", "N2"], margin=3.0), "rcsb",
     "one receptor target and one ligand"),
    (dict(mode="site", receptor=[(1, "N", "A")], ligand="auto", margin=3.0), "rcsb", "must be named"),
    (dict(mode="site", receptor=[(259, "ZN", "ZN")], ligand=["N3"], margin=3.0, pocket_ligand="AZM"), "rcsb_smiles",
     "only without anchor atoms"),
    (dict(mode="site", margin=3.0, receptor_numbering="pdb"), "rcsb", "only to anchor atoms"),
    (dict(mode="site", receptor=[(1, "N", "A")], ligand=["N1"], margin=3.0, atom_filter="polar"), "rcsb",
     "does not use"),
    (dict(mode="contacts", receptor=[(1, "N", "A"), (2, "O", "B")], ligand="auto", atom_filter="polar", n_pairs=2,
          pair_cutoff=7.0, n_needed=1), "rcsb", "not both"),
    (dict(mode="contacts", pair_cutoff=7.0, n_needed=1), "rcsb", "pairs picked from the crystal contacts needs"),
    (dict(mode="contacts", receptor=[(1, "N", "A")], pair_cutoff=7.0, n_needed=1), "rcsb", "listed pairs needs"),
    (dict(mode="contacts", search_cutoff=5.0, n_pairs=8, atom_filter="all", pair_cutoff=7.0, n_needed=6), "complex_pdb", "even number"),
    (dict(mode="contacts", receptor=[(1, "N", "A"), (2, "O", "B"), (3, "O", "C")], ligand="auto", atom_filter="polar",
          pair_cutoff=7.0, n_needed=3), "rcsb", "even number"),
    (dict(mode="contacts", receptor=[(1, "N", "A"), (2, "O", "B")], ligand=["N1", "N2"], atom_filter="polar",
          pair_cutoff=7.0, n_needed=1), "rcsb", "applies only to a ligand side"),
    (dict(mode="contacts", search_cutoff=0.0, n_pairs=2, atom_filter="all", pair_cutoff=7.0, n_needed=1), "rcsb", "above zero"),
    (dict(mode="centroid", pair_cutoff=0.0), "prmtop", "above zero"),
])
def test_invalid_blocks(c, src, words):
    with pytest.raises(SystemExit, match=words):
        bi.check_criterion(c, src)


def test_rxns_layout_is_the_table_s3_trypsin_file(tmp_path):
    p = tmp_path / "rxns.xml"
    bi.write_rxns(p, [(i, j, 7.0, 0.0) for i, j in TRYPSIN_PAIRS], 3)
    assert p.read_text() == TRYPSIN_RXNS


def test_rxns_quarter_distance_is_written_in_full(tmp_path):
    p = tmp_path / "rxns.xml"
    bi.write_rxns(p, [(10, 4, 9.25, 0.0)], 1)
    assert "<distance> 9.25 </distance>" in p.read_text()
    from pystarc.pipeline.geometry import _parse_rxns_xml_criteria
    pairs, n = _parse_rxns_xml_criteria(p, single=True)
    assert n == 1 and pairs[0].cutoff == 9.25 and (pairs[0].rec_index, pairs[0].lig_index) == (10, 4)


def test_deposited_centroid(tmp_path):
    rows = []
    for k, (x, rid) in enumerate([(0.0, 263), (2.0, 263), (10.0, 264)]):
        rows.append(f"HETATM{k + 1:5d}  C1  AZM A{rid:4d}    {x:8.3f}{1.0:8.3f}{2.0:8.3f}  1.00  0.00           C\n")
    p = tmp_path / "e.pdb"
    p.write_text("".join(rows) + "END\n")
    c, n = bi.deposited_centroid(p, "A", "AZM", 263)
    assert n == 2 and np.allclose(c, [1.0, 1.0, 2.0])
    with pytest.raises(SystemExit, match="Give pocket_ligand_resid"):
        bi.deposited_centroid(p, "A", "AZM")


def test_closest_approach_of_two_spheres():
    """One receptor atom and one ligand atom of radius 1: the wall energy
    (a / r)^12 with a = 0.9345 (1 + 1) is 2 kT at r = a 2^(-1/12). The bound pose
    sits 0.37 A off the receptor atom (an exact overlap has no wall term)."""
    from pystarc.analysis.closest_approach import closest_approach
    want = 0.9345 * 2.0 * 2.0 ** (-1.0 / 12.0)
    res = closest_approach([[0.0, 0.0, 0.0]], [1.0], [[0.3, 0.2, 0.1]], [1.0], [0.0, 0.0, 0.0], bound=True,
                           ball=4.0, n_coarse=40_000, n_dense=40_000, n_starts=64, n_steps=400)
    assert res["closest"] is not None and abs(res["closest"] - want) < 0.02
    assert res["descent"] is not None and abs(res["descent"] - want) < 0.06


def _atoms(rows):
    return [dict(name=n, el=n[:1], resname=rn, resid=ri, xyz=np.array(x, float), q=0.0, r=1.5) for n, rn, ri, x in rows]


def test_site_pair_of_anchor_atoms(tmp_path):
    """Anchor atoms give the pair of the named atoms, read in PDB numbering
    through the residue map, and place no ghost atom."""
    rec = _atoms([("N", "GLY", 1, (0.0, 0.0, 0.0)), ("ZN", "ZN", 2, (1.0, 0.0, 0.0))])
    lig = _atoms([("C1", "LIG", 1, (5.0, 0.0, 0.0)), ("N3", "LIG", 1, (4.0, 0.0, 0.0))])
    c = dict(mode="site", receptor=[(262, "ZN", "ZN")], ligand=[("N3",)], receptor_numbering="pdb", pair_cutoff=6.0)
    out = {}
    pair = bi.site_pair(str(tmp_path), rec, lig, c, dict(ligand_source="rcsb"), out, {261: 1, 262: 2})
    assert pair == (1, 1, 6.0, 3.0) and out["anchors"] == dict(receptor=[2, "ZN", 2, "ZN"], ligand=[2, "N3"])
    assert not list(tmp_path.iterdir())
    with pytest.raises(SystemExit, match="no ZN2 ZN"):
        bi.site_pair(str(tmp_path), rec[:1], lig, c, dict(ligand_source="rcsb"), {}, {262: 2})


def test_closest_approach_of_an_anchor_atom():
    """A dumbbell ligand, two atoms of radius 1 that are 6 A apart, against one
    receptor atom of radius 1. The anchor atom stops at the 2 kT contact distance
    of the two-sphere case, while the centroid can sit on the receptor atom."""
    from pystarc.analysis.closest_approach import closest_approach
    want = 0.9345 * 2.0 * 2.0 ** (-1.0 / 12.0)
    lig = [[0.3, 0.2, 0.1], [0.3, 0.2, 6.1]]
    kw = dict(bound=True, ball=4.0, n_coarse=40_000, n_dense=40_000, n_starts=64, n_steps=400)
    res = closest_approach([[0.0, 0.0, 0.0]], [1.0], lig, [1.0, 1.0], [0.0, 0.0, 0.0], anchor=0, **kw)
    assert res["anchor"] == 0 and abs(res["closest"] - want) < 0.02
    assert res["descent"] is not None and abs(res["descent"] - want) < 0.15
    assert closest_approach([[0.0, 0.0, 0.0]], [1.0], lig, [1.0, 1.0], [0.0, 0.0, 0.0], **kw)["closest"] < 0.5


def test_anchor_at_the_centroid_gives_the_centroid_scan():
    """With the anchor atom at the ligand centroid, the scans and the descent
    give the centroid results to the last bit."""
    from pystarc.analysis.closest_approach import closest_approach
    rec = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    lig = [[-1.25, 0.5, 0.75], [0.25, 0.5, 0.75], [1.75, 0.5, 0.75]]
    kw = dict(bound=True, ball=6.0, n_coarse=20_000, n_dense=20_000, n_starts=32, n_steps=200)
    a = closest_approach(rec, [1.5] * 3, lig, [1.2] * 3, [0.5, 0.5, 0.5], anchor=1, **kw)
    b = closest_approach(rec, [1.5] * 3, lig, [1.2] * 3, [0.5, 0.5, 0.5], **kw)
    keys = ("closest", "coarse", "dense", "descent", "accessible_coarse", "accessible_dense", "descent_reached")
    assert b["descent"] is not None and [a[k] for k in keys] == [b[k] for k in keys]


CA_ENTRY = os.path.join(os.path.dirname(__file__), "..", "examples", "carbonic_anhydrase_inhibitors", "ca13_azm",
                        "3CZV.pdb")
RADII = {"C": 1.7, "N": 1.55, "O": 1.5, "S": 1.8, "ZN": 1.5}


def test_anchor_scan_on_a_carbonic_anhydrase_complex():
    """Zn 262 of 3CZV (carbonic anhydrase XIII) and N1, the sulfonamide N of the
    bound acetazolamide AZM 263, with the heavy atoms of chain A and the radii of
    the package PQRs. The crystal pair, 2.01 A apart, lies inside the wall. The
    pair alone reaches 2 kT at a 2^(-1/12) with a = 0.9345 (1.5 + 1.55), and every
    other atom only adds energy, so no accessible pose comes closer. The closest
    pose is the crystal ligand moved as a rigid body, its wall energy summed over
    all atom pairs is at most 2 kT, and its N1 lies at the closest approach."""
    from pystarc.analysis.closest_approach import closest_approach
    from pystarc.forces.soft_repulsion import repulsion_reference
    rec, lig = [], []
    for line in open(CA_ENTRY):
        if not line.startswith(("ATOM", "HETATM")) or line[21] != "A" or line[16] not in " A":
            continue
        el, x = line[76:78].strip().upper(), [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if (line.startswith("ATOM") and el != "H") or (line.startswith("HETATM") and line[17:20] == " ZN"):
            rec.append((line[12:16].strip(), x, RADII[el]))
        elif line.startswith("HETATM") and line[17:26] == "AZM A 263":
            lig.append((line[12:16].strip(), x, RADII[el]))
    R, rr = np.array([x for n, x, r in rec]), np.array([r for n, x, r in rec])
    L, lr = np.array([x for n, x, r in lig]), np.array([r for n, x, r in lig])
    zn = next(np.array(x) for n, x, r in rec if n == "ZN")
    j = next(k for k, (n, x, r) in enumerate(lig) if n == "N1")
    res = closest_approach(R, rr, L, lr, zn, anchor=j, bound=True, ball=8.0, n_coarse=4_000, n_dense=4_000,
                           n_starts=32, n_steps=300)
    pose = np.array(res["pose"])
    _, e, _ = repulsion_reference(pose[None], lr, R, rr)
    pairs = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
    floor = 0.9345 * (1.5 + 1.55) * 2.0 ** (-1.0 / 12.0)
    assert len(lig) == 13 and abs(np.linalg.norm(L[j] - zn) - 2.01) < 0.01 and res["descent"] is not None
    assert res["closest"] >= floor - 0.005 and float(e.sum()) <= 2.0 + 1e-9
    assert np.allclose(pairs(pose), pairs(L), atol=1e-9) and abs(np.linalg.norm(pose[j] - zn) - res["closest"]) <= 0.005
