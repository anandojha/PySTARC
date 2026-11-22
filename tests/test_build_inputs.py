"""pystarc.pipeline.build_inputs, scripts/prepare_pdb.py and scripts/find_reaction_criteria.py:
criterion modes contacts and centroid, the pair-count rule, the rxns.xml layout,
strict .pdb+.pdb or .pqr+.pqr two-solute structure schema, and standalone pre-config scripts."""

import hashlib
import importlib.util
import os
import shutil

import numpy as np
import pytest

from pystarc.pipeline import build_inputs as bi

TRYPSIN_RXNS = "<top>\n  <first_state> start </first_state>\n  <reactions>\n    <reaction>\n      <name> association </name>\n      <state_before> start </state_before>\n      <state_after> end </state_after>\n      <criterion>\n        <molecules>\n          <molecule0> rec rec </molecule0>\n          <molecule1> lig lig </molecule1>\n        </molecules>\n        <n_needed> 3 </n_needed>\n        <pair>\n          <atoms> 2767 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2743 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2497 8 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2924 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2498 8 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n        <pair>\n          <atoms> 2768 9 </atoms>\n          <distance> 7.0 </distance>\n        </pair>\n      </criterion>\n    </reaction>\n  </reactions>\n</top>\n"
TRYPSIN_PAIRS = [(2766, 8), (2742, 8), (2496, 7), (2923, 8), (2497, 7), (2767, 8)]


def _load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "n,need,ok",
    [
        (1, 1, True),
        (6, 3, True),
        (4, 2, True),
        (2, 1, True),
        (3, 3, False),
        (8, 6, False),
        (4, 3, False),
        (1, 0, False),
        (2, 2, False),
        (5, 2, False),
    ],
)
def test_pair_count_rule(n, need, ok):
    assert bi.pair_count_ok(n, need) is ok


def test_modes():
    assert sorted(bi.MODES) == ["centroid", "contacts"]
    assert bi.CRITERION == ("mode", "receptor", "ligand", "pair_cutoff", "n_needed")


@pytest.mark.parametrize(
    "c,model,has_rp",
    [
        (dict(mode="centroid", pair_cutoff=8.0, n_needed=0), "rigid", False),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "ALA")],
                ligand=[("N1",)],
                pair_cutoff=7.0,
                n_needed=1,
            ),
            "rigid",
            False,
        ),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "ALA"), (2, "O", "GLY")],
                ligand=[("N1",), ("O2",)],
                pair_cutoff=7.0,
                n_needed=1,
            ),
            "rigid",
            False,
        ),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "ALA")],
                ligand=[(5, "OD1", "ASP")],
                pair_cutoff=7.0,
                n_needed=1,
            ),
            "rigid",
            False,
        ),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "ALA"), (2, "O", "GLY"), (3, "O", "SER")],
                ligand=[(1, "N", "ALA"), (2, "O", "GLY"), (3, "O", "SER")],
                pair_cutoff=8.0,
                n_needed=2,
            ),
            "chain",
            False,
        ),
        (dict(mode="contacts", pair_cutoff=8.0, n_needed=3), "chain", True),
    ],
)
def test_valid_blocks(c, model, has_rp):
    assert bi.check_criterion(c, model, has_chain_rp_json=has_rp) == c["mode"]


@pytest.mark.parametrize(
    "c,model,words",
    [
        (dict(mode="site", pair_cutoff=9.0), "rigid", "must be one of"),
        (dict(mode="zinc", pair_cutoff=12.4), "rigid", "must be one of"),
        (dict(mode="pocket", pair_cutoff=9.0), "rigid", "must be one of"),
        (dict(mode="center", pair_cutoff=8.0), "rigid", "must be one of"),
        (dict(mode="auto", pair_cutoff=7.0, n_needed=3), "rigid", "must be one of"),
        (
            dict(
                mode="explicit",
                receptor=[(1, "N", "A")],
                ligand=["N1"],
                pair_cutoff=7.0,
                n_needed=1,
            ),
            "rigid",
            "must be one of",
        ),
        (
            dict(mode="contacts", search_cutoff=6.0, pair_cutoff=7.0, n_needed=1),
            "rigid",
            "does not accept",
        ),
        (
            dict(mode="contacts", atom_filter="polar", pair_cutoff=7.0, n_needed=1),
            "rigid",
            "does not accept",
        ),
        (dict(mode="centroid", pair_cutoff=8.0, n_needed=1), "rigid", "n_needed 0"),
        (
            dict(mode="centroid", pair_cutoff=8.0, n_needed=0),
            "chain",
            "requires criterion mode 'contacts'",
        ),
        (dict(mode="contacts", pair_cutoff=7.0, n_needed=1), "rigid", "explicit"),
        (
            dict(
                mode="contacts", receptor=[(1, "N", "A")], pair_cutoff=7.0, n_needed=1
            ),
            "rigid",
            "explicit",
        ),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "A"), (2, "O", "B")],
                ligand=[("N1",)],
                pair_cutoff=7.0,
                n_needed=1,
            ),
            "rigid",
            "against",
        ),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "A"), (2, "O", "B"), (3, "O", "C")],
                ligand=["N1", "N2", "N3"],
                pair_cutoff=7.0,
                n_needed=3,
            ),
            "rigid",
            "even number",
        ),
        (
            dict(
                mode="contacts",
                receptor=[(1, "N", "A")],
                ligand=["N1"],
                pair_cutoff=0.0,
                n_needed=1,
            ),
            "rigid",
            "above zero",
        ),
        (dict(mode="centroid", pair_cutoff=0.0, n_needed=0), "rigid", "above zero"),
    ],
)
def test_invalid_blocks(c, model, words):
    with pytest.raises(SystemExit, match=words):
        bi.check_criterion(c, model)


def test_rxns_layout_is_the_table_s3_trypsin_file(tmp_path):
    p = tmp_path / "rxns.xml"
    bi.write_rxns(p, [(i, j, 7.0, 0.0) for i, j in TRYPSIN_PAIRS], 3)
    assert p.read_text() == TRYPSIN_RXNS
    ex_trypsin = os.path.join(
        os.path.dirname(__file__), "..", "examples", "trypsin_benzamidine"
    )
    dst = tmp_path / "trypsin"
    shutil.copytree(ex_trypsin, dst)
    bi.build_from_config(str(dst / "config.xml"))
    assert (dst / "bd_sims" / "rxns.xml").read_text() == TRYPSIN_RXNS


def test_rxns_quarter_distance_is_written_in_full(tmp_path):
    p = tmp_path / "rxns.xml"
    bi.write_rxns(p, [(10, 4, 9.25, 0.0)], 1)
    assert "<distance> 9.25 </distance>" in p.read_text()
    from pystarc.pipeline.geometry import _parse_rxns_xml_criteria

    pairs, n = _parse_rxns_xml_criteria(p, single=True)
    assert (
        n == 1
        and pairs[0].cutoff == 9.25
        and (pairs[0].rec_index, pairs[0].lig_index) == (10, 4)
    )


def test_prepare_pdb_deposited_centroid_and_split(tmp_path):
    mod = _load_script("prepare_pdb")
    rows = []
    for k, (x, rid) in enumerate([(0.0, 263), (2.0, 263), (10.0, 264)]):
        rows.append(
            f"HETATM{k + 1:5d}  C1  AZM A{rid:4d}    {x:8.3f}{1.0:8.3f}{2.0:8.3f}  1.00  0.00           C\n"
        )
    p = tmp_path / "e.pdb"
    p.write_text("".join(rows) + "END\n")
    c, n = mod.deposited_centroid(str(p), "A", "AZM", 263)
    assert n == 2 and np.allclose(c, [1.0, 1.0, 2.0])
    with pytest.raises(SystemExit, match="pocket-ligand-resid"):
        mod.deposited_centroid(str(p), "A", "AZM")

    comb = tmp_path / "comb.pdb"
    comb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2 ZN    ZN A 301       1.000   0.000   0.000  1.00  0.00          ZN\n"
        "HETATM    3  C1  LIG L   1       5.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    rec_out = tmp_path / "rec.pdb"
    lig_out = tmp_path / "lig.pdb"
    mod.split_complex_pdb(str(comb), str(rec_out), str(lig_out))
    assert "ALA A   1" in rec_out.read_text() and "ZN A 301" in rec_out.read_text()
    assert "LIG L   1" in lig_out.read_text() and "ALA" not in lig_out.read_text()


def _atoms(rows):
    return [
        dict(
            name=n, el=n[:1], resname=rn, resid=ri, xyz=np.array(x, float), q=0.0, r=1.5
        )
        for n, rn, ri, x in rows
    ]


def test_contacts_listed_metal_and_multi_residue_ligand(tmp_path):
    pdb = tmp_path / "rec.pdb"
    pdb.write_text(
        "ATOM      1  N   GLY A  10       0.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    2 ZN    ZN A 262       1.000   0.000   0.000  1.00  0.00          ZN\n"
        "END\n"
    )
    rmap = bi.resid_map(str(pdb))
    assert rmap == {10: 1, 262: 2}
    rec = _atoms([("N", "GLY", 1, (0.0, 0.0, 0.0)), ("ZN", "ZN", 2, (1.0, 0.0, 0.0))])
    lig = _atoms(
        [("OD1", "ASP", 400, (5.0, 0.0, 0.0)), ("N3", "ASP", 401, (4.0, 0.0, 0.0))]
    )
    c = dict(
        mode="contacts",
        receptor=[(262, "ZN", "ZN")],
        ligand=[(401, "N3", "ASP")],
        pair_cutoff=6.0,
        n_needed=1,
    )
    pairs = bi.contacts_listed(rec, lig, c, rmap, None)
    assert pairs == [(1, 1, 6.0, 3.0)]
    with pytest.raises(SystemExit, match="no ZN262 ZN"):
        bi.contacts_listed(rec[:1], lig, c, rmap, None)


def test_mixed_pdb_and_pqr_formats_rejected(tmp_path):
    rec = tmp_path / "receptor.pqr"
    lig = tmp_path / "ligand.pdb"
    rec.write_text(
        "ATOM      1  CA  ALA     1       0.000   0.000   0.000  0.0000 1.5000\nEND\n"
    )
    lig.write_text(
        "HETATM    1  C1  LIG A   1       5.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    with pytest.raises(SystemExit, match="both be .pdb or both be .pqr"):
        bi.build(
            receptor="receptor.pqr",
            ligand="ligand.pdb",
            ligand_net_charge=0,
            model="rigid",
            criterion=dict(mode="centroid", pair_cutoff=5.0, n_needed=0),
            source_dir=str(tmp_path),
            work_dir=str(tmp_path / "bd"),
        )
    lig_pqr = tmp_path / "ligand.pqr"
    lig_pqr.write_text(
        "ATOM      1  CA  ALA     1       5.000   0.000   0.000 -1.0000 1.5000\nEND\n"
    )
    with pytest.raises(SystemExit, match="ligand_net_charge is \\+1 e"):
        bi.build(
            receptor="receptor.pqr",
            ligand="ligand.pqr",
            ligand_net_charge=1,
            model="rigid",
            criterion=dict(mode="centroid", pair_cutoff=5.0, n_needed=0),
            source_dir=str(tmp_path),
            work_dir=str(tmp_path / "bd2"),
        )


def test_two_solute_schema_and_single_conformer_guard():
    assert sorted(bi.MODELS) == ["chain", "rigid"]
    assert bi.STRUCTURE == ("receptor", "ligand", "ligand_net_charge", "model")
    assert "names" not in dict(bi.SECTIONS)
    dropped = {
        "ligand_source",
        "receptor_pqr_file",
        "ligand_pqr_file",
        "prmtop",
        "complex_pdb",
        "receptor_pdb",
        "smiles",
        "chain",
        "ligand_chain",
        "pocket_ligand",
        "pocket_ligand_resid",
        "ligand_formula",
        "receptor_resname",
        "ligand_resname",
        "search_cutoff",
        "n_pairs",
        "atom_filter",
        "excluded",
        "receptor_numbering",
        "margin",
    }
    assert not (dropped & set(bi.KEYS))
    assert not (dropped & set(bi.CRITERION))
    multi = [
        "HETATM    1  C1 ALIG A   1       0.000   0.000   0.000  0.50  0.00           C\n",
        "HETATM    2  C1 BLIG A   1       1.000   0.000   0.000  0.50  0.00           C\n",
    ]
    with pytest.raises(SystemExit, match="multiple altloc conformers"):
        bi._check_single_conformer(multi, "multi.pdb")


def test_prepare_pdb_metal_his_cyx_mutate_and_chain_charge(tmp_path, capsys):
    mod = _load_script("prepare_pdb")
    raw = tmp_path / "raw.pdb"
    rec_out = tmp_path / "receptor.pdb"
    lig_out = tmp_path / "ligand.pdb"
    raw.write_text(
        "ATOM      1  ND1 HIS A  94       3.500   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  NE2AHIS A  94       2.000   0.000   0.000  0.50  0.00           N\n"
        "ATOM      3  NE2BHIS A  94       2.100   0.000   0.000  0.50  0.00           N\n"
        "ATOM      4  ND1 HIS A 119       0.000   2.050   0.000  1.00  0.00           N\n"
        "ATOM      5  SG  CYS A  10       0.000  10.000   0.000  1.00  0.00           S\n"
        "ATOM      6  SG  CYS A  20       2.030  10.000   0.000  1.00  0.00           S\n"
        "ATOM      7  CA  TYR A 604       5.000   5.000   0.000  1.00  0.00           C\n"
        "ATOM      8  CB  TYR A 604       5.500   5.000   0.000  1.00  0.00           C\n"
        "ATOM      9  OH  TYR A 604       8.000   5.000   0.000  1.00  0.00           O\n"
        "ATOM     10  CA  ASP D   1       0.000   0.000  20.000  1.00  0.00           C\n"
        "ATOM     11  CA  GLU D   2       3.800   0.000  20.000  1.00  0.00           C\n"
        "ATOM     12  CA  LYS D   3       7.600   0.000  20.000  1.00  0.00           C\n"
        "HETATM   13 ZN    ZN A 301       0.000   0.000   0.000  1.00  0.00          ZN\n"
        "END\n"
    )
    mod.main(
        [
            "--pdb",
            str(raw),
            "--chain",
            "A",
            "--ligand-chain",
            "D",
            "--model",
            "chain",
            "--altloc",
            "B",
            "--keep-hetatm",
            "ZN",
            "--metal-his",
            "--auto-cyx",
            "--mutate",
            "604:CYS",
            "--receptor-out",
            str(rec_out),
            "--ligand-out",
            str(lig_out),
        ]
    )
    txt = rec_out.read_text()
    out_stdout = capsys.readouterr().out
    assert (
        "HID A  94" in txt
        and "HIE A 119" in txt
        and "2.100" in txt
        and "2.000" not in txt
    )
    assert "CYX A  10" in txt and "CYX A  20" in txt
    assert "CYS A 604" in txt and "OH " not in txt
    assert "<ligand_net_charge>-1</ligand_net_charge>" in out_stdout


def test_prepare_pdb_multichain_ter_and_mse_to_met(tmp_path):
    mod = _load_script("prepare_pdb")
    raw = tmp_path / "dimer.pdb"
    rec_out = tmp_path / "receptor.pdb"
    lig_out = tmp_path / "ligand.pdb"
    raw.write_text(
        "HETATM    1  CA  MSE H  10       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  SE  MSE H  10       1.800   0.000   0.000  1.00  0.00          SE\n"
        "ATOM      3  CA  ALA L  10       0.000   8.000   0.000  1.00  0.00           C\n"
        "ATOM      4  CA  GLY C   1      12.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    mod.main(
        [
            "--pdb",
            str(raw),
            "--chain",
            "H,L",
            "--ligand-chain",
            "C",
            "--mse-to-met",
            "--receptor-out",
            str(rec_out),
            "--ligand-out",
            str(lig_out),
        ]
    )
    lines = rec_out.read_text().splitlines()
    assert any("SD  MET H  10" in l for l in lines)
    ter_idx = [i for i, l in enumerate(lines) if l == "TER"]
    assert len(ter_idx) == 2 and ter_idx[0] < ter_idx[1]

    comb = tmp_path / "pp.pdb"
    comb.write_text(rec_out.read_text().replace("END\n", "") + lig_out.read_text())
    r2, l2 = tmp_path / "r2.pdb", tmp_path / "l2.pdb"
    mod.split_complex_pdb(str(comb), str(r2), str(l2), ligand_chains=("C",))
    assert "GLY C   1" in l2.read_text() and "GLY C   1" not in r2.read_text()


def test_find_reaction_criteria_multichain_and_custom_elements(tmp_path, capsys):
    frc = _load_script("find_reaction_criteria")
    rec = tmp_path / "rec.pdb"
    lig = tmp_path / "lig.pdb"
    rec.write_text(
        "ATOM      1  O   TYR H  50       0.000   0.000   0.000  1.00  0.00           O\n"
        "TER\n"
        "ATOM      2  O   TYR L  50       0.000   3.000   0.000  1.00  0.00           O\n"
        "HETATM    3 ZN    ZN H 200       0.000   6.000   0.000  1.00  0.00          ZN\n"
        "END\n"
    )
    lig.write_text(
        "HETATM    1  N1  LIG A   1       0.000   2.900   0.000  1.00  0.00           N\n"
        "HETATM    2  O1  LIG A   1       0.000   5.800   0.000  1.00  0.00           O\n"
        "END\n"
    )
    frc.main(
        [
            "--receptor",
            str(rec),
            "--ligand",
            str(lig),
            "--elements",
            "ZN,O:N,O",
            "--search-cutoff",
            "4.0",
            "--n-pairs",
            "4",
            "--pair-cutoff",
            "7.0",
        ]
    )
    out = capsys.readouterr().out
    # Duplicate resid 50 across H and L maps to sequential 1-based PQR residue numbers (2 and 3)
    assert '<target resid="2" atom="O" resname="TYR"/>' in out
    assert '<target resid="3" atom="ZN" resname="ZN"/>' in out


def test_find_reaction_criteria_contacts_targets_centroid_and_chain(tmp_path, capsys):
    frc = _load_script("find_reaction_criteria")
    assert "closest_approach" not in dir(frc) and "write_site_atom" not in dir(frc)

    ex_trypsin = os.path.join(
        os.path.dirname(__file__), "..", "examples", "trypsin_benzamidine"
    )
    frc.main(
        [
            "--receptor",
            os.path.join(ex_trypsin, "receptor.pqr"),
            "--ligand",
            os.path.join(ex_trypsin, "ligand.pqr"),
            "--atom-filter",
            "polar",
            "--n-pairs",
            "6",
            "--pair-cutoff",
            "7.0",
        ]
    )
    xml_contacts = capsys.readouterr().out
    assert '<target resid="193" atom="O" resname="TRP"/>' in xml_contacts
    assert "<n_needed>3</n_needed>" in xml_contacts

    ex_ca = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "carbonic_anhydrase_inhibitors",
        "ca13_azm",
    )
    frc.main(
        [
            "--receptor",
            os.path.join(ex_ca, "receptor.pdb"),
            "--ligand",
            os.path.join(ex_ca, "ligand.pdb"),
            "--receptor-targets",
            "199:OG1:THR,94:NE2:HID,96:NE2:HID,119:ND1:HIE",
            "--ligand-atoms",
            "N1",
            "--pair-cutoff",
            "7.0",
        ]
    )
    xml_targets = capsys.readouterr().out
    assert (
        xml_targets.count('<atom name="N1"/>') == 4
        and "<n_needed>2</n_needed>" in xml_targets
    )

    frc.main(
        [
            "--receptor",
            os.path.join(ex_trypsin, "receptor.pqr"),
            "--ligand",
            os.path.join(ex_trypsin, "ligand.pqr"),
            "--mode",
            "centroid",
            "--pair-cutoff",
            "8.0",
        ]
    )
    xml_cen = capsys.readouterr().out
    assert "<mode>centroid</mode>" in xml_cen and "<n_needed>0</n_needed>" in xml_cen

    if shutil.which("tleap") is not None:
        ex_bb = os.path.join(
            os.path.dirname(__file__), "..", "examples", "barnase_barstar"
        )
        rp_out = tmp_path / "reaction_pairs.json"
        frc.main(
            [
                "--receptor",
                os.path.join(ex_bb, "receptor.pdb"),
                "--ligand",
                os.path.join(ex_bb, "ligand.pdb"),
                "--search-cutoff",
                "6.0",
                "--pair-cutoff",
                "8.0",
                "--n-needed",
                "3",
                "--chain-pairs-out",
                str(rp_out),
            ]
        )
        want_md5 = hashlib.md5(
            open(os.path.join(ex_bb, "reaction_pairs.json"), "rb").read()
        ).hexdigest()
        got_md5 = hashlib.md5(rp_out.read_bytes()).hexdigest()
        assert got_md5 == want_md5


def test_prepare_pdb_sphere_pqr_pair_and_split_skips_bulk_ions(tmp_path, capsys):
    mod = _load_script("prepare_pdb")

    # 1. Analytical charged sphere pair
    s_rec = tmp_path / "s_rec.pqr"
    s_lig = tmp_path / "s_lig.pqr"
    mod.main(
        [
            "--sphere-pair",
            "1.0:1.0,-1.0:1.0",
            "--receptor-out",
            str(s_rec),
            "--ligand-out",
            str(s_lig),
        ]
    )
    out_s = capsys.readouterr().out
    assert "<receptor>s_rec.pqr</receptor>" in out_s
    assert "<ligand_net_charge>-1</ligand_net_charge>" in out_s
    assert s_rec.read_text().split()[-2:] == ["1.0", "1.0"]
    assert s_lig.read_text().split()[-2:] == ["-1.0", "1.0"]

    # 2. Pre-parameterized PQR pair
    p_rec = tmp_path / "p_rec.pqr"
    p_lig = tmp_path / "p_lig.pqr"
    mod.main(
        [
            "--receptor-pqr",
            str(s_rec),
            "--ligand-pqr",
            str(s_lig),
            "--receptor-out",
            str(p_rec),
            "--ligand-out",
            str(p_lig),
        ]
    )
    out_p = capsys.readouterr().out
    assert "<ligand_net_charge>-1</ligand_net_charge>" in out_p
    assert (
        p_rec.read_bytes() == s_rec.read_bytes()
        and p_lig.read_bytes() == s_lig.read_bytes()
    )

    # 3. --split-complex strips bulk NA/CL solvent ions while retaining structural ZN and printing <structure>
    comb = tmp_path / "solvated.pdb"
    comb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2 ZN    ZN A 301       1.000   0.000   0.000  1.00  0.00          ZN\n"
        "HETATM    3 NA    NA C   1      20.000   0.000   0.000  1.00  0.00          NA\n"
        "HETATM    4 CL    CL C   2      22.000   0.000   0.000  1.00  0.00          CL\n"
        "HETATM    5  C1  UNK B   1       4.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    r_out = tmp_path / "rec.pdb"
    l_out = tmp_path / "lig.pdb"
    mod.main(
        [
            "--split-complex",
            str(comb),
            "--ligand",
            "UNK",
            "--ligand-charge",
            "0",
            "--receptor-out",
            str(r_out),
            "--ligand-out",
            str(l_out),
        ]
    )
    out_c = capsys.readouterr().out
    r_txt = r_out.read_text()
    assert "ZN A 301" in r_txt and "NA C" not in r_txt and "CL C" not in r_txt
    assert "UNK B   1" in l_out.read_text()
    assert "<ligand_net_charge>0</ligand_net_charge>" in out_c


def test_write_chain_input_emits_coarse_and_fine_dime(tmp_path):
    cfg = {
        "apbs_fglen": "140.0",
        "apbs_dime": "449",
        "apbs_coarse_dime": "129",
        "apbs_fine_dime": "257",
        "bd_milestone_radius": "60.0",
        "n_trajectories": "10",
        "max_steps": "1000",
        "dt": "0.2",
        "temperature": "298.15",
        "seed": "1",
        "desolvation_alpha": "1.0",
        "convergence_check": "true",
        "convergence_interval": "10",
        "convergence_tol": "0.05",
        "save_interval": "0",
        "criterion": {"n_needed": 3},
        "chain_cfg": {},
    }
    out = tmp_path / "input.xml"
    bi.write_chain_input(str(out), cfg, str(tmp_path))
    txt = out.read_text()
    assert "<apbs_coarse_dime>129</apbs_coarse_dime>" in txt
    assert "<apbs_fine_dime>257</apbs_fine_dime>" in txt
