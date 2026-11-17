"""
Pipeline, partner-role symmetry, CLI, and multi-GPU pooling tests for GPU and
CPU flexible-chain Brownian dynamics.
"""

from __future__ import annotations

from pystarc.structures.chain_io import (
    interface_reaction_pairs,
    prepare_chain_partner,
    save_chain_to_json,
)
from pystarc.simulation.coffdrop_chain import (
    chain_from_sequence,
    place_relaxed_geometry,
)
from pystarc.pipeline.input_parser import ChainConfig, PySTARCConfig, parse
from pystarc.simulation.gpu_chain_simulator import GPUChainBDSimulator
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.multi_GPU import combine_data, multi_GPU_runs
from pystarc.pipeline import chain_pipeline
from pystarc.cli.main import cli
from click.testing import CliRunner
from types import SimpleNamespace
from pathlib import Path
import subprocess
import warnings
import numpy as np
import pytest
import json
import sys

try:
    import cupy as cp

    cp.arange(3).sum()
    _HAS_CUDA = True
except Exception:
    cp = None
    _HAS_CUDA = False


def _write_synthetic_complex_pdb_and_pqrs(tmp_path: Path):
    """Write a two-chain all-heavy-atom PDB and centered PQRs for chains A and B."""
    pdb_lines = []
    serial = 1
    # Chain A: 4 residues near x = 0; Chain B: 4 residues facing A across x ~ 4.2 A.
    seq_a = [("ALA", 0.0), ("ASP", 3.6), ("LYS", 7.2), ("GLY", 10.8)]
    seq_b = [("ARG", 0.2), ("GLU", 3.8), ("SER", 7.4), ("VAL", 11.0)]

    def _res_atoms(resname, x0, y0):
        pts = {
            "N": (x0, y0 - 1.2, 0.0),
            "CA": (x0, y0, 0.0),
            "C": (x0, y0 + 1.2, 0.0),
            "O": (x0, y0 + 2.1, 0.6),
        }
        if resname != "GLY":
            pts["CB"] = (x0 + (0.8 if x0 == 0.0 else -0.8), y0, 1.2)
        extra = {
            "ASP": [("CG", 1.4), ("OD1", 2.1), ("OD2", 2.1)],
            "LYS": [("CG", 1.3), ("CD", 2.0), ("CE", 2.6), ("NZ", 3.2)],
            "ARG": [("CG", -1.3), ("CD", -2.0), ("NE", -2.6), ("CZ", -3.1), ("NH1", -3.5), ("NH2", -3.5)],
            "GLU": [("CG", -1.3), ("CD", -2.0), ("OE1", -2.6), ("OE2", -2.6)],
            "SER": [("OG", -1.4)],
            "VAL": [("CG1", -1.4), ("CG2", -1.4)],
        }
        for k, (an, dx) in enumerate(extra.get(resname, [])):
            pts[an] = (x0 + dx, y0 + 0.2 * k, 1.4 + 0.2 * k)
        return pts

    pqr_a_atoms = []
    pqr_b_atoms = []
    for r_i, (rname, y0) in enumerate(seq_a, start=1):
        for aname, (x, y, z) in _res_atoms(rname, 0.0, y0).items():
            pdb_lines.append(
                f"ATOM  {serial:5d} {aname:<4s} {rname:3s} A{r_i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            pqr_a_atoms.append((serial, aname, rname, r_i, x, y, z, -0.25 if rname == "ASP" else 0.15, 1.8))
            serial += 1
    for r_i, (rname, y0) in enumerate(seq_b, start=1):
        for aname, (x, y, z) in _res_atoms(rname, 5.2, y0).items():
            pdb_lines.append(
                f"ATOM  {serial:5d} {aname:<4s} {rname:3s} B{r_i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            pqr_b_atoms.append((serial, aname, rname, r_i, x, y, z, 0.25 if rname == "ARG" else -0.15, 1.8))
            serial += 1
    pdb_lines.append("END\n")
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("".join(pdb_lines))

    def _write_centered_pqr(p_out: Path, records):
        xyz = np.array([[r[4], r[5], r[6]] for r in records], dtype=float)
        xyz -= xyz.mean(axis=0)
        lines = []
        for idx, (rec, c) in enumerate(zip(records, xyz), start=1):
            lines.append(
                f"ATOM  {idx:5d} {rec[1]:<4s} {rec[2]:3s} {rec[3]:5d}    "
                f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f} {rec[7]:7.4f} {rec[8]:6.4f}\n"
            )
        p_out.write_text("".join(lines))

    pqr_a = tmp_path / "partner_a.pqr"
    pqr_b = tmp_path / "partner_b.pqr"
    _write_centered_pqr(pqr_a, pqr_a_atoms)
    _write_centered_pqr(pqr_b, pqr_b_atoms)
    return pdb_path, pqr_a, pqr_b


def _write_dx(path: Path, origin, spacing: float, n: int, scale: float) -> Path:
    xs = origin[0] + spacing * np.arange(n)
    ys = origin[1] + spacing * np.arange(n)
    zs = origin[2] + spacing * np.arange(n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z + 9.0)
    grid = DXGrid(
        np.asarray(origin, dtype=float),
        np.diag([spacing] * 3),
        scale * np.exp(-r / 7.858) / r,
    )
    grid.to_file(str(path)) if hasattr(grid, "to_file") else _dump_dx(path, grid)
    return path


def _dump_dx(path: Path, grid: DXGrid) -> None:
    nx, ny, nz = grid.data.shape
    ox, oy, oz = grid.origin
    dx, dy, dz = np.diag(grid.delta)
    vals = grid.data.ravel()
    lines = [
        f"object 1 class gridpositions counts {nx} {ny} {nz}",
        f"origin {ox:.6f} {oy:.6f} {oz:.6f}",
        f"delta {dx:.6f} 0.000000 0.000000",
        f"delta 0.000000 {dy:.6f} 0.000000",
        f"delta 0.000000 0.000000 {dz:.6f}",
        f"object 2 class gridconnections counts {nx} {ny} {nz}",
        f"object 3 class array type double rank 0 items {vals.size} data follows",
    ]
    for i in range(0, vals.size, 3):
        lines.append(" ".join(f"{v:.6e}" for v in vals[i : i + 3]))
    path.write_text("\n".join(lines) + "\n")


def test_chain_input_tags_and_flexible_partner_validation(tmp_path):
    xml_rec_flex = tmp_path / "rec_flex.xml"
    xml_rec_flex.write_text(
        f"""<?xml version="1.0"?>
<pystarc>
  <ligand_pqr>partner_b.pqr</ligand_pqr>
  <work_dir>{tmp_path / 'work_rf'}</work_dir>
  <gpu>false</gpu>
  <chain>
    <flexible_partner>Receptor</flexible_partner>
    <chain_json>partner_a_chain.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
    <target_grid_dx>fine1.dx</target_grid_dx>
    <coarse_grid_dx>coarse0.dx</coarse_grid_dx>
    <born_grid_dx>born0.dx</born_grid_dx>
    <far_field>true</far_field>
    <auto_diffusion>true</auto_diffusion>
  </chain>
</pystarc>"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = parse(xml_rec_flex)
    assert cfg.chain.flexible_partner == "receptor"
    assert cfg.chain.coarse_grid_dx == "coarse0.dx"
    assert cfg.chain.far_field is True
    assert cfg.debye_length == 7.858

    # Missing ligand_pqr when flexible_partner is receptor must raise.
    bad_missing_lig = tmp_path / "missing_lig.xml"
    bad_missing_lig.write_text(
        f"""<pystarc>
  <receptor_pqr>partner_a.pqr</receptor_pqr>
  <work_dir>{tmp_path / 'w1'}</work_dir>
  <chain>
    <flexible_partner>receptor</flexible_partner>
    <chain_json>c.json</chain_json>
  </chain>
</pystarc>"""
    )
    with pytest.raises(ValueError, match="ligand_pqr"):
        parse(bad_missing_lig)

    # Invalid flexible_partner value must raise.
    bad_role = tmp_path / "bad_role.xml"
    bad_role.write_text(
        f"""<pystarc>
  <receptor_pqr>partner_a.pqr</receptor_pqr>
  <work_dir>{tmp_path / 'w2'}</work_dir>
  <chain>
    <flexible_partner>both</flexible_partner>
    <chain_json>c.json</chain_json>
  </chain>
</pystarc>"""
    )
    with pytest.raises(ValueError, match="flexible_partner"):
        parse(bad_role)


def test_prepare_chain_partner_and_interface_reaction_pairs_symmetry(tmp_path):
    pdb_path, pqr_a, pqr_b = _write_synthetic_complex_pdb_and_pqrs(tmp_path)

    # Partner B flexible (ligand role), partner A rigid.
    chain_b, pos_b = prepare_chain_partner(pdb_path, "B", chain_name="partner_b")
    assert np.allclose(pos_b.mean(axis=0), 0.0, atol=1e-12)
    pairs_ab = interface_reaction_pairs(
        pdb_path,
        rigid_chain_id="A",
        flexible_chain_id="B",
        rigid_pqr_path=pqr_a,
        chain_common=chain_b,
        interface_cutoff=6.5,
        reaction_distance=8.0,
    )
    assert len(pairs_ab) > 0

    # Partner A flexible (receptor role), partner B rigid.
    chain_a, pos_a = prepare_chain_partner(pdb_path, "A", chain_name="partner_a")
    assert np.allclose(pos_a.mean(axis=0), 0.0, atol=1e-12)
    pairs_ba = interface_reaction_pairs(
        pdb_path,
        rigid_chain_id="B",
        flexible_chain_id="A",
        rigid_pqr_path=pqr_b,
        chain_common=chain_a,
        interface_cutoff=6.5,
        reaction_distance=8.0,
    )
    assert len(pairs_ba) > 0


def test_run_chain_role_swap_rxns_xml_and_gpu_dispatch(tmp_path, monkeypatch):
    pdb_path, pqr_a, pqr_b = _write_synthetic_complex_pdb_and_pqrs(tmp_path)
    chain_a, pos_a = prepare_chain_partner(pdb_path, "A", chain_name="flex_receptor")
    chain_a_json = save_chain_to_json(chain_a, pos_a, tmp_path / "chain_a.json")

    fine_dx = _write_dx(tmp_path / "partb1.dx", [-16.0, -16.0, -16.0], 2.0, 17, -1.5)
    coarse_dx = _write_dx(tmp_path / "partb0.dx", [-32.0, -32.0, -32.0], 4.0, 17, -1.5)
    born_dx = _write_dx(tmp_path / "partb0_born.dx", [-16.0, -16.0, -16.0], 2.0, 17, 0.3)

    # rxns.xml in canonical (mol1=receptor=chain_a, mol2=ligand=pqr_b) order.
    rxns_xml = tmp_path / "rxns.xml"
    rxns_xml.write_text(
        """<reactions>
  <reaction name="assoc" n_needed="1">
    <contact molecule1_index="2" molecule2_index="0" distance="200.0"/>
  </reaction>
</reactions>"""
    )

    # CPU run with flexible_partner='receptor' and rxns.xml index swapping.
    cfg_cpu = PySTARCConfig(
        ligand_pqr=str(pqr_b),
        rxns_xml=str(rxns_xml),
        r_hydro_lig=8.5,
        n_trajectories=4,
        max_steps=6,
        bd_milestone_radius=25.0,
        gpu=False,
        seed=101,
        confidence_interval=0.90,
        convergence_tol=0.08,
        work_dir=tmp_path / "out_cpu_rec",
        chain=ChainConfig(
            flexible_partner="receptor",
            chain_json=str(chain_a_json),
            target_grid_dx=str(fine_dx),
            born_grid_dx=str(born_dx),
            auto_diffusion=True,
        ),
    ).validate()
    out_cpu = chain_pipeline.run_chain(cfg_cpu)
    rj_cpu = json.loads((out_cpu / "results.json").read_text())
    assert rj_cpu["chain"]["flexible_partner"] == "receptor"
    assert rj_cpu["summary"]["n_reacted"] == 4
    assert rj_cpu["rate"]["confidence_level"] == pytest.approx(0.90)
    assert rj_cpu["rate"]["convergence_tol"] == pytest.approx(0.08)

    if _HAS_CUDA:
        cfg_gpu = PySTARCConfig(
            ligand_pqr=str(pqr_b),
            rxns_xml=str(rxns_xml),
            r_hydro_lig=8.5,
            n_trajectories=6,
            max_steps=6,
            bd_milestone_radius=25.0,
            gpu=True,
            seed=101,
            work_dir=tmp_path / "out_gpu_rec",
            chain=ChainConfig(
                flexible_partner="receptor",
                chain_json=str(chain_a_json),
                target_grid_dx=str(fine_dx),
                coarse_grid_dx=str(coarse_dx),
                born_grid_dx=str(born_dx),
                far_field=True,
                auto_diffusion=True,
            ),
        ).validate()
        out_gpu = chain_pipeline.run_chain(cfg_gpu)
        rj_gpu = json.loads((out_gpu / "results.json").read_text())
        assert rj_gpu["summary"]["n_reacted"] == 6
        assert rj_gpu["chain"]["flexible_partner"] == "receptor"
    else:
        # On a CPU-only machine, explicit <gpu>true</gpu> in XML raises RuntimeError.
        xml_gpu = tmp_path / "explicit_gpu.xml"
        xml_gpu.write_text(
            f"""<pystarc>
  <ligand_pqr>{pqr_b}</ligand_pqr>
  <rxns_xml>{rxns_xml}</rxns_xml>
  <r_hydro_lig>8.5</r_hydro_lig>
  <gpu>true</gpu>
  <work_dir>{tmp_path / 'fail_gpu'}</work_dir>
  <chain>
    <flexible_partner>receptor</flexible_partner>
    <chain_json>{chain_a_json}</chain_json>
    <auto_diffusion>true</auto_diffusion>
  </chain>
</pystarc>"""
        )
        cfg_exp = parse(xml_gpu)
        with pytest.raises(RuntimeError, match="<gpu>true</gpu>"):
            chain_pipeline.run_chain(cfg_exp)

        # Verify GPU dispatch wiring on CPU via xp=np wrapper.
        class _NPBackedGPUSim(GPUChainBDSimulator):
            def __init__(self, *args, **kwargs):
                kwargs["xp"] = np
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(chain_pipeline, "GPUChainBDSimulator", _NPBackedGPUSim)
        cfg_exp.work_dir = tmp_path / "sim_gpu_np"
        out_sim = chain_pipeline.run_chain(cfg_exp)
        rj_sim = json.loads((out_sim / "results.json").read_text())
        assert rj_sim["summary"]["n_reacted"] == cfg_exp.n_trajectories


def test_run_pystarc_chain_summary_and_multi_gpu_pooling(tmp_path, monkeypatch):
    _, pqr_a, _ = _write_synthetic_complex_pdb_and_pqrs(tmp_path)
    common = chain_from_sequence("AKRD", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    pos0 -= pos0.mean(axis=0)
    c_json = save_chain_to_json(common, pos0, tmp_path / "chain.json")
    rp_json = tmp_path / "rp.json"
    rp_json.write_text(json.dumps([[0, 0, 12.0], [1, 1, 12.0]]))
    dx_file = _write_dx(tmp_path / "parta1.dx", [-18.0, -18.0, -18.0], 3.0, 13, -2.0)

    gpu_flag = "true" if _HAS_CUDA else "false"
    xml_path = tmp_path / "input.xml"
    xml_path.write_text(
        f"""<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>partner_a.pqr</receptor_pqr>
  <r_hydro_rec>8.0</r_hydro_rec>
  <n_trajectories>9</n_trajectories>
  <max_steps>15</max_steps>
  <bd_milestone_radius>16.0</bd_milestone_radius>
  <dt>0.2</dt>
  <gpu>{gpu_flag}</gpu>
  <seed>50</seed>
  <confidence_interval>0.90</confidence_interval>
  <convergence_check>true</convergence_check>
  <convergence_tol>0.15</convergence_tol>
  <work_dir>bd_single</work_dir>
  <outputs>
    <save_interval>2</save_interval>
  </outputs>
  <chain>
    <flexible_partner>ligand</flexible_partner>
    <chain_json>chain.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
    <target_grid_dx>parta1.dx</target_grid_dx>
    <dt_chain>0.05</dt_chain>
    <chain_steps_per_outer>4</chain_steps_per_outer>
    <reaction_n_needed>1</reaction_n_needed>
    <auto_diffusion>true</auto_diffusion>
  </chain>
</pystarc>"""
    )

    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "run_pystarc.py"
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}
    proc = subprocess.run(
        [sys.executable, str(runner), str(xml_path)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Flexible role  : ligand" in proc.stdout
    assert "90% CI" in proc.stdout
    single_dir = tmp_path / "bd_single"
    assert (single_dir / "results.json").is_file()
    assert (single_dir / "convergence.json").is_file()

    # Now test multi_GPU_runs.py + shard execution + combine_data.py on a <chain> run.
    split_xml = tmp_path / "input_split.xml"
    split_xml.write_text(
        xml_path.read_text().replace("<work_dir>bd_single</work_dir>", "<work_dir>bd_multi</work_dir>")
    )
    monkeypatch.setattr(
        sys, "argv", ["multi_GPU_runs.py", str(split_xml), "--n-splits", "3"]
    )
    assert multi_GPU_runs.main() == 0
    bd_multi = tmp_path / "bd_multi"
    split_meta = json.loads((bd_multi / "split.json").read_text())
    assert [s["n_trajectories"] for s in split_meta["shards"]] == [3, 3, 3]

    for shard in split_meta["shards"]:
        sdir = bd_multi / shard["dir"]
        sproc = subprocess.run(
            [sys.executable, str(runner), "input.xml"],
            cwd=str(sdir),
            env=env,
            capture_output=True,
            text=True,
        )
        assert sproc.returncode == 0, sproc.stderr + sproc.stdout

    monkeypatch.setattr(
        sys, "argv", ["combine_data.py", "--bd-sims", str(bd_multi)]
    )
    assert combine_data.main() == 0
    pooled = json.loads((bd_multi / "results.json").read_text())
    assert pooled["summary"]["n_trajectories"] == 9
    assert pooled["chain"]["flexible_partner"] == "ligand"
    assert pooled["rate"]["confidence_level"] == pytest.approx(0.90)
    assert (bd_multi / "trajectories.csv").is_file()
    assert (bd_multi / "paths.npz").is_file()
    assert (bd_multi / "energetics.npz").is_file()
    assert (bd_multi / "convergence.json").is_file()
    paths_npz = np.load(bd_multi / "paths.npz")
    assert set(np.unique(paths_npz["traj_id"])).issubset(set(range(9)))
    assert int(paths_npz["traj_id"].max()) >= 6


def test_cli_chain_simulation_options(tmp_path):
    _, pqr_a, _ = _write_synthetic_complex_pdb_and_pqrs(tmp_path)
    common = chain_from_sequence("AKRD", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    pos0 -= pos0.mean(axis=0)
    c_json = save_chain_to_json(common, pos0, tmp_path / "c.json")
    born_dx = _write_dx(tmp_path / "b0.dx", [-15.0, -15.0, -15.0], 3.0, 11, 0.2)
    rxn_xml = tmp_path / "r.xml"
    rxn_xml.write_text(
        """<reactions><reaction name="r" n_needed="1">
        <contact molecule1_index="0" molecule2_index="0" distance="200.0"/>
        </reaction></reactions>"""
    )

    runner = CliRunner()
    args = [
        "chain_simulation",
        "--chain",
        str(c_json),
        "--target",
        str(pqr_a),
        "--rxn",
        str(rxn_xml),
        "--born-dx",
        str(born_dx),
        "--target-hydro-radius",
        "7.5",
        "--auto-diffusion",
        "--n",
        "3",
        "--max-steps",
        "5",
        "--r-start",
        "20.0",
        "--seed",
        "9",
        "-o",
        str(tmp_path / "cli_out"),
    ]
    if _HAS_CUDA:
        args.append("--gpu")
    res = runner.invoke(cli, args)
    assert res.exit_code == 0, res.output
    assert (tmp_path / "cli_out" / "results.json").is_file()
