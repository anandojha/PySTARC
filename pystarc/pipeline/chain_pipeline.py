"""Orchestrator for chain Brownian-dynamics simulations launched from input.xml.

When pystarc.pipeline.input_parser.parse() returns a PySTARCConfig whose .chain
field is set, meaning the input.xml contained a <chain> block, run_pystarc.py
delegates to run_chain() defined here. This follows the same flow as
pystarc.pipeline.pipeline.run() for rigid-body BD. It loads the chain topology,
the target structure, the reaction pairs, and the APBS grids, constructs a
ChainBDSimulator, runs the trajectories, and writes outputs to the work_dir.

Outputs are written to work_dir via write_chain_results (results.json,
trajectories.csv, encounters, near misses, radial/angular distributions,
contact frequencies, milestone fluxes, paths, and energetics).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import json
import shutil
import time

from pystarc.global_defs.constants import KB_KCAL
from pystarc.hydrodynamics.rotne_prager import (
    stokes_rotational_diffusion,
    stokes_translational_diffusion,
)
from pystarc.simulation.chain_simulator import ChainBDParameters, ChainBDSimulator
from pystarc.structures.molecules import ContactPair, ReactionCriteria
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.pipeline.chain_output_writer import write_chain_results
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.structures.chain_io import load_chain_from_json
from pystarc.xml_io.simulation_io import parse_reaction_xml
from pystarc.simulation.coffdrop_chain import elastic_network
from pystarc.pipeline.input_parser import PySTARCConfig
from pystarc.pipeline.geometry import analyse_molecule
from pystarc.global_defs.defaults import DT_RXN, GHOST_NAMES
from pystarc.structures.pqr_io import parse_pqr
import numpy as np

_DEFAULT_CHAIN_SIM = ChainBDSimulator


def _rigid_partner_pqr_and_r_hydro(config: PySTARCConfig) -> Tuple[str, float, str]:
    """Return (rigid_pqr_path, r_hydro_override, tag_name) for the rigid partner."""
    role = (config.chain.flexible_partner or "ligand").strip().lower()
    if role == "receptor":
        return config.ligand_pqr, float(config.r_hydro_lig), "ligand_pqr"
    return config.receptor_pqr, float(config.r_hydro_rec), "receptor_pqr"


def _load_reaction_pairs_json(path: str) -> List[Tuple[int, int, float]]:
    """Load reaction pairs from a JSON file.

    The file holds a list of entries of the form
    [[target_atom_idx, chain_atom_idx, distance_A], ...]. The function returns a
    list of (target_idx, chain_idx, distance_cutoff) three-tuples, where the
    cutoff distance is given in angstrom.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reaction_pairs_json not found: {p}")
    with open(p) as fh:
        data = json.load(fh)
    pairs = []
    for entry in data:
        if len(entry) != 3:
            raise ValueError(
                f"reaction_pairs_json entry has length {len(entry)}; "
                f"expected 3 (target_atom_idx, chain_atom_idx, distance_A)"
            )
        pairs.append((int(entry[0]), int(entry[1]), float(entry[2])))
    return pairs


def _build_pathway_set(
    reaction_pairs: List[Tuple[int, int, float]],
    n_needed: int,
) -> PathwaySet:
    """Convert a list of (target_idx, chain_idx, cutoff_A) tuples into a
    single-reaction PathwaySet. The reaction criterion is satisfied once
    n_needed of the listed contact pairs are simultaneously within their cutoff
    distances.
    """
    pairs = [
        ContactPair(t_idx, c_idx, cutoff) for (t_idx, c_idx, cutoff) in reaction_pairs
    ]
    # n_needed = -1 means "require all pairs" (the ReactionCriteria default). Any
    # positive value larger than the number of pairs can never be reached, so the
    # reaction criterion would be silently unsatisfiable and k_on would come out
    # as exactly 0. Surface that misconfiguration instead of failing quietly.
    if n_needed != -1 and int(n_needed) > len(pairs):
        raise ValueError(
            f"reaction_n_needed={int(n_needed)} exceeds the number of reaction "
            f"pairs ({len(pairs)}); the criterion can never be satisfied. Set "
            f"reaction_n_needed <= {len(pairs)} (or -1 to require all pairs)."
        )
    crit = ReactionCriteria(
        name="association",
        pairs=pairs,
        n_needed=int(n_needed),
    )
    rxn = ReactionInterface(name="association", criteria=crit)
    return PathwaySet([rxn])


def _dx_file_looks_complete(path: Path) -> bool:
    """Return True when an OpenDX file is non-empty, has a valid header, and is not truncated."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False
    try:
        with open(path, "rb") as fh:
            head_bytes = fh.read(16384)
            tail_seek = max(0, size - 4096)
            fh.seek(tail_seek)
            tail_bytes = fh.read(4096)
    except OSError:
        return False
    head = head_bytes.decode("utf-8", errors="replace")
    expected = None
    has_obj3 = False
    for raw_line in head.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("object 1"):
            parts = line.split()
            if len(parts) < 3:
                return False
            try:
                nx, ny, nz = int(parts[-3]), int(parts[-2]), int(parts[-1])
            except ValueError:
                return False
            if nx <= 0 or ny <= 0 or nz <= 0:
                return False
            expected = nx * ny * nz
        elif line.startswith("object 3"):
            has_obj3 = True
            break
    if expected is None or not has_obj3:
        return False
    # Each ASCII value needs at least one character plus one separator byte.
    if size < expected * 2:
        return False

    tail = tail_bytes.decode("utf-8", errors="replace")
    tail_lines = [
        line.strip()
        for line in tail.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tail_lines:
        return False
    has_trailer = any(
        line.startswith(("attribute ", 'object "', "component ")) for line in tail_lines
    )
    if has_trailer:
        for line in reversed(tail_lines):
            parts = line.split()
            if (
                len(parts) == 4
                and parts[0] == "component"
                and parts[1] == '"data"'
                and parts[2] == "value"
                and parts[3].isdigit()
            ):
                return True
        return False

    # Minimal DX files without an OpenDX trailer must end with a newline
    # and hold the full expected grid value count.
    if not tail_bytes.endswith((b"\n", b"\r")):
        return False
    count = 0
    last_data_line = ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            in_data = False
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if not in_data:
                    if line.startswith("object 3"):
                        in_data = True
                    continue
                if line.startswith("object") or line.startswith("attribute"):
                    break
                toks = line.split()
                count += len(toks)
                last_data_line = line
    except OSError:
        return False
    if count != expected or not last_data_line:
        return False
    try:
        vals = [float(v) for v in last_data_line.split()]
    except ValueError:
        return False
    return all(np.isfinite(v) for v in vals)


def _ensure_chain_apbs_grids(config: PySTARCConfig) -> None:
    """Generate APBS electrostatic grids when missing or when cache hash mismatches."""
    if config.chain is None:
        return
    cc = config.chain
    if not cc.target_grid_dx:
        if cc.born_grid_dx or cc.coarse_born_grid_dx:
            raise ValueError(
                "<chain><born_grid_dx> requires <chain><target_grid_dx> to be "
                "set as well; the APBS run that produces the Born grid is keyed "
                "off the target grid path. Set target_grid_dx alongside "
                "born_grid_dx."
            )
        return

    target_dx = Path(cc.target_grid_dx)
    coarse_dx = Path(cc.coarse_grid_dx) if cc.coarse_grid_dx else None
    born_dx = Path(cc.born_grid_dx) if cc.born_grid_dx else None
    coarse_born_dx = Path(cc.coarse_born_grid_dx) if cc.coarse_born_grid_dx else None

    apbs_work_dir = target_dx.parent
    stem = target_dx.stem
    mol_name = stem.rstrip("0123456789")

    required_dx = [
        p for p in (target_dx, coarse_dx, born_dx, coarse_born_dx) if p is not None
    ]
    all_exist = all(p.exists() for p in required_dx)
    stale_key = False
    corrupt_dx: List[Path] = []
    key_path = apbs_work_dir / f"{mol_name}_grids.key" if mol_name else None
    if all_exist:
        corrupt_dx = [p for p in required_dx if not _dx_file_looks_complete(p)]
        if key_path is None or not key_path.is_file():
            if not corrupt_dx:
                return
        else:
            rigid_pqr, _, pqr_tag = _rigid_partner_pqr_and_r_hydro(config)
            src_pqr = Path(rigid_pqr)
            if not src_pqr.exists():
                raise FileNotFoundError(
                    f"{pqr_tag} not found: {src_pqr} (needed for APBS)"
                )
            from pystarc.pipeline.run_apbs import _grid_key

            expected_key = _grid_key(
                src_pqr,
                dict(
                    ion_conc=config.ion_concentration,
                    debye_length=config.debye_length,
                    pdie=config.pdie,
                    sdie=config.sdie,
                    srad=config.srad,
                    temp=config.temperature,
                    dime=getattr(config, "apbs_dime", 129),
                    coarse_dime=getattr(config, "apbs_coarse_dime", 0),
                    fine_dime=getattr(config, "apbs_fine_dime", 0),
                    cglen=getattr(config, "apbs_cglen", 0.0),
                    fglen=getattr(config, "apbs_fglen", 0.0),
                    ion_radius_pos=getattr(config, "ion_radius_pos", 0.95),
                    ion_radius_neg=getattr(config, "ion_radius_neg", 1.81),
                ),
            )
            if not corrupt_dx and key_path.read_text().strip() == expected_key:
                return
            if not corrupt_dx:
                stale_key = True

    if not mol_name:
        raise ValueError(
            f"Cannot derive APBS mol_name from {target_dx.name}; "
            f"expected '{{mol_name}}1.dx' pattern (e.g. 'target1.dx')"
        )

    produced_elec_names = {f"{mol_name}0.dx", f"{mol_name}1.dx"}
    if target_dx.name not in produced_elec_names:
        raise ValueError(
            f"target_grid_dx '{target_dx.name}' does not match the "
            f"'{{mol_name}}1.dx' naming pattern produced by APBS. With "
            f"mol_name='{mol_name}', APBS writes the electrostatic grids "
            f"'{mol_name}0.dx' (coarse) and '{mol_name}1.dx' (fine) in "
            f"{apbs_work_dir}, neither of which is the requested file, so it "
            f"would be absent after the APBS run. Rename target_grid_dx to "
            f"'{mol_name}1.dx' (the fine grid used for force evaluation), or "
            f"point it at an existing DX grid."
        )

    target_dir_resolved = apbs_work_dir.resolve()
    for tag, grid_path in (
        ("coarse_grid_dx", coarse_dx),
        ("born_grid_dx", born_dx),
        ("coarse_born_grid_dx", coarse_born_dx),
    ):
        if grid_path is not None and grid_path.parent.resolve() != target_dir_resolved:
            raise ValueError(
                f"<chain><{tag}> parent directory ({grid_path.parent}) does not "
                f"match <chain><target_grid_dx> parent directory ({apbs_work_dir}). "
                f"APBS writes all grids into {apbs_work_dir}. Put all generated "
                f"grid paths in the same directory or point them at existing DX files."
            )

    apbs_work_dir.mkdir(parents=True, exist_ok=True)
    rigid_pqr, _, pqr_tag = _rigid_partner_pqr_and_r_hydro(config)
    src_pqr = Path(rigid_pqr)
    if not src_pqr.exists():
        raise FileNotFoundError(f"{pqr_tag} not found: {src_pqr} (needed for APBS)")
    dst_pqr = apbs_work_dir / src_pqr.name
    if src_pqr.resolve() != dst_pqr.resolve():
        shutil.copy(str(src_pqr), str(dst_pqr))

    if corrupt_dx:
        if key_path is not None:
            key_path.unlink(missing_ok=True)
        names = ", ".join(p.name for p in corrupt_dx)
        print(
            f"APBS DX grid empty or truncated ({names}); regenerating in {apbs_work_dir}/ ..."
        )
    elif stale_key:
        print(f"APBS DX grid key mismatch; regenerating in {apbs_work_dir}/ ...")
    else:
        print(f"APBS DX grids missing; generating in {apbs_work_dir}/ ...")
    from pystarc.pipeline.run_apbs import run_apbs

    run_apbs(
        pqr_path=src_pqr,
        mol_name=mol_name,
        work_dir=apbs_work_dir,
        ion_conc=config.ion_concentration,
        debye_length=config.debye_length,
        dielectric_in=config.pdie,
        dielectric_out=config.sdie,
        srad=config.srad,
        temp=config.temperature,
        ion_radius_pos=getattr(config, "ion_radius_pos", 0.95),
        ion_radius_neg=getattr(config, "ion_radius_neg", 1.81),
        cglen_override=getattr(config, "apbs_cglen", 0.0),
        fglen_override=getattr(config, "apbs_fglen", 0.0),
        dime=getattr(config, "apbs_dime", 129),
        coarse_dime=getattr(config, "apbs_coarse_dime", 0),
        fine_dime=getattr(config, "apbs_fine_dime", 0),
    )


def run_chain(
    config: PySTARCConfig,
    input_xml_path: Optional[Path] = None,
) -> Path:
    """Run a chain BD simulation from a parsed PySTARCConfig.

    The config argument is a parsed configuration with config.chain populated.
    Validation guarantees that chain_json and receptor_pqr are present. The
    optional input_xml_path is the path to the original input.xml. When it is
    given, a copy is placed in the work_dir for reproducibility, following the
    rigid-body convention.

    The function returns the work_dir directory that holds the output files.
    """
    if config.chain is None:
        raise ValueError("run_chain() requires config.chain to be set")
    cc = config.chain

    # Resolve the work_dir, following the rigid-body single-run convention.
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy input.xml into work_dir unless already on disk or a temporary grid-gen XML.
    if input_xml_path is not None and not (
        Path(input_xml_path).name.startswith("_grid_gen")
        and Path(input_xml_path).suffix == ".xml"
    ):
        dst_xml = work_dir / "input.xml"
        if Path(input_xml_path).resolve() != dst_xml.resolve():
            shutil.copy(str(input_xml_path), str(dst_xml))

    # Load the chain topology and centered initial atom positions.
    print(f"Loading chain from {cc.chain_json} ...")
    chain_template, body_positions = load_chain_from_json(cc.chain_json)
    if (
        abs(float(config.temperature) - 298.15) > 1e-6 or cc.nb_attractive_scale != 1.0
    ) and chain_template.coffdrop_params is not None:
        cp_cur = chain_template.coffdrop_params
        files = getattr(cp_cur, "files", None) or {}
        req_keys = ("ff_xml", "mapping_xml", "connectivity_xml", "charges_xml")
        if all(k in files and Path(files[k]).is_file() for k in req_keys):
            from pystarc.simulation.coffdrop_params import COFFDROPParams

            chain_template.coffdrop_params = COFFDROPParams.load(
                ff_xml=files["ff_xml"],
                mapping_xml=files["mapping_xml"],
                connectivity_xml=files["connectivity_xml"],
                charges_xml=files["charges_xml"],
                nb_attractive_scale=cc.nb_attractive_scale,
                temperature=config.temperature,
            )
        else:
            from pystarc.simulation.coffdrop_params import TabulatedPotential

            target_temp = float(config.temperature)
            if target_temp <= 0.0:
                raise ValueError(f"temperature must be positive, got {target_temp}")
            cur_temp = float(getattr(cp_cur, "temperature", 298.15))
            temp_ratio = cur_temp / target_temp
            cur_scale = float(getattr(cp_cur, "nb_attractive_scale", 1.0))
            attr_ratio = (
                float(cc.nb_attractive_scale) / cur_scale
                if cur_scale != 0.0
                else float(cc.nb_attractive_scale)
            )

            def _rescale_pots(pots, apply_attr: bool):
                out = []
                for pot in pots:
                    vals = pot.values.copy() * temp_ratio
                    if apply_attr and attr_ratio != 1.0:
                        vals[vals < 0.0] *= attr_ratio
                    out.append(
                        TabulatedPotential(
                            x_min=pot.x_min,
                            x_max=pot.x_max,
                            values=vals,
                            residues=pot.residues,
                            atoms=pot.atoms,
                            orders=pot.orders,
                            index=pot.index,
                            kind=pot.kind,
                            eq_length=getattr(pot, "eq_length", None),
                        )
                    )
                return out

            cp_cur.pair_pots = _rescale_pots(cp_cur.pair_pots, apply_attr=True)
            if abs(temp_ratio - 1.0) > 1e-12:
                cp_cur.angle_pots = _rescale_pots(cp_cur.angle_pots, apply_attr=False)
                cp_cur.dihedral_pots = _rescale_pots(
                    cp_cur.dihedral_pots, apply_attr=False
                )
                if getattr(cp_cur, "bond_pots", None):
                    cp_cur.bond_pots = _rescale_pots(cp_cur.bond_pots, apply_attr=False)
            cp_cur.nb_attractive_scale = float(cc.nb_attractive_scale)
            cp_cur.temperature = target_temp
            cp_cur._pot_index_cache.clear()
            if hasattr(cp_cur, "_diff_index_cache"):
                cp_cur._diff_index_cache.clear()
            cp_cur.__dict__.pop("_tables", None)
    print(
        f"  chain: {chain_template.name} "
        f"({len(chain_template.atoms)} atoms, "
        f"{len(chain_template.bonds)} bonds, "
        f"{len(chain_template.angles)} angles, "
        f"{len(chain_template.torsions)} torsions)"
    )
    # The COFFDROP tables come with the chain JSON. Without them the angles
    # and torsions are harmonic and there are no pair potentials.
    tables = chain_template.coffdrop_params is not None and bool(
        chain_template.pair_lookups
    )
    if cc.coffdrop_active and not tables:
        raise ValueError(
            f"<chain><coffdrop_active> is true but {cc.chain_json} carries no "
            "COFFDROP tables; rebuild the chain JSON"
        )
    if tables:
        n_ang = sum(a.type_idx >= 0 for a in chain_template.angles)
        n_tor = sum(t.type_idx >= 0 for t in chain_template.torsions)
        print(
            f"  COFFDROP tables: {n_ang} of {len(chain_template.angles)} angles, "
            f"{n_tor} of {len(chain_template.torsions)} torsions, "
            f"{len(chain_template.pair_lookups)} pairs"
        )
    else:
        print("  COFFDROP tables: none (harmonic angles and cosine torsions)")
    if cc.elastic_network_cutoff > 0.0:
        chain_template.network = elastic_network(
            chain_template,
            body_positions,
            cc.elastic_network_cutoff,
            cc.elastic_network_k,
        )
        print(
            f"  elastic network: {len(chain_template.network)} springs, cutoff "
            f"{cc.elastic_network_cutoff} A, k {cc.elastic_network_k} kBT/A^2"
        )

    role = (cc.flexible_partner or "ligand").strip().lower()
    if role not in ("ligand", "receptor"):
        raise ValueError(
            f"<chain><flexible_partner> must be 'ligand' or 'receptor', got {cc.flexible_partner!r}"
        )
    rigid_pqr, r_hydro_override, pqr_tag = _rigid_partner_pqr_and_r_hydro(config)
    if config.gpu:
        print(
            "Note: main executes flexible-chain BD on CPU "
            "(GPU flexible-chain support lives on branch pystarc_flexible_gpu); "
            "running CPU ChainBDSimulator."
        )

    # Load the target structure as a Molecule.
    print(f"Loading rigid target ({pqr_tag}) from {rigid_pqr} ...")
    target_mol = parse_pqr(rigid_pqr)
    print(f"  target: {target_mol}  (flexible_partner={role})")

    # Load the reaction pairs from JSON or rxns_xml and convert to a PathwaySet.
    if cc.reaction_pairs_json:
        print(f"Loading reaction pairs from {cc.reaction_pairs_json} ...")
        reaction_pairs = _load_reaction_pairs_json(cc.reaction_pairs_json)
        pathway_set = _build_pathway_set(reaction_pairs, cc.reaction_n_needed)
        print(
            f"  reactions: {len(reaction_pairs)} contact pairs "
            f"(n_needed={cc.reaction_n_needed})"
        )
    elif config.rxns_xml:
        print(f"Loading reaction XML from {config.rxns_xml} ...")
        pathway_set = parse_reaction_xml(config.rxns_xml)
        if role == "receptor":
            for rxn in pathway_set.reactions:
                for pair in rxn.criteria.pairs:
                    pair.mol1_atom_index, pair.mol2_atom_index = (
                        pair.mol2_atom_index,
                        pair.mol1_atom_index,
                    )
        print(f"  reactions: {pathway_set}")
    else:
        raise ValueError("<chain><reaction_pairs_json> is required in chain BD mode")

    # Generate missing or stale APBS grids before loading them.
    _ensure_chain_apbs_grids(config)

    # Load the APBS grids when specified.
    target_grid = None
    coarse_grid = None
    born_grid = None
    elec_grids = []
    born_grids = []
    seen_elec_paths = set()
    if cc.target_grid_dx:
        t_path = Path(cc.target_grid_dx)
        print(f"Loading target grid {cc.target_grid_dx} ...")
        target_grid = DXGrid.from_file(cc.target_grid_dx)
        elec_grids.append(target_grid)
        seen_elec_paths.add(t_path.resolve())
    if cc.coarse_grid_dx:
        c_path = Path(cc.coarse_grid_dx)
        if c_path.exists() and c_path.resolve() not in seen_elec_paths:
            print(f"Loading coarse grid {cc.coarse_grid_dx} ...")
            coarse_grid = DXGrid.from_file(cc.coarse_grid_dx)
            elec_grids.append(coarse_grid)
            seen_elec_paths.add(c_path.resolve())
    if coarse_grid is None and cc.target_grid_dx:
        t_path = Path(cc.target_grid_dx)
        if t_path.stem.endswith("1"):
            auto_coarse_e = t_path.parent / f"{t_path.stem[:-1]}0.dx"
            if (
                auto_coarse_e.exists()
                and auto_coarse_e.resolve() not in seen_elec_paths
            ):
                print(f"Loading coarse grid {auto_coarse_e} ...")
                coarse_grid = DXGrid.from_file(str(auto_coarse_e))
                elec_grids.append(coarse_grid)
                seen_elec_paths.add(auto_coarse_e.resolve())
    seen_born_paths = set()
    if cc.born_grid_dx:
        b_path = Path(cc.born_grid_dx)
        if b_path.stem.endswith("0_born"):
            auto_fine_b = b_path.parent / f"{b_path.stem[:-6]}1_born.dx"
            if auto_fine_b.exists() and auto_fine_b.resolve() not in seen_born_paths:
                print(f"Loading fine Born grid {auto_fine_b} ...")
                born_grids.append(DXGrid.from_file(str(auto_fine_b)))
                seen_born_paths.add(auto_fine_b.resolve())
        print(f"Loading Born grid {cc.born_grid_dx} ...")
        born_grids.append(DXGrid.from_file(cc.born_grid_dx))
        seen_born_paths.add(b_path.resolve())
    if cc.coarse_born_grid_dx:
        cb_path = Path(cc.coarse_born_grid_dx)
        if cb_path.exists() and cb_path.resolve() not in seen_born_paths:
            print(f"Loading coarse Born grid {cc.coarse_born_grid_dx} ...")
            born_grids.append(DXGrid.from_file(cc.coarse_born_grid_dx))
            seen_born_paths.add(cb_path.resolve())
    elif cc.born_grid_dx:
        b_path = Path(cc.born_grid_dx)
        if b_path.stem.endswith("1_born"):
            auto_coarse = b_path.parent / f"{b_path.stem[:-6]}0_born.dx"
            if auto_coarse.exists() and auto_coarse.resolve() not in seen_born_paths:
                print(f"Loading coarse Born grid {auto_coarse} ...")
                born_grids.append(DXGrid.from_file(str(auto_coarse)))
                seen_born_paths.add(auto_coarse.resolve())
    born_grid = born_grids[0] if born_grids else None

    multipole_exp = None
    if cc.far_field and config.multipole_fallback and len(target_mol.atoms) > 0:
        t_pos = np.array([[a.x, a.y, a.z] for a in target_mol.atoms], dtype=float)
        t_chg = np.array(
            [float(getattr(a, "charge", 0.0)) for a in target_mol.atoms], dtype=float
        )
        multipole_exp = MultipoleExpansion(
            t_pos,
            t_chg,
            debye_length=config.debye_length,
            sdie=config.sdie,
            kT_scale=config.temperature / 298.15,
        )

    # Resolve the r_escape sentinel. A value of 0 means 2.0 × r_start, the LMZ
    # convention, which matches the rigid-body pipeline.
    r_escape = cc.r_escape if cc.r_escape > 0.0 else (config.bd_milestone_radius * 2.0)

    # Validate the diffusion mode.
    if cc.auto_diffusion and (cc.D_trans > 0.0 or cc.D_rot > 0.0):
        raise ValueError(
            "auto_diffusion cannot be combined with explicit D_trans or D_rot. "
            "Either enable auto_diffusion (compute D from chain geometry) or "
            "set D_trans and D_rot explicitly."
        )
    if not cc.auto_diffusion and (cc.D_trans <= 0.0 or cc.D_rot <= 0.0):
        raise ValueError(
            "<chain><D_trans> and <chain><D_rot> must both be > 0 when "
            "auto_diffusion is off, got D_trans="
            f"{cc.D_trans} and D_rot={cc.D_rot}. Set both, or enable "
            "auto_diffusion to compute D from the chain geometry."
        )

    # The target rotates about the frame origin, so its atom centroid must sit
    # there. The setup scripts center the PQR, and an uncentered PQR would turn
    # the target about the wrong point.
    target_xyz = np.array(
        [
            [a.x, a.y, a.z]
            for a in target_mol.atoms
            if str(getattr(a, "name", "")).strip().upper() not in GHOST_NAMES
        ],
        dtype=float,
    ).reshape(-1, 3)
    target_offset = float(np.linalg.norm(target_xyz.mean(axis=0)))
    if target_offset > 0.01:
        raise ValueError(
            f"the target atom centroid is {target_offset:.4f} A from the origin "
            f"in {rigid_pqr}; center the target at the origin first "
            "(the limit is 0.01 A)"
        )

    # The hydrodynamic radius of the target, from the input or from the Hansen
    # Monte Carlo radius with its cache, as in the rigid-body pipeline.
    if r_hydro_override > 0.0:
        a_target = r_hydro_override
    else:
        a_target = float(
            analyse_molecule(
                Path(rigid_pqr), srad=getattr(config, "srad", 0.0)
            ).hydrodynamic_r
        )
    if not a_target > 0.0:
        raise ValueError(f"the target hydrodynamic radius must be > 0, got {a_target}")
    print(
        f"  target hydrodynamic radius a_T = {a_target:.3f} A, "
        f"D_t,T = {stokes_translational_diffusion(a_target):.6g} A^2/ps, "
        f"D_r,T = {stokes_rotational_diffusion(a_target):.6g} rad^2/ps"
    )

    # Build the ChainBDParameters.
    dt_rxn = min(
        config.dt,
        (
            config.minimum_core_reaction_dt
            if config.minimum_core_reaction_dt > 0.0
            else DT_RXN
        ),
    )
    coffdrop_active = (
        bool(cc.coffdrop_active)
        if getattr(cc, "coffdrop_active_explicit", False)
        else bool(cc.coffdrop_active or tables)
    )
    params = ChainBDParameters(
        n_trajectories=config.n_trajectories,
        dt=config.dt,
        dt_rxn=dt_rxn,
        dt_chain=cc.dt_chain,
        chain_steps_per_outer=cc.chain_steps_per_outer,
        n_equilibration_steps=cc.n_equilibration_steps,
        max_steps=config.max_steps,
        r_start=config.bd_milestone_radius,
        r_escape=r_escape,
        use_lmz=bool(cc.use_lmz),
        seed=config.seed,
        n_threads=cc.n_workers,
        verbose=True,
        use_soft_repulsion=cc.use_soft_repulsion,
        soft_repulsion_eps=cc.soft_repulsion_eps,
        wca_core_cap_fraction=cc.wca_core_cap_fraction,
        use_self_born=cc.use_self_born,
        gb_eps_in=cc.gb_eps_in,
        gb_eps_out=cc.gb_eps_out,
        gb_obc_alpha=cc.gb_obc_alpha,
        gb_obc_beta=cc.gb_obc_beta,
        gb_obc_gamma=cc.gb_obc_gamma,
        coffdrop_active=coffdrop_active,
        intra_debye_huckel=bool(cc.intra_debye_huckel and (tables or coffdrop_active)),
        enable_born2=bool(config.enable_born2_torque),
        grid_float32=bool(cc.grid_float32),
        debye_length=config.debye_length,
        temperature_kT=KB_KCAL * config.temperature,
        dielectric=config.sdie,
        hydrodynamic_interactions=config.hydrodynamic_interactions,
        use_brownian_bridge=config.brownian_bridge,
        use_hard_sphere=cc.use_hard_sphere,
        overlap_scale=cc.overlap_scale,
        rotate_body_tensors=True,
        remove_internal_rotation=True,
        inner_field=True,
        diffusion_update_interval=(
            cc.diffusion_update_interval if cc.auto_diffusion else 0
        ),
    )

    d_trans = cc.D_trans
    d_rot = cc.D_rot

    sim_kwargs = dict(
        target=target_mol,
        chain_template=chain_template,
        chain_init_body_positions=body_positions,
        params=params,
        pathway_set=pathway_set,
        target_grid=target_grid,
        born_grid=born_grid,
        desolvation_alpha=config.desolvation_alpha,
        target_hydro_radius=a_target,
        pdie=config.pdie,
    )
    if cc.auto_diffusion:
        sim_kwargs["auto_diffusion"] = True
    else:
        sim_kwargs["D_trans"] = d_trans
        sim_kwargs["D_rot"] = d_rot
    sim_kwargs["outputs"] = config.outputs

    print("Constructing ChainBDSimulator ...")
    cpu_extra = {}
    if (
        ChainBDSimulator is _DEFAULT_CHAIN_SIM
        or len(elec_grids) > 1
        or len(born_grids) > 1
        or cc.far_field
        or bool(config.enable_born2_torque)
    ):
        cpu_extra = dict(
            elec_grids=elec_grids,
            born_grids=born_grids,
            use_far_field=cc.far_field,
            multipole_expansion=multipole_exp,
        )
    sim = ChainBDSimulator(**sim_kwargs, **cpu_extra)
    sim.flexible_partner = role
    sim.confidence_interval = config.confidence_interval
    sim.convergence_tol = config.convergence_tol

    # Run the trajectories.
    print(f"Running {config.n_trajectories} trajectories ...")
    t_start = time.time()
    results = sim.run()
    wall_time_sec = time.time() - t_start
    print(f"  done: {len(results)} trajectories in {wall_time_sec:.2f}s")

    # Write the outputs.
    written = write_chain_results(
        work_dir=work_dir,
        sim=sim,
        results=results,
        wall_time_sec=wall_time_sec,
        outputs=config.outputs,
    )
    print(f"Wrote {len(written)} output files to {work_dir}:")
    for name, path in written:
        print(f"  {name}: {path}")

    return work_dir
