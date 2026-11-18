"""Orchestrator for chain Brownian-dynamics simulations launched from input.xml.

When pystarc.pipeline.input_parser.parse() returns a PySTARCConfig whose .chain
field is set, meaning the input.xml contained a <chain> block, run_pystarc.py
delegates to run_chain() defined here. This follows the same flow as
pystarc.pipeline.pipeline.run() for rigid-body BD. It loads the chain topology,
the target structure, the reaction pairs, and the APBS grids, constructs a
ChainBDSimulator, runs the trajectories, and writes outputs to the work_dir.

In Stage 1 the output is the minimal set written by write_chain_results, namely
results.json and trajectories.csv. Stage 2 will extend the writer to produce
the same comprehensive set of files as the rigid-body work_dir.
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
from pystarc.simulation.gpu_chain_simulator import GPUChainBDSimulator
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
from pystarc.pipeline.pipeline import _require_gpu
from pystarc.global_defs.defaults import GHOST_NAMES
from pystarc.structures.pqr_io import parse_pqr
import numpy as np

_DEFAULT_CHAIN_SIM = ChainBDSimulator
_DEFAULT_GPU_CHAIN_SIM = GPUChainBDSimulator


def _gpu_available() -> bool:
    try:
        import cupy as cp

        cp.arange(3).sum()
        return True
    except Exception:
        return False


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


def _ensure_chain_apbs_grids(config: PySTARCConfig) -> None:
    """Generate the APBS electrostatic grids if they are needed but absent.

    If target_grid_dx or born_grid_dx are named in the chain config but do not
    exist on disk, run APBS to create them using parameters drawn from the
    top-level PySTARCConfig. This plays the same role as the run_apbs_both step
    in the rigid-body pipeline.run(). In chain BD there is only a single target
    molecule, since the chain itself is built from sequence and has no PQR file,
    so the single-molecule run_apbs is called here.

    By convention target_grid_dx follows the "{work_dir}/{mol_name}1.dx" pattern
    produced by run_apbs, where the trailing "1" marks the fine grid. The
    mol_name is recovered by stripping trailing digits from the filename stem.
    """
    if config.chain is None:
        return
    cc = config.chain
    if not cc.target_grid_dx:
        if cc.born_grid_dx:
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

    if (
        target_dx.exists()
        and (coarse_dx is None or coarse_dx.exists())
        and (born_dx is None or born_dx.exists())
    ):
        return

    apbs_work_dir = target_dx.parent
    apbs_work_dir.mkdir(parents=True, exist_ok=True)
    stem = target_dx.stem
    mol_name = stem.rstrip("0123456789")
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

    rigid_pqr, _, pqr_tag = _rigid_partner_pqr_and_r_hydro(config)
    src_pqr = Path(rigid_pqr)
    if not src_pqr.exists():
        raise FileNotFoundError(f"{pqr_tag} not found: {src_pqr} (needed for APBS)")
    dst_pqr = apbs_work_dir / src_pqr.name
    shutil.copy(str(src_pqr), str(dst_pqr))

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

    # Copy input.xml into the work_dir for reproducibility unless it is already there.
    if input_xml_path is not None:
        dst_xml = work_dir / "input.xml"
        if Path(input_xml_path).resolve() != dst_xml.resolve():
            shutil.copy(str(input_xml_path), str(dst_xml))

    # Load the chain topology and the initial atom positions, which are already
    # centered at the origin.
    print(f"Loading chain from {cc.chain_json} ...")
    chain_template, body_positions = load_chain_from_json(cc.chain_json)
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
    use_gpu = (
        ChainBDSimulator is _DEFAULT_CHAIN_SIM
        and bool(config.gpu)
        and (
            GPUChainBDSimulator is not _DEFAULT_GPU_CHAIN_SIM
            or _gpu_available()
        )
    )
    if (
        ChainBDSimulator is _DEFAULT_CHAIN_SIM
        and bool(config.gpu)
        and not use_gpu
        and getattr(config, "_gpu_explicit", False)
    ):
        _require_gpu()

    # Load the target structure as a Molecule.
    print(f"Loading rigid target ({pqr_tag}) from {rigid_pqr} ...")
    target_mol = parse_pqr(rigid_pqr)
    print(f"  target: {target_mol}  (flexible_partner={role})")

    # Load the reaction pairs from JSON or rxns_xml and convert them to a PathwaySet.
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

    # If the grids are named but missing on disk, generate them with APBS, just
    # as the rigid-body pipeline.run() does.
    _ensure_chain_apbs_grids(config)

    # Load the APBS grids if they were specified.
    target_grid = None
    coarse_grid = None
    born_grid = None
    elec_grids = []
    born_grids = []
    if cc.target_grid_dx:
        print(f"Loading target grid {cc.target_grid_dx} ...")
        target_grid = DXGrid.from_file(cc.target_grid_dx)
        elec_grids.append(target_grid)
    if cc.coarse_grid_dx:
        print(f"Loading coarse grid {cc.coarse_grid_dx} ...")
        coarse_grid = DXGrid.from_file(cc.coarse_grid_dx)
        elec_grids.append(coarse_grid)
    if cc.born_grid_dx:
        print(f"Loading Born grid {cc.born_grid_dx} ...")
        born_grid = DXGrid.from_file(cc.born_grid_dx)
        born_grids.append(born_grid)

    multipole_exp = None
    if cc.far_field and config.multipole_fallback and len(target_mol.atoms) > 0:
        t_pos = np.array([[a.x, a.y, a.z] for a in target_mol.atoms], dtype=float)
        t_chg = np.array([float(getattr(a, "charge", 0.0)) for a in target_mol.atoms], dtype=float)
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
    params = ChainBDParameters(
        n_trajectories=config.n_trajectories,
        dt=config.dt,
        dt_chain=cc.dt_chain,
        chain_steps_per_outer=cc.chain_steps_per_outer,
        n_equilibration_steps=cc.n_equilibration_steps,
        max_steps=config.max_steps,
        r_start=config.bd_milestone_radius,
        r_escape=r_escape,
        seed=config.seed,
        n_threads=cc.n_workers,
        verbose=True,
        use_soft_repulsion=cc.use_soft_repulsion,
        soft_repulsion_eps=cc.soft_repulsion_eps,
        use_self_born=cc.use_self_born,
        gb_eps_in=cc.gb_eps_in,
        gb_eps_out=cc.gb_eps_out,
        gb_obc_alpha=cc.gb_obc_alpha,
        gb_obc_beta=cc.gb_obc_beta,
        gb_obc_gamma=cc.gb_obc_gamma,
        coffdrop_active=tables,
        debye_length=config.debye_length,
        temperature_kT=KB_KCAL * config.temperature,
        dielectric=config.sdie,
        hydrodynamic_interactions=config.hydrodynamic_interactions,
        use_brownian_bridge=config.brownian_bridge,
        use_hard_sphere=cc.use_hard_sphere,
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
    )
    if cc.auto_diffusion:
        sim_kwargs["auto_diffusion"] = True
    else:
        sim_kwargs["D_trans"] = d_trans
        sim_kwargs["D_rot"] = d_rot
    sim_kwargs["outputs"] = config.outputs

    if use_gpu:
        conv_every = (
            max(
                1,
                int(
                    round(
                        config.n_trajectories * config.convergence_interval / 100.0
                    )
                ),
            )
            if config.convergence_interval > 0
            else 0
        )
        print("Constructing GPUChainBDSimulator ...")
        sim = GPUChainBDSimulator(
            **sim_kwargs,
            elec_grids=elec_grids,
            born_grids=born_grids,
            use_far_field=cc.far_field,
            multipole_expansion=multipole_exp,
            batch_size=config.gpu_force_batch,
            convergence_interval=conv_every,
            checkpoint_interval=config.checkpoint_interval,
            work_dir=work_dir,
        )
    else:
        print("Constructing ChainBDSimulator ...")
        cpu_extra = {}
        if ChainBDSimulator is _DEFAULT_CHAIN_SIM or len(elec_grids) > 1 or cc.far_field:
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
