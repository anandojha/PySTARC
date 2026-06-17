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

from pystarc.pipeline.input_parser import PySTARCConfig
from pystarc.simulation.chain_simulator import ChainBDParameters, ChainBDSimulator
from pystarc.pipeline.chain_output_writer import write_chain_results
from pystarc.structures.chain_io import load_chain_from_json
from pystarc.structures.pqr_io import parse_pqr
from pystarc.structures.molecules import ContactPair, ReactionCriteria
from pystarc.pathways.reaction_interface import ReactionInterface, PathwaySet
from pystarc.forces.electrostatic.grid_force import DXGrid


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
        return

    target_dx = Path(cc.target_grid_dx)
    born_dx = Path(cc.born_grid_dx) if cc.born_grid_dx else None

    if target_dx.exists() and (born_dx is None or born_dx.exists()):
        return

    apbs_work_dir = target_dx.parent
    apbs_work_dir.mkdir(parents=True, exist_ok=True)
    stem = target_dx.stem
    mol_name = stem.rstrip("0123456789")
    if not mol_name:
        raise ValueError(
            f"Cannot derive APBS mol_name from {target_dx.name}; "
            f"expected '{{mol_name}}1.dx' pattern (e.g. 'thrombin1.dx')"
        )

    src_pqr = Path(config.receptor_pqr)
    if not src_pqr.exists():
        raise FileNotFoundError(
            f"receptor_pqr not found: {src_pqr} (needed for APBS)"
        )
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

    # Copy input.xml into the work_dir for reproducibility.
    if input_xml_path is not None:
        shutil.copy(str(input_xml_path), str(work_dir / "input.xml"))

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

    # Load the target structure as a Molecule.
    print(f"Loading target from {config.receptor_pqr} ...")
    target_mol = parse_pqr(config.receptor_pqr)
    print(f"  target: {target_mol}")

    # Load the reaction pairs from JSON and convert them to a PathwaySet.
    if not cc.reaction_pairs_json:
        raise ValueError("<chain><reaction_pairs_json> is required in chain BD mode")
    print(f"Loading reaction pairs from {cc.reaction_pairs_json} ...")
    reaction_pairs = _load_reaction_pairs_json(cc.reaction_pairs_json)
    pathway_set = _build_pathway_set(reaction_pairs, cc.reaction_n_needed)
    print(
        f"  reactions: {len(reaction_pairs)} contact pairs "
        f"(n_needed={cc.reaction_n_needed})"
    )

    # If the grids are named but missing on disk, generate them with APBS, just
    # as the rigid-body pipeline.run() does.
    _ensure_chain_apbs_grids(config)

    # Load the APBS grids if they were specified.
    target_grid = None
    born_grid = None
    if cc.target_grid_dx:
        print(f"Loading target grid {cc.target_grid_dx} ...")
        target_grid = DXGrid.from_file(cc.target_grid_dx)
    if cc.born_grid_dx:
        print(f"Loading Born grid {cc.born_grid_dx} ...")
        born_grid = DXGrid.from_file(cc.born_grid_dx)

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
        coffdrop_active=cc.coffdrop_active,
        debye_length=config.debye_length,
    )

    # Resolve the scalar diffusion coefficients when auto_diffusion is off. Pass
    # None to the simulator for any value left unspecified so that it applies
    # its own defaults.
    d_trans = cc.D_trans if cc.D_trans > 0.0 else None
    d_rot = cc.D_rot if cc.D_rot > 0.0 else None

    # Construct the simulator. Automatic diffusion and explicit D are mutually
    # exclusive.
    print("Constructing ChainBDSimulator ...")
    sim_kwargs = dict(
        target=target_mol,
        chain_template=chain_template,
        chain_init_body_positions=body_positions,
        params=params,
        pathway_set=pathway_set,
        target_grid=target_grid,
        born_grid=born_grid,
        desolvation_alpha=config.desolvation_alpha,
    )
    if cc.auto_diffusion:
        sim_kwargs["auto_diffusion"] = True
    else:
        sim_kwargs["D_trans"] = d_trans
        sim_kwargs["D_rot"] = d_rot
    # Pass the OutputConfig through so the simulator records diagnostics at the
    # correct cadence, which is part of the Stage 2 instrumentation.
    sim_kwargs["outputs"] = config.outputs

    sim = ChainBDSimulator(**sim_kwargs)

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
    )
    print(f"Wrote {len(written)} output files to {work_dir}:")
    for name, path in written:
        print(f"  {name}: {path}")

    return work_dir
