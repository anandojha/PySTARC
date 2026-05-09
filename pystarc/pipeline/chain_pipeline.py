"""Orchestrator for chain BD simulations launched via input.xml.

When pystarc.pipeline.input_parser.parse() returns a PySTARCConfig
with .chain set (i.e. the input.xml contained a <chain> block),
run_pystarc.py delegates to run_chain() here. This mirrors
pystarc.pipeline.pipeline.run() for rigid-body BD: it loads the
chain topology, target structure, reaction pairs, and APBS grids;
constructs a ChainBDSimulator; runs the trajectories; and writes
outputs to the work_dir.

For Stage 1, output is the minimal set written by write_chain_results
(results.json and trajectories.csv). Stage 2 will expand the writer
to produce the same comprehensive set as the rigid-body work_dir.
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
    """Load reaction pairs from a JSON file with format
    [[target_atom_idx, chain_atom_idx, distance_A], ...].

    Returns
    -------
    list of (target_idx, chain_idx, distance_cutoff) 3-tuples.
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
    single-reaction PathwaySet with n_needed contact pairs required to
    satisfy the criterion.
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


def run_chain(
    config: PySTARCConfig,
    input_xml_path: Optional[Path] = None,
) -> Path:
    """Run a chain BD simulation from a parsed PySTARCConfig.

    Parameters
    ----------
    config : PySTARCConfig
        Parsed configuration with config.chain populated. Validation
        ensures chain_json and receptor_pqr are present.
    input_xml_path : Path, optional
        Path to the original input.xml. If provided, a copy is placed
        in the work_dir for reproducibility, matching the
        rigid-body convention.

    Returns
    -------
    Path : the work_dir directory containing output files.
    """
    if config.chain is None:
        raise ValueError("run_chain() requires config.chain to be set")
    cc = config.chain

    # Resolve work_dir (matches rigid-body single-run convention).
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy input.xml into work_dir for reproducibility.
    if input_xml_path is not None:
        shutil.copy(str(input_xml_path), str(work_dir / "input.xml"))

    # Load chain topology + initial atom positions (already centered at origin).
    print(f"Loading chain from {cc.chain_json} ...")
    chain_template, body_positions = load_chain_from_json(cc.chain_json)
    print(
        f"  chain: {chain_template.name} "
        f"({len(chain_template.atoms)} atoms, "
        f"{len(chain_template.bonds)} bonds, "
        f"{len(chain_template.angles)} angles, "
        f"{len(chain_template.torsions)} torsions)"
    )

    # Load target structure as Molecule.
    print(f"Loading target from {config.receptor_pqr} ...")
    target_mol = parse_pqr(config.receptor_pqr)
    print(f"  target: {target_mol}")

    # Load reaction pairs from JSON and convert to PathwaySet.
    if not cc.reaction_pairs_json:
        raise ValueError("<chain><reaction_pairs_json> is required in chain BD mode")
    print(f"Loading reaction pairs from {cc.reaction_pairs_json} ...")
    reaction_pairs = _load_reaction_pairs_json(cc.reaction_pairs_json)
    pathway_set = _build_pathway_set(reaction_pairs, cc.reaction_n_needed)
    print(
        f"  reactions: {len(reaction_pairs)} contact pairs "
        f"(n_needed={cc.reaction_n_needed})"
    )

    # Load APBS grids if specified.
    target_grid = None
    born_grid = None
    if cc.target_grid_dx:
        print(f"Loading target grid {cc.target_grid_dx} ...")
        target_grid = DXGrid.from_file(cc.target_grid_dx)
    if cc.born_grid_dx:
        print(f"Loading Born grid {cc.born_grid_dx} ...")
        born_grid = DXGrid.from_file(cc.born_grid_dx)

    # Resolve r_escape sentinel: 0 means 1.1 * r_start.
    r_escape = cc.r_escape if cc.r_escape > 0.0 else (config.bd_milestone_radius * 1.1)

    # Validate diffusion mode.
    if cc.auto_diffusion and (cc.D_trans > 0.0 or cc.D_rot > 0.0):
        raise ValueError(
            "auto_diffusion cannot be combined with explicit D_trans or D_rot. "
            "Either enable auto_diffusion (compute D from chain geometry) or "
            "set D_trans and D_rot explicitly."
        )

    # Build ChainBDParameters.
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

    # Resolve scalar D when auto_diffusion is False; pass None to the simulator
    # for unspecified values so it applies its own defaults.
    d_trans = cc.D_trans if cc.D_trans > 0.0 else None
    d_rot = cc.D_rot if cc.D_rot > 0.0 else None

    # Construct the simulator. auto_diffusion vs explicit D is exclusive.
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
    # Thread OutputConfig through so the simulator records the right
    # diagnostic cadence (Stage 2 instrumentation).
    sim_kwargs["outputs"] = config.outputs

    sim = ChainBDSimulator(**sim_kwargs)

    # Run.
    print(f"Running {config.n_trajectories} trajectories ...")
    t_start = time.time()
    results = sim.run()
    wall_time_sec = time.time() - t_start
    print(f"  done: {len(results)} trajectories in {wall_time_sec:.2f}s")

    # Write outputs.
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
