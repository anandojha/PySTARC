"""
PySTARC pipeline master orchestrator.

This module chains together every step needed to go from a single PDB file to
an association rate constant k_on. Step 1 extracts the ligand and receptor from
the PDB. Step 2 parameterizes the ligand with AmberTools (antechamber, parmchk2,
and tleap). Step 3 builds the gas-phase complex and writes the PQR files. Step 4
runs APBS to produce all of the DX grids, both electrostatic and Born. Step 5
computes the geometry, which includes the b-surface sphere, the hydrodynamic
radii, and the ghost atoms. Step 6 runs the PySTARC Brownian-dynamics simulation
to obtain k_on together with its 95% confidence interval. The entry point is
run(), which is called by run_pystarc.py.
"""

from __future__ import annotations
from pystarc.pipeline.prepare_bd_surface import (
    read_pqr,
    write_pqr,
    centre_at_origin,
    inject_gho as _inject_gho,
)
from pystarc.pathways.reaction_interface import (
    ReactionInterface,
    PathwaySet,
    ReactionCriteria,
    ContactPair,
)
from pystarc.pipeline.make_pqr import build_complex, make_combined_pqr, split_pqr
from pystarc.pipeline.geometry import (
    compute_geometry,
    auto_detect_reactions,
    _parse_rxns_xml_criteria,
    _parse_rxns_xml_reaction_groups,
    check_rxn_atom_numbers,
    check_rxn_atom_range,
)
from pystarc.simulation.nam_simulator import NAMSimulator, NAMParameters
from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator
from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.pipeline.input_parser import PySTARCConfig
from pystarc.pipeline.parameterize import parameterize
from pystarc.pipeline.output_writer import write_all
from pystarc.forces.engine import PySTARCEngine
from pystarc.forces.multipole import load_effective_charges
from pystarc.pipeline.run_apbs import run_apbs_both
from pystarc.structures.pqr_io import parse_pqr
from pystarc.pipeline.extract import extract
import subprocess as _sp
from pathlib import Path
import shutil
from pystarc.global_defs.defaults import (
    DEBYE_LENGTH,
    DT_RXN,
    HYDRODYNAMIC_INTERACTIONS,
    SOLVENT_DIELECTRIC,
    TEMPERATURE,
)

try:
    import cupy as cp
except ImportError:
    cp = None



def _run_key(cfg, W: Path) -> str:
    """Fingerprint a checkpoint must carry to be resumed: SHA-256 of the parsed
    settings without the run-control keys, and of the PQR, rxns and DX inputs."""
    import hashlib

    skip = {
        "n_trajectories", "seed", "max_steps", "work_dir", "n_threads",
        "gpu_force_batch", "checkpoint_interval", "convergence_interval",
        "confidence_interval", "outputs",
    }
    items = sorted((k, repr(v)) for k, v in vars(cfg).items() if k not in skip)
    h = hashlib.sha256(repr(items).encode())
    rxns = [Path(cfg.rxns_xml.strip())] if cfg.rxns_xml.strip() else []
    for p in sorted(W.glob("*.pqr")) + sorted(W.glob("*.dx")) + rxns:
        if p.exists():
            m = hashlib.md5()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    m.update(block)
            h.update(p.name.encode() + m.digest())
    return h.hexdigest()

def _checklist(cfg, params, geom, k_b, gpu_engine, cpu_engine, result):
    """Checklist lines built from the settings and the engines the run used."""
    e = gpu_engine
    alpha = cfg.desolvation_alpha
    src = lambda v: "XML" if v > 0 else "MC"
    lines = [
        "Checklist",
        f"  Temperature = {cfg.temperature:.2f} K  (kT = {cfg.temperature / 298.15:.6f})",
        f"  k_b Romberg integral           (k_b={k_b:.4f} ų/ps)"
        if e is not None
        else f"  k_b of the outer propagator    (k_db={getattr(result, 'k_db', 0.0):.4f} ų/ps)",
        f"  HI in k_b integral             ({'enabled' if params.hydrodynamic_interactions else 'disabled'})",
        f"  Hydro radii  receptor {geom.receptor.hydrodynamic_r:.4f} Å ({src(cfg.r_hydro_rec)}), "
        f"ligand {geom.ligand.hydrodynamic_r:.4f} Å ({src(cfg.r_hydro_lig)})",
    ]
    if e is not None:
        lines.append(
            f"  APBS grids finest first ({len(e._elec_grids_gpu)} elec, {len(e._born_grids_gpu)} Born)"
        )
        lines.append(
            "  Multipole far-field (monopole + dipole + quadrupole)"
            if e._has_yukawa and e._multipole is not None
            else "  Monopole far-field" if e._has_yukawa
            else "  No far-field force beyond the grids"
        )
        born1 = alpha != 0 and len(e._born_grids_gpu) > 0
        born2 = born1 and bool(e._lig_born_grids_gpu)
        if params._soft_repulsion:
            lines.append("  Soft repulsion kT (0.9345 (r_i + r_j) / r)^12 (replaces the overlap check)")
        elif params._overlap_check:
            lines.append("  Overlap check (overlapping moves are rejected)")
        else:
            lines.append("  Overlap check disabled")
        lines.append(
            "  WCA repulsive forces (LJ from PQR radii)" if e._use_lj else "  LJ forces disabled"
        )
    else:
        lines.append("  APBS grids finest first, no far-field force beyond the grids")
        born1 = alpha != 0 and bool(cpu_engine._born1)
        born2 = born1 and bool(cpu_engine._born2)
        lines.append("  Hard-sphere check (always on in the CPU model)")
    lines.append(
        f"  Born both directions            (alpha={alpha:.8f})" if born2
        else f"  Born on the ligand atoms only   (alpha={alpha:.8f})" if born1
        else "  Born desolvation off"
    )
    lines.append("  Variable dt (pair_dt + force_dt + edge_dt)")
    lines.append(
        f"  minimum_core_dt = {cfg.minimum_core_dt:g} ps"
        if cfg.minimum_core_dt > 0
        else "  No minimum_core_dt floor"
    )
    lines.append(
        "  Isotropic D(r) with HI for the BD step"
        if params.hydrodynamic_interactions
        else "  Isotropic D_rel for BD step"
    )
    lines += [
        "  Outer propagator: return_prob at r_escape",
        "  Diffusional rotation on return from r_escape",
        "  Brownian bridge at reaction surface"
        if params.use_brownian_bridge
        else "  Brownian bridge off (reaction test at the end of each step)",
        "  Position refresh after return (prevents r-overshoot)",
    ]
    return lines


def run(cfg: PySTARCConfig):
    """
    Run the full pipeline that takes a PDB file and returns k_on.

    All intermediate files are written into cfg.work_dir.
    """
    W = cfg.work_dir
    print(f"  PDB           : {cfg.pdb}")
    print(f"  Ligand resname: {cfg.ligand_resname}")
    print(f"  Work dir      : {W}")
    print(f"  b-surface     : {cfg.bd_milestone_radius:.1f} Å  (start sphere)")
    print(f"  Reactions     : {cfg.rxns_xml.strip() or 'ghost_atoms ' + cfg.ghost_atoms.strip()}")
    _resc = (
        cfg.r_escape
        if getattr(cfg, "r_escape", 0.0) > 0
        else cfg.bd_milestone_radius * 2
    )
    _resc_note = (
        "= 2 × b-surface"
        if _resc == cfg.bd_milestone_radius * 2
        else "user-set r_escape"
    )
    print(f"  Escape sphere : {_resc:.1f} Å  ({_resc_note})")
    print(f"  Trajectories  : {cfg.n_trajectories:,}")
    print(f"  Threads       : {cfg.n_threads}")
    print(f"  GPU           : {cfg.gpu}")
    print()
    # Steps 1 through 3 extract the molecules and build the PQR files. As a
    # shortcut, if receptor_pqr and ligand_pqr are both provided in the input
    # XML, we skip PDB extraction and AmberTools/tleap entirely and use the
    # pre-computed PQR files directly.
    if cfg.receptor_pqr and cfg.ligand_pqr:
        print("[1] Using pre-computed PQR files - skipping PDB extraction.")
        receptor_pqr = Path(cfg.receptor_pqr)
        ligand_pqr = Path(cfg.ligand_pqr)
        # Resolve any relative paths against the directory containing the XML file.
        xml_dir = Path(cfg.work_dir).parent
        if not receptor_pqr.is_absolute():
            # Look next to the XML file first, then fall back to the current directory.
            candidate = xml_dir / receptor_pqr
            if not candidate.exists():
                candidate = Path.cwd() / receptor_pqr
            receptor_pqr = candidate
        if not ligand_pqr.is_absolute():
            candidate = xml_dir / ligand_pqr
            if not candidate.exists():
                candidate = Path.cwd() / ligand_pqr
            ligand_pqr = candidate
        print("\n[2] Using pre-computed PQR files")
        print(f"  Receptor PQR : {receptor_pqr}")
        print(f"  Ligand PQR   : {ligand_pqr}")
        if not receptor_pqr.exists():
            raise FileNotFoundError(f"receptor_pqr not found: {receptor_pqr}")
        if not ligand_pqr.exists():
            raise FileNotFoundError(f"ligand_pqr not found: {ligand_pqr}")
        # Copy the files into the work directory so that APBS can read them. A
        # link left there is replaced by a private copy, so the centring below
        # never writes into the file it points to.
        for src, dst in ((receptor_pqr, W / "receptor.pqr"), (ligand_pqr, W / "ligand.pqr")):
            if dst.is_symlink() or src.resolve() != dst.resolve():
                data = src.read_bytes()
                dst.unlink(missing_ok=True)
                dst.write_bytes(data)
        receptor_pqr = W / "receptor.pqr"
        ligand_pqr = W / "ligand.pqr"
        print("[3] PQR files ready (pre-computed).")
    else:
        print("\n[1] Extracting ligand and receptor from PDB ...")
        # extract() returns the receptor and ligand PDB paths. Only the ligand
        # PDB is used downstream, because build_complex below takes pdb_path from
        # cfg.pdb directly. The receptor PDB is discarded.
        _receptor_pdb, ligand_pdb = extract(cfg.pdb, cfg.ligand_resname, W)
        print("\n[2] Parameterizing ligand with AmberTools ...")
        mol2_path, frcmod_path, lib_path = parameterize(
            ligand_pdb=ligand_pdb,
            ligand_resname=cfg.ligand_resname,
            ligand_charge=cfg.ligand_charge,
            work_dir=W,
            ligand_ff=cfg.ligand_ff,
        )
        print("\n[3] Building PQR files ...")
        prmtop, complex_pdb = build_complex(
            pdb_path=cfg.pdb,
            mol2_path=mol2_path,
            frcmod_path=frcmod_path,
            lib_path=lib_path,
            ligand_resname=cfg.ligand_resname,
            work_dir=W,
            protein_ff=cfg.protein_ff,
            ligand_ff=cfg.ligand_ff,
        )
        combined_pqr = make_combined_pqr(prmtop, complex_pdb, W)
        receptor_pqr, ligand_pqr = split_pqr(combined_pqr, cfg.ligand_resname, W)
    # Center the molecules and inject GHO ghost atoms. The GHO atom is placed at
    # the centroid (the origin) of each molecule. It is needed for the standard
    # b-surface reaction criterion, which fires when the GHO-to-GHO distance is
    # less than bd_milestone_radius, the radius of the b-surface.
    print("\n[3b] Centring molecules and injecting GHO ghost atoms ...")
    for pqr_path in (receptor_pqr, ligand_pqr):
        atoms = read_pqr(pqr_path)
        # Skip injection if a GHO atom is already present, for example when the
        # user supplied PQR files that were already injected.
        if any(a.name.strip().upper() == "GHO" for a in atoms):
            print(f"  {pqr_path.name}: GHO already present - skipping")
            continue
        atoms = centre_at_origin(atoms)
        atoms = _inject_gho(atoms)
        write_pqr(atoms, pqr_path)
        print(f"  {pqr_path.name}: centred + GHO injected at (0,0,0) done")
    # Step 4 runs APBS to generate the electrostatic and Born grids.
    rec_dx, lig_dx = run_apbs_both(
        receptor_pqr=receptor_pqr,
        ligand_pqr=ligand_pqr,
        work_dir=W,
        ion_conc=cfg.ion_concentration,
        debye_length=cfg.debye_length,
        dielectric_in=cfg.pdie,
        dielectric_out=cfg.sdie,
        srad=cfg.srad,
        temp=cfg.temperature,
        ion_radius_pos=getattr(cfg, "ion_radius_pos", 0.95),
        ion_radius_neg=getattr(cfg, "ion_radius_neg", 1.81),
        cglen_override=getattr(cfg, "apbs_cglen", 0.0),
        fglen_override=getattr(cfg, "apbs_fglen", 0.0),
        dime=getattr(cfg, "apbs_dime", 129),
        coarse_dime=getattr(cfg, "apbs_coarse_dime", 0),
        fine_dime=getattr(cfg, "apbs_fine_dime", 0),
        ligand_dime=cfg.apbs_ligand_dime,
    )
    # every Born level present is read, so a third, finer level is used when given
    born = lambda m: sorted(W.glob(f"{m}[0-9]_born.dx"))
    rec_e, rec_b, lig_e, lig_b = (
        [DXGrid.from_file(p) for p in g]
        for g in (rec_dx[:2], born("receptor"), lig_dx[:2], born("ligand"))
    )
    # Step 5 computes the geometry of the system.
    geom = compute_geometry(
        receptor_pqr,
        ligand_pqr,
        bd_milestone_radius=cfg.bd_milestone_radius,
        bd_milestone_radius_inner=cfg.bd_milestone_radius_inner,
        srad=getattr(cfg, "srad", 0.0),
        r_hydro_rec=getattr(cfg, "r_hydro_rec", 0.0),
        r_hydro_lig=getattr(cfg, "r_hydro_lig", 0.0),
        r_escape=getattr(cfg, "r_escape", 0.0),
    )
    rxn_result = auto_detect_reactions(
        geom,
        cfg.ghost_atoms,
        cfg.rxns_xml,
        bd_milestone_radius=cfg.bd_milestone_radius,
        bd_milestone_radius_inner=cfg.bd_milestone_radius_inner,
    )
    rxn_stages, rxn_n_needed = (
        rxn_result if isinstance(rxn_result, tuple) else (rxn_result, -1)
    )
    # rxns_xml atom numbers are positions; warn when the PQR serials differ.
    if cfg.rxns_xml.strip():
        check_rxn_atom_numbers(
            [rp for stage in rxn_stages for rp in stage], receptor_pqr, ligand_pqr
        )
    # Step 6 runs the Brownian-dynamics simulation.
    print("\n[6] Running BD simulation ...")
    # Load the receptor and ligand molecules.
    mol_rec = parse_pqr(receptor_pqr)
    mol_lig = parse_pqr(ligand_pqr)
    check_rxn_atom_range(
        [rp for stage in rxn_stages for rp in stage],
        len(mol_rec.atoms),
        len(mol_lig.atoms),
    )
    # Load the force engine, auto-detecting the fastest available backend. The
    # code prefers the GPU (CuPy), then Numba, and finally falls back to NumPy.
    engine = PySTARCEngine(
        elec_mol1=rec_e,
        elec_mol2=lig_e,
        born_mol1=rec_b,
        born_mol2=lig_b,
        eff_charges_mol1=load_effective_charges(W, "receptor", cfg.debye_length),
        debye_length=cfg.debye_length,
        desolvation_alpha=cfg.desolvation_alpha,
    )
    print(f"  {engine.summary()}")
    # If the user disabled the GPU in the configuration, fall back to Numba.
    if not cfg.gpu and engine.backend == "cupy":
        engine.backend = "numba"
        print("  GPU disabled by config - using Numba")
    # Build the mobility tensor. The Rotne-Prager-Yamakawa coupling slows the
    # relative approach as solvent is squeezed from between the two surfaces.
    _hi_flag = getattr(cfg, "hydrodynamic_interactions", HYDRODYNAMIC_INTERACTIONS)
    mob = MobilityTensor.from_radii(
        geom.receptor.hydrodynamic_r,
        geom.ligand.hydrodynamic_r,
        use_rpy=_hi_flag,
        T=cfg.temperature,
    )
    D_rel = mob.relative_translational_diffusion()
    print(
        f"  r_hydro: receptor={geom.receptor.hydrodynamic_r:.3f} Å  "
        f"ligand={geom.ligand.hydrodynamic_r:.3f} Å"
    )
    print(f"  D_rel  = {D_rel:.5f} Å²/ps")
    # Build the reaction pathway from the detected criteria. When
    # state_machine_reactions is True and a rxns_xml file is provided, we
    # preserve the per-reaction grouping and the state labels from the XML file.
    # Otherwise we fall back to the flattened-pairs path, where all contact pairs
    # are placed in a single stage.
    reactions = []
    _use_state_machine = getattr(cfg, "state_machine_reactions", False) and bool(
        cfg.rxns_xml
    )
    if _use_state_machine:
        rxn_groups, first_state = _parse_rxns_xml_reaction_groups(cfg.rxns_xml)
        if rxn_groups:
            print(
                f"  State-machine reactions: {len(rxn_groups)} reaction(s), "
                f"first_state={first_state!r}"
            )
            for rg in rxn_groups:
                pairs = [
                    ContactPair(rp.rec_index, rp.lig_index, rp.cutoff)
                    for rp in rg.pairs
                ]
                crit = ReactionCriteria(
                    name=rg.name,
                    pairs=pairs,
                    n_needed=rg.n_needed,
                    state_before=rg.state_before,
                    state_after=rg.state_after,
                )
                reactions.append(
                    ReactionInterface(
                        name=rg.name,
                        criteria=crit,
                        state_before=rg.state_before,
                        state_after=rg.state_after,
                    )
                )
                print(
                    f"    {rg.name}: {rg.state_before} -> {rg.state_after}  "
                    f"({len(pairs)} pair(s), n_needed={rg.n_needed})"
                )
        else:
            raise ValueError(f"{cfg.rxns_xml} holds no reaction")
    if not _use_state_machine:
        if cfg.rxns_xml.strip():
            _parse_rxns_xml_criteria(Path(cfg.rxns_xml.strip()).expanduser(), single=True)
        for stage_idx, stage_pairs in enumerate(rxn_stages):
            pairs = []
            for rp in stage_pairs:
                pairs.append(ContactPair(rp.rec_index, rp.lig_index, rp.cutoff))
            crit = ReactionCriteria(
                name=f"stage_{stage_idx}",
                pairs=pairs,
                n_needed=rxn_n_needed,
            )
            reactions.append(
                ReactionInterface(
                    name=f"stage_{stage_idx}",
                    criteria=crit,
                )
            )
    # When state-machine mode is active and the rxns.xml file provided a
    # <first_state> tag, pass it through to PathwaySet so that the simulator can
    # correctly initialize each trajectory's current_state. The first_state
    # variable is set in the state-machine branch above. In the flattened-pairs
    # branch it remains None.
    pathway_set = PathwaySet(
        reactions,
        first_state=first_state if "first_state" in dir() else None,
    )
    # Assemble the parameters for the near-association-mode (NAM) simulator.
    params = NAMParameters(
        n_trajectories=cfg.n_trajectories,
        dt=getattr(cfg, "dt", 0.2),
        # The step size used inside the reaction shell. Distinct from
        # minimum_core_reaction_dt, which is a floor whose zero means no floor
        # is imposed.
        dt_rxn=(getattr(cfg, "minimum_core_reaction_dt", 0.0) or DT_RXN),
        max_steps=cfg.max_steps,
        r_start=geom.r_start,
        r_escape=geom.r_escape,
        seed=cfg.seed,
        n_threads=cfg.n_threads,
        hydrodynamic_interactions=_hi_flag,
        use_hard_sphere=True,
        use_brownian_bridge=getattr(cfg, "brownian_bridge", True),
        verbose=True,
        minimum_core_dt=getattr(cfg, "minimum_core_dt", 0.0),
        minimum_core_reaction_dt=getattr(cfg, "minimum_core_reaction_dt", 0.0),
        # Forward the user's screening, temperature, and dielectric so the
        # serial OuterPropagator computes k_db for the configured conditions
        # rather than the NAMParameters defaults. kB = 0.0019872041 kcal/mol/K.
        debye_length=getattr(cfg, "debye_length", DEBYE_LENGTH),
        temperature_kT=0.0019872041 * getattr(cfg, "temperature", TEMPERATURE),
        dielectric=getattr(cfg, "sdie", SOLVENT_DIELECTRIC),
    )
    sim = NAMSimulator(mol_rec, mol_lig, mob, pathway_set, params, engine)
    # Initialize gpu_sim before the GPU branch so that later code which reads it
    # never hits an unbound name. The GPU simulator is only constructed when both
    # cfg.gpu is set and the engine actually selected the cupy backend. If the
    # user requested cfg.gpu but cupy was unavailable and the engine fell back to
    # a CPU backend, this branch is skipped and gpu_sim remains None.
    gpu_sim = None
    batch_engine = None
    if cfg.gpu and engine.backend == "cupy":
        print("\n  Using GPU Batch Simulator - all trajectories on GPU simultaneously")
        # Set up the multipole expansion used for the far-field, which adds the
        # dipole and quadrupole terms beyond the monopole.
        _mp_expansion = None
        if cfg.multipole_fallback:
            _mp_expansion = MultipoleExpansion(
                positions=mol_rec.positions_array(),
                charges=mol_rec.charges_array(),
                debye_length=cfg.debye_length,
                sdie=cfg.sdie,
                kT_scale=cfg.temperature / 298.15,
            )
            print(_mp_expansion.summary())
        batch_engine = GPUBatchForceEngine(
            elec_grids=rec_e,
            born_grids=rec_b,
            alpha=cfg.desolvation_alpha,
            receptor_charge=float(mol_rec.total_charge()),
            debye_length=cfg.debye_length,
            sdie=cfg.sdie,
            # Born desolvation acts in both directions. Here the ligand Born grid
            # is evaluated on the receptor atoms.
            lig_born_grids=lig_b,
            rec_positions=mol_rec.positions_array(),
            rec_charges=mol_rec.charges_array(),
            multipole_expansion=_mp_expansion,
            rec_radii=mol_rec.radii_array() if cfg.lj_forces else None,
            lig_radii=mol_lig.radii_array() if cfg.lj_forces else None,
            use_lj=cfg.lj_forces,
            enable_born2_torque=cfg.enable_born2_torque,
            kT_scale=cfg.temperature / 298.15,
        )
        gpu_sim = GPUBatchSimulator(
            mol_rec, mol_lig, mob, pathway_set, params, batch_engine
        )
        # Attach the configuration so that the simulator knows what to collect.
        params._output_cfg = cfg.outputs
        params.max_dt = getattr(cfg, "max_dt", 0.0)
        params.gpu_force_batch = cfg.gpu_force_batch
        params.checkpoint_interval = cfg.checkpoint_interval
        params.convergence_interval = cfg.convergence_interval
        params._work_dir = str(cfg.work_dir)
        params._kT_scale = cfg.temperature / 298.15
        params._overlap_check = cfg.overlap_check
        params._soft_repulsion = getattr(cfg, "soft_repulsion", False)
        if cfg.checkpoint_interval > 0:
            params._run_key = _run_key(cfg, W)
        result = gpu_sim.run()
        total_steps = result.total_steps
    else:
        if getattr(cfg, "soft_repulsion", False):
            raise ValueError(
                "soft_repulsion runs on the GPU path only. Set <gpu>true</gpu> "
                "on a machine with CuPy."
            )
        print(f"\n  Using CPU NAM Simulator ({cfg.n_threads} threads)")
        result = sim.run()
        total_steps = sum(r.steps for r in sim.results)
    # Print the results. First gather the hardware information for the summary footer.
    n_gpu = 0
    gpu_name = ""
    try:
        if cp is not None:
            n_gpu = cp.cuda.runtime.getDeviceCount()
            r = _sp.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                gpu_name = r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    print()
    print("=" * 64)
    print("  Results")
    print("=" * 64)
    # The GPU k_on is CONV k_b P with the Romberg k_b. The CPU result uses the
    # k_db of its outer propagator and ignores the k_b passed here.
    _k_b = gpu_sim._k_b if gpu_sim is not None else 0.0
    print(result.summary(D_rel, _k_b, cfg.confidence_interval))
    print(f"  Total steps  : {total_steps:,}")
    # Hardware footer
    if n_gpu > 0:
        gpu_info = f"  ({gpu_name})" if gpu_name else ""
        print(f"  GPU count    : {n_gpu}{gpu_info}")
    else:
        print(f"  CPU threads  : {cfg.n_threads}")
    print("=" * 64)
    print()
    for line in _checklist(cfg, params, geom, _k_b, batch_engine, engine, result):
        print(line)
    # Write the output files.
    if cfg.gpu and hasattr(result, "sim_data") and result.sim_data is not None:
        write_all(
            work_dir=cfg.work_dir,
            result=result,
            sim_data=result.sim_data,
            output_cfg=cfg.outputs,
            k_b=_k_b,
            D_rel=D_rel,
            confidence=cfg.confidence_interval,
            convergence_tol=cfg.convergence_tol,
        )
    # Clean up the temporary files.
    tmp_dir = cfg.work_dir / "tmp"
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result
