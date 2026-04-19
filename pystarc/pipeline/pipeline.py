"""
PySTARC pipeline - Master orchestrator
=====================================
Chains all steps from a single PDB file to k_on:
  Step 1: Extract ligand + receptor from PDB
  Step 2: Parameterize ligand (antechamber + parmchk2 + tleap)
  Step 3: Build gas-phase complex -> PQR files
  Step 4: Run APBS -> all DX grids (electrostatic + born)
  Step 5: Compute geometry (b-sphere, hydro radii, ghost atoms)
  Step 6: Run PySTARC BD simulation -> k_on + 95% CI
Entry point: run() called by run_pystarc.py
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
from pystarc.pipeline.geometry import compute_geometry, auto_detect_reactions
from pystarc.simulation.nam_simulator import NAMSimulator, NAMParameters
from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator
from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.pipeline.input_parser import PySTARCConfig
from pystarc.pipeline.parameterize import parameterize
from pystarc.pipeline.output_writer import write_all
from pystarc.forces.engine import load_dx_directory
from pystarc.pipeline.run_apbs import run_apbs_both
from pystarc.structures.pqr_io import parse_pqr
from pystarc.pipeline.extract import extract
import subprocess as _sp
from pathlib import Path
import shutil
import time

try:
    import cupy as cp
except ImportError:
    cp = None


def run(cfg: PySTARCConfig):
    """
    Full pipeline: PDB -> k_on.
    All intermediate files go into cfg.work_dir.
    """
    t0 = time.time()
    W = cfg.work_dir
    print(f"  PDB           : {cfg.pdb}")
    print(f"  Ligand resname: {cfg.ligand_resname}")
    print(f"  Work dir      : {W}")
    print(f"  b-surface     : {cfg.bd_milestone_radius:.1f} Å  (start sphere)")
    rxn_r = (
        cfg.bd_milestone_radius_inner
        if cfg.bd_milestone_radius_inner > 0
        else cfg.bd_milestone_radius
    )
    print(f"  Reaction (q)  : {rxn_r:.1f} Å  (reaction sphere)")
    print(f"  Escape sphere : {cfg.bd_milestone_radius * 2:.1f} Å  (= 2 × b-surface)")
    print(f"  Trajectories  : {cfg.n_trajectories:,}")
    print(f"  Threads       : {cfg.n_threads}")
    print(f"  GPU           : {cfg.gpu}")
    print()
    # Steps 1-3: Extract + Build PQR files
    # Shortcut: if receptor_pqr / ligand_pqr are provided in the input XML,
    # skip PDB extraction and AmberTools/tleap entirely.
    # Required for: pre-computed PQRs
    if cfg.receptor_pqr and cfg.ligand_pqr:
        print("[1] Using pre-computed PQR files - skipping PDB extraction.")
        receptor_pqr = Path(cfg.receptor_pqr)
        ligand_pqr = Path(cfg.ligand_pqr)
        # Resolve relative paths against the XML file's directory
        xml_dir = Path(cfg.work_dir).parent
        if not receptor_pqr.is_absolute():
            # try next to the XML first, then cwd
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
        # Copy into work dir for APBS
        if receptor_pqr.resolve() != (W / "receptor.pqr").resolve():
            shutil.copy(receptor_pqr, W / "receptor.pqr")
        if ligand_pqr.resolve() != (W / "ligand.pqr").resolve():
            shutil.copy(ligand_pqr, W / "ligand.pqr")
        receptor_pqr = W / "receptor.pqr"
        ligand_pqr = W / "ligand.pqr"
        print("[3] PQR files ready (pre-computed).")
    else:
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
    # Center molecules + inject GHO ghost atoms
    # GHO is placed at the centroid (origin) of each molecule.
    # Required for standard b-surface reaction criterion:
    # GHO-GHO distance < bd_milestone_radius (b-surface)
    print("\n[3b] Centring molecules and injecting GHO ghost atoms ...")
    for pqr_path in (receptor_pqr, ligand_pqr):
        atoms = read_pqr(pqr_path)
        # Skip if GHO already present (e.g. user supplied pre-injected PQRs)
        if any(a.name.strip().upper() == "GHO" for a in atoms):
            print(f"  {pqr_path.name}: GHO already present - skipping")
            continue
        atoms = centre_at_origin(atoms)
        atoms = _inject_gho(atoms)
        write_pqr(atoms, pqr_path)
        print(f"  {pqr_path.name}: centred + GHO injected at (0,0,0) done")
    # Step 4: Run APBS
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
    )
    # Step 5: Geometry
    geom = compute_geometry(
        receptor_pqr,
        ligand_pqr,
        bd_milestone_radius=cfg.bd_milestone_radius,
        bd_milestone_radius_inner=cfg.bd_milestone_radius_inner,
        srad=getattr(cfg, "srad", 0.0),
        r_hydro_rec=getattr(cfg, "r_hydro_rec", 0.0),
        r_hydro_lig=getattr(cfg, "r_hydro_lig", 0.0),
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
    # Step 6: BD simulation
    print("\n[6] Running BD simulation ...")
    # Load molecules
    mol_rec = parse_pqr(receptor_pqr)
    mol_lig = parse_pqr(ligand_pqr)
    # Load force engine (GPU -> Numba -> NumPy, auto-detected)
    engine = load_dx_directory(
        W,
        mol1_prefix="receptor",
        mol2_prefix="ligand",
        debye_length=cfg.debye_length,
        desolvation_alpha=cfg.desolvation_alpha,
    )
    print(f"  {engine.summary()}")
    # Override GPU if user disabled it
    if not cfg.gpu and engine.backend == "cupy":
        engine.backend = "numba"
        print("  GPU disabled by config - using Numba")
    # Mobility tensor with full RPY coupling
    mob = MobilityTensor.from_radii(
        geom.receptor.hydrodynamic_r,
        geom.ligand.hydrodynamic_r,
        use_rpy=True,
        T=cfg.temperature,
    )
    D_rel = mob.relative_translational_diffusion()
    print(
        f"  r_hydro: receptor={geom.receptor.hydrodynamic_r:.3f} Å  "
        f"ligand={geom.ligand.hydrodynamic_r:.3f} Å"
    )
    print(f"  D_rel  = {D_rel:.5f} Å²/ps")
    # Build reaction pathway from detected criteria
    reactions = []
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
    pathway_set = PathwaySet(reactions)
    # NAM parameters
    params = NAMParameters(
        n_trajectories=cfg.n_trajectories,
        dt=getattr(cfg, "dt", 0.2),
        dt_rxn=0.05,  # ps
        minimum_core_dt=getattr(cfg, "minimum_core_dt", 0.0),
        max_steps=cfg.max_steps,
        r_start=geom.r_start,
        r_escape=geom.r_escape,
        seed=cfg.seed,
        n_threads=cfg.n_threads,
        hydrodynamic_interactions=getattr(cfg, "hydrodynamic_interactions", False),
        use_hard_sphere=True,
        verbose=True,
    )
    sim = NAMSimulator(mol_rec, mol_lig, mob, pathway_set, params, engine)
    if cfg.gpu and engine.backend == "cupy":
        print("\n  Using GPU Batch Simulator - all trajectories on GPU simultaneously")

        def _load_dx(prefix, suffix):
            return [
                DXGrid.from_file(p) for p in sorted(W.glob(f"{prefix}[0-9]{suffix}"))
            ]

        # Multipole expansion for far-field (dipole + quadrupole)
        _mp_expansion = None
        if cfg.multipole_fallback:
            _mp_expansion = MultipoleExpansion(
                positions=mol_rec.positions_array(),
                charges=mol_rec.charges_array(),
                debye_length=cfg.debye_length,
                sdie=cfg.sdie,
            )
            print(_mp_expansion.summary())
        batch_engine = GPUBatchForceEngine(
            elec_grids=_load_dx("receptor", ".dx"),
            born_grids=_load_dx("receptor", "_born.dx"),
            alpha=cfg.desolvation_alpha,
            receptor_charge=float(mol_rec.total_charge()),
            debye_length=cfg.debye_length,
            sdie=cfg.sdie,
            # Born both-directions: lig Born grid on rec atoms
            lig_born_grids=_load_dx("ligand", "_born.dx"),
            rec_positions=mol_rec.positions_array(),
            rec_charges=mol_rec.charges_array(),
            multipole_expansion=_mp_expansion,
            rec_radii=mol_rec.radii_array() if cfg.lj_forces else None,
            lig_radii=mol_lig.radii_array() if cfg.lj_forces else None,
            use_lj=cfg.lj_forces,
        )
        gpu_sim = GPUBatchSimulator(
            mol_rec, mol_lig, mob, pathway_set, params, batch_engine
        )
        # Attach config so simulator knows what to collect
        params._output_cfg = cfg.outputs
        params.max_dt = getattr(cfg, "max_dt", 0.0)
        params.checkpoint_interval = cfg.checkpoint_interval
        params.convergence_interval = cfg.convergence_interval
        params._work_dir = str(cfg.work_dir)
        params._kT_scale = cfg.temperature / 298.15
        params._overlap_check = cfg.overlap_check
        result = gpu_sim.run()
        elapsed = result.elapsed_sec
        total_steps = int(result.steps_per_sec * elapsed)
    else:
        print(f"\n  Using CPU NAM Simulator ({cfg.n_threads} threads)")
        result = sim.run()
        elapsed = time.time() - t0
        total_steps = sum(r.steps for r in sim.results)
    # Print results
    # Gather hardware info for summary footer
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
    # summary() already includes: P_rxn, k_on, CI, Wall time, BD steps/sec, Backend
    # Pass Romberg k_b for two-level k_on formula (GPU: use Romberg; CPU: 0 = Smoluchowski)
    _k_b = getattr(gpu_sim, "_k_b", 0.0) if cfg.gpu else 0.0
    print(result.summary(D_rel, _k_b, cfg.confidence_interval))
    print(f"  Total steps  : {total_steps:,}")
    # Hardware footer
    if n_gpu > 0:
        gpu_info = f"  ({gpu_name})" if gpu_name else ""
        print(f"  GPU count    : {n_gpu}{gpu_info}")
    else:
        print(f"  CPU threads  : {cfg.n_threads}")
    print("=" * 64)
    # Checklist (printed to log)
    _hi = getattr(cfg, "hydrodynamic_interactions", False)
    _rh_r = getattr(cfg, "r_hydro_rec", 0.0)
    _rh_l = getattr(cfg, "r_hydro_lig", 0.0)
    _desolv = getattr(cfg, "desolvation_alpha", 0.0)
    _mcd = getattr(cfg, "minimum_core_dt", 0.0)
    print()
    print("Checklist")
    _kT_scale = cfg.temperature / 298.15
    print(f"  Temperature = {cfg.temperature:.2f} K  (kT = {_kT_scale:.6f})")
    print(f"  k_b Romberg integral           (k_b={_k_b:.4f} ų/ps)")
    print(f"  HI in k_b integral             ({'enabled' if _hi else 'disabled'})")
    print(
        f"  Hydro radii from XML           (rec={_rh_r:.4f}, lig={_rh_l:.4f})"
        if _rh_r > 0
        else "  Hydro radii from MC"
    )
    print(f"  Fine-only APBS grid + Yukawa monopole fallback")
    if cfg.multipole_fallback:
        print(f"  Multipole far-field (monopole + dipole + quadrupole)")
    else:
        print(f"  Multipole far-field disabled (monopole only)")
    if cfg.overlap_check:
        print(f"  Overlap check (elastic wall at receptor surface)")
    else:
        print(f"  Overlap check disabled")
    if cfg.lj_forces:
        print(f"  WCA repulsive forces (LJ from PQR radii)")
    else:
        print(
            f"  LJ forces disabled (use <lj_forces>true</lj_forces> for tight contact)"
        )
    print(
        f"  Born both directions            (alpha={_desolv:.8f})"
        if _desolv > 0
        else "  Born desolvation disabled (alpha=0)"
    )
    print(f"  Variable dt (pair_dt + force_dt + edge_dt)")
    print(
        f"  minimum_core_dt = {_mcd:.1f} ps"
        if _mcd > 0
        else "  No minimum_core_dt floor"
    )
    print(f"  Isotropic D_rel for BD step")
    print(f"  Outer propagator: return_prob at r_escape")
    print(f"  Diffusional rotation on return from r_escape")
    print(f"  Brownian bridge at reaction surface")
    print(f"  Position refresh after return (prevents r-overshoot)")
    # Write output files
    if cfg.gpu and hasattr(result, "sim_data") and result.sim_data is not None:
        write_all(
            work_dir=cfg.work_dir,
            result=result,
            sim_data=result.sim_data,
            output_cfg=cfg.outputs,
            k_b=_k_b,
            D_rel=D_rel,
            confidence=cfg.confidence_interval,
        )
    # Clean up temporary files
    tmp_dir = cfg.work_dir / "tmp"
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result
