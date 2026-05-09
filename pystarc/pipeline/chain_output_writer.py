"""
PySTARC chain BD output writer

Produces output files matching the rigid-body OutputConfig schema.
After Stage 2 (sub-stages 2a-2d), 11 of 14 OutputConfig flags are
implemented in chain BD mode.

  Always written:
    results.json        summary, diffusion config, chain summary, params
    trajectories.csv    one row per trajectory

  Conditional (gated on OutputConfig flags):
    encounters.csv         REACTED trajectories: pos/orientation at reaction
    near_misses.csv        non-REACTED: closest approach during trajectory
    paths.npz              per-snapshot COM + orientation + radial
    radial_density.csv     histogram of |COM| separations
    angular_map.npz        2D (theta, phi) occupancy heatmap
    fpt_distribution.csv   histogram of first-passage times
    contact_frequency.csv  aggregate (target, chain) atom-pair contacts
    milestone_flux.csv     shell crossings (outward + inward)
    energetics.npz         per-snapshot energy components

Three OutputConfig flags are accepted but currently no-op in chain BD
mode (file is not produced). Implementing each requires new analysis
code beyond simple file emission:

    p_commit.npz           commitment probability map
                            -> needs shell-conditioned committor analysis
    transition_matrix.npz  Markov state transition matrix
                            -> needs Markov state definitions for chain BD
    pose_clusters.csv      clustered encounter orientations
                            -> needs encounter quaternion clustering

These accept-but-skip flags mean a default OutputConfig (all flags
True) does not error; users see 11 files instead of 14.

This is the chain-side counterpart to pipeline/output_writer.py. The
GPU-batch NAM writer expects a sim_data dict with arrays whose shapes
differ from chain BD natural data, so we keep parallel writers.
"""

from __future__ import annotations
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
import json


def _diffusion_block(sim) -> Dict[str, Any]:
    """Serialize the simulator's diffusion configuration.

    For auto_diffusion mode, includes the full 3x3 tensors plus the
    isotropic-equivalent (trace/3) and the rigid-body hydrodynamic
    center. For scalar mode, just emits scalar D_trans, D_rot.
    """
    if sim.auto_diffusion:
        D_t = np.asarray(sim.D_trans)
        D_r = np.asarray(sim.D_rot)
        return {
            "mode": "auto_diffusion",
            "method": "Rotne-Prager full hydrodynamics",
            "D_trans_3x3": D_t.tolist(),
            "D_rot_3x3": D_r.tolist(),
            "D_trans_isotropic_equiv": float(np.trace(D_t) / 3.0),
            "D_rot_isotropic_equiv": float(np.trace(D_r) / 3.0),
            "D_trans_units": "A^2/ps",
            "D_rot_units": "rad^2/ps",
        }
    else:
        # Scalar path. D_trans / D_rot may also be (3, 3) arrays if the
        # user passed pre-computed tensors; handle both.
        D_t = sim.D_trans
        D_r = sim.D_rot
        if np.ndim(D_t) == 2:
            return {
                "mode": "tensor_explicit",
                "method": "user-supplied tensors",
                "D_trans_3x3": np.asarray(D_t).tolist(),
                "D_rot_3x3": np.asarray(D_r).tolist(),
                "D_trans_isotropic_equiv": float(np.trace(D_t) / 3.0),
                "D_rot_isotropic_equiv": float(np.trace(D_r) / 3.0),
                "D_trans_units": "A^2/ps",
                "D_rot_units": "rad^2/ps",
            }
        return {
            "mode": "scalar",
            "method": "user-supplied scalars",
            "D_trans": float(D_t),
            "D_rot": float(D_r),
            "D_trans_units": "A^2/ps",
            "D_rot_units": "rad^2/ps",
        }


def _chain_block(sim) -> Dict[str, Any]:
    """Summarize the chain template."""
    template = sim.chain_template
    return {
        "name": template.name,
        "n_atoms": template.n_atoms,
        "atom_radii": [float(a.radius) for a in template.atoms],
        "atom_charges": [float(a.charge) for a in template.atoms],
        "n_bonds": len(template.bonds),
        "n_angles": len(template.angles),
        "n_torsions": len(template.torsions),
        "n_length_constraints": len(template.length_constraints),
    }


def _params_block(sim) -> Dict[str, Any]:
    """Serialize all ChainBDParameters fields to a JSON-friendly dict.

    Uses dataclasses.asdict on the canonical ChainBDParameters dataclass
    so any new field is automatically included without touching this
    writer. Test fixtures that pass plain Python classes for sim.params
    are handled via attribute introspection so both production and test
    paths cover the same field set.
    """
    p = sim.params
    if is_dataclass(p):
        return asdict(p)
    # Plain-class fallback: collect public non-callable attributes.
    # Used by test stubs that mock sim.params without a real dataclass.
    return {
        k: getattr(p, k)
        for k in dir(p)
        if not k.startswith("_") and not callable(getattr(p, k))
    }


def _summary_stats(
    results: List[TrajectoryResult],
    max_steps_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate per-trajectory results into summary stats.

    max_steps_cap: pass sim.params.max_steps so timeout detection compares
    against the configured cap. When None (e.g. post-hoc analysis of saved
    results without a sim object), falls back to the maximum observed step
    count via _max_steps_of; this legacy behavior may misclassify the
    longest escape as a timeout when no trajectory actually times out.
    """
    n = len(results)
    n_reacted = sum(1 for r in results if r.fate == Fate.REACTED)
    n_escaped = sum(1 for r in results if r.fate == Fate.ESCAPED)
    n_max_steps = sum(1 for r in results if r.fate == Fate.MAX_STEPS)
    # Compare against the configured max_steps when available; fall back to
    # max-observed for back-compat with saved-results post-processing.
    cap = max_steps_cap if max_steps_cap is not None else _max_steps_of(results)
    n_timeouts = sum(1 for r in results if r.fate == Fate.ESCAPED and r.steps == cap)
    if n == 0:
        return {
            "n_trajectories": 0,
            "n_reacted": 0,
            "n_escaped": 0,
            "n_max_steps": 0,
        }
    times = [r.time_ps for r in results]
    steps = [r.steps for r in results]
    seps = [r.final_separation for r in results]
    summary = {
        "n_trajectories": n,
        "n_reacted": n_reacted,
        "n_escaped": n_escaped,
        "n_max_steps": n_max_steps,
        "n_escaped_via_timeout": n_timeouts,
        "fraction_reacted": n_reacted / n,
        "fraction_escaped": n_escaped / n,
        "mean_time_ps": float(np.mean(times)),
        "mean_steps": float(np.mean(steps)),
        "mean_final_separation": float(np.mean(seps)),
        "min_time_ps": float(np.min(times)),
        "max_time_ps": float(np.max(times)),
    }
    # Per-reaction tally: which reactions fired and how often.
    reaction_counts = Counter(r.reaction_name for r in results if r.reaction_name)
    if reaction_counts:
        summary["reaction_counts"] = dict(reaction_counts)
    return summary


def _max_steps_of(results: List[TrajectoryResult]) -> int:
    """Inferred max_steps cap from the results -- the largest steps value.

    Used for timeout detection. Pass sim.params.max_steps in
    instead, but inferring from results keeps the helper independent
    and slightly more permissive (works on saved results too).
    """
    if not results:
        return 0
    return max(r.steps for r in results)


def write_results_json(
    work_dir: Path,
    sim,
    results: List[TrajectoryResult],
    wall_time_sec: float,
) -> Path:
    """Write results.json: a single summary file with all the metadata
    a downstream analyst would want about this run.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_stats(results, max_steps_cap=sim.params.max_steps)
    summary["wall_time_sec"] = float(wall_time_sec)
    if wall_time_sec > 0 and summary.get("mean_steps"):
        n = summary["n_trajectories"]
        total_steps = sum(r.steps for r in results)
        summary["steps_per_sec"] = float(total_steps / wall_time_sec)
    data = {
        "summary": summary,
        "diffusion": _diffusion_block(sim),
        "chain": _chain_block(sim),
        "params": _params_block(sim),
    }
    p = work_dir / "results.json"
    p.write_text(json.dumps(data, indent=2))
    return p


def write_trajectories_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
) -> Path:
    """Write trajectories.csv: one row per trajectory."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    p = work_dir / "trajectories.csv"
    with open(p, "w") as f:
        f.write(
            "traj_id,fate,steps,time_ps,final_separation,"
            "reaction_name,energy_at_reaction\n"
        )
        for i, r in enumerate(results):
            rxn = r.reaction_name if r.reaction_name else ""
            f.write(
                f"{i},{r.fate.name},{r.steps},{r.time_ps:.4f},"
                f"{r.final_separation:.4f},{rxn},"
                f"{r.energy_at_reaction:.6f}\n"
            )
    return p


def write_encounters_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
):
    """One row per REACTED trajectory: position + orientation at reaction.

    Columns: traj_id, step, time_ps, x, y, z, q_w, q_x, q_y, q_z, reaction_name
    Returns Path to file written, or None if no REACTED trajectories with
    populated encounter_pos.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(results):
        if r.fate != Fate.REACTED:
            continue
        if r.encounter_pos is None:
            continue
        x, y, z = [float(v) for v in r.encounter_pos]
        if r.encounter_q is not None:
            qw, qx, qy, qz = [float(v) for v in r.encounter_q]
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        rxn_name = r.reaction_name or ""
        rows.append((i, r.steps, r.time_ps, x, y, z, qw, qx, qy, qz, rxn_name))
    if not rows:
        return None
    p = work_dir / "encounters.csv"
    with open(p, "w") as f:
        f.write("traj_id,step,time_ps,x,y,z,q_w,q_x,q_y,q_z,reaction_name\n")
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.4f},"
                f"{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},"
                f"{row[6]:.6f},{row[7]:.6f},{row[8]:.6f},{row[9]:.6f},"
                f"{row[10]}\n"
            )
    return p


def write_near_misses_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
):
    """One row per non-REACTED trajectory: closest approach during traj.

    Columns: traj_id, fate, near_miss_dist, x, y, z
    Returns Path or None if no near_miss data populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(results):
        if r.fate == Fate.REACTED:
            continue
        if r.near_miss_pos is None or r.near_miss_dist is None:
            continue
        x, y, z = [float(v) for v in r.near_miss_pos]
        rows.append((i, r.fate.name, float(r.near_miss_dist), x, y, z))
    if not rows:
        return None
    p = work_dir / "near_misses.csv"
    with open(p, "w") as f:
        f.write("traj_id,fate,near_miss_dist,x,y,z\n")
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.4f},"
                f"{row[3]:.4f},{row[4]:.4f},{row[5]:.4f}\n"
            )
    return p


def write_fpt_distribution_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
    n_bins: int = 50,
):
    """Histogram of first-passage times for REACTED trajectories.

    Columns: bin_lo, bin_hi, count
    Returns Path or None if no REACTED trajectories.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    times = [r.time_ps for r in results if r.fate == Fate.REACTED]
    if not times:
        return None
    counts, edges = np.histogram(times, bins=n_bins)
    p = work_dir / "fpt_distribution.csv"
    with open(p, "w") as f:
        f.write("bin_lo,bin_hi,count\n")
        for i, c in enumerate(counts):
            f.write(f"{edges[i]:.4f},{edges[i+1]:.4f},{int(c)}\n")
    return p


def write_contact_frequency_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
):
    """Aggregate (target_atom, chain_atom) contact counts across all trajectories.

    Columns: target_atom_id, chain_atom_id, total_contacts, n_trajectories
    Sorted descending by total_contacts.
    Returns Path or None if no contact data populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    n_trajs: Counter = Counter()
    for r in results:
        if r.contact_counts is None:
            continue
        for key, count in r.contact_counts.items():
            total[key] += count
            n_trajs[key] += 1
    if not total:
        return None
    p = work_dir / "contact_frequency.csv"
    rows = sorted(total.items(), key=lambda kv: -kv[1])
    with open(p, "w") as f:
        f.write("target_atom_id,chain_atom_id,total_contacts,n_trajectories\n")
        for (t_id, c_id), tot in rows:
            f.write(f"{t_id},{c_id},{tot},{n_trajs[(t_id, c_id)]}\n")
    return p


def write_energetics_npz(
    work_dir: Path,
    results: List[TrajectoryResult],
):
    """Per-trajectory energy traces in flat schema (no pickle needed).

    Stored as flat arrays keyed by snapshot:
      traj_id   (n_total,) int      trajectory index for each snapshot
      step      (n_total,) int      step number for each snapshot
      energy    (n_total, 4) float  (total, elec, born, steric)
      fate      (n_traj,) U10       fate per trajectory
      columns   (4,) U10            energy column labels
    Returns Path or None if no energy data populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    traj_id_list: List[int] = []
    step_list: List[int] = []
    energy_list: List[np.ndarray] = []
    fate_list: List[str] = []
    has_data = False
    for i, r in enumerate(results):
        fate_list.append(r.fate.name)
        if r.energy_steps is None or r.path_steps is None:
            continue
        n = len(r.path_steps)
        traj_id_list.extend([i] * n)
        step_list.extend([int(s) for s in r.path_steps])
        energy_list.append(r.energy_steps)
        has_data = True
    if not has_data:
        return None
    p = work_dir / "energetics.npz"
    np.savez(
        p,
        traj_id=np.array(traj_id_list, dtype=int),
        step=np.array(step_list, dtype=int),
        energy=np.concatenate(energy_list, axis=0),
        fate=np.array(fate_list, dtype="U10"),
        columns=np.array(["total", "elec", "born", "steric"], dtype="U10"),
    )
    return p


def write_paths_npz(
    work_dir: Path,
    results: List[TrajectoryResult],
):
    """Per-trajectory COM + orientation snapshots in flat schema (no pickle).

    Stored as flat arrays:
      traj_id  (n_total,) int      trajectory index for each snapshot
      step     (n_total,) int      step number for each snapshot
      com      (n_total, 3) float  center-of-mass position [A]
      q        (n_total, 4) float  quaternion (w, x, y, z)
      radial   (n_total,) float    |com| separation [A]
      fate     (n_traj,) U10       fate per trajectory
    Returns Path or None if no path data populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    traj_id_list: List[int] = []
    step_list: List[int] = []
    com_list: List[np.ndarray] = []
    q_list: List[np.ndarray] = []
    radial_list: List[float] = []
    fate_list: List[str] = []
    has_data = False
    for i, r in enumerate(results):
        fate_list.append(r.fate.name)
        if r.path_steps is None or r.path_com is None or r.path_q is None:
            continue
        n = len(r.path_steps)
        traj_id_list.extend([i] * n)
        step_list.extend([int(s) for s in r.path_steps])
        com_list.append(r.path_com)
        q_list.append(r.path_q)
        if r.radial_trace is not None:
            radial_list.extend([float(v) for v in r.radial_trace])
        else:
            radial_list.extend([float(np.linalg.norm(c)) for c in r.path_com])
        has_data = True
    if not has_data:
        return None
    p = work_dir / "paths.npz"
    np.savez(
        p,
        traj_id=np.array(traj_id_list, dtype=int),
        step=np.array(step_list, dtype=int),
        com=np.concatenate(com_list, axis=0),
        q=np.concatenate(q_list, axis=0),
        radial=np.array(radial_list, dtype=float),
        fate=np.array(fate_list, dtype="U10"),
    )
    return p


def write_radial_density_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
    n_bins: int = 50,
):
    """Radial density histogram of all trajectory snapshots.

    Bins |COM| separations from radial_trace across all trajectories.
    Density column is a probability density (per A): count / total / bin_width.
    Bin range: 0 to max(radial) across all trajectories.

    Columns: bin_lo, bin_hi, count, density
    Returns Path or None if no radial data populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    all_r: List[float] = []
    for r in results:
        if r.radial_trace is not None:
            all_r.extend([float(v) for v in r.radial_trace])
    if not all_r:
        return None
    r_arr = np.asarray(all_r, dtype=float)
    counts, edges = np.histogram(r_arr, bins=n_bins, range=(0.0, float(r_arr.max())))
    total = int(counts.sum())
    p = work_dir / "radial_density.csv"
    with open(p, "w") as f:
        f.write("bin_lo,bin_hi,count,density\n")
        for i, c in enumerate(counts):
            bw = edges[i + 1] - edges[i]
            density = (int(c) / total / bw) if (total > 0 and bw > 0) else 0.0
            f.write(f"{edges[i]:.4f},{edges[i+1]:.4f},{int(c)},{density:.6e}\n")
    return p


def write_angular_map_npz(
    work_dir: Path,
    results: List[TrajectoryResult],
    n_theta: int = 18,
    n_phi: int = 36,
):
    """2D histogram of ligand (theta, phi) occupancy in receptor frame.

    Uses path_com positions; converts to spherical coords:
      theta = arccos(z/r)      polar angle in [0, pi]
      phi   = arctan2(y, x)    azimuthal in [-pi, pi]

    Stored as:
      counts          (n_theta, n_phi) int    per-bin snapshot count
      theta_edges     (n_theta+1,) float      bin edges, radians
      phi_edges       (n_phi+1,) float        bin edges, radians
      total_snapshots scalar int              total snapshots binned
    Returns Path or None if no path_com data populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    coms: List[np.ndarray] = []
    for r in results:
        if r.path_com is not None:
            coms.append(r.path_com)
    if not coms:
        return None
    coms_arr = np.concatenate(coms, axis=0)
    if coms_arr.size == 0:
        return None
    norms = np.linalg.norm(coms_arr, axis=1)
    nonzero = norms > 1e-12
    if not nonzero.any():
        return None
    valid = coms_arr[nonzero]
    rs = norms[nonzero]
    theta = np.arccos(np.clip(valid[:, 2] / rs, -1.0, 1.0))
    phi = np.arctan2(valid[:, 1], valid[:, 0])
    counts, theta_edges, phi_edges = np.histogram2d(
        theta,
        phi,
        bins=[n_theta, n_phi],
        range=[[0.0, float(np.pi)], [-float(np.pi), float(np.pi)]],
    )
    p = work_dir / "angular_map.npz"
    np.savez(
        p,
        counts=counts.astype(int),
        theta_edges=theta_edges.astype(float),
        phi_edges=phi_edges.astype(float),
        total_snapshots=np.array([len(theta)], dtype=int),
    )
    return p


def write_milestone_flux_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
    n_shells: int = 10,
):
    """Aggregate outward + inward shell crossings across all trajectories.

    Shells are linearly spaced between min and max of radial_trace
    across all trajectories (excluding zero radii). For each pair of
    consecutive snapshots in a trajectory:
      - "crossed outward" if r_prev < R <= r_curr  (R is the shell radius)
      - "crossed inward"  if r_curr < R <= r_prev

    Note: crossings between sampled points (within save_interval) are
    not detected, so this gives a coarse-grained flux estimate. For
    typical save_interval=10 and shells separated by several A, the
    miss rate should be small.

    Columns: shell_radius, n_crossings_out, n_crossings_in
    Returns Path or None if no usable radial_trace data.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    radials: List[np.ndarray] = []
    for r in results:
        if r.radial_trace is not None and len(r.radial_trace) >= 2:
            radials.append(np.asarray(r.radial_trace, dtype=float))
    if not radials:
        return None
    all_r = np.concatenate(radials)
    if len(all_r) < 2:
        return None
    r_min = float(all_r.min())
    r_max = float(all_r.max())
    if r_max <= r_min:
        return None
    # n_shells internal points, excluding endpoints.
    shells = np.linspace(r_min, r_max, n_shells + 2)[1:-1]
    n_out = np.zeros(n_shells, dtype=int)
    n_in = np.zeros(n_shells, dtype=int)
    for trace in radials:
        for i in range(1, len(trace)):
            r_prev = float(trace[i - 1])
            r_curr = float(trace[i])
            for k, s in enumerate(shells):
                if r_prev < s <= r_curr:
                    n_out[k] += 1
                elif r_curr < s <= r_prev:
                    n_in[k] += 1
    p = work_dir / "milestone_flux.csv"
    with open(p, "w") as f:
        f.write("shell_radius,n_crossings_out,n_crossings_in\n")
        for k, s in enumerate(shells):
            f.write(f"{s:.4f},{int(n_out[k])},{int(n_in[k])}\n")
    return p


def write_chain_results(
    work_dir: Path,
    sim,
    results: List[TrajectoryResult],
    wall_time_sec: float = 0.0,
    outputs=None,
) -> List[Tuple[str, Path]]:
    """Write chain BD output files to work_dir.

    Always-written files (regardless of outputs flags):
      - results.json      summary + diffusion + chain + params blocks
      - trajectories.csv  one row per trajectory

    Conditional files (gated on OutputConfig flags):
      - encounters.csv         outputs.encounters_csv
      - near_misses.csv        outputs.near_misses_csv
      - paths.npz              outputs.full_paths
      - radial_density.csv     outputs.radial_density
      - angular_map.npz        outputs.angular_map
      - fpt_distribution.csv   outputs.fpt_distribution
      - contact_frequency.csv  outputs.contact_frequency
      - milestone_flux.csv     outputs.milestone_flux
      - energetics.npz         outputs.energetics

    The following OutputConfig flags are accepted but currently do not
    produce a file (silently no-op, no error):
      - outputs.p_commit
      - outputs.transition_matrix
      - outputs.pose_clusters
    See module docstring for why these are deferred.

    Each conditional writer returns None when its required data is
    absent (e.g. no REACTED trajectories -> no encounters.csv); in
    those cases the file is also skipped silently.

    Returns a list of (filename, full_path) tuples for all files
    actually written, in the order of the rigid-body output_writer
    convention.

    Parameters
    ----------
    work_dir       : output directory (created if missing)
    sim            : ChainBDSimulator instance (introspected for config)
    results        : list of TrajectoryResult from sim.run()
    wall_time_sec  : wall clock time of the run, in seconds; included
                     in the summary if > 0
    outputs        : OutputConfig controlling per-file flags. If None,
                     a default-constructed OutputConfig is used (all
                     flags True).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, Path]] = []
    # Default to a fresh OutputConfig if caller didn't supply one.
    if outputs is None:
        from pystarc.pipeline.input_parser import OutputConfig

        outputs = OutputConfig()
    # Always written.
    p = write_results_json(work_dir, sim, results, wall_time_sec)
    written.append(("results.json", p))
    p = write_trajectories_csv(work_dir, results)
    written.append(("trajectories.csv", p))
    # Sub-stage 2b: 5 conditional writers, gated on OutputConfig flags.
    if outputs.encounters_csv:
        p = write_encounters_csv(work_dir, results)
        if p is not None:
            written.append(("encounters.csv", p))
    if outputs.near_misses_csv:
        p = write_near_misses_csv(work_dir, results)
        if p is not None:
            written.append(("near_misses.csv", p))
    # Sub-stage 2c: heavier-data writers.
    if outputs.full_paths:
        p = write_paths_npz(work_dir, results)
        if p is not None:
            written.append(("paths.npz", p))
    if outputs.radial_density:
        p = write_radial_density_csv(work_dir, results)
        if p is not None:
            written.append(("radial_density.csv", p))
    if outputs.angular_map:
        p = write_angular_map_npz(work_dir, results)
        if p is not None:
            written.append(("angular_map.npz", p))
    if outputs.fpt_distribution:
        p = write_fpt_distribution_csv(work_dir, results)
        if p is not None:
            written.append(("fpt_distribution.csv", p))
    if outputs.contact_frequency:
        p = write_contact_frequency_csv(work_dir, results)
        if p is not None:
            written.append(("contact_frequency.csv", p))
    # Sub-stage 2d.
    if outputs.milestone_flux:
        p = write_milestone_flux_csv(work_dir, results)
        if p is not None:
            written.append(("milestone_flux.csv", p))
    if outputs.energetics:
        p = write_energetics_npz(work_dir, results)
        if p is not None:
            written.append(("energetics.npz", p))
    return written
