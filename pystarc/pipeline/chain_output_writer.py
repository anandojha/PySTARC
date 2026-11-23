"""
Output writer for chain Brownian-dynamics runs.

This module produces output files that match the rigid-body OutputConfig
schema. After Stage 2 (sub-stages 2a through 2d), 11 of the 14 OutputConfig
flags are implemented in chain BD mode.

Two files are always written. results.json holds the run summary, diffusion
configuration, chain summary, and parameters. trajectories.csv holds one row
per trajectory.

The remaining files are conditional and gated on OutputConfig flags.
encounters.csv records the position and orientation at reaction for each
REACTED trajectory. near_misses.csv records the closest approach during each
non-REACTED trajectory. paths.npz stores the per-snapshot center of mass,
orientation, and radial separation. radial_density.csv is a histogram of the
center-of-mass separations |COM|. angular_map.npz is a two-dimensional (theta,
phi) occupancy heatmap. fpt_distribution.csv is a histogram of first-passage
times. contact_frequency.csv aggregates target-chain atom-pair contacts.
milestone_flux.csv counts outward and inward shell crossings. energetics.npz
stores the per-snapshot energy components.

Three OutputConfig flags are accepted but currently do nothing in chain BD
mode, so no file is produced. Implementing each one requires new analysis code
beyond simple file emission. The p_commit.npz output (a commitment probability
map) needs a shell-conditioned committor analysis. The transition_matrix.npz
output (a Markov state transition matrix) needs Markov state definitions for
chain BD. The pose_clusters.csv output (clustered encounter orientations) needs
clustering of the encounter quaternions.

Because these three flags are accepted but skipped, a default OutputConfig with
all flags set to True does not raise an error. Users simply see 11 files
instead of 14.

This is the chain-side counterpart to pipeline/output_writer.py. The GPU-batch
NAM writer expects a sim_data dict whose array shapes differ from the natural
chain BD data, so we keep the two writers separate.
"""

from __future__ import annotations
import os
import statistics
import math
from pystarc.global_defs.constants import MOLAR_RATE_CONV
from pystarc.global_defs.defaults import QB_FACTOR
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
import json


def _diffusion_block(sim, results=None) -> Dict[str, Any]:
    """Serialize the simulator's diffusion configuration.

    With params.diffusion_update_interval > 0 the block also holds the start
    values of tr(D)/3, their means pooled over the trajectories with the
    trajectory time as weight, the interval, and the rebuild counts. The pooled
    means are None when no trajectory is given.

    In auto_diffusion mode this includes the full 3×3 diffusion tensors along
    with the isotropic equivalent (the trace divided by 3) and the rigid-body
    hydrodynamic center. In scalar mode it emits only the scalar D_trans and
    D_rot.
    """
    block = _diffusion_block_base(sim)
    block["n_relative_diffusion_clipped"] = int(
        sum(getattr(r, "n_relative_diffusion_clipped", 0) for r in (results or []))
    )
    interval = getattr(sim.params, "diffusion_update_interval", 0)
    if isinstance(interval, (int, np.integer)) and interval > 0:
        D_t0 = np.asarray(sim.D_trans)
        D_r0 = np.asarray(sim.D_rot)
        block["update_interval"] = int(interval)
        block["D_trans_start"] = float(np.trace(D_t0) / 3.0)
        block["D_rot_start"] = float(np.trace(D_r0) / 3.0)
        timed = [
            r
            for r in (results or [])
            if getattr(r, "mean_D_trans", None) is not None and r.time_ps > 0.0
        ]
        w = np.array([r.time_ps for r in timed], dtype=float)
        if len(timed) > 0:
            block["D_trans_mean"] = float(
                np.sum(w * np.array([r.mean_D_trans for r in timed])) / np.sum(w)
            )
            block["D_rot_mean"] = float(
                np.sum(w * np.array([r.mean_D_rot for r in timed])) / np.sum(w)
            )
        else:
            block["D_trans_mean"] = None
            block["D_rot_mean"] = None
        block["n_rebuilds"] = int(
            sum(getattr(r, "n_diffusion_rebuilds", 0) for r in (results or []))
        )
        block["n_rebuild_failures"] = int(
            sum(getattr(r, "n_diffusion_rebuild_failures", 0) for r in (results or []))
        )
    return block


def _diffusion_block_base(sim) -> Dict[str, Any]:
    """The diffusion configuration of the simulator, without the rebuild log."""
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
        # Scalar path. D_trans and D_rot may also arrive as (3, 3) arrays if
        # the user passed pre-computed tensors, so handle both cases.
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
    """Summarize the chain template into a JSON-friendly dict."""
    template = sim.chain_template
    return {
        "name": template.name,
        "flexible_partner": str(getattr(sim, "flexible_partner", "ligand")),
        "n_atoms": template.n_atoms,
        "atom_radii": [float(a.radius) for a in template.atoms],
        "atom_charges": [float(a.charge) for a in template.atoms],
        "n_bonds": len(template.bonds),
        "n_angles": len(template.angles),
        "n_torsions": len(template.torsions),
        "n_length_constraints": len(template.length_constraints),
        "n_network_springs": len(getattr(template, "network", [])),
        "n_pair_tables": (
            len(getattr(template, "pair_lookups", {}))
            if getattr(template, "coffdrop_params", None) is not None
            else 0
        ),
    }


def _params_block(sim) -> Dict[str, Any]:
    """Serialize all ChainBDParameters fields to a JSON-friendly dict.

    This calls dataclasses.asdict on the canonical ChainBDParameters dataclass
    so that any new field is included automatically without editing this
    writer. Test fixtures that pass plain Python classes for sim.params are
    handled by attribute introspection, so the production and test paths cover
    the same set of fields.
    """
    p = sim.params
    if is_dataclass(p):
        return asdict(p)
    # Plain-class fallback: collect the public, non-callable attributes. This
    # path is used by test stubs that mock sim.params without a real dataclass.
    return {
        k: getattr(p, k)
        for k in dir(p)
        if not k.startswith("_") and not callable(getattr(p, k))
    }


def _summary_stats(
    results: List[TrajectoryResult],
    max_steps_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate per-trajectory results into summary statistics.

    Pass sim.params.max_steps as max_steps_cap so that timeout detection
    compares against the configured cap. When max_steps_cap is None, for
    example during post-hoc analysis of saved results without a sim object,
    the function falls back to the largest observed step count from
    _max_steps_of. That legacy behavior can misclassify the longest escape as
    a timeout even when no trajectory actually timed out.
    """
    n = len(results)
    n_reacted = sum(1 for r in results if r.fate == Fate.REACTED)
    n_escaped = sum(1 for r in results if r.fate == Fate.ESCAPED)
    n_max_steps = sum(1 for r in results if r.fate == Fate.MAX_STEPS)
    # Compare against the configured max_steps when it is available. Fall back
    # to the maximum observed value to stay compatible with post-processing of
    # saved results.
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
        "n_hard_sphere_redraws": int(
            sum(getattr(r, "n_hard_sphere_redraws", 0) for r in results)
        ),
        "n_hard_sphere_rejections": int(
            sum(getattr(r, "n_hard_sphere_rejections", 0) for r in results)
        ),
    }
    # Tally how often each named reaction fired.
    reaction_counts = Counter(r.reaction_name for r in results if r.reaction_name)
    if reaction_counts:
        summary["reaction_counts"] = dict(reaction_counts)
    return summary


def _max_steps_of(results: List[TrajectoryResult]) -> int:
    """Infer the max_steps cap from the results as the largest step value.

    This is used for timeout detection. It is better to pass
    sim.params.max_steps directly, but inferring the cap from the results
    keeps this helper independent and slightly more permissive, since it also
    works on saved results.
    """
    if not results:
        return 0
    return max(r.steps for r in results)


def _rate_block(sim, results) -> dict:
    """k_on = N_A k_b P_rxn, or a reason why it cannot be reported.

    P_rxn is reacted over every trajectory that was run. A trajectory that
    reached the step limit counts as not reacted, as in BrownDye2, and the
    bound gives P_rxn and k_on with every such trajectory counted as reacted
    instead. k_b is the encounter rate at the b-surface, computed by the outer
    propagator. Without that propagator there is no k_b and so no rate.
    """
    prop = getattr(sim, "_outer_prop", None)
    k_b = getattr(prop, "k_b", None) if prop is not None else None
    n_r = sum(1 for r in results if r.fate == Fate.REACTED)
    n_e = sum(1 for r in results if r.fate == Fate.ESCAPED)
    n_c = sum(1 for r in results if r.fate == Fate.MAX_STEPS)
    out = {
        "n_reacted": n_r,
        "n_escaped": n_e,
        "n_censored": n_c,
        "censored_fraction": (n_c / (n_r + n_e + n_c)) if (n_r + n_e + n_c) else 0.0,
        "use_lmz": bool(getattr(sim.params, "use_lmz", False)),
        "outer_propagator": prop is not None,
    }
    if k_b is None:
        out["k_on"] = None
        out["reason"] = "no outer propagator, so k_b is undefined"
        return out
    n = n_r + n_e + n_c
    if n == 0:
        out["k_on"] = None
        out["reason"] = "no trajectory was run"
        return out
    conf = float(getattr(sim, "confidence_interval", 0.95))
    conv_tol = float(getattr(sim, "convergence_tol", 0.05))
    z = (
        1.96
        if abs(conf - 0.95) < 1.0e-9
        else statistics.NormalDist().inv_cdf(0.5 + 0.5 * conf)
    )
    p = n_r / n
    lo, hi = _wilson(n_r, n, z=z)
    p_hi_bound = (n_r + n_c) / n
    use_lmz = bool(getattr(sim.params, "use_lmz", False))
    b_rad = float(getattr(prop, "b_radius", 0.0))
    r_esc = float(getattr(sim.params, "r_escape", 0.0))
    nam_omega = 0.0
    if not use_lmz and r_esc > b_rad > 0.0 and hasattr(prop, "_relative_rate"):
        rate_esc = float(prop._relative_rate(r_esc))
        if rate_esc > 0.0:
            nam_omega = min(max(float(k_b) / rate_esc, 0.0), 1.0 - 1.0e-12)

    def _nam_beta(prob: float) -> float:
        if nam_omega <= 0.0:
            return prob
        denom = 1.0 - (1.0 - prob) * nam_omega
        return prob / denom if denom > 0.0 else 1.0

    beta = _nam_beta(p)
    beta_lo = _nam_beta(lo)
    beta_hi = _nam_beta(hi)
    beta_bound_hi = _nam_beta(p_hi_bound)
    conv = MOLAR_RATE_CONV  # A^3/ps to M^-1 s^-1
    out.update(
        {
            "k_b": float(k_b),
            "k_b_units": "A^3/ps",
            "b_radius": b_rad,
            "q_radius": float(getattr(prop, "qradius", 0.0)),
            "return_prob": float(getattr(prop, "return_prob", 0.0)),
            "nam_omega": float(nam_omega),
            "P_rxn_raw": float(p),
            "P_rxn": float(beta),
            "P_rxn_low": float(beta_lo),
            "P_rxn_high": float(beta_hi),
            "k_on": float(conv * k_b * beta),
            "k_on_low": float(conv * k_b * beta_lo),
            "k_on_high": float(conv * k_b * beta_hi),
            "k_on_units": "M-1 s-1",
            "P_rxn_bound_low": float(beta),
            "P_rxn_bound_high": float(beta_bound_hi),
            "k_on_bound_low": float(conv * k_b * beta),
            "k_on_bound_high": float(conv * k_b * beta_bound_hi),
            "confidence_level": conf,
            "convergence_tol": conv_tol,
        }
    )
    return out


def _wilson(reacted: int, n: int, z: float = 1.96):
    """Wilson score interval on the reaction probability."""
    if n == 0:
        return 0.0, 0.0
    p = reacted / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def write_results_json(
    work_dir: Path,
    sim,
    results: List[TrajectoryResult],
    wall_time_sec: float,
) -> Path:
    """Write results.json, a single summary file that holds all the metadata a
    downstream analyst would want about this run.
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
        "rate": _rate_block(sim, results),
        "diffusion": _diffusion_block(sim, results),
        "chain": _chain_block(sim),
        "params": _params_block(sim),
    }
    p = work_dir / "results.json"
    tmp = work_dir / f"results.json.tmp.{os.getpid()}"
    payload = json.dumps(data, indent=2)
    try:
        tmp.write_text(payload)
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return p


def write_trajectories_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
) -> Path:
    """Write trajectories.csv with one row per trajectory."""
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
    """Write one row per REACTED trajectory giving the position and orientation
    at reaction.

    The columns are traj_id, step, time_ps, x, y, z, q_w, q_x, q_y, q_z, and
    reaction_name. The function returns the Path to the file it wrote, or None
    when there are no REACTED trajectories with a populated encounter_pos.
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
    """Write one row per non-REACTED trajectory giving the closest approach
    reached during the trajectory.

    The columns are traj_id, fate, near_miss_dist, x, y, and z. The function
    returns the Path to the file, or None when no near-miss data is populated.
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
    """Write a histogram of first-passage times for the REACTED trajectories.

    The columns are bin_lo, bin_hi, and count. The function returns the Path to
    the file, or None when there are no REACTED trajectories.
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
    """Aggregate the (target_atom, chain_atom) contact counts across all
    trajectories.

    The columns are target_atom_id, chain_atom_id, total_contacts, and
    n_trajectories, sorted in descending order by total_contacts. The function
    returns the Path to the file, or None when no contact data is populated.
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
    """Write the per-trajectory energy traces in a flat schema that needs no
    pickling.

    The data is stored as flat arrays keyed by snapshot. The array traj_id of
    shape (n_total,) holds the trajectory index for each snapshot, and step of
    shape (n_total,) holds the step number for each snapshot. The array energy
    of shape (n_total, 4) holds the total, electrostatic, Born, and steric
    energy components. The array fate of shape (n_traj,) holds the fate of each
    trajectory, and columns of shape (4,) holds the energy column labels. The
    function returns the Path to the file, or None when no energy data is
    populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    traj_id_parts: List[np.ndarray] = []
    step_parts: List[np.ndarray] = []
    energy_parts: List[np.ndarray] = []
    fate_list: List[str] = []
    for i, r in enumerate(results):
        fate_list.append(r.fate.name)
        if r.energy_steps is None or r.path_steps is None:
            continue
        steps_arr = np.asarray(r.path_steps, dtype=int)
        n = len(steps_arr)
        traj_id_parts.append(np.full(n, i, dtype=int))
        step_parts.append(steps_arr)
        energy_parts.append(np.asarray(r.energy_steps))
    if not energy_parts:
        return None
    p = work_dir / "energetics.npz"
    np.savez(
        p,
        traj_id=np.concatenate(traj_id_parts, axis=0),
        step=np.concatenate(step_parts, axis=0),
        energy=np.concatenate(energy_parts, axis=0),
        fate=np.array(fate_list, dtype="U10"),
        columns=np.array(["total", "elec", "born", "steric"], dtype="U10"),
    )
    return p


def write_paths_npz(
    work_dir: Path,
    results: List[TrajectoryResult],
):
    """Write the per-trajectory center-of-mass and orientation snapshots in a
    flat schema that needs no pickling.

    The data is stored as flat arrays. The array traj_id of shape (n_total,)
    holds the trajectory index for each snapshot, and step of shape (n_total,)
    holds the step number for each snapshot. The array com of shape (n_total,
    3) holds the center-of-mass position in Å, and q of shape (n_total, 4)
    holds the orientation quaternion (w, x, y, z). The array radial of shape
    (n_total,) holds the separation |com| in Å, and fate of shape (n_traj,)
    holds the fate of each trajectory. The function returns the Path to the
    file, or None when no path data is populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    traj_id_parts: List[np.ndarray] = []
    step_parts: List[np.ndarray] = []
    com_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    radial_parts: List[np.ndarray] = []
    returns_parts: List[np.ndarray] = []
    has_returns = False
    fate_list: List[str] = []
    for i, r in enumerate(results):
        fate_list.append(r.fate.name)
        if r.path_steps is None or r.path_com is None or r.path_q is None:
            continue
        steps_arr = np.asarray(r.path_steps, dtype=int)
        com_arr = np.asarray(r.path_com)
        q_arr = np.asarray(r.path_q)
        n = len(steps_arr)
        traj_id_parts.append(np.full(n, i, dtype=int))
        step_parts.append(steps_arr)
        com_parts.append(com_arr)
        q_parts.append(q_arr)
        if r.radial_trace is not None:
            radial_parts.append(np.asarray(r.radial_trace, dtype=float))
        else:
            radial_parts.append(
                np.linalg.norm(com_arr, axis=1).astype(float)
                if n > 0
                else np.empty(0, dtype=float)
            )
        ret_arr = getattr(r, "radial_returns", None)
        if ret_arr is not None and len(ret_arr) == n:
            returns_parts.append(np.asarray(ret_arr, dtype=int))
            has_returns = True
        else:
            returns_parts.append(np.zeros(n, dtype=int))
    if not com_parts:
        return None
    p = work_dir / "paths.npz"
    save_kw = dict(
        traj_id=np.concatenate(traj_id_parts, axis=0),
        step=np.concatenate(step_parts, axis=0),
        com=np.concatenate(com_parts, axis=0),
        q=np.concatenate(q_parts, axis=0),
        radial=np.concatenate(radial_parts, axis=0),
        fate=np.array(fate_list, dtype="U10"),
    )
    if has_returns:
        save_kw["radial_returns"] = np.concatenate(returns_parts, axis=0)
    np.savez(p, **save_kw)
    return p


def write_radial_density_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
    n_bins: int = 50,
):
    """Write a radial density histogram over all trajectory snapshots.

    The function bins the |COM| separations taken from radial_trace across all
    trajectories. The density column is a probability density per Å, computed
    as count / total / bin_width. The bins span from 0 to the maximum radial
    value seen across all trajectories.

    The columns are bin_lo, bin_hi, count, and density. The function returns
    the Path to the file, or None when no radial data is populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    max_r = -math.inf
    has_data = False
    for r in results:
        if r.radial_trace is not None and len(r.radial_trace) > 0:
            arr = np.asarray(r.radial_trace, dtype=float)
            m = float(arr.max())
            if m > max_r:
                max_r = m
            has_data = True
    if not has_data:
        return None
    counts = np.zeros(n_bins, dtype=np.int64)
    edges = None
    hist_range = (0.0, max_r)
    for r in results:
        if r.radial_trace is not None and len(r.radial_trace) > 0:
            arr = np.asarray(r.radial_trace, dtype=float)
            c_i, edges = np.histogram(arr, bins=n_bins, range=hist_range)
            counts += c_i
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
    """Write a two-dimensional histogram of ligand (theta, phi) occupancy in
    the receptor frame.

    The function converts path_com positions to spherical coordinates. The
    polar angle in [0, π] is

        theta = arccos(z / r)

    and the azimuthal angle in [-π, π] is

        phi = arctan2(y, x).

    Here r is the radial distance of the center of mass, and x, y, z are its
    Cartesian components in the receptor frame.

    The data is stored as several arrays. The array counts of shape (n_theta,
    n_phi) holds the snapshot count in each bin. The array theta_edges of shape
    (n_theta+1,) and phi_edges of shape (n_phi+1,) hold the bin edges in
    radians, and total_snapshots is a scalar count of the snapshots binned. The
    function returns the Path to the file, or None when no path_com data is
    populated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = np.zeros((n_theta, n_phi), dtype=np.int64)
    theta_edges = None
    phi_edges = None
    total_snapshots = 0
    hist_range = [[0.0, float(np.pi)], [-float(np.pi), float(np.pi)]]
    for r in results:
        if r.path_com is None:
            continue
        com_arr = np.asarray(r.path_com, dtype=float)
        if com_arr.size == 0:
            continue
        norms = np.linalg.norm(com_arr, axis=1)
        nonzero = norms > 1e-12
        if not nonzero.any():
            continue
        valid = com_arr[nonzero]
        rs = norms[nonzero]
        theta = np.arccos(np.clip(valid[:, 2] / rs, -1.0, 1.0))
        phi = np.arctan2(valid[:, 1], valid[:, 0])
        c_i, theta_edges, phi_edges = np.histogram2d(
            theta,
            phi,
            bins=[n_theta, n_phi],
            range=hist_range,
        )
        counts += c_i.astype(np.int64)
        total_snapshots += len(theta)
    if total_snapshots == 0:
        return None
    p = work_dir / "angular_map.npz"
    np.savez(
        p,
        counts=counts.astype(int),
        theta_edges=theta_edges.astype(float),
        phi_edges=phi_edges.astype(float),
        total_snapshots=np.array([total_snapshots], dtype=int),
    )
    return p


def write_milestone_flux_csv(
    work_dir: Path,
    results: List[TrajectoryResult],
    n_shells: int = 10,
    b_radius: Optional[float] = None,
):
    """Aggregate the outward and inward shell crossings across all trajectories.

    The shells are linearly spaced between the minimum and maximum of
    radial_trace across all trajectories, with zero radii excluded. For each
    pair of consecutive snapshots in a trajectory, a crossing is counted as
    outward when r_prev < R <= r_curr and as inward when r_curr < R <= r_prev.
    Here R is the shell radius, r_prev is the separation at the earlier
    snapshot, and r_curr is the separation at the later snapshot.

    A return of the outer propagator moves the chain from QB_FACTOR × b to b in
    one jump, which crosses no shell, as in the rigid GPU path. When
    radial_returns records returns between two snapshots and b_radius is given,
    the interval is split into the walk from r_prev out to QB_FACTOR × b, one
    walk from b out to QB_FACTOR × b per further return, and the walk from b to
    r_curr.

    Crossings that happen between sampled points, that is within one
    save_interval, are not counted, so this is a coarse-grained flux estimate.
    For a typical save_interval of 10 and shells separated by several Å, the
    miss rate should be small.

    The columns are shell_radius, n_crossings_out, and n_crossings_in. The
    function returns the Path to the file, or None when there is no usable
    radial_trace data.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    r_min = math.inf
    r_max = -math.inf
    has_trace = False
    for r in results:
        rt = r.radial_trace
        if rt is not None and len(rt) >= 2:
            arr = np.asarray(rt, dtype=float)
            lo = float(arr.min())
            hi = float(arr.max())
            if lo < r_min:
                r_min = lo
            if hi > r_max:
                r_max = hi
            has_trace = True
    if not has_trace or r_max <= r_min:
        return None
    # Interior shell radii excluding the two endpoints.
    shells = np.linspace(r_min, r_max, n_shells + 2)[1:-1]
    n_out = np.zeros(n_shells, dtype=np.int64)
    n_in = np.zeros(n_shells, dtype=np.int64)

    def _accum_crossings(a: np.ndarray, b: np.ndarray) -> None:
        out_m = b > a
        if out_m.any():
            lo_idx = np.searchsorted(shells, a[out_m], side="right")
            hi_idx = np.searchsorted(shells, b[out_m], side="right")
            diff = np.bincount(lo_idx, minlength=n_shells + 1) - np.bincount(
                hi_idx, minlength=n_shells + 1
            )
            n_out[:] += np.cumsum(diff)[:-1]
        in_m = b < a
        if in_m.any():
            lo_idx = np.searchsorted(shells, b[in_m], side="right")
            hi_idx = np.searchsorted(shells, a[in_m], side="right")
            diff = np.bincount(lo_idx, minlength=n_shells + 1) - np.bincount(
                hi_idx, minlength=n_shells + 1
            )
            n_in[:] += np.cumsum(diff)[:-1]

    b_val = float(b_radius) if b_radius is not None else None
    q_val = QB_FACTOR * b_val if b_val is not None else None
    q_idx = (
        int(np.searchsorted(shells, q_val, side="right")) if q_val is not None else 0
    )
    bq_out_mask = (
        ((b_val < shells) & (shells <= q_val)).astype(np.int64)
        if b_val is not None and b_val < q_val
        else None
    )
    bq_in_mask = (
        ((q_val < shells) & (shells <= b_val)).astype(np.int64)
        if b_val is not None and q_val < b_val
        else None
    )

    for r in results:
        rt = r.radial_trace
        if rt is None or len(rt) < 2:
            continue
        trace = np.asarray(rt, dtype=float)
        r_prev = trace[:-1]
        r_curr = trace[1:]
        m = len(r_prev)
        ret = getattr(r, "radial_returns", None)
        if ret is not None and b_val is not None:
            ret_arr = np.asarray(ret, dtype=np.int64)
            if len(ret_arr) >= m + 1:
                n_ret_arr = ret_arr[1 : m + 1]
            else:
                n_ret_arr = np.zeros(m, dtype=np.int64)
                k_copy = min(len(ret_arr) - 1, m)
                if k_copy > 0:
                    n_ret_arr[:k_copy] = ret_arr[1 : 1 + k_copy]
            ret_mask = n_ret_arr > 0
            if ret_mask.any():
                prev_ret = r_prev[ret_mask]
                sub_lt = prev_ret[prev_ret < q_val]
                if sub_lt.size > 0:
                    lo_idx = np.searchsorted(shells, sub_lt, side="right")
                    diff = np.bincount(lo_idx, minlength=n_shells + 1)
                    diff[q_idx] -= sub_lt.size
                    n_out[:] += np.cumsum(diff)[:-1]
                extra_walks = int(np.sum(n_ret_arr[ret_mask] - 1))
                if extra_walks > 0:
                    if bq_out_mask is not None:
                        n_out[:] += extra_walks * bq_out_mask
                    elif bq_in_mask is not None:
                        n_in[:] += extra_walks * bq_in_mask
                r_prev = np.where(ret_mask, b_val, r_prev)
        _accum_crossings(r_prev, r_curr)

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
    """Write the chain BD output files to work_dir.

    Two files are always written regardless of the output flags. trajectories.csv
    holds one row per trajectory, and results.json holds the summary, diffusion,
    chain, and parameters blocks. results.json is written last via an atomic
    rename so resume and combiner checks never observe a partial shard.

    The remaining files are conditional and gated on OutputConfig flags. The
    outputs.encounters_csv flag controls encounters.csv, outputs.near_misses_csv
    controls near_misses.csv, and outputs.full_paths controls paths.npz. The
    outputs.radial_density flag controls radial_density.csv, outputs.angular_map
    controls angular_map.npz, and outputs.fpt_distribution controls
    fpt_distribution.csv. The outputs.contact_frequency flag controls
    contact_frequency.csv, outputs.milestone_flux controls milestone_flux.csv,
    and outputs.energetics controls energetics.npz.

    Three OutputConfig flags are accepted but currently do not produce a file.
    The outputs.p_commit, outputs.transition_matrix, and outputs.pose_clusters
    flags silently do nothing and raise no error. See the module docstring for
    why these are deferred.

    Each conditional writer returns None when its required data is absent, for
    example when there are no REACTED trajectories so no encounters.csv can be
    written. In those cases the file is skipped silently.

    The function returns a list of (filename, full_path) tuples for every file
    that was written.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, Path]] = []
    if outputs is None:
        from pystarc.pipeline.input_parser import OutputConfig

        outputs = OutputConfig()
    # Write trajectories.csv and all conditional outputs before atomic results.json.
    p = write_trajectories_csv(work_dir, results)
    written.append(("trajectories.csv", p))
    if outputs.encounters_csv:
        p = write_encounters_csv(work_dir, results)
        if p is not None:
            written.append(("encounters.csv", p))
    if outputs.near_misses_csv:
        p = write_near_misses_csv(work_dir, results)
        if p is not None:
            written.append(("near_misses.csv", p))
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
    if outputs.milestone_flux:
        p = write_milestone_flux_csv(
            work_dir, results, b_radius=getattr(sim.params, "r_start", None)
        )
        if p is not None:
            written.append(("milestone_flux.csv", p))
    if outputs.energetics:
        p = write_energetics_npz(work_dir, results)
        if p is not None:
            written.append(("energetics.npz", p))
    # Commit results.json last via atomic rename as the completion marker.
    p = write_results_json(work_dir, sim, results, wall_time_sec)
    written.append(("results.json", p))
    return written
