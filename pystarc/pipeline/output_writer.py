"""
PySTARC output writer.

This module writes all of the simulation output files into the bd_sims/
directory. The pipeline calls it once the Brownian-dynamics simulation has
finished.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import json
import math


def write_all(
    work_dir: Path,
    result,
    sim_data: Dict[str, Any],
    output_cfg,
    k_b: float,
    D_rel: float,
    confidence: float = 0.95,
):
    """
    Write every enabled output file into work_dir.

    work_dir is the output directory, normally bd_sims/. result is the
    GPUBatchResult object holding the simulation outcome. output_cfg is an
    OutputConfig whose flags decide which files are written. k_b is the
    Romberg k_b value, D_rel is the relative translational diffusion
    coefficient, and confidence is the confidence level used for the
    reported intervals.

    sim_data is a dictionary of trajectory data collected by the simulator.
    Its keys describe N trajectories, M recorded reactions, and K recorded
    near misses. The per-trajectory arrays are 'start_pos' with shape (N, 3)
    giving the starting positions, 'start_q' with shape (N, 4) giving the
    starting orientations, 'outcome' with shape (N,) where 1 means reacted,
    2 means escaped, and 3 means the trajectory hit the maximum step count,
    'n_steps' giving the number of steps taken in each trajectory, 'min_dist'
    giving the closest-approach distance, 'step_at_min' giving the step at
    which that closest approach occurred, 'total_time_ps' giving the
    accumulated simulation time, 'n_returns' counting how many times the
    trajectory returned from the escape sphere, and 'bb_triggered' which is
    True when a Brownian bridge caught the reaction.

    The reaction (encounter) arrays cover the M trajectories that reacted.
    'encounter_pos' with shape (M, 3) is the ligand centroid at reaction,
    'encounter_q' with shape (M, 4) is the ligand orientation at reaction,
    'encounter_traj' gives the trajectory indices, 'encounter_step' gives the
    step numbers, and 'encounter_n_pairs' gives the number of contact pairs
    satisfied. 'rxn_pair_dists_at_encounter' with shape (M, n_pairs) holds the
    pair distances at the moment of reaction.

    The near-miss arrays cover the K escaped trajectories at their closest
    approach. 'near_miss_pos' with shape (K, 3) is the centroid at closest
    approach, 'near_miss_q' with shape (K, 4) is the orientation there,
    'near_miss_traj' gives the trajectory indices, and 'near_miss_dist' gives
    the minimum-distance values.

    'path_steps' is a list of (n_recorded, 8) arrays with columns
    [traj_id, step, x, y, z, q0, q1, q2], and 'energy_steps' is a list of
    (n_recorded, 6) arrays with columns [traj_id, step, fx, fy, fz, dt].

    The histogram arrays are 'radial_bins' with shape (n_bins,) for the bin
    edges, 'radial_counts' with shape (n_bins-1,) for the histogram counts,
    'angular_theta' and 'angular_phi' for the theta and phi bin centers, and
    'angular_counts' with shape (n_theta, n_phi) for the two-dimensional
    angular histogram.

    The milestone arrays are 'milestone_radii' for the milestone radii,
    'milestone_flux_out' for the outward crossings, and 'milestone_flux_in'
    for the inward crossings.

    The contact and transition data are 'contact_pair_counts' with shape
    (n_pairs,) giving how often each pair was within the cutoff,
    'contact_total_steps' giving the total number of steps counted,
    'trans_bins' giving the radial bin edges for the transition matrix, and
    'trans_matrix' with shape (n_bins, n_bins) giving the count matrix.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # 1. results.json
    if output_cfg.results_json:
        k_on = result.rate_constant(D_rel, k_b)
        k_lo, k_hi = result.rate_constant_ci(D_rel, k_b, confidence)
        p = result.reaction_probability
        p_lo, p_hi = result.reaction_probability_ci(confidence)
        data = {
            "k_on": k_on,
            "k_on_low": k_lo,
            "k_on_high": k_hi,
            "k_on_units": "M-1 s-1",
            "P_rxn": p,
            "P_rxn_low": p_lo,
            "P_rxn_high": p_hi,
            "k_b": k_b,
            "k_b_units": "A3/ps",
            "D_rel": D_rel,
            "D_rel_units": "A2/ps",
            "n_trajectories": result.n_trajectories,
            "n_reacted": result.n_reacted,
            "n_escaped": result.n_escaped,
            "n_max_steps": result.n_max_steps,
            "r_start": result.r_start,
            "r_escape": result.r_escape,
            "wall_time_sec": result.elapsed_sec,
            "steps_per_sec": result.steps_per_sec,
            "confidence_level": confidence,
            # Total recorded contact steps, the denominator of the
            # contact_frequency column. Persisted so multi-GPU combine can pool
            # the frequency correctly across shards.
            "contact_total_steps": int(sim_data.get("contact_total_steps", 0)),
        }
        if k_on > 0:
            data["log10_k_on"] = math.log10(k_on)
        # State-machine runs carry a per-reaction count list. Pass it through
        # to results.json so downstream tools can read which reactions fired
        # and how many times.
        if "completed_reactions" in sim_data:
            data["completed_reactions"] = sim_data["completed_reactions"]
        p = work_dir / "results.json"
        p.write_text(json.dumps(data, indent=2))
        written.append(("results.json", p))
    # 2. trajectories.csv
    if output_cfg.trajectories_csv and "outcome" in sim_data:
        N = len(sim_data["outcome"])
        p = work_dir / "trajectories.csv"
        with open(p, "w") as f:
            f.write(
                "traj_id,outcome,n_steps,start_x,start_y,start_z,"
                "start_q0,start_q1,start_q2,start_q3,"
                "min_distance,step_at_min,total_time_ps,"
                "n_returns,bb_triggered\n"
            )
            outcome = sim_data["outcome"]
            n_steps = sim_data["n_steps"]
            spos = sim_data["start_pos"]
            sq = sim_data["start_q"]
            mdist = sim_data["min_dist"]
            smin = sim_data["step_at_min"]
            ttime = sim_data["total_time_ps"]
            nret = sim_data["n_returns"]
            bb = sim_data["bb_triggered"]
            outcome_map = {1: "reacted", 2: "escaped", 3: "max_steps", 0: "running"}
            for i in range(N):
                ostr = outcome_map.get(int(outcome[i]), "unknown")
                f.write(
                    f"{i},{ostr},{int(n_steps[i])},"
                    f"{spos[i,0]:.6f},{spos[i,1]:.6f},{spos[i,2]:.6f},"
                    f"{sq[i,0]:.6f},{sq[i,1]:.6f},{sq[i,2]:.6f},{sq[i,3]:.6f},"
                    f"{mdist[i]:.6f},{int(smin[i])},{ttime[i]:.4f},"
                    f"{int(nret[i])},{int(bb[i])}\n"
                )
        written.append(("trajectories.csv", p))
    # 3. encounters.csv
    if output_cfg.encounters_csv and "encounter_pos" in sim_data:
        epos = sim_data["encounter_pos"]
        if len(epos) > 0:
            p = work_dir / "encounters.csv"
            eq = sim_data["encounter_q"]
            etraj = sim_data["encounter_traj"]
            estep = sim_data["encounter_step"]
            enpairs = sim_data["encounter_n_pairs"]
            with open(p, "w") as f:
                f.write("traj_id,step,x,y,z,q0,q1,q2,q3,n_pairs_satisfied\n")
                for i in range(len(epos)):
                    f.write(
                        f"{int(etraj[i])},{int(estep[i])},"
                        f"{epos[i,0]:.6f},{epos[i,1]:.6f},{epos[i,2]:.6f},"
                        f"{eq[i,0]:.6f},{eq[i,1]:.6f},{eq[i,2]:.6f},{eq[i,3]:.6f},"
                        f"{int(enpairs[i])}\n"
                    )
            written.append(("encounters.csv", p))
    # 4. near_misses.csv
    if output_cfg.near_misses_csv and "near_miss_pos" in sim_data:
        npos = sim_data["near_miss_pos"]
        if len(npos) > 0:
            p = work_dir / "near_misses.csv"
            nq = sim_data["near_miss_q"]
            ntraj = sim_data["near_miss_traj"]
            ndist = sim_data["near_miss_dist"]
            with open(p, "w") as f:
                f.write("traj_id,min_distance,x,y,z,q0,q1,q2,q3\n")
                for i in range(len(npos)):
                    f.write(
                        f"{int(ntraj[i])},{ndist[i]:.6f},"
                        f"{npos[i,0]:.6f},{npos[i,1]:.6f},{npos[i,2]:.6f},"
                        f"{nq[i,0]:.6f},{nq[i,1]:.6f},{nq[i,2]:.6f},{nq[i,3]:.6f}\n"
                    )
            written.append(("near_misses.csv", p))
    # 5. paths.npz
    if output_cfg.full_paths and "path_steps" in sim_data:
        pdata = sim_data["path_steps"]
        if len(pdata) > 0:
            all_paths = np.vstack(pdata)  # (total_records, 8)
            p = work_dir / "paths.npz"
            np.savez_compressed(
                p,
                data=all_paths,
                columns=np.array(["traj_id", "step", "x", "y", "z", "q0", "q1", "q2"]),
            )
            written.append(("paths.npz", p))
    # 6. radial_density.csv
    if output_cfg.radial_density and "radial_bins" in sim_data:
        bins = sim_data["radial_bins"]
        counts = sim_data["radial_counts"]
        p = work_dir / "radial_density.csv"
        with open(p, "w") as f:
            f.write("r_center,r_low,r_high,count,density\n")
            total = counts.sum() if counts.sum() > 0 else 1
            for i in range(len(counts)):
                rc = 0.5 * (bins[i] + bins[i + 1])
                dr = bins[i + 1] - bins[i]
                vol = 4.0 / 3.0 * math.pi * (bins[i + 1] ** 3 - bins[i] ** 3)
                rho = counts[i] / (total * vol) if vol > 0 else 0
                f.write(
                    f"{rc:.4f},{bins[i]:.4f},{bins[i+1]:.4f},"
                    f"{int(counts[i])},{rho:.8e}\n"
                )
        written.append(("radial_density.csv", p))
    # 7. angular_map.npz
    if output_cfg.angular_map and "angular_counts" in sim_data:
        p = work_dir / "angular_map.npz"
        np.savez_compressed(
            p,
            counts=sim_data["angular_counts"],
            theta_centers=sim_data["angular_theta"],
            phi_centers=sim_data["angular_phi"],
        )
        written.append(("angular_map.npz", p))
    # 8. fpt_distribution.csv
    if output_cfg.fpt_distribution and "outcome" in sim_data:
        outcome = sim_data["outcome"]
        ttime = sim_data["total_time_ps"]
        mdist = sim_data["min_dist"]
        reacted_mask = outcome == 1
        if reacted_mask.any():
            p = work_dir / "fpt_distribution.csv"
            with open(p, "w") as f:
                f.write("traj_id,first_passage_time_ps,min_distance\n")
                for i in np.where(reacted_mask)[0]:
                    f.write(f"{i},{ttime[i]:.4f},{mdist[i]:.6f}\n")
            written.append(("fpt_distribution.csv", p))
    # 9. contact_frequency.csv
    if output_cfg.contact_frequency and "contact_pair_counts" in sim_data:
        counts = sim_data["contact_pair_counts"]
        total = sim_data.get("contact_total_steps", 1)
        p = work_dir / "contact_frequency.csv"
        with open(p, "w") as f:
            f.write("pair_index,n_contacts,frequency\n")
            for i, c in enumerate(counts):
                freq = c / total if total > 0 else 0
                f.write(f"{i},{int(c)},{freq:.8e}\n")
        written.append(("contact_frequency.csv", p))
    # 10. milestone_flux.csv
    if output_cfg.milestone_flux and "milestone_radii" in sim_data:
        radii = sim_data["milestone_radii"]
        fout = sim_data["milestone_flux_out"]
        fin = sim_data["milestone_flux_in"]
        p = work_dir / "milestone_flux.csv"
        with open(p, "w") as f:
            f.write("radius,flux_outward,flux_inward,net_flux\n")
            for i in range(len(radii)):
                f.write(
                    f"{radii[i]:.4f},{int(fout[i])},{int(fin[i])},"
                    f"{int(fout[i]-fin[i])}\n"
                )
        written.append(("milestone_flux.csv", p))
    # 11. p_commit.npz
    if output_cfg.p_commit and "outcome" in sim_data:
        outcome = sim_data["outcome"]
        spos = sim_data["start_pos"]
        # Bin the starting positions by their radius and compute the
        # reaction probability within each bin.
        r_start = np.linalg.norm(spos, axis=1)
        n_bins = 50
        bins = np.linspace(r_start.min(), r_start.max(), n_bins + 1)
        p_react = np.zeros(n_bins)
        n_count = np.zeros(n_bins)
        for i in range(len(outcome)):
            idx = min(np.searchsorted(bins, r_start[i]) - 1, n_bins - 1)
            idx = max(idx, 0)
            n_count[idx] += 1
            if outcome[i] == 1:
                p_react[idx] += 1
        p_commit_arr = np.divide(
            p_react, n_count, out=np.zeros_like(p_react), where=n_count > 0
        )
        p_path = work_dir / "p_commit.npz"
        np.savez_compressed(
            p_path, r_bins=bins, p_commit=p_commit_arr, n_samples=n_count
        )
        written.append(("p_commit.npz", p_path))
    # 12. transition_matrix.npz
    if output_cfg.transition_matrix and "trans_matrix" in sim_data:
        p = work_dir / "transition_matrix.npz"
        np.savez_compressed(
            p, bins=sim_data["trans_bins"], counts=sim_data["trans_matrix"]
        )
        written.append(("transition_matrix.npz", p))
    # 13. energetics.npz
    if output_cfg.energetics and "energy_steps" in sim_data:
        edata = sim_data["energy_steps"]
        if len(edata) > 0:
            all_e = np.vstack(edata)  # (total_records, 6)
            p = work_dir / "energetics.npz"
            np.savez_compressed(
                p,
                data=all_e,
                columns=np.array(["traj_id", "step", "fx", "fy", "fz", "dt"]),
            )
            written.append(("energetics.npz", p))
    # 14. pose_clusters.csv
    if output_cfg.pose_clusters and "encounter_q" in sim_data:
        eq = sim_data["encounter_q"]
        if len(eq) > 5:
            # Simple angular clustering. Convert each quaternion to the
            # rotation angle θ and bin the angles into octants, where
            # θ = 2 arccos(|q0|).
            angles = 2.0 * np.arccos(np.clip(np.abs(eq[:, 0]), 0, 1))
            n_clusters = min(8, len(eq))
            bins = np.linspace(0, np.pi, n_clusters + 1)
            labels = np.digitize(angles, bins) - 1
            labels = np.clip(labels, 0, n_clusters - 1)
            p = work_dir / "pose_clusters.csv"
            with open(p, "w") as f:
                f.write("traj_id,cluster,angle_rad,q0,q1,q2,q3\n")
                etraj = sim_data["encounter_traj"]
                for i in range(len(eq)):
                    f.write(
                        f"{int(etraj[i])},{int(labels[i])},{angles[i]:.6f},"
                        f"{eq[i,0]:.6f},{eq[i,1]:.6f},"
                        f"{eq[i,2]:.6f},{eq[i,3]:.6f}\n"
                    )
            written.append(("pose_clusters.csv", p))
    # Summary
    if written:
        print(f"\n  Output files ({len(written)})")
        for name, fp in written:
            size = fp.stat().st_size
            if size > 1e6:
                print(f"  {name:<30s}  {size/1e6:.1f} MB")
            elif size > 1e3:
                print(f"  {name:<30s}  {size/1e3:.1f} KB")
            else:
                print(f"  {name:<30s}  {size} B")
    return written
