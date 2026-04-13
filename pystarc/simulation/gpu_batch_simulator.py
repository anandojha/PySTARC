"""
PySTARC batch GPU BD simulator
==============================

Runs all trajectories simultaneously on GPU.

Key difference from NAMSimulator:
- NAMSimulator: Python loop, one trajectory at a time, GPU called per step
- GPUBatchSimulator: All N_traj trajectories as GPU arrays, entire BD loop
runs on GPU, Python only checks termination

Architecture:
- positions : (N_traj, 3)      - ligand centroid position per trajectory
- orientations: (N_traj, 4)    - quaternion per trajectory
- All random numbers pre-generated on GPU
- Force, BD step, reaction check - all GPU operations
- Python loop only: check escape/reaction masks, recycle finished trajs

This saturates the GPU: N_traj x N_atoms threads per kernel launch.
"""

from __future__ import annotations
from pystarc.simulation.nam_simulator import NAMParameters, SimulationResult
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.lib.numerical import romberg_integrate
from typing import List, Optional, Tuple, Dict
from scipy.special import erf as scipy_erf
from scipy.special import erf, erfc
from dataclasses import dataclass
import scipy.stats as stats
from pathlib import Path
import numpy as np
import math
import copy
import time

try:
    import cupy as cp

    _CUPY = True
except ImportError:
    _CUPY = False


@dataclass
class GPUBatchResult:
    """Results from batch GPU BD simulation."""

    n_trajectories: int
    n_reacted: int
    n_escaped: int
    n_max_steps: int
    reaction_counts: Dict[str, int]
    r_start: float
    r_escape: float
    dt: float
    elapsed_sec: float
    steps_per_sec: float
    sim_data: Dict = None  # collected trajectory data for output_writer

    @property
    def n_completed(self) -> int:
        return self.n_reacted + self.n_escaped

    @property
    def reaction_probability(self) -> float:
        n = self.n_completed
        return self.n_reacted / n if n > 0 else 0.0

    def reaction_probability_ci(self, confidence: float = 0.95):
        n = self.n_completed
        if n == 0:
            return (0.0, 1.0)
        z = float(stats.norm.ppf(0.5 + confidence / 2.0))
        p = self.reaction_probability
        z2n = z**2 / n
        denom = 1.0 + z2n
        centre = (p + z2n / 2.0) / denom
        margin = z * math.sqrt(max(p * (1 - p) / n + z2n / (4.0 * n), 0)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def rate_constant(self, D_rel: float, k_b: float = 0.0) -> float:
        P = self.reaction_probability
        if P == 0.0:
            return 0.0
        # CONV = N_A * 1e-30 / 1e-12 / 1e-3 = 6.022e8  (A^3/ps -> M^-1 s^-1)
        CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        if k_b > 0.0:
            # k_on = CONV * k_b * P_rxn
            # k_b = Romberg integral of exp(U(r))/D
            return CONV * k_b * P
        else:
            # Smoluchowski fallback (no potential)
            k_D = 4.0 * math.pi * D_rel * self.r_start
            beta = self.r_start / self.r_escape
            return CONV * k_D * P / (1.0 - P * (1.0 - beta))

    def rate_constant_ci(
        self, D_rel: float, k_b: float = 0.0, confidence: float = 0.95
    ):
        p_lo, p_hi = self.reaction_probability_ci(confidence)

        def _k(P):
            if P <= 0:
                return 0.0
            CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
            if k_b > 0.0:
                return CONV * k_b * P
            k_D = 4.0 * math.pi * D_rel * self.r_start
            beta = self.r_start / self.r_escape
            return CONV * k_D * P / (1 - P * (1 - beta))

        return (_k(p_lo), _k(p_hi))

    def summary(self, D_rel: float, k_b: float = 0.0, confidence: float = 0.95) -> str:
        k = self.rate_constant(D_rel, k_b)
        k_lo, k_hi = self.rate_constant_ci(D_rel, k_b, confidence)
        p = self.reaction_probability
        p_lo, p_hi = self.reaction_probability_ci(confidence)
        pct = int(confidence * 100)
        lines = [
            f"  Trajectories : {self.n_trajectories:,}",
            f"  Completed    : {self.n_completed:,}  "
            f"({self.n_reacted:,} reacted + {self.n_escaped:,} escaped)",
        ]
        if self.n_max_steps:
            lines.append(f"  Max-steps    : {self.n_max_steps:,}  (excluded)")
        lines += [
            f"  P_rxn        : {p:.6f}  ({pct}% CI: [{p_lo:.6f}, {p_hi:.6f}])",
            f"  k_on         : {k:.4e} M-1 s-1",
            f"  {pct}% CI     : [{k_lo:.4e}, {k_hi:.4e}] M-1 s-1",
        ]
        if k > 0:
            lines.append(f"  log10(k_on)  : {math.log10(k):.3f}")
        lines += [
            f"  Wall time    : {self.elapsed_sec:.1f} s",
            f"  BD steps/sec : {self.steps_per_sec:,.0f}",
            f"  Backend      : CUPY (batch GPU)",
        ]
        return "\n".join(lines)


class GPUBatchSimulator:
    """
    Batch GPU Brownian Dynamics simulator.
    All N_traj trajectories run simultaneously on GPU.
    GPU-Util will be high (>80%) unlike sequential NAMSimulator.

    Parameters
    ----------
    mol1        : receptor Molecule (fixed)
    mol2        : ligand Molecule (diffusing)
    mob         : MobilityTensor
    pathway_set : reaction criteria
    params      : NAMParameters
    batch_engine: GPUBatchForceEngine
    """

    def __init__(
        self, mol1, mol2, mob, pathway_set, params: NAMParameters, batch_engine
    ):
        if not _CUPY:
            raise RuntimeError("CuPy required for GPU batch simulation.")
        self.mol1 = mol1
        self.mol2 = mol2
        self.mob = mob
        self.pathway_set = pathway_set
        self.params = params
        self.engine = batch_engine
        # Extract ligand geometry
        c0 = mol2.centroid()
        self._mol2_pos0 = cp.asarray(
            mol2.positions_array() - c0, dtype=cp.float64
        )  # (N_lig, 3)
        self._charges = cp.asarray(mol2.charges_array(), dtype=cp.float64)  # (N_lig,)
        self._N_lig = len(mol2.atoms)
        # Outer propagator (LMZ) parameters - computed once before run()
        self._return_prob = 0.0  # set in run(): P(return to b | at trigger_r)
        self._k_b = (
            0.0  # set in run(): Romberg k_b (encounter rate from Romberg integral)
        )
        self._qb_factor = 1.1  # trigger at 1.1 × b_sphere (standard qb_factor = 1.1)
        self._bradius_cover = 0.0  # set in run(): cover zone boundary
        self._use_hi = getattr(params, "hydrodynamic_interactions", False)
        # Ghost atom indices for reaction check
        self._rec_gho_indices = []
        self._lig_gho_indices = []
        self._rxn_cutoffs = []
        for rxn in pathway_set.reactions:
            for pair in rxn.criteria.pairs:
                self._rec_gho_indices.append(pair.mol1_atom_index)
                self._lig_gho_indices.append(pair.mol2_atom_index)
                self._rxn_cutoffs.append(pair.distance_cutoff)
        self._rxn_cutoffs = np.array(self._rxn_cutoffs)
        self._rxn_cutoff_min = (
            float(self._rxn_cutoffs.min()) if len(self._rxn_cutoffs) else 5.0
        )
        # n_needed: minimum pairs that must fire simultaneously
        # -1 means all pairs; set from first reaction's ReactionCriteria
        if pathway_set.reactions:
            nn = pathway_set.reactions[0].criteria.n_needed
            total_pairs = len(self._rxn_cutoffs)
            self._rxn_n_needed = total_pairs if (nn < 0 or nn > total_pairs) else nn
        else:
            self._rxn_n_needed = 1
        # Receptor ghost atom positions (fixed - receptor never moves)
        if self._rec_gho_indices:
            rec_pos = mol1.positions_array()
            self._rec_gho_pos = cp.asarray(
                rec_pos[self._rec_gho_indices], dtype=cp.float64
            )
            self._rxn_cutoffs_gpu = cp.asarray(self._rxn_cutoffs, dtype=cp.float64)
        else:
            self._rec_gho_pos = None
            self._rxn_cutoffs_gpu = None

    def _compute_return_prob(self) -> float:
        """
        Compute P(return to b | currently at trigger_r) and k_b (b-sphere rate).
        P_return(r->b) = h(trigger_r) / h(b)
        where h(r) = integral_r^q_outer exp(U(s)) / s^2 ds
        Also computes and stores self._k_b via the Romberg integral:
            k_b = 4pi / integral_0^{1/b} exp(U(1/s)) / D ds
        """
        b = self.params.r_start
        # q_out = r_escape - return_prob is now applied at r_escape,
        # so the Romberg integral must use r_escape as the outer boundary.
        # the reference uses qradius = 20*max_mol_radius, but applies return_prob
        # at qradius. Since PySTARC applies at r_escape, q_out = r_escape.
        q_out = self.params.r_escape
        trigger = self._qb_factor * b
        q_rec = float(self.mol1.total_charge())
        q_lig = float(self.mol2.total_charge())
        debye = getattr(self.engine, "_debye", 7.858)
        eps0 = 0.000142
        sdie = 78.0
        eps = sdie * eps0
        D_t = float(self.mob.relative_translational_diffusion())
        # kT scaling: V(r) is in kBT_298 units. At T≠298.15K,
        # Boltzmann factor is exp(V/kT_scale) where kT_scale = T/298.15.
        _kT_scale = getattr(self.params, "_kT_scale", 1.0)
        if abs(q_rec * q_lig) < 1e-9:
            # Pure diffusion fallback
            self._k_b = 4.0 * math.pi * D_t * b
            return b / q_out  # P_return = k_b(b)/k_b(q) for free diffusion

        def U(r):
            return q_rec * q_lig / (4.0 * math.pi * eps * r) * math.exp(-r / debye)

        # With HI: D_parallel(r) = dpre*(ainv - 3/r + 2*a2/r³)
        # Without: D_parallel(r) = dpre*ainv = D_t (constant)
        # the reference variables:
        #   D_factor = kT/mu,  dpre = D_factor/(6π)
        #   a0, a1 = hydro radii
        #   ainv = 1/a0 + 1/a1
        #   a2 = 0.5*(a0² + a1²)
        #   At s=0: integrand = 1/(dpre*ainv) regardless of HI
        use_hi = self._use_hi
        if use_hi:
            a0 = float(self.mob.radius1)  # receptor hydro radius
            a1 = float(self.mob.radius2)  # ligand hydro radius
            a2 = 0.5 * (a0**2 + a1**2)
            # dpre*ainv = D_t, so dpre = D_t / ainv
            ainv = 1.0 / a0 + 1.0 / a1
            dpre = D_t / ainv

        def intgd_romberg(s):
            if s == 0.0:
                return 1.0 / D_t  # same with or without HI
            r = 1.0 / s
            v = U(r) / _kT_scale  # scale V to kBT_T units
            if use_hi:
                D_para = dpre * (ainv - 3.0 / r + 2.0 * a2 / (r**3))
                return math.exp(v) / D_para
            else:
                return math.exp(v) / D_t

        try:
            # k_b(b) = Romberg integral to b
            self._k_b = (
                4.0 * math.pi / romberg_integrate(intgd_romberg, 0.0, 1.0 / b, tol=1e-8)
            )
            # k_b(q_out) = Romberg integral to escape sphere
            k_q = (
                4.0
                * math.pi
                / romberg_integrate(intgd_romberg, 0.0, 1.0 / q_out, tol=1e-8)
            )
            # return_prob = k_b(b) / k_b(q_out)
            rp = self._k_b / k_q if k_q > 0 else 0.5
            # Compute bradius_cover
            # Finds x0 where P(exceeding linear force region) < 0.01
            # then bradius_cover = b + x0
            self._bradius_cover = self._compute_bradius_cover(
                b, q_rec, q_lig, eps, debye, D_t
            )
            return float(max(0.0, min(1.0, rp)))
        except Exception:
            self._k_b = 4.0 * math.pi * D_t * b
            self._bradius_cover = b * 1.05  # fallback: 5% above b
            return b / q_out

    def _compute_bradius_cover(self, b, q_rec, q_lig, eps, debye, D_t) -> float:
        """
        Compute bradius_cover: b-sphere + cover zone width.
        Cover zone boundary computation.
        """
        V_factor = q_rec * q_lig / (4.0 * math.pi * eps)

        def force_yukawa(r):
            rm1 = 1.0 / r
            V = V_factor * math.exp(-r / debye) * rm1
            return V * (rm1 + 1.0 / debye)

        # ts_boundary: where Yukawa force curvature > curve_tol=0.05
        curve_tol = 0.05

        def reldiff(r):
            L = debye
            rd = (2 * L**2 + 2 * r * L + r**2) * abs(r - b) / (r * L * (L + r))
            return rd - curve_tol

        rb = 2.0 * b
        while reldiff(rb) < 0.0:
            rb *= 2.0
        rlo, rhi = b, rb
        while rhi - rlo > 1e-6 * b:
            rm = 0.5 * (rlo + rhi)
            if reldiff(rm) * reldiff(rhi) < 0.0:
                rlo = rm
            else:
                rhi = rm
        L_linear = min(0.5 * (rlo + rhi) - b, 0.1 * b)

        # bisect for x0 where P(exceed linear region) < thresh=0.01
        F_b = force_yukawa(b)

        def prob_exceed(x0):
            if x0 <= 1e-10:
                return 1.0
            try:
                arg1 = (L_linear - x0**2 * F_b - x0) / (2.0 * x0)
                arg2 = (x0 * F_b + 1.0) / 2.0
                return erfc(arg1) / (erf(arg2) + 1.0 + 1e-30)
            except Exception:
                return 0.0

        lo, hi = 0.0, L_linear
        for _ in range(100):
            if hi - lo < 1e-6 * max(L_linear, 1e-10):
                break
            mid = 0.5 * (lo + hi)
            if prob_exceed(mid) > 0.01:
                hi = mid
            else:
                lo = mid
        return float(b + 0.5 * (lo + hi))

    def run(self) -> GPUBatchResult:
        """Run all trajectories as a GPU batch."""
        # GPU warmup: force CuPy/CUDA kernel JIT compilation BEFORE timing starts.
        # Without this, the first run pays ~2-5s JIT cost inside the timed region.
        _w = cp.zeros(1, dtype=cp.float64)
        _w + 1.0
        cp.cuda.Stream.null.synchronize()
        del _w
        t0 = time.time()
        N = self.params.n_trajectories
        rng = np.random.default_rng(self.params.seed)
        D_t = float(self.mob.relative_translational_diffusion())
        D_r = float(self.mob.relative_rotational_diffusion())
        dt = self.params.dt
        dt_rxn = self.params.dt_rxn
        r_b = self.params.r_start
        r_esc = self.params.r_escape
        print(
            f"  GPU Batch BD: {N:,} trajectories x up to "
            f"{self.params.max_steps:,} steps"
        )
        print(f"  D_rel={D_t:.5f} A2/ps  dt={dt}ps  dt_rxn={dt_rxn}ps")
        print(f"  Progress columns:")
        print(f"    step      = BD time-step counter (all trajectories combined)")
        print(f"    done      = trajectories finished (reacted + escaped) / total")
        print(f"    reacted   = ligand entered reaction zone (criterion fired)")
        print(f"    escaped   = ligand diffused beyond escape radius")
        print(f"    steps/sec = GPU throughput (all active trajectories)")
        # Outer propagator (LMZ) setup
        # Compute P_return = h(trigger_r) / h(b)  [Smoluchowski hitting prob]
        # trigger_r = QB_FACTOR × b = 1.1 × b
        # When r > trigger_r: snap to b with P_return, else escape.
        trigger_r = self._qb_factor * r_b  # kept for reference only
        self._return_prob = self._compute_return_prob()
        print(
            f"  Outer propagator (LMZ): return_prob={self._return_prob:.4f}"
            f"  (applied at r_esc={r_esc:.1f}Å)  k_b={self._k_b:.4f} Å³/ps"
        )
        # Initialise all trajectories on GPU
        # Random start positions on b-sphere
        v = cp.asarray(rng.standard_normal((N, 3)), dtype=cp.float64)
        v /= cp.linalg.norm(v, axis=1, keepdims=True)
        pos = v * r_b  # (N, 3) centroid positions
        # Random initial quaternions (w, x, y, z)
        q = self._random_quaternions_gpu(N, rng)  # (N, 4)
        # Trajectory status: 0=running, 1=reacted, 2=escaped, 3=max_steps
        status = cp.zeros(N, dtype=cp.int32)
        n_steps = cp.zeros(N, dtype=cp.int64)
        # Track whether each trajectory is currently in the outer region
        n_reacted = 0
        n_escaped = 0
        n_maxsteps = 0
        total_steps = 0
        _stall_done = 0  # for stall detection
        _resume_step = 0  # step offset if resuming from checkpoint
        # Resume from checkpoint if available
        _ckpt_dir_resume = Path(getattr(self.params, "_work_dir", "bd_sims"))
        _ckpt_path_resume = _ckpt_dir_resume / "checkpoint.npz"
        if _ckpt_path_resume.exists():
            try:
                ckpt = np.load(str(_ckpt_path_resume), allow_pickle=True)
                pos = cp.asarray(ckpt["pos"])
                q = cp.asarray(ckpt["q"])
                status = cp.asarray(ckpt["status"])
                n_steps = cp.asarray(ckpt["n_steps"])
                n_reacted = int(ckpt["n_reacted"])
                n_escaped = int(ckpt["n_escaped"])
                n_maxsteps = int(ckpt["n_maxsteps"])
                total_steps = int(ckpt["total_steps"])
                _resume_step = int(ckpt["step"])
                _done_resume = int((status != 0).sum())
                print(
                    f"  Resumed from checkpoint: step={_resume_step}, "
                    f"{_done_resume}/{N} done "
                    f"({n_reacted} react, {n_escaped} esc)"
                )
                # Offset RNG to avoid repeating the same random sequence
                rng = np.random.default_rng(self.params.seed + _resume_step)
            except Exception as e:
                print(f" Checkpoint load failed: {e}. Starting fresh.")
        # Data collection arrays
        # Per-trajectory tracking (always collected - lightweight)
        _output_cfg = getattr(self.params, "_output_cfg", None)
        _save_interval = 10
        if _output_cfg is not None:
            _save_interval = max(getattr(_output_cfg, "save_interval", 10), 1)
        _start_pos = cp.asnumpy(pos.copy())  # (N, 3)
        _start_q = cp.asnumpy(q.copy())  # (N, 4)
        _min_dist = np.full(N, 1e30, dtype=np.float64)  # closest approach
        _step_at_min = np.zeros(N, dtype=np.int64)
        _total_time = np.zeros(N, dtype=np.float64)  # accumulated sim time (ps)
        _n_returns = np.zeros(N, dtype=np.int64)
        _bb_triggered = np.zeros(N, dtype=np.int64)
        _prev_r = cp.linalg.norm(pos, axis=1)  # for milestone crossing detection
        # Encounter snapshots (appended when reaction happens)
        _enc_pos = []
        _enc_q = []
        _enc_traj = []
        _enc_step = []
        _enc_npairs = []
        # Near-miss tracking: store pos/q at min_dist (updated each step)
        _nm_pos = cp.asnumpy(pos.copy())
        _nm_q = cp.asnumpy(q.copy())
        # Path recording (every save_interval steps)
        _path_data = []
        _energy_data = []
        # Radial density histogram
        _rad_bins = np.linspace(0, float(r_esc * 1.2), 201)
        _rad_counts = np.zeros(len(_rad_bins) - 1, dtype=np.int64)
        # Angular occupancy
        _n_theta = 36
        _n_phi = 72
        _ang_counts = np.zeros((_n_theta, _n_phi), dtype=np.int64)
        # Milestone flux: shells at 10%, 20%, ..., 100% of r_esc
        _ms_radii = np.linspace(float(r_b), float(r_esc), 11)
        _ms_flux_out = np.zeros(len(_ms_radii), dtype=np.int64)
        _ms_flux_in = np.zeros(len(_ms_radii), dtype=np.int64)
        # Contact frequency
        _n_rxn_pairs = len(self._rxn_cutoffs) if self._rxn_cutoffs.size > 0 else 0
        _contact_counts = np.zeros(max(_n_rxn_pairs, 1), dtype=np.int64)
        _contact_total_steps = 0
        # Transition matrix (radial bins)
        _trans_n = 50
        _trans_bins = np.linspace(0, float(r_esc * 1.2), _trans_n + 1)
        _trans_mat = np.zeros((_trans_n, _trans_n), dtype=np.int64)
        # Pre-compute adaptive dt constants (avoids Python overhead per step)
        _adt_debye = float(getattr(self.engine, "_debye", 7.828))
        _adt_eps = 78.0 * 0.000142
        _adt_V0 = (
            float(self.mol1.total_charge())
            * float(self.mol2.total_charge())
            / (4.0 * math.pi * _adt_eps)
        )
        _adt_has_force = abs(_adt_V0) > 1e-9
        # the reference minimum_core_dt: floor on adaptive dt
        _min_core_dt = float(getattr(self.params, "minimum_core_dt", 0.0))
        # Pre-compute D_eff for Brownian bridge
        # Pair distances change due to translation AND rotation.
        # D_eff = D_trans + D_rot_rec * L_rec² + D_rot_lig * L_lig²
        # where L is the lever arm (distance from centroid to GHO atom).
        # For charged_spheres: L=0, D_eff=D_trans. Exact.
        # For thrombin: D_eff ≈ 2×D_trans (rotation doubles effective D).
        if self._rec_gho_pos is not None and len(self._rec_gho_indices) > 0:
            r_h_rec = float(getattr(self.mob, "radius1", 1.0))
            r_h_lig = float(getattr(self.mob, "radius2", 1.0))
            # Per-molecule translational D (not D_rel which is the sum)
            # D_trans_rec = kT/(6πμa_rec) = D_rel × a_lig/(a_rec+a_lig)
            # D_rot = D_trans × 3/(4a²) (Stokes-Einstein-Debye)
            a_sum = r_h_rec + r_h_lig if (r_h_rec + r_h_lig) > 0 else 1.0
            D_trans_rec = D_t * r_h_lig / a_sum
            D_trans_lig = D_t * r_h_rec / a_sum
            D_rot_rec = D_trans_rec * 3.0 / (4.0 * r_h_rec**2) if r_h_rec > 0 else 0.0
            D_rot_lig = D_trans_lig * 3.0 / (4.0 * r_h_lig**2) if r_h_lig > 0 else 0.0
            # Lever arms: distance from centroid to GHO atom
            rec_gho_np = cp.asnumpy(self._rec_gho_pos)  # (n_pairs, 3)
            lig_gho_body = cp.asnumpy(self._mol2_pos0[self._lig_gho_indices])
            L_rec = np.linalg.norm(rec_gho_np, axis=1)  # (n_pairs,)
            L_lig = np.linalg.norm(lig_gho_body, axis=1)  # (n_pairs,)
            D_eff_np = D_t + D_rot_rec * L_rec**2 + D_rot_lig * L_lig**2
            self._bb_D_eff = cp.asarray(D_eff_np, dtype=cp.float64)  # (n_pairs,)
            print(
                f"  BB D_eff: D_trans={D_t:.5f}, D_rot_rec={D_rot_rec:.6f}, "
                f"D_rot_lig={D_rot_lig:.6f}"
            )
            print(
                f"  BB D_eff per pair: min={D_eff_np.min():.5f}, "
                f"max={D_eff_np.max():.5f}, mean={D_eff_np.mean():.5f} Å²/ps "
                f"({D_eff_np.mean()/D_t:.1f}x D_trans)"
            )
        else:
            self._bb_D_eff = None
        # -- VERBOSE: Startup diagnostics
        print(f"\n  Diagnostic dump (verbose mode)")
        print(
            f"  Receptor: Q={float(self.mol1.total_charge()):+.2f} e, "
            f"n_atoms={len(self.mol1.atoms)}"
        )
        print(
            f"  Ligand:   Q={float(self.mol2.total_charge()):+.2f} e, "
            f"n_atoms={len(self.mol2.atoms)}, n_lig_gpu={self._N_lig}"
        )
        _q = cp.asnumpy(self._charges)
        print(f"  Charges GPU: {len(_q)} atoms, range=[{_q.min():.2f}, {_q.max():.2f}]")
        _m2p = cp.asnumpy(self._mol2_pos0)
        print(
            f"  Mol2 pos0: {_m2p.shape[0]} atoms, "
            f"extent=[{_m2p.min(0).tolist()}, {_m2p.max(0).tolist()}]"
        )
        print(f"  Yukawa V_factor (Q_rec*Q_lig/(4pi*eps)): {_adt_V0:.6f} kBT·Å")
        print(f"  Has electrostatic force: {_adt_has_force}")
        print(
            f"  Engine has Yukawa fallback: {getattr(self.engine, '_has_yukawa', False)}"
        )
        print(
            f"  Engine V_factor (Q_rec only): "
            f"{getattr(self.engine, '_V_factor', 'N/A')}"
        )
        print(
            f"  Engine receptor_charge: "
            f"{getattr(self.engine, '_rec_charge', 'N/A')}"
        )
        print(f"  Debye length: {_adt_debye:.3f} Å")
        print(f"  r_b (b-sphere): {r_b:.3f} Å")
        print(f"  r_esc (escape): {r_esc:.3f} Å")
        print(f"  return_prob: {self._return_prob:.6f}")
        print(f"  k_b: {self._k_b:.6f} Å³/ps")
        # Grid info
        if hasattr(self.engine, "_elec_grids_gpu"):
            print(f"  Elec grids on GPU: {len(self.engine._elec_grids_gpu)}")
            for i, g in enumerate(self.engine._elec_grids_gpu):
                lo_np = (
                    g["lo"] if isinstance(g["lo"], np.ndarray) else np.array(g["lo"])
                )
                hi_np = (
                    g["hi"] if isinstance(g["hi"], np.ndarray) else np.array(g["hi"])
                )
                print(
                    f"    Grid {i}: nx={g['nx']} ny={g['ny']} nz={g['nz']}  "
                    f"lo=[{lo_np[0]:.3f},{lo_np[1]:.3f},{lo_np[2]:.3f}]  "
                    f"hi=[{hi_np[0]:.3f},{hi_np[1]:.3f},{hi_np[2]:.3f}]"
                )
        if hasattr(self.engine, "_born_grids_gpu"):
            print(f"  Born grids on GPU: {len(self.engine._born_grids_gpu)}")
        # Reaction criterion info
        print(f"  Reaction pairs: {len(self._rxn_cutoffs)}")
        print(f"  n_needed: {self._rxn_n_needed}")
        if self._rec_gho_pos is not None:
            _gho = cp.asnumpy(self._rec_gho_pos)
            print(
                f"  Rec GHO: {_gho.shape[0]} positions, "
                f"r_range=[{float(cp.linalg.norm(self._rec_gho_pos, axis=1).min()):.1f}, "
                f"{float(cp.linalg.norm(self._rec_gho_pos, axis=1).max()):.1f}] Å from origin"
            )
        _cuts = self._rxn_cutoffs
        if len(set(_cuts)) == 1:
            print(f"  Rxn cutoffs: all {_cuts[0]:.1f} Å ({len(_cuts)} pairs)")
        else:
            print(f"  Rxn cutoffs: {_cuts}")
        print(f" End startup diagnostics\n")
        # Pre-compute HI constants (avoids Python overhead per step)
        if self._use_hi:
            _hi_mu = 0.243
            _hi_Df = 1.0 / _hi_mu
            _hi_pi6 = 6.0 * math.pi
            _hi_pi8 = 8.0 * math.pi
            _hi_a0 = float(getattr(self.mob, "radius1", 0.0))
            _hi_a1 = float(getattr(self.mob, "radius2", 0.0))
            _hi_a2 = 0.5 * (_hi_a0**2 + _hi_a1**2)
            _hi_ainv = (
                _hi_Df / (_hi_pi6 * _hi_a0) + _hi_Df / (_hi_pi6 * _hi_a1)
            ) / _hi_Df
        # Convergence milestones & checkpoint setup
        _conv_interval = getattr(self.params, "convergence_interval", 10)
        if _conv_interval > 0:
            _conv_milestones = set(
                int(N * p / 100) for p in range(_conv_interval, 100, _conv_interval)
            )
        else:
            _conv_milestones = set()
        _conv_CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        _last_conv_done = 0
        _progress_interval_sec = 10.0  # print progress every N seconds
        _last_progress_time = time.time()
        _ckpt_interval = getattr(self.params, "checkpoint_interval", 0)
        _ckpt_dir = None
        if _ckpt_interval > 0:
            _ckpt_dir = Path(getattr(self.params, "_work_dir", "bd_sims"))
            _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _last_ckpt_done = 0
        # Overlap check setup
        _use_overlap = getattr(self.params, "_overlap_check", True)
        _rec_pos_overlap = None
        _n_overlap_rejected = 0
        _overlap_threshold = 1.5
        if _use_overlap:
            rec_pos_np = self.mol1.positions_array()  # (N_rec, 3)
            _rec_pos_overlap = cp.asarray(rec_pos_np, dtype=cp.float64)
            print(
                f"  Overlap check: enabled (threshold={_overlap_threshold} Å, "
                f"{len(rec_pos_np)} receptor atoms)"
            )
        # Main BD loop
        for step in range(_resume_step, self.params.max_steps):
            running = status == 0  # (N,) bool mask
            n_run = int(running.sum())
            if n_run == 0:
                break
            # Place ligand atoms for all running trajectories
            # For memory efficiency with large ligands,
            # we avoid creating the full (N_run, N_lig, 3) array.
            # Instead, compute GHO positions for reaction check (small),
            # and batch the full pos_lig for force evaluation.
            run_idx = cp.where(running)[0]
            pos_run = pos[run_idx]  # (N_run, 3) centroid positions
            q_run = q[run_idx]  # (N_run, 4) orientations
            # Rotate ligand atoms: R(q) @ mol2_pos0 + centroid
            R = self._quats_to_rotmats(q_run)  # (N_run, 3, 3)
            # Memory-safe batch size for force evaluation
            # Engine creates ~6 arrays of (batch, N_lig, 3): pos, forces,
            # energies, masks, etc. Total ≈ batch × N_lig × 150 bytes.
            N_lig = self._mol2_pos0.shape[0]
            _cfg_batch = getattr(self.params, "gpu_force_batch", 0)
            if _cfg_batch > 0:
                _force_batch = _cfg_batch
            else:
                _force_batch = max(1000, int(4 * 1024**3 / max(1, N_lig * 150)))
            _force_batch = min(_force_batch, n_run)  # never exceed running count
            _n_force_batches = (n_run + _force_batch - 1) // _force_batch
            if step == 0:
                print(
                    f"  Force batch: {_force_batch:,} traj/batch "
                    f"({_n_force_batches} batches for {n_run:,} running)"
                )
            # Reaction check (only GHO atoms - small)
            if self._rec_gho_pos is not None:
                # Only compute GHO atom positions, not all ligand atoms
                _gho_mol2 = self._mol2_pos0[self._lig_gho_indices]  # (n_gho, 3)
                _gho_pos = cp.einsum("nij,kj->nki", R, _gho_mol2) + pos_run[:, None, :]
                # Build mini pos_lig with only GHO atoms for reaction check
                reacted = self._check_reactions_gpu_gho(_gho_pos)  # (N_run,) bool
                del _gho_pos
            else:
                reacted = cp.zeros(len(run_idx), dtype=cp.bool_)
            # Mark newly reacted (only those still running)
            still_running_mask = status[run_idx] == 0
            newly_reacted = run_idx[reacted & still_running_mask]
            if len(newly_reacted) > 0:
                status[newly_reacted] = 1
                n_reacted += int(len(newly_reacted))
                # Record encounter snapshots
                _nr_np = cp.asnumpy(newly_reacted)
                _enc_traj.extend(_nr_np.tolist())
                _enc_step.extend([step] * len(_nr_np))
                _enc_pos.append(cp.asnumpy(pos[newly_reacted]))
                _enc_q.append(cp.asnumpy(q[newly_reacted]))
                _enc_npairs.extend([self._rxn_n_needed] * len(_nr_np))
            r_mag = cp.linalg.norm(pos_run, axis=1)  # (N_run,)
            # Trajectories reaching r_escape: apply return_prob
            at_escape = (r_mag >= r_esc) & still_running_mask & ~reacted
            if cp.any(at_escape):
                esc_idx = run_idx[at_escape]
                u_outer = cp.asarray(rng.random(int(at_escape.sum())), dtype=cp.float64)
                returns = u_outer < self._return_prob
                # Snapped back to b-sphere
                if cp.any(returns):
                    ret_idx = esc_idx[returns]
                    new_ret = ret_idx[status[ret_idx] == 0]
                    if len(new_ret) > 0:
                        dirs = pos[new_ret] / cp.linalg.norm(
                            pos[new_ret], axis=1, keepdims=True
                        )
                        pos[new_ret] = dirs * r_b
                        # rot = diffusional_rotation(t * Drot) where
                        # t is accumulated time in outer propagator. For large t
                        # (trajectory diffused to r_escape and back), this is
                        # effectively a random rotation.
                        q[new_ret] = self._random_quaternions_gpu(
                            int(len(new_ret)), rng
                        )
                        # Track returns
                        _ret_np = cp.asnumpy(new_ret)
                        _n_returns[_ret_np] += 1
                # True escapes
                if cp.any(~returns):
                    true_esc = esc_idx[~returns]
                    new_esc = true_esc[status[true_esc] == 0]
                    if len(new_esc) > 0:
                        status[new_esc] = 2
                        n_escaped += int(len(new_esc))
            # Still running = not reacted, not escaped, not done
            still_running = status[run_idx] == 0
            # Refresh positions after escape/return handling
            pos_run = pos[run_idx]
            q_run = q[run_idx]
            R = self._quats_to_rotmats(q_run)
            r_mag = cp.linalg.norm(pos_run, axis=1)
            # pos_lig not computed here - batched during force evaluation
            # Force calculation - GPU batch (memory-safe)
            sr_mask = still_running
            sr_idx = run_idx[sr_mask]
            if len(sr_idx) == 0:
                continue
            R_sr = R[sr_mask]  # (N_sr, 3, 3)
            cen_sr = pos_run[sr_mask]  # (N_sr, 3) centroids
            N_sr = len(sr_idx)
            # Batch force computation to limit GPU memory
            forces_gpu = cp.zeros((N_sr, 3), dtype=cp.float64)
            torques_gpu = cp.zeros((N_sr, 3), dtype=cp.float64)
            for _fb0 in range(0, N_sr, _force_batch):
                _fb1 = min(_fb0 + _force_batch, N_sr)
                _R_b = R_sr[_fb0:_fb1]
                _cen_b = cen_sr[_fb0:_fb1]
                _pos_lig_b = (
                    cp.einsum("nij,kj->nki", _R_b, self._mol2_pos0) + _cen_b[:, None, :]
                )
                _f_b, _t_b, _ = self.engine(
                    _pos_lig_b, self._charges, R_matrices=_R_b, centroids=_cen_b
                )
                forces_gpu[_fb0:_fb1] = _f_b
                torques_gpu[_fb0:_fb1] = _t_b
                del _pos_lig_b
            # forces_gpu: (N_sr, 3)
            # Save old pair distances for Brownian bridge
            # After the BD step, we will check if the continuous Brownian path
            # crossed the reaction surface even if both endpoints are above it.
            # Exact formula: P(cross) = exp(-x₀×x₁/(D×dt))
            # where x = pair_distance - cutoff (distance above reaction surface)
            _old_pair_dists = None
            if self._rec_gho_pos is not None:
                # Compute GHO positions only (small: N_sr × n_gho × 3)
                _gho_mol2 = self._mol2_pos0[self._lig_gho_indices]
                _lig_gho_old = (
                    cp.einsum("nij,kj->nki", R_sr, _gho_mol2) + cen_sr[:, None, :]
                )
                _rec_gho_bcast = self._rec_gho_pos[None, :, :]
                _old_pair_dists = cp.linalg.norm(
                    _lig_gho_old - _rec_gho_bcast, axis=2
                )  # (N_sr, n_pairs)
            # Adaptive time step             #
            # 1. pair_dt:
            #      dt_pair = (frac²/2) × r² / D,  frac = 0.1
            #    This is the max safe dt at separation r. Ensures mean
            #    displacement < 10% of separation per step. At r=172:
            #    dt_pair = 0.005 × 172² / 0.0187 = 7912 ps -> σ=17 Å.
            #    At r=10: dt_pair = 0.005 × 100 / 0.434 = 1.15 ps.
            # 2. dt_force:
            #      dt_force = alpha / |D × Fr1|,  alpha = 0.01
            #    Controls force accuracy. Only active near Yukawa region.
            # 3. dt_edge :
            #      dt_edge = min(r-b, q-r)² / (18D)
            #    Prevents overshooting b-sphere and r_escape boundaries.
            # Final: dt = min(dt_pair, dt_force, dt_edge)
            r_sr = r_mag[sr_mask]
            in_outer = r_sr > (self._qb_factor * r_b)  # 1.1 * b
            _frac = 0.1
            dt_pair = (_frac * _frac / 2.0) * r_sr * r_sr / D_t
            # Force-change criterion (only where Yukawa force is significant)
            _force_alpha = 0.01
            if _adt_has_force:
                _expf = cp.exp(-r_sr / _adt_debye)
                _Fr0 = _adt_V0 * _expf * (1.0 / r_sr**2 + 1.0 / (r_sr * _adt_debye))
                _Fr1 = -_adt_V0 * _expf / _adt_debye**2 - 2.0 * _Fr0 / r_sr
                _DFr1 = cp.abs(D_t * _Fr1)
                _dt_force = cp.where(
                    (_DFr1 > 1e-15) & (r_sr < 3.0 * _adt_debye),
                    _force_alpha / _DFr1,
                    dt_pair,  # no significant force -> pair_dt is the limit
                )
            else:
                _dt_force = dt_pair  # no force at all -> pair_dt
            # Boundary proximity (outer zone only)
            dist_b = cp.maximum(r_sr - r_b, cp.full_like(r_sr, 1e-3))
            dist_esc = cp.maximum(r_esc - r_sr, cp.full_like(r_sr, 1e-3))
            dt_edge = cp.minimum(dist_b, dist_esc) ** 2 / (18.0 * D_t)
            # PySTARC uses Brownian bridge, which catches mid-step
            # crossings without shrinking dt. This avoids the trapping problem.
            # Combine:
            #   outer zone (r > 1.1*b): min(dt_force, dt_edge, dt_pair)
            #   inner zone (r <= 1.1*b): min(dt_force, dt_pair)
            dt_outer = cp.minimum(cp.minimum(_dt_force, dt_edge), dt_pair)
            dt_inner = cp.minimum(_dt_force, dt_pair)
            dt_arr = cp.where(in_outer, dt_outer, dt_inner)
            # Cap adaptive dt to prevent drift >> noise at large separations
            _max_dt = float(getattr(self.params, "max_dt", 0))
            if _max_dt > 0:
                dt_arr = cp.minimum(dt_arr, cp.full_like(dt_arr, _max_dt))
            # the reference time_step_tolerances: minimum_core_dt floors the timestep
            if _min_core_dt > 0:
                dt_arr = cp.maximum(dt_arr, cp.full_like(dt_arr, _min_core_dt))
            # Diffusion coefficient for BD step
            #   D_parallel(r) = D_factor/pi6 × (ainv + Di(r))
            # where Di captures the Oseen hydrodynamic coupling.
            # Use D_parallel isotropically (same for all 3 noise components).
            # not anisotropic radial/tangential split - that belongs in the
            # outer propagator loop where PySTARC employs with return_prob.
            if self._use_hi:
                _r_sr = r_mag[sr_mask]
                _rm1 = 1.0 / cp.maximum(_r_sr, cp.full_like(_r_sr, 0.01))
                _rm3 = _rm1**3
                _Di = (
                    -2.0 * _hi_Df * (2.0 * _rm1 - (4.0 / 3.0) * _hi_a2 * _rm3) / _hi_pi8
                )
                D_r_arr = _hi_Df * _hi_ainv + _Di
                D_r_arr = cp.maximum(D_r_arr, cp.full_like(D_r_arr, 1e-6))
            else:
                D_r_arr = cp.full(int(sr_mask.sum()), D_t, dtype=cp.float64)
            # Translational update (Ermak-McCammon)
            noise_t = cp.asarray(
                rng.standard_normal((int(sr_mask.sum()), 3)), dtype=cp.float64
            )
            sigma_t = cp.sqrt(2.0 * D_r_arr * dt_arr)[:, None]
            drift = D_r_arr[:, None] * dt_arr[:, None] * forces_gpu
            # Save old positions for overlap rejection
            _old_pos_sr = pos[sr_idx].copy() if _use_overlap else None
            _old_q_sr = q[sr_idx].copy() if _use_overlap else None
            pos[sr_idx] += drift + sigma_t * noise_t
            # Rotational update
            noise_r = cp.asarray(
                rng.standard_normal((int(sr_mask.sum()), 3)), dtype=cp.float64
            )
            sigma_r = cp.sqrt(2.0 * D_r * dt_arr)[:, None]
            omega = D_r * dt_arr[:, None] * torques_gpu + sigma_r * noise_r
            q[sr_idx] = self._apply_rotation_gpu(q[sr_idx], omega)
            # Overlap check (elastic wall)
            # If ligand centroid penetrated the receptor volume, reject step.
            # Check: min distance from ligand centroid to any receptor atom.
            # Chunked to limit GPU memory: (chunk, N_rec, 3) per batch.
            if _use_overlap and _rec_pos_overlap is not None:
                new_pos_sr = pos[sr_idx]  # (N_sr, 3)
                N_sr_ov = new_pos_sr.shape[0]
                N_rec_ov = _rec_pos_overlap.shape[0]
                # Chunk size: limit to ~500 MB
                _ov_chunk = max(1, int(500 * 1024 * 1024 / (N_rec_ov * 3 * 8)))
                _ov_chunk = min(_ov_chunk, N_sr_ov)
                _inside_all = cp.zeros(N_sr_ov, dtype=cp.bool_)
                for _ov_c0 in range(0, N_sr_ov, _ov_chunk):
                    _ov_c1 = min(_ov_c0 + _ov_chunk, N_sr_ov)
                    _diff = (
                        new_pos_sr[_ov_c0:_ov_c1, None, :]
                        - _rec_pos_overlap[None, :, :]
                    )
                    _dists = cp.linalg.norm(_diff, axis=2)
                    _min_dists = _dists.min(axis=1)
                    _inside_all[_ov_c0:_ov_c1] = _min_dists < _overlap_threshold
                    del _diff, _dists, _min_dists
                if cp.any(_inside_all):
                    _inside_idx = cp.where(_inside_all)[0]
                    pos[sr_idx[_inside_idx]] = _old_pos_sr[_inside_idx]
                    q[sr_idx[_inside_idx]] = _old_q_sr[_inside_idx]
                    _n_overlap_rejected += int(_inside_idx.shape[0])
            # Brownian bridge crossing check at reaction surface
            # To catch reactions that discrete endpoint checks miss, PySTARC approximates
            # this with the exact Brownian bridge crossing probability:
            #   P(path crossed boundary | start x₀, end x₁, both > 0)
            #       = exp(-x₀ × x₁ / (D × dt))
            # where x = pair_distance - cutoff (height above reaction surface).
            # Exact for free diffusion, accurate when drift << noise.
            # For each reaction pair, independently sample whether the continuous
            # path crossed the reaction cutoff. If n_crossed >= n_needed, react.
            if (
                _old_pair_dists is not None
                and len(sr_idx) > 0
                and self._rec_gho_pos is not None
            ):
                # Compute GHO positions after step (small: N_sr × n_gho × 3)
                R_new = self._quats_to_rotmats(q[sr_idx])
                _gho_mol2 = self._mol2_pos0[self._lig_gho_indices]
                _lig_gho_new = (
                    cp.einsum("nij,kj->nki", R_new, _gho_mol2) + pos[sr_idx][:, None, :]
                )
                _rec_gho_bcast = self._rec_gho_pos[None, :, :]
                _new_pair_dists = cp.linalg.norm(
                    _lig_gho_new - _rec_gho_bcast, axis=2
                )  # (N_sr, n_pairs)
                # x₀ = old distance above reaction cutoff
                # x₁ = new distance above reaction cutoff
                _cutoffs = self._rxn_cutoffs_gpu[None, :]  # (1, n_pairs)
                _x0 = _old_pair_dists - _cutoffs  # (N_sr, n_pairs)
                _x1 = _new_pair_dists - _cutoffs
                # Only apply to pairs where both endpoints are above cutoff
                # (if either is below, the endpoint check already caught it)
                _both_above = (_x0 > 0) & (_x1 > 0)
                if cp.any(_both_above):
                    # Crossing probability: exp(-x₀*x₁/(D_eff*dt))
                    # D_eff includes translational + rotational diffusion
                    # contribution to pair distance fluctuations.
                    if self._bb_D_eff is not None:
                        _Deff = self._bb_D_eff[None, :]  # (1, n_pairs)
                        _Ddt = _Deff * dt_arr[:, None]  # (N_sr, n_pairs)
                    else:
                        _Ddt = D_t * dt_arr[:, None]  # (N_sr, n_pairs)
                    _Ddt = cp.maximum(_Ddt, cp.full_like(_Ddt, 1e-30))
                    _p_cross = cp.exp(-_x0 * _x1 / _Ddt)
                    _p_cross = cp.where(_both_above, _p_cross, cp.zeros_like(_p_cross))
                    # Sample crossing for each pair
                    _u_bb = cp.asarray(rng.random(_p_cross.shape), dtype=cp.float64)
                    _crossed = _u_bb < _p_cross  # (N_sr, n_pairs) bool
                    # Also count pairs already below cutoff (from new positions)
                    _below = _new_pair_dists < _cutoffs
                    _total_fired = (_crossed | _below).sum(axis=1)  # (N_sr,)
                    # React if total fired pairs >= n_needed
                    _bb_reacted = _total_fired >= self._rxn_n_needed
                    # Only mark trajectories still running
                    if cp.any(_bb_reacted):
                        _bb_global = sr_idx[_bb_reacted]
                        _bb_new = _bb_global[status[_bb_global] == 0]
                        if len(_bb_new) > 0:
                            status[_bb_new] = 1
                            n_reacted += int(len(_bb_new))
                            # -- Record BB encounters --
                            _bb_np = cp.asnumpy(_bb_new)
                            _bb_triggered[_bb_np] = 1
                            _enc_traj.extend(_bb_np.tolist())
                            _enc_step.extend([step] * len(_bb_np))
                            _enc_pos.append(cp.asnumpy(pos[_bb_new]))
                            _enc_q.append(cp.asnumpy(q[_bb_new]))
                            _enc_npairs.extend([self._rxn_n_needed] * len(_bb_np))
            n_steps[sr_idx] += 1
            total_steps += int(sr_mask.sum())
            # Per-step data collection
            _cur_r = cp.asnumpy(cp.linalg.norm(pos[sr_idx], axis=1))
            _sr_np = cp.asnumpy(sr_idx)
            # Update min distance
            for j, ti in enumerate(_sr_np):
                if _cur_r[j] < _min_dist[ti]:
                    _min_dist[ti] = _cur_r[j]
                    _step_at_min[ti] = step
                    _nm_pos[ti] = cp.asnumpy(pos[sr_idx[j]])
                    _nm_q[ti] = cp.asnumpy(q[sr_idx[j]])
            # Accumulate time
            _dt_np = cp.asnumpy(dt_arr)
            for j, ti in enumerate(_sr_np):
                _total_time[ti] += _dt_np[j]
            # Radial density histogram
            np.add.at(
                _rad_counts,
                np.clip(
                    np.searchsorted(_rad_bins, _cur_r) - 1, 0, len(_rad_counts) - 1
                ),
                1,
            )
            # Angular occupancy (theta, phi of centroid direction)
            _cen_np = cp.asnumpy(pos[sr_idx])
            _r_safe = np.maximum(_cur_r, 1e-30)
            _theta = np.arccos(np.clip(_cen_np[:, 2] / _r_safe, -1, 1))
            _phi = np.arctan2(_cen_np[:, 1], _cen_np[:, 0]) + np.pi
            _ti = np.clip((_theta / np.pi * _n_theta).astype(int), 0, _n_theta - 1)
            _pi = np.clip((_phi / (2 * np.pi) * _n_phi).astype(int), 0, _n_phi - 1)
            np.add.at(_ang_counts, (_ti, _pi), 1)
            # Milestone flux
            _prev_r_np = cp.asnumpy(_prev_r[sr_idx])
            for mi in range(len(_ms_radii)):
                mr = _ms_radii[mi]
                _out = (_prev_r_np < mr) & (_cur_r >= mr)
                _in = (_prev_r_np >= mr) & (_cur_r < mr)
                _ms_flux_out[mi] += int(_out.sum())
                _ms_flux_in[mi] += int(_in.sum())
            # Transition matrix
            _old_bin = np.clip(
                np.searchsorted(_trans_bins, _prev_r_np) - 1, 0, _trans_n - 1
            )
            _new_bin = np.clip(
                np.searchsorted(_trans_bins, _cur_r) - 1, 0, _trans_n - 1
            )
            for j in range(len(_old_bin)):
                _trans_mat[_old_bin[j], _new_bin[j]] += 1
            # Contact frequency
            if self._rec_gho_pos is not None and _n_rxn_pairs > 0 and len(sr_idx) > 0:
                _gho_mol2_cf = self._mol2_pos0[self._lig_gho_indices]
                _lig_gho_cf = (
                    cp.einsum("nij,kj->nki", R_sr, _gho_mol2_cf) + cen_sr[:, None, :]
                )
                _lig_gho = cp.asnumpy(_lig_gho_cf)
                _rec_gho = cp.asnumpy(self._rec_gho_pos)
                for pi in range(_n_rxn_pairs):
                    _pdist = np.linalg.norm(
                        _lig_gho[:, pi, :] - _rec_gho[pi, :], axis=1
                    )
                    _contact_counts[pi] += int((_pdist < self._rxn_cutoffs[pi]).sum())
                _contact_total_steps += len(sr_idx)
            # Update previous r for milestone tracking
            _prev_r[sr_idx] = cp.asarray(_cur_r)
            # Record paths and energetics every save_interval steps
            if step % _save_interval == 0:
                if _output_cfg is None or getattr(_output_cfg, "full_paths", True):
                    _snap = np.column_stack(
                        [
                            _sr_np.astype(np.float64),
                            np.full(len(_sr_np), step, dtype=np.float64),
                            cp.asnumpy(pos[sr_idx]),
                            cp.asnumpy(q[sr_idx])[:, :3],  # q0, q1, q2 (q3 from norm)
                        ]
                    )  # (n_running, 8)
                    _path_data.append(_snap)

                if _output_cfg is None or getattr(_output_cfg, "energetics", True):
                    _esnap = np.column_stack(
                        [
                            _sr_np.astype(np.float64),
                            np.full(len(_sr_np), step, dtype=np.float64),
                            cp.asnumpy(forces_gpu),
                            _dt_np,
                        ]
                    )  # (n_running, 6)
                    _energy_data.append(_esnap)
            # Progress report
            done = int((status != 0).sum())
            # Early exit: two conditions to avoid stragglers dragging the run.
            # 1. Stop when >= 99.5% done (mark remaining as max_steps)
            # 2. Stall detection: if >95% done and no progress in 10k steps, stop
            #    (catches trajectories trapped in deep potential wells)
            if done >= int(N * 0.995):
                break
            if step > 0 and step % 10000 == 0:
                if done > int(N * 0.95) and done - _stall_done < max(2, int(N * 0.001)):
                    print(
                        f"  Stall detected: {N-done} trajectories trapped "
                        f"(+{done-_stall_done} in last 10k steps). Stopping."
                    )
                    break
                _stall_done = done
            # Live convergence report
            if done in _conv_milestones and done > _last_conv_done:
                _last_conv_done = done
                _n_comp = n_reacted + n_escaped
                if _n_comp > 0:
                    _p_live = n_reacted / _n_comp
                    _k_live = _conv_CONV * self._k_b * _p_live if self._k_b > 0 else 0
                    _pct = done / N * 100
                    _elapsed = time.time() - t0
                    print(
                        f"  -- {_pct:.0f}% converged --  "
                        f"P_rxn={_p_live:.5f}  "
                        f"k_on={_k_live:.3e}  "
                        f"({n_reacted:,} react / {n_escaped:,} esc)  "
                        f"[{_elapsed:.0f}s]"
                    )
            # Time-based progress (every 10s)
            _now = time.time()
            if _now - _last_progress_time >= _progress_interval_sec:
                _last_progress_time = _now
                _n_comp = n_reacted + n_escaped
                _pct_done = done / N * 100
                _elapsed = _now - t0
                if _n_comp > 0:
                    _p_live = n_reacted / _n_comp
                    _k_live = _conv_CONV * self._k_b * _p_live if self._k_b > 0 else 0
                    # Format k_on with error
                    if _k_live > 0:
                        _exp = int(math.floor(math.log10(_k_live)))
                        _man = _k_live / 10**_exp
                        print(
                            f"  [T] {_elapsed:6.0f}s  "
                            f"done={done:,}/{N:,} ({_pct_done:.1f}%)  "
                            f"reacted={n_reacted:,}  escaped={n_escaped:,}  "
                            f"P_rxn={_p_live:.5f}  "
                            f"k_on={_man:.2f}e{_exp}"
                        )
                    else:
                        print(
                            f"  [T] {_elapsed:6.0f}s  "
                            f"done={done:,}/{N:,} ({_pct_done:.1f}%)  "
                            f"reacted={n_reacted:,}  escaped={n_escaped:,}"
                        )
            # Checkpoint
            if (
                _ckpt_interval > 0
                and _ckpt_dir is not None
                and done - _last_ckpt_done >= _ckpt_interval
            ):
                _last_ckpt_done = done
                _ckpt_path = _ckpt_dir / "checkpoint.npz"
                np.savez_compressed(
                    str(_ckpt_path),
                    pos=cp.asnumpy(pos),
                    q=cp.asnumpy(q),
                    status=cp.asnumpy(status),
                    n_steps=cp.asnumpy(n_steps),
                    n_reacted=n_reacted,
                    n_escaped=n_escaped,
                    n_maxsteps=n_maxsteps,
                    total_steps=total_steps,
                    step=step,
                )
                print(f" Checkpoint saved ({done:,}/{N:,} done) -> {_ckpt_path}")
            # VERBOSE: Detailed diagnostics every 1000 steps
            if step % 1000 == 0:
                elapsed = time.time() - t0
                sps = total_steps / elapsed if elapsed > 0 else 0
                # Force statistics
                f_mag = float(cp.linalg.norm(forces_gpu, axis=1).mean())
                f_max = float(cp.linalg.norm(forces_gpu, axis=1).max())
                f_min = float(cp.linalg.norm(forces_gpu, axis=1).min())
                # Position statistics
                r_mean = float(r_sr.mean())
                r_min = float(r_sr.min())
                r_max = float(r_sr.max())
                # dt statistics
                dt_mean = float(dt_arr.mean())
                dt_min = float(dt_arr.min())
                dt_max_ = float(dt_arr.max())
                # Drift and noise magnitude
                drift_mag = float(cp.linalg.norm(drift, axis=1).mean())
                noise_mag = float((sigma_t[:, 0]).mean())
                # How many near b vs far from b
                n_near_b = int((~in_outer).sum())
                n_far = int(in_outer.sum())
                # Outer propagator events this step
                n_at_esc = int(at_escape.sum()) if "at_escape" in dir() else 0
                print(
                    f"  step {step:6d} | done={done:6d}/{N} | "
                    f"reacted={n_reacted} escaped={n_escaped} | "
                    f"{sps:,.0f} steps/sec"
                )
                print(
                    f"    r: mean={r_mean:.2f} min={r_min:.2f} max={r_max:.2f} Å  "
                    f"(b={r_b:.1f}, esc={r_esc:.1f})"
                )
                print(
                    f"    |F|: mean={f_mag:.6f} max={f_max:.6f} min={f_min:.6f} kBT/Å"
                )
                print(
                    f"    dt: mean={dt_mean:.4f} min={dt_min:.4f} max={dt_max_:.4f} ps"
                )
                print(
                    f"    drift: {drift_mag:.6f} Å  noise(sigma): {noise_mag:.4f} Å  "
                    f"drift/noise={drift_mag/(noise_mag+1e-30):.4f}"
                )
                print(
                    f"    zone: {n_near_b} inner (r<=b) + {n_far} outer (r>b)  "
                    f"| at_escape: {n_at_esc}"
                )
                # First 3 steps: dump individual trajectory details
                if step <= 2:
                    n_show = min(5, int(sr_mask.sum()))
                    print(f"    [TRACE] First {n_show} running trajectories:")
                    for ti in range(n_show):
                        _pos = cp.asnumpy(pos[sr_idx[ti]])
                        _r = float(cp.linalg.norm(pos[sr_idx[ti]]))
                        _f = cp.asnumpy(forces_gpu[ti])
                        _dt = float(dt_arr[ti])
                        _drft = cp.asnumpy(drift[ti])
                        print(
                            f"      traj[{int(sr_idx[ti])}]: pos=({_pos[0]:.4f},"
                            f"{_pos[1]:.4f},{_pos[2]:.4f}) r={_r:.4f}  "
                            f"F=({_f[0]:.6f},{_f[1]:.6f},{_f[2]:.6f})  "
                            f"|F|={np.linalg.norm(_f):.6f}  dt={_dt:.4f}  "
                            f"drift=({_drft[0]:.6f},{_drft[1]:.6f},{_drft[2]:.6f})"
                        )
        # Max steps for any still-running after loop ends
        n_maxsteps = int((status == 0).sum())
        # Also count those that never finished
        n_reacted = int((status == 1).sum())
        n_escaped = int((status == 2).sum())
        elapsed = time.time() - t0
        sps = total_steps / elapsed if elapsed > 0 else 0
        print(
            f"  BD complete: {n_reacted} reacted, {n_escaped} escaped, "
            f"{n_maxsteps} max-steps"
        )
        print(f"  Total steps: {total_steps:,}  ({sps:,.0f} steps/sec)")
        if _n_overlap_rejected > 0:
            print(f"  Overlap rejections: {_n_overlap_rejected:,}")
        # Final convergence report
        _n_comp_final = n_reacted + n_escaped
        if _n_comp_final > 0 and self._k_b > 0:
            _p_final = n_reacted / _n_comp_final
            _k_final = _conv_CONV * self._k_b * _p_final
            print(f"  Final: P_rxn={_p_final:.6f}  k_on={_k_final:.4e} M-1 s-1")
        # Delete checkpoint on successful completion
        if _ckpt_dir is not None:
            _ckpt_path = _ckpt_dir / "checkpoint.npz"
            if _ckpt_path.exists():
                _ckpt_path.unlink()
                print(f"  Checkpoint removed (run completed successfully)")
        # Package collected data
        _outcome = cp.asnumpy(status)
        _nsteps_np = cp.asnumpy(n_steps)
        # Near-miss: only escaped trajectories
        _esc_mask = _outcome == 2
        _nm_traj_arr = np.where(_esc_mask)[0]
        # Encounter arrays: stack lists
        if _enc_pos:
            _enc_pos_arr = np.vstack(_enc_pos)
            _enc_q_arr = np.vstack(_enc_q)
        else:
            _enc_pos_arr = np.zeros((0, 3))
            _enc_q_arr = np.zeros((0, 4))
        sim_data = {
            "outcome": _outcome,
            "n_steps": _nsteps_np,
            "start_pos": _start_pos,
            "start_q": _start_q,
            "min_dist": _min_dist,
            "step_at_min": _step_at_min,
            "total_time_ps": _total_time,
            "n_returns": _n_returns,
            "bb_triggered": _bb_triggered,
            "encounter_pos": _enc_pos_arr,
            "encounter_q": _enc_q_arr,
            "encounter_traj": np.array(_enc_traj, dtype=np.int64),
            "encounter_step": np.array(_enc_step, dtype=np.int64),
            "encounter_n_pairs": np.array(_enc_npairs, dtype=np.int64),
            "near_miss_pos": _nm_pos[_esc_mask],
            "near_miss_q": _nm_q[_esc_mask],
            "near_miss_traj": _nm_traj_arr,
            "near_miss_dist": _min_dist[_esc_mask],
            "path_steps": _path_data,
            "energy_steps": _energy_data,
            "radial_bins": _rad_bins,
            "radial_counts": _rad_counts,
            "angular_theta": np.linspace(0, np.pi, _n_theta),
            "angular_phi": np.linspace(0, 2 * np.pi, _n_phi),
            "angular_counts": _ang_counts,
            "milestone_radii": _ms_radii,
            "milestone_flux_out": _ms_flux_out,
            "milestone_flux_in": _ms_flux_in,
            "contact_pair_counts": _contact_counts,
            "contact_total_steps": _contact_total_steps,
            "trans_bins": _trans_bins,
            "trans_matrix": _trans_mat,
        }

        return GPUBatchResult(
            n_trajectories=N,
            n_reacted=n_reacted,
            n_escaped=n_escaped,
            n_max_steps=n_maxsteps,
            reaction_counts={"stage_0": n_reacted},
            r_start=self.params.r_start,
            r_escape=self.params.r_escape,
            dt=self.params.dt,
            elapsed_sec=elapsed,
            steps_per_sec=sps,
            sim_data=sim_data,
        )

    # GPU helpers
    def _random_quaternions_gpu(self, N: int, rng) -> "cp.ndarray":
        """Generate N random unit quaternions on GPU."""
        u = rng.random((N, 3)).astype(np.float64)
        q = np.stack(
            [
                np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1]),
                np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1]),
                np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2]),
                np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2]),
            ],
            axis=1,
        )
        return cp.asarray(q, dtype=cp.float64)

    def _quats_to_rotmats(self, q: "cp.ndarray") -> "cp.ndarray":
        """Convert (N,4) quaternions to (N,3,3) rotation matrices on GPU."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        N = q.shape[0]
        R = cp.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            axis=1,
        ).reshape(N, 3, 3)
        return R

    def _apply_rotation_gpu(self, q: "cp.ndarray", omega: "cp.ndarray") -> "cp.ndarray":
        """Apply rotation vector omega to quaternions q, return normalised."""
        angle = cp.linalg.norm(omega, axis=1, keepdims=True)  # (N,1)
        safe = angle > 1e-10
        axis = cp.where(safe, omega / cp.where(safe, angle, 1.0), 0.0)
        half = angle * 0.5
        dq = cp.stack(
            [
                cp.cos(half[:, 0]),
                axis[:, 0] * cp.sin(half[:, 0]),
                axis[:, 1] * cp.sin(half[:, 0]),
                axis[:, 2] * cp.sin(half[:, 0]),
            ],
            axis=1,
        )  # (N,4)
        # Quaternion multiplication q x dq
        w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        w2, x2, y2, z2 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
        qnew = cp.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            axis=1,
        )
        # Normalise
        qnew /= cp.linalg.norm(qnew, axis=1, keepdims=True)
        return qnew

    def _check_reactions_gpu(self, pos_lig: "cp.ndarray") -> "cp.ndarray":
        """
        Check reaction criteria for all running trajectories.
        pos_lig: (N_run, N_lig, 3)
        Returns (N_run,) bool - True if reacted.
        """
        N_run = pos_lig.shape[0]
        n_pairs = len(self._rec_gho_indices)
        if n_pairs == 0:
            return cp.zeros(N_run, dtype=cp.bool_)
        # Ligand GHO positions for all trajectories
        lig_gho = pos_lig[:, self._lig_gho_indices, :]  # (N_run, n_pairs, 3)
        # Receptor GHO positions (fixed, broadcast)
        rec_gho = self._rec_gho_pos[None, :, :]  # (1, n_pairs, 3)
        # Distances
        dists = cp.linalg.norm(lig_gho - rec_gho, axis=2)  # (N_run, n_pairs)
        satisfied = dists < self._rxn_cutoffs_gpu[None, :]  # (N_run, n_pairs) bool
        n_satisfied = satisfied.sum(axis=1)  # (N_run,) int
        reacted = n_satisfied >= self._rxn_n_needed  # (N_run,) bool
        return reacted

    def _check_reactions_gpu_gho(self, gho_pos: "cp.ndarray") -> "cp.ndarray":
        """
        Check reaction criteria using pre-computed GHO positions.
        gho_pos: (N_run, n_pairs, 3) - already rotated+translated GHO atoms.
        Returns (N_run,) bool - True if reacted.
        """
        N_run = gho_pos.shape[0]
        n_pairs = len(self._rec_gho_indices)
        if n_pairs == 0:
            return cp.zeros(N_run, dtype=cp.bool_)
        rec_gho = self._rec_gho_pos[None, :, :]
        dists = cp.linalg.norm(gho_pos - rec_gho, axis=2)
        satisfied = dists < self._rxn_cutoffs_gpu[None, :]
        n_satisfied = satisfied.sum(axis=1)
        return n_satisfied >= self._rxn_n_needed
