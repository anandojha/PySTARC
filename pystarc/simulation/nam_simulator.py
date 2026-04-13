"""
PySTARC NAM simulator
===================

Northrup-Allison-McCammon Brownian dynamics

Physics
-------
- Ermak-McCammon translational + rotational integrator
- Adaptive time step: 0.2 ps normal, 0.05 ps near reaction boundary
- Ghost atom (GHO) reaction criteria with n_needed AND logic
- NAM k_on formula (Northrup, Allison, McCammon 1984)
- b-sphere setup and escape sphere termination

Parallelism
-----------
- n_threads=1 : serial (default, reproducible)
- n_threads>1 : Python multiprocessing pool
- Each worker seeded as seed + trajectory_index (fully reproducible)

"""

from __future__ import annotations
from pystarc.motion.do_bd_step import (
    bd_step,
    bd_step_wiener,
    bd_step_adaptive,
    backstep_due_to_force,
    escape_radius,
)
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.motion.adaptive_time_step import AdaptiveTimeStep
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet
from typing import Callable, Dict, List, Optional, Tuple
from pystarc.structures.molecules import Molecule, Atom
from dataclasses import dataclass
import multiprocessing as mp
import scipy.stats as _stats
import numpy as np
import math
import copy


def _check_hard_sphere_overlap(mol1: Molecule, mol2: Molecule) -> bool:
    """
    Check if any atoms from mol1 and mol2 overlap (hard-sphere exclusion).
    Returns True if any pair of atoms from different molecules have their
    centres closer than the sum of their radii.
    If overlap detected after a BD step, the step is rejected
    and a new random displacement is drawn.
    Ghost atoms (radius=0) never overlap - they are zero-radius reference
    points and are correctly excluded by the radius sum check.
    """
    for a1 in mol1.atoms:
        if a1.radius < 1e-10:
            continue  # ghost atom - no hard-sphere interaction
        p1 = np.array([a1.x, a1.y, a1.z])
        r1 = a1.radius
        for a2 in mol2.atoms:
            if a2.radius < 1e-10:
                continue
            p2 = np.array([a2.x, a2.y, a2.z])
            d = float(np.linalg.norm(p1 - p2))
            if d < r1 + a2.radius:
                return True  # overlap detected
    return False


ForceFunction = Callable[[Molecule, Molecule], Tuple[np.ndarray, np.ndarray, float]]


def zero_force(mol1: Molecule, mol2: Molecule):
    return np.zeros(3), np.zeros(3), 0.0


# Parameters
@dataclass
class NAMParameters:
    """
    BD simulation parameters.
    dt     = minimum_core_dt            (0.2 ps)
    dt_rxn = minimum_core_reaction_dt  (0.05 ps)
    """

    n_trajectories: int = 1_000
    dt: float = 0.2  # ps  normal minimum time step
    dt_rxn: float = 0.05  # ps  near-reaction minimum time step
    max_steps: int = 1_000_000  # max_n_steps
    r_start: float = 100.0  # Å  b-sphere radius
    r_escape: float = 0.0  # Å  0 = auto (2 × r_start)
    seed: Optional[int] = None
    n_threads: int = 1
    use_hard_sphere: bool = True  # reject steps with atom overlap (default)
    hydrodynamic_interactions: bool = False  # hydrodynamic interactions (Rotne-Prager)
    verbose: bool = False

    def __post_init__(self):
        if self.r_escape == 0.0:
            # the outer propagator triggers at qb_factor=1.1 × b_sphere,
            # not at 2×b. Trajectories only need to diffuse 10% above b
            # before return_prob is applied - critical for large b-spheres.
            self.r_escape = self.r_start * 1.1


# Per-trajectory worker (top-level for multiprocessing)
def _run_trajectory_worker(args):
    """
    Run one BD trajectory. Top-level function required by multiprocessing.
    Each worker gets its own RNG: seed + trajectory_index.
    """
    (
        mol1,
        mol2,
        mol2_pos0,
        mob,
        pathway_set,
        params,
        force_fn,
        rxn_cutoffs,
        traj_idx,
    ) = args
    rng = np.random.default_rng((params.seed or 0) + traj_idx)
    # Private scratch molecule for this worker
    mol2_scratch = copy.copy(mol2)
    mol2_scratch.atoms = [copy.copy(a) for a in mol2.atoms]
    D_r = mob.relative_rotational_diffusion()
    # Random start on b-sphere
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    pos = v * params.r_start
    ori = random_quaternion(rng)
    for step in range(params.max_steps):
        # Place mol2 (vectorised: rotate pre-centred positions + translate)
        R = ori.to_rotation_matrix()
        new_pos = (R @ mol2_pos0.T).T + pos
        for atom, p in zip(mol2_scratch.atoms, new_pos):
            atom.x = float(p[0])
            atom.y = float(p[1])
            atom.z = float(p[2])
        # Reaction check (GHO AND logic built into PathwaySet)
        rxn = pathway_set.check_all(mol1, mol2_scratch, rng)
        if rxn is not None:
            return TrajectoryResult(
                Fate.REACTED, step, step * params.dt, float(np.linalg.norm(pos)), rxn
            )
        # Escape check
        r = float(np.linalg.norm(pos))
        if r >= params.r_escape:
            return TrajectoryResult(Fate.ESCAPED, step, step * params.dt, r)
        # Forces via engine
        force, torque, _ = force_fn(mol1, mol2_scratch)
        # RPY: position-dependent D_rel
        D_t = mob.relative_translational_diffusion(pos)
        # Choose dt (adaptive two-value scheme)
        rxn_min = min(rxn_cutoffs) if rxn_cutoffs else 5.0
        dt = params.dt_rxn if r < 1.5 * rxn_min else params.dt
        # Draw Wiener increments
        dW_t = math.sqrt(dt) * rng.standard_normal(3)
        dW_r = math.sqrt(dt) * rng.standard_normal(3)
        pos_old, ori_old, force_old = pos, ori, force.copy()
        # BD step with pre-drawn Wiener increments
        pos, ori = bd_step_wiener(pos, ori, force, torque, D_t, D_r, dt, dW_t, dW_r)
        # Force-change backstep check
        # If forces changed too much, subdivide the Wiener step
        if params.use_hard_sphere or True:
            # get forces at new position to check
            R_trial = ori.to_rotation_matrix()
            trial_pos_arr = (R_trial @ mol2_pos0.T).T + pos
            for atom, p in zip(mol2_scratch.atoms, trial_pos_arr):
                atom.x = float(p[0])
                atom.y = float(p[1])
                atom.z = float(p[2])
            force_new, torque_new, _ = force_fn(mol1, mol2_scratch)
            if backstep_due_to_force(
                force_new,
                force_old,
                pos,
                pos_old,
                dt,
                params.dt_rxn,
                radius=mob.radius2,
            ):
                # Wiener subdivision: split dW at midpoint (unbiased)
                s = math.sqrt(dt / 4.0)  # standard half-step noise
                dW_mid_t = 0.5 * dW_t + s * rng.standard_normal(3)
                dW_mid_r = 0.5 * dW_r + s * rng.standard_normal(3)
                hdt = dt / 2.0
                # First half-step
                pos, ori = bd_step_wiener(
                    pos_old,
                    ori_old,
                    force_old,
                    torque,
                    D_t,
                    D_r,
                    hdt,
                    dW_mid_t,
                    dW_mid_r,
                )
                # Second half-step (using new forces)
                R2 = ori.to_rotation_matrix()
                p2 = (R2 @ mol2_pos0.T).T + pos
                for atom, p in zip(mol2_scratch.atoms, p2):
                    atom.x = float(p[0])
                    atom.y = float(p[1])
                    atom.z = float(p[2])
                f2, t2, _ = force_fn(mol1, mol2_scratch)
                dW_2nd_t = dW_t - dW_mid_t
                dW_2nd_r = dW_r - dW_mid_r
                D_t2 = mob.relative_translational_diffusion(pos)
                pos, ori = bd_step_wiener(
                    pos, ori, f2, t2, D_t2, D_r, hdt, dW_2nd_t, dW_2nd_r
                )
        # Hard-sphere collision rejection
        if params.use_hard_sphere:
            R_hs = ori.to_rotation_matrix()
            hs_pos = (R_hs @ mol2_pos0.T).T + pos
            for atom, p in zip(mol2_scratch.atoms, hs_pos):
                atom.x = float(p[0])
                atom.y = float(p[1])
                atom.z = float(p[2])
            if _check_hard_sphere_overlap(mol1, mol2_scratch):
                # redraw from same starting point
                dW_t2 = math.sqrt(dt) * rng.standard_normal(3)
                dW_r2 = math.sqrt(dt) * rng.standard_normal(3)
                D_t_old = mob.relative_translational_diffusion(pos_old)
                pos, ori = bd_step_wiener(
                    pos_old, ori_old, force_old, torque, D_t_old, D_r, dt, dW_t2, dW_r2
                )
    return TrajectoryResult(
        Fate.MAX_STEPS,
        params.max_steps,
        params.max_steps * params.dt,
        float(np.linalg.norm(pos)),
    )


# NAM simulator
class NAMSimulator:
    """
    Northrup-Allison-McCammon Brownian dynamics simulator.
    Molecule 1 is fixed at the origin.
    Molecule 2 diffuses from a random point on the b-sphere.
    """

    def __init__(
        self,
        mol1: Molecule,
        mol2: Molecule,
        mobility: MobilityTensor,
        pathway_set: PathwaySet,
        params: NAMParameters,
        force_fn: Optional[ForceFunction] = None,
    ):
        self.mol1 = mol1
        self.mol2 = mol2
        self.mobility = mobility
        self.pathway_set = pathway_set
        self.params = params
        self.force_fn = force_fn or zero_force
        self.rng = np.random.default_rng(params.seed)
        # Pre-centre mol2 positions for fast placement (avoid copies per step)
        c0 = mol2.centroid()
        self._mol2_pos0 = mol2.positions_array() - c0
        self._mol2_scratch = copy.copy(mol2)
        self._mol2_scratch.atoms = [copy.copy(a) for a in mol2.atoms]
        # Reaction cutoff distances - used by adaptive dt
        self._rxn_cutoffs = [
            pair.distance_cutoff
            for rxn in pathway_set.reactions
            for pair in rxn.criteria.pairs
        ]
        # Accumulators
        self.results: List[TrajectoryResult] = []
        self.n_reacted = 0
        self.n_escaped = 0
        self.reaction_counts: Dict[str, int] = {}
        # Geometry-based adaptive time step
        self._dt_ctrl = AdaptiveTimeStep()
        # Outer propagator (LMZ) - set up if mobility info is available
        self._outer_prop = None
        try:
            from pystarc.simulation.outer_propagator import OuterPropagator, OPGroupInfo

            g0 = OPGroupInfo(
                q=mol1.total_charge(),
                Dtrans=mobility.D_trans1,
                Drot=mobility.D_rot1,
            )
            g1 = OPGroupInfo(
                q=mol2.total_charge(),
                Dtrans=mobility.D_trans2,
                Drot=mobility.D_rot2,
            )
            # physical constants in PySTARC units (A, ps, kBT)
            kT = 0.5961  # kBT in kcal/mol at 298.15 K
            viscosity = 1.002e-3 * 1e-4 / 1e-12  # Pa.s -> kcal.ps/A^3 (20°C water)
            dielectric = 78.54
            vacuum_perm = 1.0 / (4 * math.pi * 332.0636)  # e^2/(kcal.A)
            debye_len = 8.0  # A, physiological ionic strength
            max_mol_r = max(mol1.bounding_radius(), mol2.bounding_radius())
            self._outer_prop = OuterPropagator(
                b_radius=params.r_start,
                max_radius=max_mol_r,
                has_hi=True,
                kT=kT,
                viscosity=viscosity,
                dielectric=dielectric,
                vacuum_perm=vacuum_perm,
                debye_len=debye_len,
                g0=g0,
                g1=g1,
            )
        except Exception:
            # If outer propagator fails to set up, fall back to simple escape
            self._outer_prop = None

    def _place_mol2(self, pos: np.ndarray, ori: Quaternion) -> Molecule:
        R = ori.to_rotation_matrix()
        new_pos = (R @ self._mol2_pos0.T).T + pos
        mol = self._mol2_scratch
        for atom, p in zip(mol.atoms, new_pos):
            atom.x = float(p[0])
            atom.y = float(p[1])
            atom.z = float(p[2])
        return mol

    def run_one(self) -> TrajectoryResult:
        """Run a single trajectory - used by serial path."""
        v = self.rng.standard_normal(3)
        v /= np.linalg.norm(v)
        pos = v * self.params.r_start
        ori = random_quaternion(self.rng)
        D_r = self.mobility.relative_rotational_diffusion()
        r_h1 = self.mobility.radius1
        r_h2 = self.mobility.radius2
        # Reset adaptive dt controller for this trajectory
        self._dt_ctrl.reset()
        for step in range(self.params.max_steps):
            mol2 = self._place_mol2(pos, ori)
            rxn = self.pathway_set.check_all(self.mol1, mol2, self.rng)
            if rxn is not None:
                return TrajectoryResult(
                    Fate.REACTED,
                    step,
                    step * self.params.dt,
                    float(np.linalg.norm(pos)),
                    rxn,
                )
            r = float(np.linalg.norm(pos))
            # Outer propagator (LMZ): triggers at r > qb_factor * b_sphere
            # qb_factor = 1.1 (from motion/qb_factor.hh)
            QB_FACTOR = 1.1
            if self._outer_prop is not None and r >= QB_FACTOR * self.params.r_start:
                reached_b, pos, ori_arr = self._outer_prop.new_state(
                    pos, ori.to_array(), self.rng
                )
                ori = Quaternion(
                    w=float(ori_arr[0]),
                    x=float(ori_arr[1]),
                    y=float(ori_arr[2]),
                    z=float(ori_arr[3]),
                )
                if not reached_b:
                    return TrajectoryResult(
                        Fate.ESCAPED, step, step * self.params.dt, r
                    )
                # returned to b-sphere - continue BD
                continue
            # Fallback simple escape check (when no outer propagator)
            if self._outer_prop is None and r >= self.params.r_escape:
                return TrajectoryResult(Fate.ESCAPED, step, step * self.params.dt, r)
            force, torque, _ = self.force_fn(self.mol1, mol2)
            # RPY: position-dependent D_rel (hydrodynamic_interactions=true)
            D_t = self.mobility.relative_translational_diffusion(pos)
            r = float(np.linalg.norm(pos))
            # Geometry-based adaptive dt (step_variable_dt)
            dt = self._dt_ctrl.get_dt(
                r,
                D_t,
                D_r,
                r_h1,
                r_h2,
                self._rxn_cutoffs,
                dt_min=self.params.dt_rxn,
                dt_rxn_min=self.params.dt_rxn / 4.0,
            )
            # Hard-sphere rejection: save old state before stepping
            pos_old, ori_old = pos, ori
            # Draw Wiener increments and step
            dW_t = math.sqrt(dt) * self.rng.standard_normal(3)
            dW_r = math.sqrt(dt) * self.rng.standard_normal(3)
            pos, ori = bd_step_wiener(pos, ori, force, torque, D_t, D_r, dt, dW_t, dW_r)
            # Force-change backstep with Wiener subdivision
            mol2_new = self._place_mol2(pos, ori)
            force_new, _, _ = self.force_fn(self.mol1, mol2_new)
            if backstep_due_to_force(
                force_new,
                force,
                pos,
                pos_old,
                dt,
                self.params.dt_rxn,
                radius=self.mobility.radius2,
            ):
                hdt = dt / 2.0
                s = math.sqrt(hdt / 2.0)
                dW_mid_t = 0.5 * dW_t + s * self.rng.standard_normal(3)
                dW_mid_r = 0.5 * dW_r + s * self.rng.standard_normal(3)
                pos, ori = bd_step_wiener(
                    pos_old, ori_old, force, torque, D_t, D_r, hdt, dW_mid_t, dW_mid_r
                )
                mol2_mid = self._place_mol2(pos, ori)
                f2, t2, _ = self.force_fn(self.mol1, mol2_mid)
                D_t2 = self.mobility.relative_translational_diffusion(pos)
                pos, ori = bd_step_wiener(
                    pos, ori, f2, t2, D_t2, D_r, hdt, dW_t - dW_mid_t, dW_r - dW_mid_r
                )
                self._dt_ctrl._last_dt = hdt  # record actual dt used
            # Hard-sphere collision rejection
            if self.params.use_hard_sphere:
                mol2_trial = self._place_mol2(pos, ori)
                if _check_hard_sphere_overlap(self.mol1, mol2_trial):
                    dW_t2 = math.sqrt(dt) * self.rng.standard_normal(3)
                    dW_r2 = math.sqrt(dt) * self.rng.standard_normal(3)
                    D_t_old = self.mobility.relative_translational_diffusion(pos_old)
                    pos, ori = bd_step_wiener(
                        pos_old, ori_old, force, torque, D_t_old, D_r, dt, dW_t2, dW_r2
                    )
        return TrajectoryResult(
            Fate.MAX_STEPS,
            self.params.max_steps,
            self.params.max_steps * self.params.dt,
            float(np.linalg.norm(pos)),
        )

    def run(self) -> "SimulationResult":
        """Run all trajectories, serial or parallel."""
        self.results.clear()
        self.reaction_counts.clear()
        self.n_reacted = 0
        self.n_escaped = 0
        n = self.params.n_trajectories
        if self.params.n_threads > 1 and n > 1:
            self._run_parallel(n)
        else:
            self._run_serial(n)
        return SimulationResult.from_simulator(self)

    def _run_serial(self, n: int):
        for i in range(n):
            if self.params.verbose and i % 1 == 0:
                print(
                    f"  Trajectory {i+1}/{n}  "
                    f"(reacted={self.n_reacted}, escaped={self.n_escaped})"
                )
            self._record(self.run_one())

    def _run_parallel(self, n: int):
        n_workers = min(self.params.n_threads, n, mp.cpu_count())
        if self.params.verbose:
            print(f"  Parallel: {n_workers} workers, {n} trajectories")
        args = [
            (
                self.mol1,
                self.mol2,
                self._mol2_pos0,
                self.mobility,
                self.pathway_set,
                self.params,
                self.force_fn,
                self._rxn_cutoffs,
                i,
            )
            for i in range(n)
        ]
        with mp.Pool(n_workers) as pool:
            for result in pool.map(_run_trajectory_worker, args):
                self._record(result)

    def _record(self, result: TrajectoryResult):
        self.results.append(result)
        if result.reacted:
            self.n_reacted += 1
            name = result.reaction_name or "unnamed"
            self.reaction_counts[name] = self.reaction_counts.get(name, 0) + 1
        elif result.escaped:
            self.n_escaped += 1


# Simulation result
@dataclass
class SimulationResult:
    """Aggregated NAM BD results with k_on calculation."""

    n_trajectories: int
    n_reacted: int
    n_escaped: int
    n_max_steps: int
    reaction_counts: Dict[str, int]
    r_start: float
    r_escape: float
    dt: float
    k_db: float = 0.0  # LMZ rate from outer_propagator.relative_rate(b)

    @classmethod
    def from_simulator(cls, sim: NAMSimulator) -> "SimulationResult":
        n_max = sum(1 for r in sim.results if r.fate == Fate.MAX_STEPS)
        # Get LMZ k_db from outer propagator if available
        k_db = 0.0
        if sim._outer_prop is not None:
            try:
                k_db = sim._outer_prop._relative_rate(sim.params.r_start)
            except Exception:
                k_db = 0.0
        return cls(
            n_trajectories=sim.params.n_trajectories,
            n_reacted=sim.n_reacted,
            n_escaped=sim.n_escaped,
            n_max_steps=n_max,
            reaction_counts=dict(sim.reaction_counts),
            r_start=sim.params.r_start,
            r_escape=sim.params.r_escape,
            dt=sim.params.dt,
            k_db=k_db,
        )

    @property
    def reaction_probability(self) -> float:
        n = self.n_reacted + self.n_escaped
        return self.n_reacted / n if n > 0 else 0.0

    def rate_constant(self, D_rel: float, k_db: float = 0.0) -> float:
        """
        NAM k_on (Northrup, Allison, McCammon 1984).
        The LMZ formulation gives k_db = relative_rate(b_sphere)
        from the outer propagator (which includes electrostatic effects):
            k_on = conv_factor * k_db * P
        where conv_factor = 6.02e8 converts to M^-1 s^-1.
        If k_db is not provided (k_db=0), falls back to Smoluchowski:
            k_D = 4*pi*D_rel*r_b*N_A
            k_on = k_D * P / (1 - P*(1 - r_b/r_esc))
        Parameters
        ----------
        D_rel : relative translational diffusion (Å²/ps)
        k_db  : LMZ rate from outer_propagator.relative_rate(b_sphere)
                in the internal units. If 0.0, uses Smoluchowski.
        """
        P = self.reaction_probability
        if P == 0.0:
            return 0.0

        if k_db > 0.0:
            # k_on = conv_factor * k_db * P
            # conv_factor = 6.02e8 L/(mol) converts A^3/ps to M^-1 s^-1
            # k_db is in A^3/ps units from relative_rate()
            # Units: [A^3/ps] * [6.02e23/L] * [1e-27 L/A^3] * [1e12 ps/s] = M^-1 s^-1
            CONV = 6.022e23 * 1e-27 * 1e12  # A^3/ps -> M^-1 s^-1
            return CONV * k_db * P
        else:
            # Smoluchowski approximation (no outer propagator)
            # k_D = 4π·D·b  in Å³/ps, then convert to M⁻¹s⁻¹
            # CONV = N_A [/mol] * 1e-30 [m³/Å³] / 1e-12 [s/ps] / 1e-3 [m³/L]
            #      = 6.022e23 * 1e-30 / 1e-12 / 1e-3 = 6.022e8  (Å³/ps -> M⁻¹s⁻¹)
            CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3  # = 6.022e8
            k_D = 4.0 * math.pi * D_rel * self.r_start  # Å³/ps
            beta = self.r_start / self.r_escape
            denom = 1.0 - P * (1.0 - beta)
            return CONV_A3ps * k_D * P / denom

    def __repr__(self):
        return (
            f"SimulationResult(N={self.n_trajectories}, "
            f"reacted={self.n_reacted}, escaped={self.n_escaped}, "
            f"P_rxn={self.reaction_probability:.4f})"
        )


# Confidence interval helpers (added to SimulationResult via monkey-patch)
# We extend the class after definition to avoid dataclass issues
def _n_completed(self) -> int:
    return self.n_reacted + self.n_escaped


def _reaction_probability_ci(self, confidence: float = 0.95):
    """
    Wilson score 95% CI on P_rxn -
    Wilson (1927): valid even for very small P_rxn.
    """
    n = self.n_reacted + self.n_escaped
    if n == 0:
        return (0.0, 1.0)
    z = float(_stats.norm.ppf(0.5 + confidence / 2.0))
    p = self.reaction_probability
    z2n = z**2 / n
    denom = 1.0 + z2n
    ctr = (p + z2n / 2.0) / denom
    mar = z * math.sqrt(max(p * (1 - p) / n + z2n / 4.0, 0.0)) / denom
    return (max(0.0, ctr - mar), min(1.0, ctr + mar))


def _k_from_P(self, P: float, D_rel: float) -> float:
    if P <= 0.0:
        return 0.0
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3  # Å³/ps -> M⁻¹s⁻¹ = 6.022e8
    if self.k_db > 0.0:
        return CONV_A3ps * self.k_db * P
    k_D = 4.0 * math.pi * D_rel * self.r_start  # Å³/ps
    beta = self.r_start / self.r_escape
    return CONV_A3ps * k_D * P / (1.0 - P * (1.0 - beta))


def _rate_constant_ci(self, D_rel: float, confidence: float = 0.95):
    """95% CI on k_on propagated from Wilson CI on P_rxn."""
    p_lo, p_hi = self.reaction_probability_ci(confidence)
    return (_k_from_P(self, p_lo, D_rel), _k_from_P(self, p_hi, D_rel))


def _result_summary(
    self, D_rel: float, k_b: float = 0.0, confidence: float = 0.95
) -> str:
    k = self.rate_constant(D_rel)
    k_lo, k_hi = self.rate_constant_ci(D_rel, confidence)
    p = self.reaction_probability
    p_lo, p_hi = self.reaction_probability_ci(confidence)
    pct = int(confidence * 100)
    n_comp = self.n_reacted + self.n_escaped
    lines = [
        f"  Trajectories : {self.n_trajectories:,}",
        f"  Completed    : {n_comp:,}  ({self.n_reacted:,} reacted + {self.n_escaped:,} escaped)",
    ]
    if self.n_max_steps:
        lines.append(f"  Max-steps    : {self.n_max_steps:,}  (excluded from P_rxn)")
    lines += [
        f"  P_rxn        : {p:.6f}  ({pct}% CI: [{p_lo:.6f}, {p_hi:.6f}])",
        f"  k_on         : {k:.4e} M-1 s-1",
        f"  {pct}% CI     : [{k_lo:.4e}, {k_hi:.4e}] M-1 s-1",
    ]
    if k > 0:
        lines.append(f"  log10(k_on)  : {math.log10(k):.3f}")
    return "\n".join(lines)


# Attach methods to SimulationResult
SimulationResult.reaction_probability_ci = _reaction_probability_ci
SimulationResult.rate_constant_ci = _rate_constant_ci
SimulationResult.summary = _result_summary
