"""
PySTARC NAM simulator.

This module implements Northrup-Allison-McCammon Brownian dynamics for
bimolecular association.

On the physics, the integrator uses the Ermak-McCammon scheme for both
translation and rotation. The time step is adaptive, taking 0.2 ps in the
normal regime and 0.05 ps near the reaction boundary. Reaction criteria are
defined through ghost atoms (GHO) and combined with the n_needed AND logic.
The association rate constant k_on is computed from the NAM formula of
Northrup, Allison, and McCammon (1984). Trajectories begin on the b-sphere
and terminate when they reach the escape sphere.

On parallelism, setting n_threads=1 runs the trajectories serially, which is
the reproducible default. Setting n_threads greater than 1 distributes the
trajectories across a Python multiprocessing pool. Each worker is seeded as
seed + trajectory_index so that results are fully reproducible regardless of
the number of threads.
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
from pystarc.simulation.chain_simulator import (
    compute_pair_distances,
    check_reaction_with_bridge,
)
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
    Check whether any atom of mol1 overlaps any atom of mol2 under hard-sphere
    exclusion. The function returns True when any pair of atoms from different
    molecules have their centres closer than the sum of their radii. When an
    overlap is detected after a BD step, the step is rejected and a new random
    displacement is drawn. Ghost atoms have radius 0 and never overlap, because
    they are zero-radius reference points that the radius-sum check correctly
    excludes.
    """
    for a1 in mol1.atoms:
        if a1.radius < 1e-10:
            continue  # Ghost atom, which has no hard-sphere interaction.
        p1 = np.array([a1.x, a1.y, a1.z])
        r1 = a1.radius
        for a2 in mol2.atoms:
            if a2.radius < 1e-10:
                continue
            p2 = np.array([a2.x, a2.y, a2.z])
            d = float(np.linalg.norm(p1 - p2))
            if d < r1 + a2.radius:
                return True  # Overlap detected.
    return False


ForceFunction = Callable[[Molecule, Molecule], Tuple[np.ndarray, np.ndarray, float]]


def zero_force(mol1: Molecule, mol2: Molecule):
    return np.zeros(3), np.zeros(3), 0.0


def _mol2_positions(mol) -> np.ndarray:
    """Return an (n_atoms, 3) array of atom positions from a Molecule.

    The Brownian bridge code uses this to pass mol2's current positions to
    compute_pair_distances, which expects a flat positions array. The
    positions are read from each atom's x, y, and z fields.
    """
    return np.array(
        [[a.x, a.y, a.z] for a in mol.atoms],
        dtype=float,
    )


@dataclass
class NAMParameters:
    """
    Parameters that control a Brownian dynamics simulation. The field dt is the
    normal minimum time step of 0.2 ps, and dt_rxn is the minimum time step of
    0.05 ps used near the reaction surface.
    """

    n_trajectories: int = 1_000
    dt: float = 0.2  # Normal minimum time step in ps.
    dt_rxn: float = 0.05  # Minimum time step in ps near the reaction surface.
    minimum_core_dt: float = (
        0.0  # Hard floor on the adaptive dt away from the reaction surface (0 means no floor).
    )
    minimum_core_reaction_dt: float = (
        0.0  # Hard floor on the adaptive dt near the reaction surface (0 means no floor).
    )
    max_steps: int = 1_000_000  # Maximum number of steps per trajectory.
    r_start: float = 100.0  # b-sphere radius in Å.
    r_escape: float = 0.0  # Escape radius in Å. 0 selects the automatic value of 2 × r_start.
    seed: Optional[int] = None
    n_threads: int = 1
    use_hard_sphere: bool = True  # Reject steps in which atoms overlap. This is the default.
    hydrodynamic_interactions: bool = False  # Include Rotne-Prager hydrodynamic interactions.
    # When use_brownian_bridge is True, the reaction check supplements the
    # endpoint test by evaluating the path-crossing probability for every
    # contact pair whose distance lies above its cutoff both before and after a
    # step. This catches reactions that occur between discrete BD steps. It is
    # on by default because it closes a gap in the method, and with an empty
    # PathwaySet the bridge code path is simply skipped.
    use_brownian_bridge: bool = True
    verbose: bool = False

    def __post_init__(self):
        if self.r_escape == 0.0:
            # The outer propagator triggers at qb_factor = 1.1 × b-sphere
            # rather than at 2 × b. A trajectory only needs to diffuse 10%
            # above b before the return probability is applied, which matters
            # for large b-spheres.
            self.r_escape = self.r_start * 1.1


# Module-level state for the multiprocessing workers. Each worker process builds
# one NAMSimulator in its initializer and then reuses it for every trajectory it
# runs. This lets the parallel path call the same run_one method as the serial
# path, so the two paths share identical physics, including the
# Luty-McCammon-Zhou outer-region recycling. Only the random seed differs between
# trajectories.
_WORKER_SIM = None


def _worker_init(mol1, mol2, mobility, pathway_set, params, force_fn):
    """
    Build the NAMSimulator that a multiprocessing worker reuses for every
    trajectory it runs. This runs once when each worker process starts.
    """
    global _WORKER_SIM
    _WORKER_SIM = NAMSimulator(mol1, mol2, mobility, pathway_set, params, force_fn)


def _run_trajectory_worker(traj_idx):
    """
    Run one Brownian dynamics trajectory in a worker process by delegating to
    NAMSimulator.run_one, so the parallel path uses exactly the same physics as
    the serial path, including the Luty-McCammon-Zhou recycling at the outer
    boundary. The random number generators are reseeded as seed + traj_idx so
    that each trajectory is independent and reproducible.
    """
    sim = _WORKER_SIM
    base_seed = sim.params.seed if sim.params.seed is not None else 0
    sim.rng = np.random.default_rng(base_seed + traj_idx)
    sim.rng_bb = np.random.default_rng(base_seed + traj_idx + 0xBB)
    return sim.run_one()


class NAMSimulator:
    """
    Northrup-Allison-McCammon Brownian dynamics simulator. Molecule 1 is held
    fixed at the origin, while molecule 2 diffuses inward from a random point
    on the b-sphere.
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
        # A separate RNG stream is used for Brownian bridge sampling (offset by
        # 0xBB). Keeping bridge sampling off the main rng ensures that runs with
        # the bridge on and runs with it off, at the same seed, produce
        # identical trajectories and differ only in the bridge samples.
        _bb_seed = (params.seed if params.seed is not None else 0) + 0xBB
        self.rng_bb = np.random.default_rng(_bb_seed)
        # Pre-centre the mol2 positions so that placement is fast and avoids
        # making copies on every step.
        c0 = mol2.centroid()
        self._mol2_pos0 = mol2.positions_array() - c0
        self._mol2_scratch = copy.copy(mol2)
        self._mol2_scratch.atoms = [copy.copy(a) for a in mol2.atoms]
        # Reaction cutoff distances, which the adaptive time step uses.
        self._rxn_cutoffs = [
            pair.distance_cutoff
            for rxn in pathway_set.reactions
            for pair in rxn.criteria.pairs
        ]
        # Accumulators for the trajectory outcomes.
        self.results: List[TrajectoryResult] = []
        self.n_reacted = 0
        self.n_escaped = 0
        self.reaction_counts: Dict[str, int] = {}
        # Geometry-based adaptive time step controller.
        self._dt_ctrl = AdaptiveTimeStep()
        # The outer propagator (LMZ) is set up when mobility information is
        # available.
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
            # Physical constants expressed in PySTARC units of Å, ps, and kBT.
            kT = 0.5961  # kBT in kcal/mol at 298.15 K.
            viscosity = 1.002e-3 * 1e-4 / 1e-12  # Viscosity of water at 20 °C, converted from Pa·s to kcal·ps/Å³.
            dielectric = 78.54
            vacuum_perm = 1.0 / (4 * math.pi * 332.0636)  # Vacuum permittivity in e²/(kcal·Å).
            debye_len = 8.0  # Debye length in Å at physiological ionic strength.
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
            # If the outer propagator cannot be set up, fall back to the simple
            # escape check.
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
        """Run a single trajectory. This is used by the serial path."""
        v = self.rng.standard_normal(3)
        v /= np.linalg.norm(v)
        pos = v * self.params.r_start
        ori = random_quaternion(self.rng)
        D_r = self.mobility.relative_rotational_diffusion()
        r_h1 = self.mobility.radius1
        r_h2 = self.mobility.radius2
        # Reset the adaptive dt controller for this trajectory.
        self._dt_ctrl.reset()
        # The Brownian bridge state holds the pair distances from the top of the
        # previous iteration, the dt actually used in the previous outer step,
        # and the diffusion at the start of that step. It is None on the first
        # iteration. The bridge code path runs only when use_brownian_bridge is
        # True and the pathway_set has reactions to track.
        _has_reactions = (
            self.pathway_set is not None and len(self.pathway_set.reactions) > 0
        )
        _bb_active = self.params.use_brownian_bridge and _has_reactions
        prev_pair_dists = None
        prev_dt_outer = 0.0
        prev_D_eff_bb = 0.0
        for step in range(self.params.max_steps):
            mol2 = self._place_mol2(pos, ori)
            # Compute the current pair distances when the bridge is active.
            if _bb_active:
                cur_pair_dists = compute_pair_distances(
                    self.mol1,
                    _mol2_positions(mol2),
                    self.pathway_set,
                )
            else:
                cur_pair_dists = None
            # The reaction check is bridge-aware when prior state is available.
            if (
                _bb_active
                and prev_pair_dists is not None
                and cur_pair_dists is not None
                and prev_dt_outer > 0.0
                and prev_D_eff_bb > 0.0
            ):
                rxn = check_reaction_with_bridge(
                    self.mol1,
                    mol2,
                    self.pathway_set,
                    prev_pair_dists,
                    cur_pair_dists,
                    prev_D_eff_bb,
                    prev_dt_outer,
                    self.rng,
                    rng_bb=self.rng_bb,
                )
            else:
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
            # The outer propagator (LMZ) is triggered when r exceeds
            # qb_factor × b-sphere, where qb_factor is 1.1 (taken from
            # motion/qb_factor.hh).
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
                # The particle returned to the b-sphere, so continue the BD
                # propagation.
                continue
            # Simple fallback escape check, used when there is no outer
            # propagator.
            if self._outer_prop is None and r >= self.params.r_escape:
                return TrajectoryResult(Fate.ESCAPED, step, step * self.params.dt, r)
            force, torque, _ = self.force_fn(self.mol1, mol2)
            # With Rotne-Prager-Yamakawa hydrodynamics the relative
            # translational diffusion depends on position.
            D_t = self.mobility.relative_translational_diffusion(pos)
            r = float(np.linalg.norm(pos))
            # Geometry-based adaptive time step.
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
            # Save the old state before stepping so a hard-sphere rejection can
            # redraw from it.
            pos_old, ori_old = pos, ori
            # Draw the Wiener increments and take the step.
            dW_t = math.sqrt(dt) * self.rng.standard_normal(3)
            dW_r = math.sqrt(dt) * self.rng.standard_normal(3)
            pos, ori = bd_step_wiener(pos, ori, force, torque, D_t, D_r, dt, dW_t, dW_r)
            # Backstep on a large force change, subdividing the Wiener path.
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
                self._dt_ctrl._last_dt = hdt  # Record the dt actually used.
            # Reject the step if it produces a hard-sphere collision.
            if self.params.use_hard_sphere:
                mol2_trial = self._place_mol2(pos, ori)
                if _check_hard_sphere_overlap(self.mol1, mol2_trial):
                    dW_t2 = math.sqrt(dt) * self.rng.standard_normal(3)
                    dW_r2 = math.sqrt(dt) * self.rng.standard_normal(3)
                    D_t_old = self.mobility.relative_translational_diffusion(pos_old)
                    pos, ori = bd_step_wiener(
                        pos_old, ori_old, force, torque, D_t_old, D_r, dt, dW_t2, dW_r2
                    )
            # Save the state for the next iteration's Brownian bridge check.
            # The pair distances in cur_pair_dists were captured before this
            # step. The effective dt is whatever the dt controller actually
            # applied, which is set on every call to get_dt and overwritten to
            # hdt when a backstep fires. The diffusion at the start of this step
            # (at pos_old) is the correct bridge diffusion over the interval
            # from pos_old to pos.
            if _bb_active:
                prev_pair_dists = cur_pair_dists
                prev_dt_outer = float(self._dt_ctrl._last_dt or 0.0)
                prev_D_eff_bb = float(
                    self.mobility.relative_translational_diffusion(pos_old)
                )
        return TrajectoryResult(
            Fate.MAX_STEPS,
            self.params.max_steps,
            self.params.max_steps * self.params.dt,
            float(np.linalg.norm(pos)),
        )

    def run(self) -> "SimulationResult":
        """Run all trajectories, either serially or in parallel."""
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
        # Each worker process builds its own NAMSimulator once, through the
        # initializer, and then runs trajectories by index. Every trajectory is
        # handled by the same run_one method the serial path uses, so the
        # parallel path reproduces the serial physics, including the
        # Luty-McCammon-Zhou recycling, and differs only in the per-trajectory
        # random seed.
        init_args = (
            self.mol1,
            self.mol2,
            self.mobility,
            self.pathway_set,
            self.params,
            self.force_fn,
        )
        with mp.Pool(
            n_workers, initializer=_worker_init, initargs=init_args
        ) as pool:
            for result in pool.map(_run_trajectory_worker, range(n)):
                self._record(result)

    def _record(self, result: TrajectoryResult):
        self.results.append(result)
        if result.reacted:
            self.n_reacted += 1
            name = result.reaction_name or "unnamed"
            self.reaction_counts[name] = self.reaction_counts.get(name, 0) + 1
        elif result.escaped:
            self.n_escaped += 1


@dataclass
class SimulationResult:
    """Aggregated NAM Brownian dynamics results together with the k_on calculation."""

    n_trajectories: int
    n_reacted: int
    n_escaped: int
    n_max_steps: int
    reaction_counts: Dict[str, int]
    r_start: float
    r_escape: float
    dt: float
    k_db: float = 0.0  # LMZ rate from outer_propagator.relative_rate(b).

    @classmethod
    def from_simulator(cls, sim: NAMSimulator) -> "SimulationResult":
        n_max = sum(1 for r in sim.results if r.fate == Fate.MAX_STEPS)
        # Obtain the LMZ k_db from the outer propagator when one is available.
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
        Compute the NAM association rate constant k_on from Northrup, Allison,
        and McCammon (1984). The LMZ formulation supplies k_db, the relative
        rate at the b-sphere from the outer propagator, which already includes
        electrostatic effects, and gives

            k_on = conv_factor × k_db × P

        where conv_factor = 6.02e8 converts the result to M⁻¹ s⁻¹. Here P is the
        reaction probability. When k_db is not provided (k_db = 0), the method
        falls back to the Smoluchowski expression

            k_D = 4π × D_rel × r_b × N_A
            k_on = k_D × P / (1 − P × (1 − r_b / r_esc))

        where D_rel is the relative translational diffusion coefficient, r_b is
        the b-sphere radius, r_esc is the escape radius, and N_A is Avogadro's
        number.

        The argument D_rel is the relative translational diffusion coefficient
        in Å²/ps. The argument k_db is the LMZ rate from
        outer_propagator.relative_rate(b_sphere) in internal units; when it is
        0.0 the Smoluchowski form is used instead.
        """
        P = self.reaction_probability
        if P == 0.0:
            return 0.0

        if k_db > 0.0:
            # Here k_on = conv_factor × k_db × P. The conversion factor
            # 6.02e8 L/mol takes the rate from Å³/ps to M⁻¹ s⁻¹, and k_db comes
            # from relative_rate() in Å³/ps. The unit chain is
            # [Å³/ps] × [6.02e23/L] × [1e-27 L/Å³] × [1e12 ps/s] = M⁻¹ s⁻¹.
            CONV = 6.022e23 * 1e-27 * 1e12  # Converts Å³/ps to M⁻¹ s⁻¹.
            return CONV * k_db * P
        else:
            # Smoluchowski approximation, used when there is no outer
            # propagator. Here k_D = 4π × D × b in Å³/ps is converted to
            # M⁻¹ s⁻¹. The factor is N_A [/mol] × 1e-30 [m³/Å³] / 1e-12 [s/ps]
            # / 1e-3 [m³/L] = 6.022e23 × 1e-30 / 1e-12 / 1e-3 = 6.022e8.
            CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3  # Equals 6.022e8, converting Å³/ps to M⁻¹ s⁻¹.
            k_D = 4.0 * math.pi * D_rel * self.r_start  # In Å³/ps.
            beta = self.r_start / self.r_escape
            denom = 1.0 - P * (1.0 - beta)
            return CONV_A3ps * k_D * P / denom

    def __repr__(self):
        return (
            f"SimulationResult(N={self.n_trajectories}, "
            f"reacted={self.n_reacted}, escaped={self.n_escaped}, "
            f"P_rxn={self.reaction_probability:.4f})"
        )


# Confidence interval helpers, attached to SimulationResult by monkey-patching.
# The class is extended after its definition to avoid dataclass issues.
def _n_completed(self) -> int:
    return self.n_reacted + self.n_escaped


def _reaction_probability_ci(self, confidence: float = 0.95):
    """
    Compute the Wilson score 95% confidence interval on the reaction
    probability P_rxn. The Wilson (1927) interval is valid even when P_rxn is
    very small.
    """
    n = self.n_reacted + self.n_escaped
    if n == 0:
        return (0.0, 1.0)
    z = float(_stats.norm.ppf(0.5 + confidence / 2.0))
    p = self.reaction_probability
    z2n = z**2 / n
    denom = 1.0 + z2n
    ctr = (p + z2n / 2.0) / denom
    # The Wilson (1927) margin is z × √(p(1−p)/n + z²/(4n²)) / (1 + z²/n).
    # With z2n = z²/n, the second term inside the square root is z2n/(4n), not
    # z2n/4. The latter overstates the margin by a factor of n, which made the
    # intervals roughly 22 times too wide for typical PySTARC inputs of
    # n = 1000 and p = 1e-3.
    mar = z * math.sqrt(max(p * (1 - p) / n + z2n / (4.0 * n), 0.0)) / denom
    return (max(0.0, ctr - mar), min(1.0, ctr + mar))


def _k_from_P(self, P: float, D_rel: float) -> float:
    if P <= 0.0:
        return 0.0
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3  # Equals 6.022e8, converting Å³/ps to M⁻¹ s⁻¹.
    if self.k_db > 0.0:
        return CONV_A3ps * self.k_db * P
    k_D = 4.0 * math.pi * D_rel * self.r_start  # In Å³/ps.
    beta = self.r_start / self.r_escape
    return CONV_A3ps * k_D * P / (1.0 - P * (1.0 - beta))


def _rate_constant_ci(self, D_rel: float, confidence: float = 0.95):
    """Compute the 95% confidence interval on k_on by propagating the Wilson
    confidence interval on the reaction probability P_rxn."""
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


# Attach these helper methods to SimulationResult.
SimulationResult.reaction_probability_ci = _reaction_probability_ci
SimulationResult.rate_constant_ci = _rate_constant_ci
SimulationResult.summary = _result_summary
