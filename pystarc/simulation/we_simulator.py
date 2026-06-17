"""
Weighted Ensemble Brownian dynamics simulator for PySTARC.

This module implements the Huber and McCammon (1996) weighted ensemble
algorithm. For rare-event problems it reduces the required number of
trajectories by roughly 10,000 to 100,000 times compared with brute-force
Northrup-Allison-McCammon (NAM) sampling.

The algorithm works as follows. The progress coordinate is the separation
distance r between the two molecules. The interval from the b-sphere radius
r_b to the escape radius r_esc is divided into N_bins bins, usually spaced
logarithmically. The method maintains a target of n_per_bin trajectories in
each bin. On every Brownian-dynamics step the trajectories are advanced by
the time step Δt, any bin holding too many trajectories has its excess split
by cloning, any bin holding too few merges trajectories by combining their
weights, and the probability weights are tracked so that they always sum to 1.
Reacted and escaped trajectories are collected together with their weights,
and k_on is obtained from the weighted reaction probability using the same
formula as NAM.

The method is effective because in rare-event systems, where the reaction
probability is far less than 1, most trajectories escape quickly. Weighted
ensemble BD forces roughly uniform sampling across all separations by cloning
the trajectories that drift toward the binding site and removing those that
escape too quickly. Because the weights track the true probability, the
resulting k_on is unbiased.

A typical use looks like this. Build a force engine with make_fast_engine,
construct a WEParameters object specifying the trajectories per bin, the
number of bins, the number of iterations, the time step in picoseconds, the
b-sphere radius and escape radius in angstrom, and a random seed, then create
a WESimulator from the receptor and ligand molecules, the mobility tensor,
the pathway set, the parameters, and the engine, and call run(). The returned
result provides the rate constant through result.rate_constant(D_rel).

    from pystarc.simulation.we_simulator import WESimulator, WEParameters
    from pystarc.forces.fast_force import make_fast_engine
    engine = make_fast_engine("/path/to/b_surface_trp/")
    params = WEParameters(
        n_per_bin=10,
        n_bins=40,
        n_iterations=500,
        dt=0.2,
        r_start=38.101,
        r_escape=76.202,
        seed=1523,
    )
    result = WESimulator(mol_rec, mol_lig, mobility, pathway_set,
                         params, engine).run()
    print(f"k_on = {result.rate_constant(D_rel):.3e} M-1s-1")
"""

from __future__ import annotations
from pystarc.simulation.nam_simulator import (
    ForceFunction,
    zero_force,
    SimulationResult,
    _check_hard_sphere_overlap,
    _mol2_positions,
)
from pystarc.simulation.chain_simulator import (
    compute_pair_distances,
    check_reaction_with_bridge,
)
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.motion.do_bd_step import bd_step, bd_step_adaptive
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.structures.molecules import Molecule
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
import math
import copy
import warnings


# Weighted ensemble parameters
@dataclass
class WEParameters:
    """
    Parameters for the weighted ensemble Brownian dynamics run.

    The field n_per_bin is the number of trajectories kept per bin, typically
    10 to 20. The field n_bins is the number of bins along the progress
    coordinate, typically 20 to 50. The field n_iterations is the number of
    weighted ensemble iterations to run. The field dt is the BD time step in
    picoseconds. The field r_start is the b-sphere radius in angstrom and
    r_escape is the escape radius in angstrom. The field bin_scheme selects
    'log' for logarithmic spacing or 'linear' for uniform spacing. Logarithmic
    spacing is recommended for binding problems, where most of the dynamics
    happens near the b-sphere radius r_b.
    """

    n_per_bin: int = 10
    n_bins: int = 40
    n_iterations: int = 500
    dt: float = 0.2  # time step in picoseconds
    dt_rxn: float = 0.05  # smaller time step in picoseconds used near a reaction
    r_start: float = 100.0  # b-sphere radius in angstrom
    r_escape: float = 0.0  # escape radius in angstrom (0 means choose automatically)
    seed: Optional[int] = None
    adaptive_dt: bool = True
    steps_per_iteration: int = (
        100  # BD steps per weighted ensemble iteration before resampling
    )
    bin_scheme: str = "log"  # spacing of the bins, either 'log' or 'linear'
    verbose: bool = False
    # When use_brownian_bridge is True the reaction check supplements the
    # endpoint test with the closed-form Brownian-bridge crossing probability
    # for every contact pair whose distance lies above its cutoff both before
    # and after a step, the same test the NAM path applies in run_one. This
    # catches reactions that occur between discrete BD steps so that the
    # weighted ensemble counts reactive flux the same way NAM does. With an
    # empty PathwaySet the bridge code path is simply skipped.
    use_brownian_bridge: bool = True
    # When use_hard_sphere is True a step that produces a hard-sphere overlap
    # is rejected and the displacement is redrawn, matching the NAM path.
    use_hard_sphere: bool = True

    def __post_init__(self):
        if self.r_escape == 0.0:
            self.r_escape = self.r_start * 2.0


# A single weighted ensemble trajectory


@dataclass
class WETrajectory:
    """
    One trajectory in the weighted ensemble.

    It carries the current position, the current orientation, the probability
    weight, and the index of the bin it currently occupies.
    """

    position: np.ndarray  # current position as a length-3 vector
    orientation: Quaternion  # current orientation
    weight: float  # probability weight, summing to 1 over all trajectories
    bin_idx: int  # index of the bin currently occupied
    steps: int = 0  # total number of BD steps taken
    time_ps: float = 0.0  # total simulation time in picoseconds
    # Brownian-bridge state carried from the previous step, used to evaluate the
    # path-crossing probability on the current step. The fields hold the contact
    # pair distances captured at the top of the previous step, the dt actually
    # used on that step, and the relative diffusion at the start of that step.
    # They are None or zero until the first step has been taken, in which case
    # the bridge check falls back to the endpoint-only test.
    prev_pair_dists: "Optional[list]" = None
    prev_dt: float = 0.0
    prev_D_eff: float = 0.0

    def copy(self) -> "WETrajectory":
        return WETrajectory(
            position=self.position.copy(),
            orientation=Quaternion(
                self.orientation.w,
                self.orientation.x,
                self.orientation.y,
                self.orientation.z,
            ),
            weight=self.weight,
            bin_idx=self.bin_idx,
            steps=self.steps,
            time_ps=self.time_ps,
            prev_pair_dists=(
                [d.copy() for d in self.prev_pair_dists]
                if self.prev_pair_dists is not None
                else None
            ),
            prev_dt=self.prev_dt,
            prev_D_eff=self.prev_D_eff,
        )


# Result of a weighted ensemble run
@dataclass
class WEResult:
    """Results from a weighted ensemble BD simulation."""

    n_iterations: int
    n_per_bin: int
    n_bins: int
    flux_reaction: float  # weighted flux into the reaction state, per picosecond
    flux_escape: float  # weighted flux into the escape state, per picosecond
    weight_reacted: float  # total probability weight of the reacted trajectories
    weight_escaped: float  # total probability weight of the escaped trajectories
    r_start: float
    r_escape: float
    dt: float
    iteration_fluxes: List[float] = field(default_factory=list)

    @property
    def reaction_probability(self) -> float:
        """
        Return the reaction probability P_rxn, defined as the weight of the
        reacted trajectories divided by the combined weight of the reacted and
        escaped trajectories.
        """
        total = self.weight_reacted + self.weight_escaped
        return self.weight_reacted / total if total > 0 else 0.0

    def rate_constant(self, D_rel: float) -> float:
        """
        Return k_on in M^-1 s^-1 from the weighted ensemble run.

        The weighted ensemble reaction probability P_rxn is combined with the
        diffusion-limited encounter rate at the b-surface through the
        Northrup-Allison-McCammon finite-escape expression. The encounter rate is
        the Smoluchowski steady-state rate k_b = 4 pi D r in units of A^3/ps,
        evaluated from the relative diffusion coefficient D_rel and the b-surface
        radius r_start. The factor 6.022e8 converts A^3/ps to M^-1 s^-1, the same
        convention used in the single-trajectory pipeline.
        """
        P = self.reaction_probability
        if P == 0.0:
            return 0.0
        k_b = 4.0 * math.pi * D_rel * self.r_start  # A^3/ps
        beta = self.r_start / self.r_escape
        denom = 1.0 - P * (1.0 - beta)
        return 6.022e8 * k_b * P / denom

    def __repr__(self) -> str:
        return (
            f"WEResult(iters={self.n_iterations}, "
            f"P_rxn={self.reaction_probability:.4e}, "
            f"flux_rxn={self.flux_reaction:.4e} /ps)"
        )


# The weighted ensemble simulator
class WESimulator:
    """
    Weighted ensemble Brownian dynamics simulator.

    For rare-event binding problems this reduces the required trajectory count
    by roughly 10,000 to 100,000 times compared with NAM. The progress
    coordinate is the separation distance r = |pos|, the distance from the
    fixed receptor to the centroid of the mobile ligand.
    """

    def __init__(
        self,
        mol1: Molecule,
        mol2: Molecule,
        mobility: MobilityTensor,
        pathway_set: PathwaySet,
        params: WEParameters,
        force_fn: Optional[ForceFunction] = None,
    ):
        self.mol1 = mol1
        self.mol2 = mol2
        self.mobility = mobility
        self.pathway_set = pathway_set
        self.params = params
        self.force_fn = force_fn or zero_force
        self.rng = np.random.default_rng(params.seed)
        # A separate RNG stream is used for the Brownian bridge sampling, offset
        # by 0xBB. Keeping bridge sampling off the main rng means the bridge acts
        # as a purely additive operator on the trajectory, the same convention
        # the NAM simulator uses.
        _bb_seed = (params.seed if params.seed is not None else 0) + 0xBB
        self.rng_bb = np.random.default_rng(_bb_seed)
        # The Brownian bridge code path runs only when use_brownian_bridge is set
        # and the pathway set has reactions to track. With an empty PathwaySet it
        # is skipped, matching the NAM path.
        self._bb_active = bool(params.use_brownian_bridge) and bool(
            pathway_set is not None and len(pathway_set.reactions) > 0
        )
        # Cache the ligand atom positions relative to its centroid so it can
        # be placed quickly during the simulation.
        c0 = mol2.centroid()
        self._mol2_pos0 = mol2.positions_array() - c0
        self._mol2_scratch = copy.copy(mol2)
        self._mol2_scratch.atoms = [copy.copy(a) for a in mol2.atoms]
        # Collect the reaction distance cutoffs, used to decide when to switch
        # to the smaller adaptive time step.
        self._rxn_cutoffs = [
            p.distance_cutoff
            for rxn in pathway_set.reactions
            for p in rxn.criteria.pairs
        ]
        # Build the bin edges along the progress coordinate.
        self._bins = self._make_bins()
        # Running totals accumulated over the simulation.
        self.weight_reacted = 0.0
        self.weight_escaped = 0.0
        self.iteration_fluxes: List[float] = []

    # Construction of the bin edges
    def _make_bins(self) -> np.ndarray:
        """
        Build the array of bin edges for the binding progress coordinate.

        For association the progress coordinate is the separation distance r,
        which decreases as binding proceeds. The bins span the interval from
        r_contact to r_start, where r_contact is the smallest reaction cutoff
        distance and r_start is the b-sphere radius. Trajectories begin in the
        rightmost bin near r_start and move leftward toward smaller r to reach
        the reaction zone. Any trajectory that drifts past r_escape is
        terminated.
        """
        # Warn when there are no reactions defined. In that case the loop below
        # never updates r_contact, so the bins collapse to the narrow interval
        # [0.9 * r_start, r_start] and provide no resolution in the reaction
        # zone. The bins themselves are left unchanged.
        if not self.pathway_set.reactions:
            warnings.warn(
                "PathwaySet has no reactions; weighted ensemble bins span only "
                "[0.9 * r_start, r_start] and provide no reaction-zone "
                "resolution.",
                stacklevel=2,
            )
        # Find the smallest reaction cutoff across all criteria, falling back
        # to r_start if there are none.
        r_contact = self.params.r_start  # fallback
        for rxn in self.pathway_set.reactions:
            for pair in rxn.criteria.pairs:
                r_contact = min(r_contact, pair.distance_cutoff)
        r_lo = max(
            r_contact * 0.9, 1.0
        )  # place the lower edge just below the reaction cutoff
        r_hi = self.params.r_start
        n = self.params.n_bins + 1  # n bins require n+1 edges
        if self.params.bin_scheme == "log":
            bins = np.logspace(np.log10(r_lo), np.log10(r_hi), n)
        else:
            bins = np.linspace(r_lo, r_hi, n)
        return bins

    def _bin_of(self, r: float) -> int:
        """Return the bin index for separation r.

        A separation below the innermost edge r_lo is clamped to bin 0, the
        innermost bin, because a walker that drifts past r_lo has moved deeper
        into the reaction zone and belongs in the innermost bin rather than
        being pinned to its previous bin. A separation at or beyond the
        outermost edge r_hi lies in the escape region and returns -1, since
        those walkers are handled by the escape test rather than by a bin.
        """
        idx = int(np.searchsorted(self._bins, r, side="right")) - 1
        if idx < 0:
            return 0  # below r_lo, clamp to the innermost bin
        if idx >= self.params.n_bins:
            return -1  # at or beyond r_hi, the escape region
        return idx

    # Placement of the mobile ligand
    def _place_mol2(self, pos: np.ndarray, ori: Quaternion) -> Molecule:
        R = ori.to_rotation_matrix()
        new_pos = (R @ self._mol2_pos0.T).T + pos
        mol = self._mol2_scratch
        for atom, p in zip(mol.atoms, new_pos):
            atom.x = float(p[0])
            atom.y = float(p[1])
            atom.z = float(p[2])
        return mol

    # Initialisation of the ensemble on the b-sphere
    def _init_ensemble(self) -> List[WETrajectory]:
        """
        Place n_per_bin trajectories uniformly on the b-sphere. Every
        trajectory starts with the same weight, equal to 1 / (n_per_bin × n_bins).
        """
        n_total = self.params.n_per_bin * self.params.n_bins
        w0 = 1.0 / n_total
        trajs = []
        for _ in range(n_total):
            v = self.rng.standard_normal(3)
            v /= np.linalg.norm(v)
            pos = v * self.params.r_start
            ori = random_quaternion(self.rng)
            r = float(np.linalg.norm(pos))
            b = self._bin_of(r)
            trajs.append(WETrajectory(pos, ori, w0, max(b, 0)))
        return trajs

    # A single BD step for one trajectory
    def _step_traj(self, traj: WETrajectory) -> Tuple[WETrajectory, str]:
        """
        Advance one weighted ensemble trajectory by a single BD step. Returns
        the pair (updated_traj, outcome), where the outcome is 'ongoing',
        'reacted', or 'escaped'.
        """
        # The relative translational diffusion is evaluated at the current pair
        # separation so the Rotne-Prager-Yamakawa position dependence is applied,
        # matching the NAM path. Without the position the bare D0 would be used
        # even when use_rpy is True. The pair separation vector is the ligand
        # centroid position relative to the receptor fixed at the origin.
        D_t = self.mobility.relative_translational_diffusion(traj.position)
        D_r = self.mobility.relative_rotational_diffusion()
        mol2_placed = self._place_mol2(traj.position, traj.orientation)
        # Compute the current contact pair distances when the bridge is active so
        # the reaction check can evaluate the path-crossing probability.
        if self._bb_active:
            cur_pair_dists = compute_pair_distances(
                self.mol1,
                _mol2_positions(mol2_placed),
                self.pathway_set,
            )
        else:
            cur_pair_dists = None
        # Check whether the trajectory has reacted. The check is bridge-aware
        # when prior-step state is available, using the closed-form crossing
        # probability P = exp(-x0 * x1 / (D_eff * dt)) for every pair that stayed
        # above its cutoff at both endpoints, exactly as NAM run_one does.
        # Otherwise it falls back to the endpoint-only test.
        if (
            self._bb_active
            and traj.prev_pair_dists is not None
            and cur_pair_dists is not None
            and traj.prev_dt > 0.0
            and traj.prev_D_eff > 0.0
        ):
            rxn = check_reaction_with_bridge(
                self.mol1,
                mol2_placed,
                self.pathway_set,
                traj.prev_pair_dists,
                cur_pair_dists,
                traj.prev_D_eff,
                traj.prev_dt,
                self.rng,
                rng_bb=self.rng_bb,
            )
        else:
            rxn = self.pathway_set.check_all(self.mol1, mol2_placed, self.rng)
        if rxn is not None:
            return traj, "reacted"
        # Check whether the trajectory has escaped.
        r = float(np.linalg.norm(traj.position))
        if r >= self.params.r_escape:
            return traj, "escaped"
        # Evaluate the forces and torques.
        force, torque, _ = self.force_fn(self.mol1, mol2_placed)
        # Take the BD step, using the adaptive time step when reaction cutoffs
        # are available, otherwise the fixed time step.
        if self.params.adaptive_dt and self._rxn_cutoffs:
            new_pos, new_ori, dt_used = bd_step_adaptive(
                traj.position,
                traj.orientation,
                force,
                torque,
                D_t,
                D_r,
                self.rng,
                self._rxn_cutoffs,
                self.params.dt,
                self.params.dt_rxn,
            )
        else:
            new_pos, new_ori = bd_step(
                traj.position,
                traj.orientation,
                force,
                torque,
                D_t,
                D_r,
                self.params.dt,
                self.rng,
            )
            dt_used = self.params.dt

        # Reject the step if it produces a hard-sphere overlap, then redraw a
        # fresh displacement from the previous position and keep redrawing until
        # the new position is free of overlap, up to a fixed number of attempts.
        # This mirrors the NAM run_one rejection. The diffusion D_t was already
        # evaluated at traj.position, which is the start of the redrawn step.
        if self.params.use_hard_sphere:
            mol2_trial = self._place_mol2(new_pos, new_ori)
            if _check_hard_sphere_overlap(self.mol1, mol2_trial):
                HS_MAX_REDRAWS = 5
                for _ in range(HS_MAX_REDRAWS):
                    pos_try, ori_try = bd_step(
                        traj.position,
                        traj.orientation,
                        force,
                        torque,
                        D_t,
                        D_r,
                        dt_used,
                        self.rng,
                    )
                    mol2_try = self._place_mol2(pos_try, ori_try)
                    if not _check_hard_sphere_overlap(self.mol1, mol2_try):
                        new_pos, new_ori = pos_try, ori_try
                        break
                else:
                    # No overlap-free redraw was found within the attempt cap, so
                    # the ligand stays at the previous non-overlapping position.
                    new_pos, new_ori = traj.position, traj.orientation

        new_r = float(np.linalg.norm(new_pos))
        new_bin = self._bin_of(new_r)
        new_traj = WETrajectory(
            position=new_pos,
            orientation=new_ori,
            weight=traj.weight,
            bin_idx=new_bin if new_bin >= 0 else traj.bin_idx,
            steps=traj.steps + 1,
            time_ps=traj.time_ps + dt_used,
        )
        # Save the bridge state for the next step. The pair distances in
        # cur_pair_dists were captured before this step, dt_used is the time step
        # actually applied, and D_t is the relative diffusion at the start of the
        # step (at traj.position), which is the correct bridge diffusion over the
        # interval from traj.position to new_pos.
        if self._bb_active:
            new_traj.prev_pair_dists = cur_pair_dists
            new_traj.prev_dt = float(dt_used)
            new_traj.prev_D_eff = float(D_t)
        return new_traj, "ongoing"

    # Splitting and merging, the weighted ensemble resampling step
    def _resample(self, trajs: List[WETrajectory]) -> List[WETrajectory]:
        """
        Resample the trajectories so that each bin again holds n_per_bin of
        them.

        A bin with more than n_per_bin trajectories has its excess merged by
        combining weights and keeping fewer trajectories. A bin with fewer than
        n_per_bin trajectories splits trajectories by cloning and halving their
        weight. In both cases the total probability weight is conserved exactly.
        """
        n_target = self.params.n_per_bin
        new_trajs: List[WETrajectory] = []
        # Group the trajectories by the bin they occupy.
        bins: Dict[int, List[WETrajectory]] = {}
        for t in trajs:
            bins.setdefault(t.bin_idx, []).append(t)
        for b_idx, group in bins.items():
            n = len(group)
            if n == n_target:
                new_trajs.extend(group)
            elif n > n_target:
                # Merge the excess. Sort by descending weight so the heaviest
                # trajectories are kept and the dominant weight survives, then
                # redistribute the weight of the n - n_target lightest extras
                # into randomly chosen kept trajectories. Adding the extra
                # weight to a donor conserves the total weight.
                group.sort(key=lambda t: -t.weight)
                keep = group[:n_target]
                extra = group[n_target:]
                for t in extra:
                    # Transfer the extra weight into a randomly chosen kept trajectory.
                    donor = keep[int(self.rng.integers(0, n_target))]
                    donor.weight += t.weight
                new_trajs.extend(keep)

            else:
                # Split, following Huber and McCammon. Repeatedly take the
                # heaviest trajectory in the bin, clone it, and divide its
                # weight evenly between the original and the clone, until the
                # bin holds n_target trajectories. Re-selecting the current
                # heaviest trajectory on each split distributes the cloning
                # across the trajectories carrying the most weight, so the bin
                # ends with n_target trajectories of nearly even weight rather
                # than a geometric spread. Because the clone receives w/2 while
                # the original keeps w/2, the total weight is conserved.
                while len(group) < n_target:
                    heaviest = max(range(len(group)), key=lambda i: group[i].weight)
                    t = group[heaviest]
                    clone = t.copy()
                    clone.weight = t.weight / 2.0
                    t.weight = t.weight / 2.0
                    group.append(clone)
                new_trajs.extend(group)
        return new_trajs

    # The main weighted ensemble loop
    def run(self) -> WEResult:
        """
        Run the weighted ensemble BD simulation.

        Each iteration advances all trajectories by one BD step, collects the
        reacted and escaped trajectories and adds their weight to the flux,
        replaces those reacted and escaped trajectories with new ones launched
        from the b-sphere, and resamples so that each bin again holds n_per_bin
        trajectories. This repeats for n_iterations iterations.
        """
        self.weight_reacted = 0.0
        self.weight_escaped = 0.0
        self.iteration_fluxes = []
        trajs = self._init_ensemble()
        self.total_time_ps = 0.0
        for iteration in range(self.params.n_iterations):
            new_trajs: List[WETrajectory] = []
            iter_flux = 0.0
            iter_time_ps = 0.0
            for traj in trajs:
                # Advance each trajectory for steps_per_iteration steps before
                # resampling, which gives the trajectories time to cross bin
                # boundaries.
                current = traj
                final_outcome = "ongoing"
                time_at_start = current.time_ps
                for _ in range(self.params.steps_per_iteration):
                    current, outcome = self._step_traj(current)
                    if outcome != "ongoing":
                        final_outcome = outcome
                        break
                # The time genuinely simulated for this trajectory during the
                # iteration is the change in its own maintained clock, summing
                # the real per-step time taken. Trajectories that reach a
                # reacted or escaped outcome stop early, and steps taken near a
                # reaction boundary use the smaller adaptive time step, so this
                # difference may be shorter than steps_per_iteration × dt. The
                # ensemble advances by the longest such span, the genuine
                # elapsed time of the iteration.
                iter_time_ps = max(iter_time_ps, current.time_ps - time_at_start)
                if final_outcome == "reacted":
                    self.weight_reacted += current.weight
                    iter_flux += current.weight
                    # Recycle the trajectory by launching a new one on the b-sphere.
                    v = self.rng.standard_normal(3)
                    v /= np.linalg.norm(v)
                    pos = v * self.params.r_start
                    ori = random_quaternion(self.rng)
                    b = self._bin_of(float(np.linalg.norm(pos)))
                    new_trajs.append(WETrajectory(pos, ori, current.weight, max(b, 0)))
                elif final_outcome == "escaped":
                    self.weight_escaped += current.weight
                    v = self.rng.standard_normal(3)
                    v /= np.linalg.norm(v)
                    pos = v * self.params.r_start
                    ori = random_quaternion(self.rng)
                    b = self._bin_of(float(np.linalg.norm(pos)))
                    new_trajs.append(WETrajectory(pos, ori, current.weight, max(b, 0)))
                else:
                    new_trajs.append(current)
            self.iteration_fluxes.append(iter_flux)
            # Advance the ensemble clock by the time genuinely simulated during
            # the iteration, taken from the trajectories' own maintained clocks.
            # This accumulates the real per-step time actually taken, which
            # reflects early termination at a reacted or escaped outcome and the
            # smaller adaptive time step used near a reaction boundary.
            self.total_time_ps += iter_time_ps
            # Resample so that each bin again holds n_per_bin trajectories.
            trajs = self._resample(new_trajs)
            if (
                self.params.verbose
                and iteration % max(1, self.params.n_iterations // 10) == 0
            ):
                n_bins_occupied = len({t.bin_idx for t in trajs})
                print(
                    f"  WE iter {iteration+1}/{self.params.n_iterations}  "
                    f"w_react={self.weight_reacted:.4e}  "
                    f"w_escape={self.weight_escaped:.4e}  "
                    f"bins_occupied={n_bins_occupied}/{self.params.n_bins}"
                )
        # Compute the flux as the accumulated weight per unit of simulation time.
        flux_rxn = (
            self.weight_reacted / self.total_time_ps if self.total_time_ps > 0 else 0.0
        )
        flux_esc = (
            self.weight_escaped / self.total_time_ps if self.total_time_ps > 0 else 0.0
        )
        return WEResult(
            n_iterations=self.params.n_iterations,
            n_per_bin=self.params.n_per_bin,
            n_bins=self.params.n_bins,
            flux_reaction=flux_rxn,
            flux_escape=flux_esc,
            weight_reacted=self.weight_reacted,
            weight_escaped=self.weight_escaped,
            r_start=self.params.r_start,
            r_escape=self.params.r_escape,
            dt=self.params.dt,
            iteration_fluxes=self.iteration_fluxes,
        )
