"""
Weighted Ensemble Brownian dynamics simulator for PySTARC.

Implements the Huber & McCammon (1996) weighted ensemble algorithm,
which reduces the required number of trajectories by 10,000-100,000x
compared to brute-force NAM for rare-event problems.

Algorithm
---------
1. Define progress coordinate: separation distance r between molecules
2. Divide [r_b, r_esc] into N_bins bins (logarithmically spaced)
3. Maintain a target of n_per_bin trajectories in each bin
4. Each BD step:
   a. Advance all trajectories by dt
   b. For each bin with too many trajectories: split (clone) the excess
   c. For each bin with too few trajectories:  merge (combine) with weight sum
   d. Track probability weights - they sum to 1 at all times
5. Collect reacted/escaped trajectories with their weights
6. k_on from weighted P_rxn, same formula as NAM

Why it works
------------
Rare-event systems (P_rxn << 1) have most trajectories quickly escaping.
WE-BD forces uniform sampling across all distances by cloning trajectories
that drift toward the binding site and killing those that escape too fast.
The weights track the true probability so k_on is unbiased.

Usage
-----
    from pystarc.simulation.we_simulator import WESimulator, WEParameters
    from pystarc.forces.fast_force import make_fast_engine
    engine = make_fast_engine("/path/to/b_surface_trp/")
    params = WEParameters(
        n_per_bin=10,           # trajectories per bin (10-20 typical)
        n_bins=40,              # bins along progress coordinate
        n_iterations=500,       # WE iterations
        dt=0.2,                 # ps
        r_start=38.101,         # b-sphere radius (Å)
        r_escape=76.202,        # escape radius (Å)
        seed=1523,
    )
    result = WESimulator(mol_rec, mol_lig, mobility, pathway_set,
                         params, engine).run()
    print(f"k_on = {result.rate_constant(D_rel):.3e} M-1s-1")
"""

from __future__ import annotations
from pystarc.simulation.nam_simulator import ForceFunction, zero_force, SimulationResult
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

# WE Parameters
@dataclass
class WEParameters:
    """
    Parameters for Weighted Ensemble BD.
    Key parameters
    --------------
    n_per_bin   : number of trajectories per bin (10-20 typical)
    n_bins      : number of bins along progress coordinate (20-50 typical)
    n_iterations: number of WE iterations to run
    dt          : BD time step in ps
    r_start     : b-sphere radius in Å
    r_escape    : escape radius in Å
    bin_scheme  : 'log' (logarithmic) or 'linear' - log recommended for
                  binding problems where most action is near r_b
    """
    n_per_bin:   int   = 10
    n_bins:      int   = 40
    n_iterations: int  = 500
    dt:          float = 0.2      # ps
    dt_rxn:      float = 0.05     # ps near reaction
    r_start:     float = 100.0    # Å
    r_escape:    float = 0.0      # Å (0 = auto)
    seed:        Optional[int] = None
    adaptive_dt: bool  = True
    steps_per_iteration: int  = 100     # BD steps per WE iteration before resampling
    bin_scheme:  str   = 'log'    # 'log' or 'linear'
    verbose:     bool  = False

    def __post_init__(self):
        if self.r_escape == 0.0:
            self.r_escape = self.r_start * 2.0

# WE Trajectory

@dataclass
class WETrajectory:
    """
    One trajectory in the weighted ensemble.
    Carries position, orientation, probability weight, and bin index.
    """
    position:    np.ndarray          # (3,) current position
    orientation: Quaternion          # current orientation
    weight:      float               # probability weight (sums to 1 over all traj)
    bin_idx:     int                 # current bin
    steps:       int = 0             # total BD steps taken
    time_ps:     float = 0.0         # total simulation time

    def copy(self) -> "WETrajectory":
        return WETrajectory(
            position=self.position.copy(),
            orientation=Quaternion(self.orientation.w, self.orientation.x,
                                   self.orientation.y, self.orientation.z),
            weight=self.weight,
            bin_idx=self.bin_idx,
            steps=self.steps,
            time_ps=self.time_ps,
        )

# WE result
@dataclass
class WEResult:
    """Results from a WE-BD simulation."""
    n_iterations:     int
    n_per_bin:        int
    n_bins:           int
    flux_reaction:    float   # weighted flux into reaction state (per ps)
    flux_escape:      float   # weighted flux into escape state (per ps)
    weight_reacted:   float   # total probability weight of reacted trajectories
    weight_escaped:   float   # total probability weight of escaped trajectories
    r_start:          float
    r_escape:         float
    dt:               float
    iteration_fluxes: List[float] = field(default_factory=list)

    @property
    def reaction_probability(self) -> float:
        """P_rxn = weight_reacted / (weight_reacted + weight_escaped)"""
        total = self.weight_reacted + self.weight_escaped
        return self.weight_reacted / total if total > 0 else 0.0

    def rate_constant(self, D_rel: float) -> float:
        """
        k_on from WE-BD using NAM Smoluchowski formula with WE P_rxn.
        """
        P = self.reaction_probability
        if P == 0.0:
            return 0.0
        D_cm2_s = D_rel * 1e-4
        r_cm    = self.r_start * 1e-8
        k_D     = 4.0 * math.pi * D_cm2_s * r_cm * 6.022e23
        beta    = self.r_start / self.r_escape
        denom   = 1.0 - P * (1.0 - beta)
        return k_D * P / denom

    def __repr__(self) -> str:
        return (f"WEResult(iters={self.n_iterations}, "
                f"P_rxn={self.reaction_probability:.4e}, "
                f"flux_rxn={self.flux_reaction:.4e} /ps)")

# WE simulator 
class WESimulator:
    """
    Weighted Ensemble Brownian dynamics simulator.
    Reduces required trajectory count by 10,000-100,000x versus NAM
    for rare-event binding problems.
    The progress coordinate is the separation distance r = |pos|
    (distance from fixed receptor to mobile ligand centroid).
    """
    def __init__(self,
                 mol1: Molecule,
                 mol2: Molecule,
                 mobility: MobilityTensor,
                 pathway_set: PathwaySet,
                 params: WEParameters,
                 force_fn: Optional[ForceFunction] = None):
        self.mol1        = mol1
        self.mol2        = mol2
        self.mobility    = mobility
        self.pathway_set = pathway_set
        self.params      = params
        self.force_fn    = force_fn or zero_force
        self.rng         = np.random.default_rng(params.seed)
        # Pre-cache mol2 for fast placement
        c0 = mol2.centroid()
        self._mol2_pos0 = mol2.positions_array() - c0
        self._mol2_scratch = copy.copy(mol2)
        self._mol2_scratch.atoms = [copy.copy(a) for a in mol2.atoms]
        # Reaction cutoffs for adaptive dt
        self._rxn_cutoffs = [
            p.distance_cutoff
            for rxn in pathway_set.reactions
            for p in rxn.criteria.pairs
        ]
        # Build bin edges
        self._bins = self._make_bins()
        # Accumulators
        self.weight_reacted = 0.0
        self.weight_escaped = 0.0
        self.iteration_fluxes: List[float] = []
            
    # Bin construction 
    def _make_bins(self) -> np.ndarray:
        """
        Build bin edge array for the binding progress coordinate.
        For association (binding), the progress coordinate is the
        separation distance r, which DECREASES as binding occurs.
        Bins span [r_contact, r_start] where:
          r_contact = minimum reaction cutoff distance
          r_start   = b-sphere radius
        Trajectories start in the rightmost bin (near r_start) and
        progress leftwards toward smaller r to reach the reaction zone.
        Trajectories that drift past r_escape are terminated.
        """
        # Get the minimum reaction cutoff across all criteria
        r_contact = self.params.r_start  # fallback
        for rxn in self.pathway_set.reactions:
            for pair in rxn.criteria.pairs:
                r_contact = min(r_contact, pair.distance_cutoff)
        r_lo = max(r_contact * 0.9, 1.0)   # slightly below reaction cutoff
        r_hi = self.params.r_start
        n = self.params.n_bins + 1   # n+1 edges for n bins
        if self.params.bin_scheme == 'log':
            bins = np.logspace(np.log10(r_lo), np.log10(r_hi), n)
        else:
            bins = np.linspace(r_lo, r_hi, n)
        return bins

    def _bin_of(self, r: float) -> int:
        """Return bin index for separation r. -1 if outside all bins."""
        idx = int(np.searchsorted(self._bins, r, side='right')) - 1
        if idx < 0 or idx >= self.params.n_bins:
            return -1
        return idx
    
    # Molecule placement 
    def _place_mol2(self, pos: np.ndarray, ori: Quaternion) -> Molecule:
        R = ori.to_rotation_matrix()
        new_pos = (R @ self._mol2_pos0.T).T + pos
        mol = self._mol2_scratch
        for atom, p in zip(mol.atoms, new_pos):
            atom.x = float(p[0]); atom.y = float(p[1]); atom.z = float(p[2])
        return mol
    
    # Initialise ensemble on b-sphere 
    def _init_ensemble(self) -> List[WETrajectory]:
        """
        Place n_per_bin trajectories uniformly on the b-sphere.
        Each starts with equal weight 1/(n_per_bin * n_bins).
        """
        n_total = self.params.n_per_bin * self.params.n_bins
        w0 = 1.0 / n_total
        trajs = []
        for _ in range(n_total):
            v = self.rng.standard_normal(3); v /= np.linalg.norm(v)
            pos = v * self.params.r_start
            ori = random_quaternion(self.rng)
            r   = float(np.linalg.norm(pos))
            b   = self._bin_of(r)
            trajs.append(WETrajectory(pos, ori, w0, max(b, 0)))
        return trajs
    
    # One BD step for one trajectory 
    def _step_traj(self, traj: WETrajectory) -> Tuple[WETrajectory, str]:
        """
        Advance one WE trajectory by one BD step.
        Returns (updated_traj, outcome) where outcome is
        'ongoing', 'reacted', or 'escaped'.
        """
        D_t = self.mobility.relative_translational_diffusion()
        D_r = self.mobility.relative_rotational_diffusion()
        mol2_placed = self._place_mol2(traj.position, traj.orientation)
        # Reaction check
        rxn = self.pathway_set.check_all(self.mol1, mol2_placed, self.rng)
        if rxn is not None:
            return traj, 'reacted'
        # Escape check
        r = float(np.linalg.norm(traj.position))
        if r >= self.params.r_escape:
            return traj, 'escaped'
        # Forces
        force, torque, _ = self.force_fn(self.mol1, mol2_placed)
        # BD step
        if self.params.adaptive_dt and self._rxn_cutoffs:
            new_pos, new_ori, dt_used = bd_step_adaptive(
                traj.position, traj.orientation, force, torque,
                D_t, D_r, self.rng, self._rxn_cutoffs,
                self.params.dt, self.params.dt_rxn,
            )
        else:
            new_pos, new_ori = bd_step(
                traj.position, traj.orientation, force, torque,
                D_t, D_r, self.params.dt, self.rng)
            dt_used = self.params.dt

        new_r   = float(np.linalg.norm(new_pos))
        new_bin = self._bin_of(new_r)
        new_traj = WETrajectory(
            position=new_pos,
            orientation=new_ori,
            weight=traj.weight,
            bin_idx=new_bin if new_bin >= 0 else traj.bin_idx,
            steps=traj.steps + 1,
            time_ps=traj.time_ps + dt_used,
        )
        return new_traj, 'ongoing'

    # Split and merge (the WE resampling step) 
    def _resample(self, trajs: List[WETrajectory]) -> List[WETrajectory]:
        """
        Resample trajectories to maintain n_per_bin per bin.

        - Bins with > n_per_bin: split excess trajectories (clone + halve weight)
        - Bins with < n_per_bin: merge pairs (combine weights, keep one)
        - Total probability weight is conserved exactly.
        """
        n_target = self.params.n_per_bin
        new_trajs: List[WETrajectory] = []
        # Group by bin
        bins: Dict[int, List[WETrajectory]] = {}
        for t in trajs:
            bins.setdefault(t.bin_idx, []).append(t)
        for b_idx, group in bins.items():
            n = len(group)
            if n == n_target:
                new_trajs.extend(group)
            elif n > n_target:
                # Split: clone the n - n_target extras, halving their weights
                # Sort by weight descending so we split the heaviest first
                group.sort(key=lambda t: -t.weight)
                keep = group[:n_target]
                extra = group[n_target:]
                for t in extra:
                    # Redistribute its weight to a random kept trajectory
                    donor = keep[int(self.rng.integers(0, n_target))]
                    donor.weight += t.weight
                new_trajs.extend(keep)

            else:
                # Merge: combine pairs until we reach n_target
                # Merge lightest pairs first
                group.sort(key=lambda t: t.weight)
                while len(group) < n_target:
                    # Clone the lightest trajectory (split weight)
                    t = group[0]
                    clone = t.copy()
                    clone.weight = t.weight / 2.0
                    t.weight     = t.weight / 2.0
                    group.append(clone)
                new_trajs.extend(group)
        return new_trajs

    # Main WE loop 
    def run(self) -> WEResult:
        """
        Run the Weighted Ensemble BD simulation.
        Each iteration:
        1. Advance all trajectories by one BD step
        2. Collect reacted/escaped trajectories (add their weight to flux)
        3. Replace reacted/escaped with new trajectories from b-sphere
        4. Resample to maintain n_per_bin per bin
        5. Repeat for n_iterations
        """
        self.weight_reacted = 0.0
        self.weight_escaped = 0.0
        self.iteration_fluxes = []
        trajs = self._init_ensemble()
        total_time_ps = 0.0
        for iteration in range(self.params.n_iterations):
            new_trajs: List[WETrajectory] = []
            iter_flux = 0.0
            for traj in trajs:
                # Advance each trajectory for steps_per_iteration steps
                # before resampling - gives trajectories time to cross bin boundaries
                current = traj
                final_outcome = 'ongoing'
                for _ in range(self.params.steps_per_iteration):
                    current, outcome = self._step_traj(current)
                    if outcome != 'ongoing':
                        final_outcome = outcome
                        break
                if final_outcome == 'reacted':
                    self.weight_reacted += current.weight
                    iter_flux += current.weight
                    # Recycle: spawn new trajectory on b-sphere
                    v = self.rng.standard_normal(3); v /= np.linalg.norm(v)
                    pos = v * self.params.r_start
                    ori = random_quaternion(self.rng)
                    b   = self._bin_of(float(np.linalg.norm(pos)))
                    new_trajs.append(WETrajectory(pos, ori, current.weight,
                                                   max(b, 0)))
                elif final_outcome == 'escaped':
                    self.weight_escaped += current.weight
                    v = self.rng.standard_normal(3); v /= np.linalg.norm(v)
                    pos = v * self.params.r_start
                    ori = random_quaternion(self.rng)
                    b   = self._bin_of(float(np.linalg.norm(pos)))
                    new_trajs.append(WETrajectory(pos, ori, current.weight,
                                                   max(b, 0)))
                else:
                    new_trajs.append(current)
            self.iteration_fluxes.append(iter_flux)
            total_time_ps += self.params.dt
            # Resample to maintain n_per_bin per bin
            trajs = self._resample(new_trajs)
            if self.params.verbose and iteration % max(1, self.params.n_iterations // 10) == 0:
                n_bins_occupied = len({t.bin_idx for t in trajs})
                print(f"  WE iter {iteration+1}/{self.params.n_iterations}  "
                      f"w_react={self.weight_reacted:.4e}  "
                      f"w_escape={self.weight_escaped:.4e}  "
                      f"bins_occupied={n_bins_occupied}/{self.params.n_bins}")
        # Compute flux (probability per unit time)
        flux_rxn = (self.weight_reacted / total_time_ps
                    if total_time_ps > 0 else 0.0)
        flux_esc = (self.weight_escaped / total_time_ps
                    if total_time_ps > 0 else 0.0)
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