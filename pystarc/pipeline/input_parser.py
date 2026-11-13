"""
PySTARC input file parser.

This module reads a pystarc_input.xml file and returns a PySTARCConfig
dataclass in which every pipeline parameter has been validated and given a
sensible default.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
import difflib
import math
import warnings
import xml.etree.ElementTree as ET
from typing import List, Optional
from pathlib import Path
from pystarc.global_defs.constants import AVOGADRO, E_CHARGE, EPS0_SI, KB_SI
from pystarc.global_defs.defaults import (
    DEBYE_LENGTH,
    DESOLVATION_ALPHA,
    DT,
    HYDRODYNAMIC_INTERACTIONS,
    INPUT_DEFAULTS,
    QB_FACTOR,
    SAVE_INTERVAL,
    SOLVENT_PROBE_RADIUS,
    TEMPERATURE,
)


@dataclass
class OutputConfig:
    """Controls what output files PySTARC writes to bd_sims/."""

    results_json: bool = (
        True  # results.json holds k_on, P_rxn, the confidence interval, and related summary numbers.
    )
    trajectories_csv: bool = (
        True  # trajectories.csv records the outcome of each individual trajectory.
    )
    encounters_csv: bool = (
        True  # encounters.csv records position and orientation at the end of the reacting step.
    )
    near_misses_csv: bool = (
        True  # near_misses.csv records position and orientation at the closest approach.
    )
    full_paths: bool = (
        True  # paths.npz stores position and quaternion every save_interval steps.
    )
    radial_density: bool = (
        True  # radial_density.csv stores the BD steps and the BD time per radial shell.
    )
    angular_map: bool = True  # angular_map.npz stores the BD steps and the BD time per (θ, φ) bin.
    bd_time_to_reaction: bool = (
        True  # bd_time_to_reaction.csv stores the BD time inside r_esc of each reacting trajectory.
    )
    contact_frequency: bool = (
        True  # contact_frequency.csv stores the steps and the time each pair spends inside its cutoff.
    )
    milestone_flux: bool = (
        True  # milestone_flux.csv stores the flux through concentric shells.
    )
    transition_matrix: bool = (
        True  # transition_matrix.npz stores Markov state transitions.
    )
    energetics: bool = True  # energetics.npz stores the force and the time step every save_interval steps.
    pose_clusters: bool = (
        True  # pose_clusters.csv stores the rotation angle of the encounter poses in 8 fixed bins.
    )
    save_interval: int = (
        SAVE_INTERVAL  # Record full paths every N steps. A value of 0 saves endpoints only.
    )

    @property
    def fpt_distribution(self) -> bool:
        return self.bd_time_to_reaction


@dataclass
class ChainConfig:
    """Configuration specific to chain Brownian dynamics.

    When PySTARCConfig.chain is set to something other than None, run_pystarc.py
    dispatches to the chain BD pipeline instead of the rigid-body pipeline. Chain
    BD reuses the common PySTARCConfig fields (n_trajectories, seed, max_steps,
    dt, temperature, work_dir, gpu, desolvation_alpha, bd_milestone_radius, and
    the convergence settings) and adds the chain-specific fields defined here.
    Rigid-body input.xml files that have no <chain> block keep working unchanged.
    """

    # Input files.
    chain_json: str = (
        ""  # Path to chain.json, which holds the topology and the initial positions.
    )
    reaction_pairs_json: str = (
        ""  # Path to reaction_pairs.json, a list of [target_atom_idx, chain_atom_idx, distance_A] triplets.
    )
    target_grid_dx: str = (
        ""  # Path to the electrostatic DX grid. An empty string means no field is applied.
    )
    born_grid_dx: str = (
        ""  # Path to the Born desolvation DX grid. An empty string means no Born term is applied.
    )

    # Geometry.
    r_escape: float = (
        0.0  # Escape sphere radius in angstrom. A value of 0 means 2.0 × bd_milestone_radius.
    )
    reaction_n_needed: int = (
        3  # Minimum number of contact pairs that must be satisfied for a reaction.
    )

    # Inner integration, the internal-coordinate dynamics taken within each outer step.
    dt_chain: float = 0.05  # Inner internal-coordinate timestep in picoseconds.
    chain_steps_per_outer: int = (
        4  # Number of inner steps taken per outer rigid-body step.
    )
    n_equilibration_steps: int = (
        0  # Number of pre-equilibration steps for the chain internal coordinates. A value of 0 means none.
    )

    # Diffusion model.
    auto_diffusion: bool = (
        False  # If True, build RPY tensors from the geometry. If False, use a scalar diffusion coefficient.
    )
    D_trans: float = (
        0.0  # Translational diffusion coefficient in Å²/ps. A value of 0 uses the default of 0.1.
    )
    D_rot: float = (
        0.0  # Rotational diffusion coefficient in rad²/ps. A value of 0 uses the default of 0.01.
    )

    # Soft repulsion using the WCA potential.
    use_soft_repulsion: bool = False  # If True, add a WCA chain-target steering layer.
    soft_repulsion_eps: float = 1.0  # WCA epsilon in units of kBT.

    # Generalized Born self-Born terms.
    use_self_born: bool = False  # If True, add chain-internal generalized Born forces.
    gb_eps_in: float = 1.0  # Interior dielectric constant (vacuum).
    gb_eps_out: float = 78.5  # Exterior dielectric constant (water at 300 K).
    gb_obc_alpha: float = 1.0  # OBC2 set-II coefficient.
    gb_obc_beta: float = 0.8  # OBC2 set-II coefficient.
    gb_obc_gamma: float = 4.85  # OBC2 set-II coefficient.
    coffdrop_active: bool = (
        False  # If True, use diagonal-only generalized Born (Path B with COFFDROP).
    )

    # Parallelism.
    n_workers: int = (
        1  # Number of parallel workers for trajectory dispatch. A value of 1 runs serially.
    )


@dataclass
class PySTARCConfig:
    # System.
    pdb: Path = (
        None  # Input PDB file. Optional when both receptor_pqr and ligand_pqr are provided.
    )
    ligand_resname: str = ""  # Residue name of the ligand, for example 'BEN'.
    ligand_charge: int = 0  # Net formal charge of the ligand.
    work_dir: Path = Path("bd_sims")  # Directory where all output files are written.
    # Force field.
    protein_ff: str = "ff14SB"
    ligand_ff: str = "gaff"
    # Brownian dynamics simulation.
    n_trajectories: int = 10000
    n_threads: int = 24
    gpu: bool = True
    seed: int = 1523
    confidence_interval: float = 0.95
    # Trajectories start on the b surface and the escape sphere sits at twice its
    # radius. The reaction criterion comes from rxns_xml, else from ghost_atoms,
    # else from the GHO atoms with bd_milestone_radius_inner as the cutoff.
    bd_milestone_radius: float = (
        30.0  # b-sphere starting radius in angstrom. Should be at least 3 × (r_rec + r_lig).
    )
    bd_milestone_radius_inner: float = (
        0.0  # GHO-GHO reaction distance in angstrom, between 0 and bd_milestone_radius. Used only without rxns_xml and ghost_atoms.
    )
    r_escape: float = (
        0.0  # Escape sphere radius in angstrom. A value of 0 means 2.0 × bd_milestone_radius. A set value must exceed 1.1 × bd_milestone_radius.
    )
    # Without state-machine reactions rxns_xml must hold one reaction, whose pairs
    # and n_needed form the criterion. With them, a trajectory carries a state label
    # and a reaction fires only in its state_before, advancing it to state_after.
    state_machine_reactions: bool = False
    # Ghost atoms. The value 'auto' detects GHO atoms from the PQR files automatically.
    # Otherwise this is a list of [rec_idx, lig_idx, cutoff_ang] triplets.
    ghost_atoms: str = "auto"
    rxns_xml: str = ""  # Reaction criterion file in the BrownDye 2 rxns format.
    receptor_pqr: str = (
        ""  # Pre-computed receptor PQR. Supplying it skips AmberTools and tleap.
    )
    ligand_pqr: str = (
        ""  # Pre-computed ligand PQR. Supplying it skips AmberTools and tleap.
    )
    desolvation_alpha: float = (
        # Multiplier on the Born cavity self energy. The *_born.dx grids
        # carry the rigorous Kirkwood normalisation, so a charge sees
        # dG = alpha q^2 a^3 / r^4 and the correct base is unity.
        DESOLVATION_ALPHA
    )
    max_steps: int = 1000000  # Step limit per trajectory. A trajectory runs until it reacts, escapes or reaches this limit; at the limit it counts as not reacted. Choose it from a pilot run with pystarc.analysis.step_limit.
    debye_length: float = (
        7.858  # Debye length in angstrom. parse() derives it from ion_concentration, sdie and temperature when the tag is omitted, and a given value must agree within 1 %.
    )
    ion_concentration: float = 0.150  # Salt concentration in molar (150 mM NaCl).
    ion_radius_pos: float = 0.95  # Positive ion radius in angstrom (Na⁺ is 0.95 Å).
    ion_radius_neg: float = 1.81  # Negative ion radius in angstrom (Cl⁻ is 1.81 Å).
    apbs_cglen: float = (
        0.0  # Coarse receptor electrostatic grid length in angstrom. A value of 0 derives it.
    )
    apbs_fglen: float = (
        0.0  # Fine receptor electrostatic grid length in angstrom. A value of 0 derives it.
    )
    apbs_dime: int = (
        129  # Grid points per axis of the receptor grids, both electrostatic levels and the Born grids.
    )
    apbs_coarse_dime: int = (
        0  # Grid points per axis of the coarse receptor electrostatic grid. A value of 0 uses apbs_dime.
    )
    apbs_fine_dime: int = (
        0  # Grid points per axis of the fine receptor electrostatic grid. A value of 0 uses apbs_dime.
    )
    apbs_ligand_dime: int = (
        129  # Grid points per axis of the ligand grids, which are sized to the ligand.
    )
    gpu_force_batch: int = (
        0  # Trajectories per GPU force batch. A value of 0 chooses automatically from the ligand size and a 4 GB target.
    )
    pdie: float = 4.0  # Solute dielectric constant (standard protein interior).
    sdie: float = 78.0  # Solvent dielectric constant (water at 298 K).
    srad: float = SOLVENT_PROBE_RADIUS  # Solvent probe radius in angstrom.
    temperature: float = TEMPERATURE  # Temperature in kelvin.
    dt: float = DT  # Fixed outer time step of chain BD in ps. Rigid runs ignore it; their step is bounded by minimum_core_dt and max_dt.
    # Whether the Rotne-Prager-Yamakawa correction to the relative mobility is applied.
    hydrodynamic_interactions: bool = HYDRODYNAMIC_INTERACTIONS
    r_hydro_rec: float = (
        0.0  # Receptor hydrodynamic radius. A value of 0 computes it from the PQR.
    )
    r_hydro_lig: float = (
        0.0  # Ligand hydrodynamic radius. A value of 0 computes it from the PQR.
    )
    minimum_core_dt: float = (
        0.0  # Floor of the adaptive time step in ps. A value of 0 imposes no floor.
    )
    minimum_core_reaction_dt: float = (
        0.0  # Floor of the reaction refinement in state-machine mode in ps. Flattened criteria ignore it. A value of 0 imposes no floor.
    )
    max_dt: float = 0.0  # Ceiling on the timestep. A value of 0 imposes no cap.
    # Physics extensions.
    overlap_check: bool = True  # Reject a move that brings a ligand atom within 0.5 (r_i + r_j) of a receptor atom; the clock still advances. A rigid input must name overlap_check or soft_repulsion.
    brownian_bridge: bool = (
        True  # Count crossings of the reaction surface between two steps. False checks the end of each step only.
    )
    soft_repulsion: bool = (
        False  # Soft excluded volume of BrownDye 2, V = kT (0.9345 (r_i + r_j) / r)^12 between receptor and ligand atoms; replaces the hard overlap check.
    )
    multipole_fallback: bool = (
        True  # Use a dipole and quadrupole far-field expansion beyond the APBS grid.
    )
    lj_forces: bool = (
        False  # Apply WCA repulsive forces from the PQR radii to handle tight contact.
    )
    enable_born2_torque: bool = (
        True  # Born desolvation direction 2 of SI Table S1 Step 6, the ligand Born field on the receptor charges, whose force and torque act on the ligand. False drops both.
    )
    # Checkpointing and convergence.
    checkpoint_interval: int = (
        0  # Save a checkpoint every N completed trajectories. A value of 0 disables checkpointing.
    )
    convergence_interval: int = 10  # Print the live k_on every N% of completion.
    convergence_check: bool = (
        True  # Run convergence analysis after the BD run completes.
    )
    convergence_tol: float = 0.05  # Relative standard-error threshold (0.05 means 5%).
    # Outputs.
    outputs: OutputConfig = None
    # Chain BD configuration. When this is not None, the simulation runs in chain BD
    # mode instead of rigid-body mode. See ChainConfig above.
    chain: Optional[ChainConfig] = None

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = OutputConfig()
        # The PDB is optional when both receptor_pqr and ligand_pqr are provided.
        if self.pdb is not None:
            self.pdb = Path(self.pdb)
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Check required inputs exist and values are sane."""
        if self.chain is not None:
            # In chain BD mode the required inputs are chain_json (the topology)
            # and receptor_pqr (the target). ligand_pqr is not used because the
            # flexible chain plays the role of the ligand.
            if not self.chain.chain_json:
                raise ValueError("<chain><chain_json> is required in chain BD mode")
            if not self.receptor_pqr:
                raise ValueError(
                    "<receptor_pqr> is required in chain BD mode (target structure)"
                )
            # Sanity-check the chain BD numbers, because bad values silently produce
            # garbage results. For example, dt_chain=0 gives zero noise and a static
            # chain, gb_eps_in=0 causes a divide-by-zero in the OBC2 self-Born term,
            # and negative diffusion gives NaN positions. Catch all of these at parse time.
            cc = self.chain
            if cc.dt_chain <= 0:
                raise ValueError(f"chain.dt_chain must be > 0, got {cc.dt_chain}")
            if cc.chain_steps_per_outer < 1:
                raise ValueError(
                    f"chain.chain_steps_per_outer must be >= 1, got "
                    f"{cc.chain_steps_per_outer}"
                )
            if cc.n_equilibration_steps < 0:
                raise ValueError(
                    f"chain.n_equilibration_steps must be >= 0, got "
                    f"{cc.n_equilibration_steps}"
                )
            if cc.D_trans < 0:
                raise ValueError(f"chain.D_trans must be >= 0, got {cc.D_trans}")
            if cc.D_rot < 0:
                raise ValueError(f"chain.D_rot must be >= 0, got {cc.D_rot}")
            if cc.r_escape < 0:
                raise ValueError(
                    f"chain.r_escape must be >= 0 (0 = default), got " f"{cc.r_escape}"
                )
            if cc.reaction_n_needed < 1:
                raise ValueError(
                    f"chain.reaction_n_needed must be >= 1, got "
                    f"{cc.reaction_n_needed}"
                )
            if cc.soft_repulsion_eps < 0:
                raise ValueError(
                    f"chain.soft_repulsion_eps must be >= 0, got "
                    f"{cc.soft_repulsion_eps}"
                )
            if cc.gb_eps_in <= 0:
                raise ValueError(f"chain.gb_eps_in must be > 0, got {cc.gb_eps_in}")
            if cc.gb_eps_out <= 0:
                raise ValueError(f"chain.gb_eps_out must be > 0, got {cc.gb_eps_out}")
            if cc.gb_eps_in > cc.gb_eps_out:
                raise ValueError(
                    f"chain.gb_eps_in ({cc.gb_eps_in}) must be <= "
                    f"chain.gb_eps_out ({cc.gb_eps_out}); interior "
                    f"dielectric is conventionally smaller than exterior."
                )
            if cc.n_workers < 1:
                raise ValueError(f"chain.n_workers must be >= 1, got {cc.n_workers}")
        else:
            has_pqrs = bool(self.receptor_pqr and self.ligand_pqr)
            if not has_pqrs and self.pdb is None:
                raise ValueError(
                    "Either <pdb> or both <receptor_pqr> and <ligand_pqr> must be specified"
                )
        if self.pdb is not None and not Path(self.pdb).exists():
            raise FileNotFoundError(f"PDB not found: {self.pdb}")
        if self.n_trajectories < 1:
            raise ValueError("n_trajectories must be >= 1")
        # Written as not x > 0 so that NaN fails too; inf means no salt.
        if not (self.debye_length > 0 and self.sdie > 0):
            raise ValueError(
                f"debye_length and sdie must be positive, got {self.debye_length} "
                f"and {self.sdie}"
            )
        if not self.ion_concentration >= 0:
            raise ValueError(f"ion_concentration must be >= 0, got {self.ion_concentration}")
        q = QB_FACTOR * self.bd_milestone_radius
        if not (self.r_escape == 0 or self.r_escape > q):
            raise ValueError(
                f"r_escape must be 0 (2 x bd_milestone_radius) or above "
                f"{QB_FACTOR} x bd_milestone_radius = {q:g} Å, got {self.r_escape}"
            )
        if not self.desolvation_alpha >= 0:
            raise ValueError(f"desolvation_alpha must be >= 0, got {self.desolvation_alpha}")
        if self.gpu_force_batch < 0:
            raise ValueError(
                f"gpu_force_batch must be >= 0 (0 chooses automatically), got "
                f"{self.gpu_force_batch}"
            )
        for k in ("confidence_interval", "convergence_tol"):
            if not 0 < getattr(self, k) < 1:
                raise ValueError(f"{k} must lie between 0 and 1, got {getattr(self, k)}")
        if self.outputs.save_interval < 0:
            raise ValueError(
                f"save_interval must be >= 0 (0 keeps the start and end poses), got "
                f"{self.outputs.save_interval}"
            )
        from pystarc.pipeline.run_apbs import _is_valid_apbs_dime

        if not _is_valid_apbs_dime(self.apbs_ligand_dime):
            raise ValueError(
                f"apbs_ligand_dime must be odd, at least 5 and 1 above a multiple "
                f"of 4, got {self.apbs_ligand_dime}"
            )
        if self.temperature != TEMPERATURE:
            warnings.warn(
                f"temperature {self.temperature} K scales kBT only. The viscosity "
                "stays that of water at 25 °C (0.890 mPa·s).",
                stacklevel=2,
            )
        if self.soft_repulsion and self.lj_forces:
            # Both add a repulsion between receptor and ligand atoms, so together
            # they would count the excluded volume twice.
            raise ValueError(
                "<soft_repulsion> and <lj_forces> both add a repulsion between "
                "receptor and ligand atoms, so together they count it twice. "
                "Use <soft_repulsion>true</soft_repulsion> with "
                "<lj_forces>false</lj_forces>."
            )
        return self


_FLAGS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
_RUN_RULE = (
    "every trajectory runs until it reacts, escapes or reaches max_steps, and a "
    "trajectory at the limit counts as not reacted"
)
_SPACING = (
    "the APBS spacing is the grid length over (dime - 1), set with apbs_fglen, "
    "apbs_cglen and apbs_dime"
)
_RETIRED = {
    "stall_check": _RUN_RULE,
    "unfinished_in_p_rxn": _RUN_RULE,
    "apbs_grid_spacing": _SPACING,
    "apbs_coarse_spacing": _SPACING,
}
_RETIRED_OUTPUTS = {"p_commit": "no committor is computed"}
# Tags that only prepare_bd_surface reads from the same input file.
_PREPARE_ONLY = (
    "parm7", "ligand_atom_per_residue", "inject_gho", "apbs_fine_spacing",
    "apbs_n_grids", "apbs_srfm", "apbs_chgm", "apbs_srad", "ion_type",
    "max_n_steps", "desolvation_parameter",
)


def parse_flag(tag, text):
    """true, yes or 1 and false, no or 0 in any case. Anything else raises."""
    v = _FLAGS.get(text.strip().lower())
    if v is None:
        raise ValueError(f"<{tag}> must be true or false, got {text.strip()!r}")
    return v


def debye_from_ions(conc, sdie, T):
    """Debye length in Å of a 1:1 salt at conc mol/L, infinite without salt."""
    if conc <= 0:
        return math.inf
    return 1e10 * math.sqrt(
        EPS0_SI * sdie * KB_SI * T / (2e3 * AVOGADRO * E_CHARGE**2 * conc)
    )


def _reader(block, seen):
    """Reader of the children of block that records every tag it is asked for."""

    def get(tag, default=None, cast=str):
        seen.add(tag)
        el = block.find(tag)
        if el is None or not (el.text or "").strip():
            return default
        if cast is bool:
            return parse_flag(tag, el.text)
        try:
            return cast(el.text.strip())
        except ValueError:
            raise ValueError(
                f"<{tag}> must be {cast.__name__}, got {el.text.strip()!r}"
            ) from None

    return get


def _check_tags(block, seen, retired={}, quiet=(), prepare=()):
    """Stop on a child of block that no reader took."""
    homes = {
        "<outputs>": {f.name for f in fields(OutputConfig)},
        "<chain>": {f.name for f in fields(ChainConfig)},
        "the top level": {f.name for f in fields(PySTARCConfig)},
    }
    where = f"<{block.tag}>"
    other = []
    for el in block:
        t = el.tag
        if t in seen or t in quiet:
            continue
        if t in retired:
            warnings.warn(f"<{t}> is no longer used and is ignored: {retired[t]}.", stacklevel=3)
        elif t in prepare:
            other.append(t)
        else:
            home = [h for h, names in homes.items() if t in names and h != where]
            hint = difflib.get_close_matches(t, seen, n=1)
            raise ValueError(
                f"Unknown tag <{t}> in {where}"
                + (f": it belongs inside {home[0]}." if home and home[0] != "the top level"
                   else ": it belongs at the top level." if home
                   else f": did you mean <{hint[0]}>?" if hint else ".")
            )
    if other:
        warnings.warn(
            f"{', '.join(f'<{t}>' for t in other)} are read by prepare_bd_surface "
            "only, not by the PySTARC run.",
            stacklevel=3,
        )


def parse(xml_path: str | Path) -> PySTARCConfig:
    """
    Parse a pystarc_input.xml file and return a validated PySTARCConfig.

    A minimal example input file looks like this.

        <?xml version="1.0" ?>
        <pystarc_input>
            <receptor_pqr>receptor.pqr</receptor_pqr>
            <ligand_pqr>ligand.pqr</ligand_pqr>
            <rxns_xml>rxns.xml</rxns_xml>
            <work_dir>bd_sims/</work_dir>
            <n_trajectories>10000</n_trajectories>
            <gpu>true</gpu>
            <seed>1523</seed>
            <!-- b-surface radius in angstrom. Trajectories start on it and
                 the escape sphere sits at twice its radius. -->
            <bd_milestone_radius>30.0</bd_milestone_radius>
            <ion_concentration>0.15</ion_concentration>
            <soft_repulsion>true</soft_repulsion>
        </pystarc_input>

    Every tag must be one a reader takes. A flag is true, yes, 1, false, no or
    0 in any case, and a blank element counts as omitted.
    """
    root = ET.parse(xml_path).getroot()
    seen = {"outputs", "chain"}
    _get = _reader(root, seen)

    def get(tag, default=None, cast=str):
        # The registry is authoritative for any tag it names, so this
        # reader and prepare_bd_surface resolve an omitted tag alike.
        return _get(tag, INPUT_DEFAULTS.get(tag, default), cast)

    kw = dict(
        pdb=get("pdb", cast=str),
        desolvation_alpha=get("desolvation_alpha", cast=float),
        debye_length=_get("debye_length", cast=float),
        max_steps=get("max_steps", cast=int, default=1000000),
        ligand_resname=get("ligand_resname", cast=str),
        ligand_charge=get("ligand_charge", default=0, cast=int),
        work_dir=get("work_dir", default="bd_sims", cast=str),
        protein_ff=get("protein_ff", default="ff14SB", cast=str),
        ligand_ff=get("ligand_ff", default="gaff", cast=str),
        n_trajectories=get("n_trajectories", default=10000, cast=int),
        n_threads=get("n_threads", default=24, cast=int),
        gpu=get("gpu", default=True, cast=bool),
        seed=get("seed", default=1523, cast=int),
        confidence_interval=get("confidence_interval", default=0.95, cast=float),
        bd_milestone_radius=get("bd_milestone_radius", cast=float),
        bd_milestone_radius_inner=get(
            "bd_milestone_radius_inner", default=0.0, cast=float
        ),
        r_escape=get("r_escape", default=0.0, cast=float),
        state_machine_reactions=get(
            "state_machine_reactions", default=False, cast=bool
        ),
        ghost_atoms=get("ghost_atoms", default="auto", cast=str),
        rxns_xml=get("rxns_xml", default="", cast=str),
        receptor_pqr=get("receptor_pqr", default="", cast=str),
        ligand_pqr=get("ligand_pqr", default="", cast=str),
        ion_concentration=get("ion_concentration", default=0.150, cast=float),
        ion_radius_pos=get("ion_radius_pos", default=0.95, cast=float),
        ion_radius_neg=get("ion_radius_neg", default=1.81, cast=float),
        apbs_cglen=get("apbs_cglen", default=0.0, cast=float),
        apbs_fglen=get("apbs_fglen", default=0.0, cast=float),
        apbs_dime=get("apbs_dime", default=129, cast=int),
        apbs_coarse_dime=get("apbs_coarse_dime", default=0, cast=int),
        apbs_fine_dime=get("apbs_fine_dime", default=0, cast=int),
        apbs_ligand_dime=get("apbs_ligand_dime", default=129, cast=int),
        gpu_force_batch=get("gpu_force_batch", default=0, cast=int),
        pdie=get("pdie", cast=float),
        sdie=get("sdie", cast=float),
        srad=get("srad", cast=float),
        temperature=get("temperature", cast=float),
        dt=get("dt", cast=float),
        hydrodynamic_interactions=get(
            "hydrodynamic_interactions", default=False, cast=bool
        ),
        r_hydro_rec=get("r_hydro_rec", default=0.0, cast=float),
        r_hydro_lig=get("r_hydro_lig", default=0.0, cast=float),
        minimum_core_dt=get("minimum_core_dt", cast=float),
        minimum_core_reaction_dt=get(
            "minimum_core_reaction_dt", default=0.0, cast=float
        ),
        max_dt=get("max_dt", default=0.0, cast=float),
        overlap_check=get("overlap_check", default=True, cast=bool),
        brownian_bridge=get("brownian_bridge", default=True, cast=bool),
        soft_repulsion=get("soft_repulsion", default=False, cast=bool),
        multipole_fallback=get("multipole_fallback", default=True, cast=bool),
        lj_forces=get("lj_forces", default=False, cast=bool),
        enable_born2_torque=get("enable_born2_torque", default=True, cast=bool),
        checkpoint_interval=get("checkpoint_interval", default=0, cast=int),
        convergence_interval=get("convergence_interval", default=10, cast=int),
        convergence_check=get("convergence_check", default=True, cast=bool),
        convergence_tol=get("convergence_tol", default=0.05, cast=float),
    )
    out_el = root.find("outputs")
    outputs = None
    if out_el is not None:
        oseen = set()
        oget = _reader(out_el, oseen)
        bd_time = oget("bd_time_to_reaction", None, bool)
        outputs = OutputConfig(
            results_json=oget("results_json", True, bool),
            trajectories_csv=oget("trajectories_csv", True, bool),
            encounters_csv=oget("encounters_csv", True, bool),
            near_misses_csv=oget("near_misses_csv", True, bool),
            full_paths=oget("full_paths", True, bool),
            radial_density=oget("radial_density", True, bool),
            angular_map=oget("angular_map", True, bool),
            bd_time_to_reaction=(
                oget("fpt_distribution", True, bool) if bd_time is None else bd_time
            ),
            contact_frequency=oget("contact_frequency", True, bool),
            milestone_flux=oget("milestone_flux", True, bool),
            transition_matrix=oget("transition_matrix", True, bool),
            energetics=oget("energetics", True, bool),
            pose_clusters=oget("pose_clusters", True, bool),
            save_interval=oget("save_interval", SAVE_INTERVAL, int),
        )
        _check_tags(out_el, oseen, _RETIRED_OUTPUTS)
    # The <chain> block selects chain BD mode. Rigid-body inputs have none.
    chain_el = root.find("chain")
    chain = None
    if chain_el is not None:
        cseen = set()
        cget = _reader(chain_el, cseen)
        chain = ChainConfig(
            chain_json=cget("chain_json", default="", cast=str),
            reaction_pairs_json=cget("reaction_pairs_json", default="", cast=str),
            target_grid_dx=cget("target_grid_dx", default="", cast=str),
            born_grid_dx=cget("born_grid_dx", default="", cast=str),
            r_escape=cget("r_escape", default=0.0, cast=float),
            reaction_n_needed=cget("reaction_n_needed", default=3, cast=int),
            auto_diffusion=cget("auto_diffusion", default=False, cast=bool),
            D_trans=cget("D_trans", default=0.0, cast=float),
            D_rot=cget("D_rot", default=0.0, cast=float),
            use_soft_repulsion=cget("use_soft_repulsion", default=False, cast=bool),
            soft_repulsion_eps=cget("soft_repulsion_eps", default=1.0, cast=float),
            use_self_born=cget("use_self_born", default=False, cast=bool),
            gb_eps_in=cget("gb_eps_in", default=1.0, cast=float),
            gb_eps_out=cget("gb_eps_out", default=78.5, cast=float),
            gb_obc_alpha=cget("gb_obc_alpha", default=1.0, cast=float),
            gb_obc_beta=cget("gb_obc_beta", default=0.8, cast=float),
            gb_obc_gamma=cget("gb_obc_gamma", default=4.85, cast=float),
            coffdrop_active=cget("coffdrop_active", default=False, cast=bool),
            n_workers=cget("n_workers", default=1, cast=int),
            dt_chain=cget("dt_chain", default=0.05, cast=float),
            chain_steps_per_outer=cget("chain_steps_per_outer", default=4, cast=int),
            n_equilibration_steps=cget("n_equilibration_steps", default=0, cast=int),
        )
        _check_tags(chain_el, cseen)
    if get("relative_viscosity", 1.0, float) != 1.0:
        raise ValueError(
            "<relative_viscosity> must be 1: PySTARC uses water at 25 °C "
            "(0.890 mPa·s) at every temperature."
        )
    _check_tags(root, seen, _RETIRED, quiet=("receptor_resname",), prepare=_PREPARE_ONLY)
    lam, conc, sdie, T = kw["debye_length"], kw["ion_concentration"], kw["sdie"], kw["temperature"]
    if sdie > 0 and T > 0 and conc >= 0:
        lam_ions = debye_from_ions(conc, sdie, T)
        if lam is None and chain is None:
            lam = lam_ions
        elif lam is not None and abs(lam / lam_ions - 1) > 0.01:
            raise ValueError(
                f"<debye_length> {lam} Å differs by more than 1 % from the "
                f"{lam_ions:.4g} Å that <ion_concentration> {conc} M, <sdie> and "
                "<temperature> give. Omit <debye_length> to derive it."
            )
    kw["debye_length"] = DEBYE_LENGTH if lam is None else lam
    named = lambda t: bool((root.findtext(t) or "").strip())
    if chain is None:
        if not (named("soft_repulsion") or named("overlap_check")):
            raise ValueError(
                "Name the excluded-volume model of a rigid run: "
                "<soft_repulsion>true</soft_repulsion> for the soft wall or "
                "<overlap_check>true</overlap_check> for the hard check."
            )
    return PySTARCConfig(**kw, outputs=outputs, chain=chain).validate()


def write_template(path: str | Path = "pystarc_input.xml"):
    """Write a fully commented template input file."""
    template = """<?xml version="1.0" ?>
<!--
  PySTARC Input File
  ================
  One command:  python run_pystarc.py pystarc_input.xml
  Output:       k_on + 95% CI printed to terminal
-->
<pystarc_input>

    <!-- System -->

    <!-- Path to PDB file containing protein + ligand together -->
    <pdb>hostguest.pdb</pdb>

    <!-- Residue name of the ligand in the PDB (3-letter code) -->
    <ligand_resname>BEN</ligand_resname>

    <!-- Net formal charge of the ligand (used by antechamber) -->
    <ligand_charge>1</ligand_charge>

    <!-- Directory where all output files will be written -->
    <work_dir>bd_sims/</work_dir>

    <!-- Force field -->

    <!-- Protein force field: ff14SB, ff19SB, ff03 -->
    <protein_ff>ff14SB</protein_ff>

    <!-- Ligand force field: gaff, gaff2 -->
    <ligand_ff>gaff</ligand_ff>

    <!-- APBS electrostatics. The spacing is the grid length over (dime - 1). -->

    <!-- Fine receptor grid length in Angstrom (0 derives it) -->
    <apbs_fglen>0</apbs_fglen>

    <!-- Grid points per axis of the receptor grids -->
    <apbs_dime>129</apbs_dime>

    <!-- Salt concentration in molar. The Debye length follows from it. -->
    <ion_concentration>0.15</ion_concentration>

    <!-- Excluded volume: soft wall (true) or the hard overlap check -->
    <soft_repulsion>true</soft_repulsion>

    <!-- BD simulation -->

    <!-- Number of BD trajectories -->
    <n_trajectories>10000</n_trajectories>

    <!-- Number of CPU threads (set to number of cores for HPC) -->
    <n_threads>24</n_threads>

    <!-- Use GPU if available (requires: pip install cupy-cuda12x) -->
    <gpu>true</gpu>

    <!-- Random seed for reproducibility -->
    <seed>1523</seed>

    <!-- Confidence level for Wilson CI on k_on (0.95 = 95%) -->
    <confidence_interval>0.95</confidence_interval>

    <!-- Reaction criterion file (BrownDye 2 rxns format) -->
    <rxns_xml>rxns.xml</rxns_xml>

    <!-- Ghost atoms, used only without rxns_xml -->
    <!--
    auto   = GHO-GHO distance below bd_milestone_radius_inner, which must
             lie between 0 and bd_milestone_radius

    Or specify manually as: rec_atom_index,lig_atom_index,cutoff_angstrom
    One pair per line. Example for trypsin-benzamidine:
        <ghost_atoms>
            3220,18,17.0
            3221,18,10.0
        </ghost_atoms>
    -->
    <ghost_atoms>auto</ghost_atoms>

    <!-- Chain BD mode -->
    <!--
    When the <chain> block is present, the simulation runs as
    flexible chain BD (e.g. peptide-protein association) instead
    of rigid-body. The chain plays the role of the ligand;
    receptor_pqr is the target structure, and chain_json provides
    the chain topology and initial atom positions.

    Required fields:
      <chain_json>          path to chain JSON (topology + positions)

    Recommended fields:
      <reaction_pairs_json> path to reaction_pairs.json (list of
                            [target_atom_idx, chain_atom_idx, dist_A])

    Optional fields:
      <target_grid_dx>      APBS electrostatic potential DX file
      <born_grid_dx>        APBS Born desolvation DX file
      <r_escape>            escape sphere radius (A); 0 = 2 * bd_milestone_radius
      <reaction_n_needed>   minimum contact pairs to satisfy (default 3)
      <auto_diffusion>      true = RPY tensors from chain geometry
      <D_trans>             translational D used when auto_diffusion is false
      <D_rot>               rotational D used when auto_diffusion is false
      <use_soft_repulsion>  WCA chain-target steering layer
      <soft_repulsion_eps>  WCA epsilon (kBT)
      <n_workers>           parallel workers for trajectory dispatch

    Example (uncomment to enable):

    <chain>
        <chain_json>chain.json</chain_json>
        <reaction_pairs_json>reaction_pairs.json</reaction_pairs_json>
        <target_grid_dx>apbs_output/target0.dx</target_grid_dx>
        <born_grid_dx>apbs_output/target0_born.dx</born_grid_dx>
        <r_escape>120.0</r_escape>
        <reaction_n_needed>3</reaction_n_needed>
        <auto_diffusion>true</auto_diffusion>
        <use_soft_repulsion>false</use_soft_repulsion>
        <soft_repulsion_eps>1.0</soft_repulsion_eps>
        <n_workers>1</n_workers>
    </chain>
    -->

</pystarc_input>
"""
    Path(path).write_text(template)
    print(f"Template written to: {path}")
