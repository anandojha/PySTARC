"""
PySTARC input file parser
=======================
Reads pystarc_input.xml and returns a PySTARCConfig dataclass
with all pipeline parameters validated and defaulted.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from typing import List, Optional
from pathlib import Path


@dataclass
class OutputConfig:
    """Controls what output files PySTARC writes to bd_sims/."""

    results_json: bool = True  # results.json - k_on, P_rxn, CI, etc.
    trajectories_csv: bool = True  # trajectories.csv - per-trajectory outcomes
    encounters_csv: bool = True  # encounters.csv - pos/orientation at reaction
    near_misses_csv: bool = (
        True  # near_misses.csv - pos/orientation at closest approach
    )
    full_paths: bool = True  # paths.npz - pos/q every save_interval steps
    radial_density: bool = True  # radial_density.csv - ρ(r) histogram
    angular_map: bool = True  # angular_map.npz - (θ, φ) occupancy heatmap
    fpt_distribution: bool = True  # fpt_distribution.csv - first passage times
    contact_frequency: bool = True  # contact_frequency.csv - pair contact counts
    milestone_flux: bool = True  # milestone_flux.csv - flux through concentric shells
    p_commit: bool = True  # p_commit.npz - commitment probability map
    transition_matrix: bool = True  # transition_matrix.npz - Markov state transitions
    energetics: bool = True  # energetics.npz - force/energy traces
    pose_clusters: bool = True  # pose_clusters.csv - clustered encounter orientations
    save_interval: int = 10  # record full paths every N steps (0 = endpoints only)


@dataclass
class ChainConfig:
    """Chain BD-specific configuration.

    When PySTARCConfig.chain is set (not None), run_pystarc.py
    dispatches to the chain BD pipeline instead of the rigid-body
    pipeline. Chain BD reuses common PySTARCConfig fields
    (n_trajectories, seed, max_steps, dt, temperature, work_dir, gpu,
    desolvation_alpha, bd_milestone_radius, convergence_*) and adds
    chain-specific fields here. Rigid-body input.xml files (without
    a <chain> block) keep working unchanged.
    """

    # Inputs
    chain_json: str = ""  # path to chain.json (topology + initial positions)
    reaction_pairs_json: str = (
        ""  # path to reaction_pairs.json: list of [target_atom_idx, chain_atom_idx, distance_A]
    )
    target_grid_dx: str = ""  # path to electrostatic DX grid (or "" -> no field)
    born_grid_dx: str = ""  # path to Born desolvation DX grid (or "" -> no Born)

    # Geometry
    r_escape: float = 0.0  # escape sphere radius (A); 0 -> 2.0 * bd_milestone_radius
    reaction_n_needed: int = 3  # minimum contact pairs to satisfy for reaction

    # Inner integration (internal-coordinate dynamics within each outer step)
    dt_chain: float = 0.05  # inner internal-coordinate timestep (ps)
    chain_steps_per_outer: int = 4  # number of inner steps per outer rigid-body step
    n_equilibration_steps: int = (
        0  # pre-equilibration steps for chain internal coords (0 = none)
    )

    # Diffusion mode
    auto_diffusion: bool = False  # True -> RPY tensors from geometry; False -> scalar D
    D_trans: float = 0.0  # translational D (A^2/ps); 0 -> default 0.1
    D_rot: float = 0.0  # rotational D (rad^2/ps); 0 -> default 0.01

    # Soft repulsion (WCA)
    use_soft_repulsion: bool = False  # True -> WCA chain-target steering layer
    soft_repulsion_eps: float = 1.0  # WCA epsilon (kBT)

    # GB self-Born / generalized Born
    use_self_born: bool = False  # True -> add chain-internal GB forces
    gb_eps_in: float = 1.0  # interior dielectric (vacuum)
    gb_eps_out: float = 78.5  # exterior dielectric (water at 300 K)
    gb_obc_alpha: float = 1.0  # OBC2 set-II coefficient
    gb_obc_beta: float = 0.8  # OBC2 set-II coefficient
    gb_obc_gamma: float = 4.85  # OBC2 set-II coefficient
    coffdrop_active: bool = False  # True -> diagonal-only GB (Path B with COFFDROP)

    # Parallelism
    n_workers: int = 1  # parallel workers for trajectory dispatch (1 = serial)


@dataclass
class PySTARCConfig:
    # System
    pdb: Path = None  # input PDB - optional when receptor_pqr+ligand_pqr provided
    ligand_resname: str = ""  # residue name of ligand, e.g. 'BEN'
    ligand_charge: int = 0  # net formal charge of ligand
    work_dir: Path = Path("bd_sims")  # all output files written here
    # Force field
    protein_ff: str = "ff14SB"
    ligand_ff: str = "gaff"
    # APBS
    apbs_grid_spacing: float = 0.5  # Å fine grid spacing
    apbs_coarse_spacing: float = 2.0  # Å coarse grid spacing
    # BD simulation
    n_trajectories: int = 10000
    n_threads: int = 24
    gpu: bool = True
    seed: int = 1523
    confidence_interval: float = 0.95
    # Milestone / b-surface
    # b-surface = b-surface
    # Reaction criterion: GHO-GHO distance < bd_milestone_radius (b-surface)
    # Escape sphere: 2 × bd_milestone_radius
    bd_milestone_radius: float = 30.0  # Å - b-sphere start (>= 3×(r_rec+r_lig))
    bd_milestone_radius_inner: float = 0.0  # Å - inner milestone (0 = disabled)
    # State-machine reactions: multi-reaction tracking per trajectory.
    # False -> all reactions flattened; first distance match ends the trajectory.
    # True  -> trajectory carries a state label; reactions fire only when the
    #          trajectory is in the matching state_before and advance it to state_after.
    state_machine_reactions: bool = False
    # Ghost atoms
    # 'auto'  -> detect GHO atoms from PQR automatically
    # or list of [rec_idx, lig_idx, cutoff_ang] triplets
    ghost_atoms: str = "auto"
    rxns_xml: str = ""  # path to the rxns XML for auto GHO injection
    receptor_pqr: str = ""  # pre-computed receptor PQR - skips AmberTools+tleap
    ligand_pqr: str = ""  # pre-computed ligand PQR  - skips AmberTools+tleap
    desolvation_alpha: float = (
        0.07957747  # Born desolvation parameter (1/(4*pi) ≈ 0.0796)
    )
    max_steps: int = 1000000  # max BD steps per trajectory
    debye_length: float = 7.858  # Debye screening length Å (~150mM NaCl at 298K)
    ion_concentration: float = 0.150  # salt concentration M (150mM NaCl)
    ion_radius_pos: float = 0.95  # positive ion radius Å (Na+: 0.95 Å)
    ion_radius_neg: float = 1.81  # negative ion radius Å (Cl-: 1.81 Å)
    apbs_cglen: float = 0.0  # APBS coarse grid length Å (0=auto from formula)
    apbs_fglen: float = 0.0  # APBS fine grid length Å   (0=auto from formula)
    apbs_dime: int = (
        129  # APBS grid dimension (129=standard, 257=high-res for large proteins)
    )
    apbs_coarse_dime: int = 0  # APBS coarse grid dime (0=use global apbs_dime)
    apbs_fine_dime: int = 0  # APBS fine grid dime   (0=use global apbs_dime)
    gpu_force_batch: int = (
        0  # trajectories per GPU force batch (0=auto from ligand size & 4GB target)
    )
    pdie: float = 4.0  # solute dielectric (standard protein interior)
    sdie: float = 78.0  # solvent dielectric (water at 298K)
    srad: float = 1.5  # solvent probe radius Å (standard water probe)
    temperature: float = 298.15  # temperature K
    dt: float = 0.2  # ps max time step (reference minimum_core_dt)
    hydrodynamic_interactions: bool = False  # hydrodynamic_interactions flag
    r_hydro_rec: float = 0.0  # receptor hydro radius (0=compute from PQR)
    r_hydro_lig: float = 0.0  # ligand hydro radius   (0=compute from PQR)
    minimum_core_dt: float = 0.0  # minimum_core_dt (0=no floor)
    minimum_core_reaction_dt: float = (
        0.0  # dt floor near reaction surface (0=no floor; SEEKR2 default 0.05)
    )
    max_dt: float = 0.0  # max_dt ceiling (0=no cap)
    # Physics extensions
    overlap_check: bool = True  # prevent ligand entering receptor volume
    multipole_fallback: bool = True  # dipole+quadrupole far-field (beyond APBS grid)
    lj_forces: bool = False  # WCA repulsive forces from PQR radii (for tight contact)
    enable_born2_torque: bool = True  # BORN2 reciprocal torque (lig Born on rec atoms; ~3% of total torque, ~50x slower for proteins)
    # Checkpointing & convergence
    checkpoint_interval: int = 0  # save checkpoint every N completed traj (0=disabled)
    convergence_interval: int = 10  # print live k_on every N% completion
    convergence_check: bool = True  # run convergence analysis after BD completes
    convergence_tol: float = 0.05  # relative SE threshold (0.05 = 5%)
    # Outputs
    outputs: OutputConfig = None
    # Chain BD configuration. When present (not None), the simulation
    # runs in chain BD mode instead of rigid-body. See ChainConfig above.
    chain: Optional[ChainConfig] = None

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = OutputConfig()
        # pdb is optional when receptor_pqr + ligand_pqr are both provided
        if self.pdb is not None:
            self.pdb = Path(self.pdb)
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Check required inputs exist and values are sane."""
        if self.chain is not None:
            # Chain BD mode: chain_json (topology) and receptor_pqr (target)
            # are the required inputs. ligand_pqr is not used because the
            # flexible chain plays the role of the ligand.
            if not self.chain.chain_json:
                raise ValueError("<chain><chain_json> is required in chain BD mode")
            if not self.receptor_pqr:
                raise ValueError(
                    "<receptor_pqr> is required in chain BD mode (target structure)"
                )
            # Sanity-check chain BD numerics: bad values silently produce
            # garbage results (e.g. dt_chain=0 -> zero noise -> static chain;
            # gb_eps_in=0 -> divide-by-zero in OBC2 self-Born; negative
            # diffusion -> NaN positions). Catch them at parse time.
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
        return self


def parse(xml_path: str | Path) -> PySTARCConfig:
    """
    Parse a pystarc_input.xml file and return a validated PySTARCConfig.
    Example XML::

        <?xml version="1.0" ?>
        <pystarc_input>
            <pdb>hostguest.pdb</pdb>
            <ligand_resname>BEN</ligand_resname>
            <ligand_charge>1</ligand_charge>
            <work_dir>bd_sims/</work_dir>
            <protein_ff>ff14SB</protein_ff>
            <ligand_ff>gaff</ligand_ff>
            <apbs_grid_spacing>0.5</apbs_grid_spacing>
            <apbs_coarse_spacing>2.0</apbs_coarse_spacing>
            <n_trajectories>10000</n_trajectories>
            <n_threads>24</n_threads>
            <gpu>true</gpu>
            <seed>1523</seed>
            <!-- b-surface radius = b-surface radius (Å) -->
            <!-- Reaction criterion: GHO-GHO distance < bd_milestone_radius (b-surface) -->
            <!-- Escape sphere: 2 x bd_milestone_radius -->
            <bd_milestone_radius>30.0</bd_milestone_radius>
            <confidence_interval>0.95</confidence_interval>
            <ghost_atoms>auto</ghost_atoms>
        </pystarc_input>
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get(tag, default=None, cast=str):
        el = root.find(tag)
        if el is None or el.text is None:
            return default
        text = el.text.strip()
        if cast is bool:
            return text.lower() in ("true", "1", "yes")
        return cast(text)

    cfg = PySTARCConfig(
        pdb=get("pdb", cast=str),
        desolvation_alpha=get("desolvation_alpha", cast=float, default=0.07957747),
        debye_length=get("debye_length", cast=float, default=7.858),
        max_steps=get("max_steps", cast=int, default=1000000),
        ligand_resname=get("ligand_resname", cast=str),
        ligand_charge=get("ligand_charge", default=0, cast=int),
        work_dir=get("work_dir", default="bd_sims", cast=str),
        protein_ff=get("protein_ff", default="ff14SB", cast=str),
        ligand_ff=get("ligand_ff", default="gaff", cast=str),
        apbs_grid_spacing=get("apbs_grid_spacing", default=0.5, cast=float),
        apbs_coarse_spacing=get("apbs_coarse_spacing", default=2.0, cast=float),
        n_trajectories=get("n_trajectories", default=10000, cast=int),
        n_threads=get("n_threads", default=24, cast=int),
        gpu=get("gpu", default=True, cast=bool),
        seed=get("seed", default=1523, cast=int),
        confidence_interval=get("confidence_interval", default=0.95, cast=float),
        bd_milestone_radius=get("bd_milestone_radius", default=30.0, cast=float),
        bd_milestone_radius_inner=get(
            "bd_milestone_radius_inner", default=0.0, cast=float
        ),
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
        gpu_force_batch=get("gpu_force_batch", default=0, cast=int),
        pdie=get("pdie", default=4.0, cast=float),
        sdie=get("sdie", default=78.0, cast=float),
        srad=get("srad", default=1.5, cast=float),
        temperature=get("temperature", default=298.15, cast=float),
        dt=get("dt", default=0.2, cast=float),
        hydrodynamic_interactions=get(
            "hydrodynamic_interactions", default=False, cast=bool
        ),
        r_hydro_rec=get("r_hydro_rec", default=0.0, cast=float),
        r_hydro_lig=get("r_hydro_lig", default=0.0, cast=float),
        minimum_core_dt=get("minimum_core_dt", default=0.0, cast=float),
        minimum_core_reaction_dt=get(
            "minimum_core_reaction_dt", default=0.0, cast=float
        ),
        max_dt=get("max_dt", default=0.0, cast=float),
        overlap_check=get("overlap_check", default=True, cast=bool),
        multipole_fallback=get("multipole_fallback", default=True, cast=bool),
        lj_forces=get("lj_forces", default=False, cast=bool),
        enable_born2_torque=get("enable_born2_torque", default=False, cast=bool),
        checkpoint_interval=get("checkpoint_interval", default=0, cast=int),
        convergence_interval=get("convergence_interval", default=10, cast=int),
        convergence_check=get("convergence_check", default=True, cast=bool),
        convergence_tol=get("convergence_tol", default=0.05, cast=float),
    )
    # Parse <outputs> block
    out_el = root.find("outputs")
    if out_el is not None:

        def oget(tag, default=True, cast=bool):
            el = out_el.find(tag)
            if el is None or el.text is None:
                return default
            text = el.text.strip()
            if cast is bool:
                return text.lower() in ("true", "1", "yes")
            return cast(text)

        cfg.outputs = OutputConfig(
            results_json=oget("results_json", True, bool),
            trajectories_csv=oget("trajectories_csv", True, bool),
            encounters_csv=oget("encounters_csv", True, bool),
            near_misses_csv=oget("near_misses_csv", True, bool),
            full_paths=oget("full_paths", True, bool),
            radial_density=oget("radial_density", True, bool),
            angular_map=oget("angular_map", True, bool),
            fpt_distribution=oget("fpt_distribution", True, bool),
            contact_frequency=oget("contact_frequency", True, bool),
            milestone_flux=oget("milestone_flux", True, bool),
            p_commit=oget("p_commit", True, bool),
            transition_matrix=oget("transition_matrix", True, bool),
            energetics=oget("energetics", True, bool),
            pose_clusters=oget("pose_clusters", True, bool),
            save_interval=oget("save_interval", 10, int),
        )
    # Parse <chain> block (chain BD mode). When present, dispatches the
    # simulation to the chain BD pipeline. Optional and additive:
    # rigid-body input.xml files keep cfg.chain = None unchanged.
    chain_el = root.find("chain")
    if chain_el is not None:

        def cget(tag, default=None, cast=str):
            el = chain_el.find(tag)
            if el is None or el.text is None:
                return default
            text = el.text.strip()
            if cast is bool:
                return text.lower() in ("true", "1", "yes")
            return cast(text)

        cfg.chain = ChainConfig(
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
    return cfg.validate()


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

    <!-- APBS electrostatics -->

    <!-- Fine grid spacing in Angstrom (default 0.5) -->
    <apbs_grid_spacing>0.5</apbs_grid_spacing>

    <!-- Coarse grid spacing in Angstrom (default 2.0) -->
    <apbs_coarse_spacing>2.0</apbs_coarse_spacing>

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

    <!-- Ghost atoms -->
    <!--
    auto   = detect GHO atoms from PQR files automatically
             (looks for atoms with name GHO or radius=0)

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
