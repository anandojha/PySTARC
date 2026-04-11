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
    results_json:      bool = True   # results.json - k_on, P_rxn, CI, etc.
    trajectories_csv:  bool = True   # trajectories.csv - per-trajectory outcomes
    encounters_csv:    bool = True   # encounters.csv - pos/orientation at reaction
    near_misses_csv:   bool = True   # near_misses.csv - pos/orientation at closest approach
    full_paths:        bool = True   # paths.npz - pos/q every save_interval steps
    radial_density:    bool = True   # radial_density.csv - ρ(r) histogram
    angular_map:       bool = True   # angular_map.npz - (θ, φ) occupancy heatmap
    fpt_distribution:  bool = True   # fpt_distribution.csv - first passage times
    contact_frequency: bool = True   # contact_frequency.csv - pair contact counts
    milestone_flux:    bool = True   # milestone_flux.csv - flux through concentric shells
    p_commit:          bool = True   # p_commit.npz - commitment probability map
    transition_matrix: bool = True   # transition_matrix.npz - Markov state transitions
    energetics:        bool = True   # energetics.npz - force/energy traces
    pose_clusters:     bool = True   # pose_clusters.csv - clustered encounter orientations
    save_interval:     int  = 10     # record full paths every N steps (0 = endpoints only)

@dataclass
class PySTARCConfig:
    # System 
    pdb:              Path = None  # input PDB - optional when receptor_pqr+ligand_pqr provided
    ligand_resname:   str  = ""    # residue name of ligand, e.g. 'BEN'
    ligand_charge:    int  = 0     # net formal charge of ligand
    work_dir:         Path = Path("bd_sims")  # all output files written here
    # Force field 
    protein_ff:       str = "ff14SB"
    ligand_ff:        str = "gaff"
    # APBS
    apbs_grid_spacing:   float = 0.5    # Å fine grid spacing
    apbs_coarse_spacing: float = 2.0    # Å coarse grid spacing
    # BD simulation 
    n_trajectories:      int   = 10000
    n_threads:           int   = 24
    gpu:                 bool  = True
    seed:                int   = 1523
    confidence_interval: float = 0.95
    # Milestone / b-surface 
    # b-surface = b-surface 
    # Reaction criterion: GHO-GHO distance < bd_milestone_radius (b-surface)
    # Escape sphere: 2 × bd_milestone_radius
    bd_milestone_radius:       float = 30.0   # Å - b-sphere start (>= 3×(r_rec+r_lig))
    bd_milestone_radius_inner: float = 0.0    # Å - inner milestone (0 = disabled)
    # Ghost atoms 
    # 'auto'  -> detect GHO atoms from PQR automatically
    # or list of [rec_idx, lig_idx, cutoff_ang] triplets
    ghost_atoms:  str = "auto"
    rxns_xml:            str   = ""           # path to the rxns XML for auto GHO injection
    receptor_pqr:        str   = ""           # pre-computed receptor PQR - skips AmberTools+tleap
    ligand_pqr:          str   = ""           # pre-computed ligand PQR  - skips AmberTools+tleap
    desolvation_alpha:   float = 0.07957747   # Born desolvation parameter (1/(4*pi) ≈ 0.0796)
    max_steps:           int   = 1000000      # max BD steps per trajectory
    debye_length:        float = 7.858        # Debye screening length Å (~150mM NaCl at 298K)
    ion_concentration:   float = 0.150        # salt concentration M (150mM NaCl)
    ion_radius_pos:      float = 0.95         # positive ion radius Å (Na+: 0.95 Å)
    ion_radius_neg:      float = 1.81         # negative ion radius Å (Cl-: 1.81 Å)
    apbs_cglen:          float = 0.0          # APBS coarse grid length Å (0=auto from formula)
    apbs_fglen:          float = 0.0          # APBS fine grid length Å   (0=auto from formula)
    apbs_dime:           int   = 129          # APBS grid dimension (129=standard, 257=high-res for large proteins)
    apbs_coarse_dime:    int   = 0            # APBS coarse grid dime (0=use global apbs_dime)
    apbs_fine_dime:      int   = 0            # APBS fine grid dime   (0=use global apbs_dime)
    gpu_force_batch:     int   = 0            # trajectories per GPU force batch (0=auto from ligand size & 4GB target)
    pdie:                float = 4.0          # solute dielectric (standard protein interior)
    sdie:                float = 78.0         # solvent dielectric (water at 298K)
    srad:                float = 1.5          # solvent probe radius Å (standard water probe)
    temperature:         float = 298.15       # temperature K
    dt:                  float = 0.2          # ps max time step (reference minimum_core_dt)
    hydrodynamic_interactions: bool = False   # hydrodynamic_interactions flag
    r_hydro_rec:         float = 0.0          # receptor hydro radius (0=compute from PQR)
    r_hydro_lig:         float = 0.0          # ligand hydro radius   (0=compute from PQR)
    minimum_core_dt:     float = 0.0          # minimum_core_dt (0=no floor)
    # Physics extensions 
    overlap_check:       bool  = True         # prevent ligand entering receptor volume
    multipole_fallback:  bool  = True         # dipole+quadrupole far-field (beyond APBS grid)
    lj_forces:           bool  = False        # WCA repulsive forces from PQR radii (for tight contact)
    # Checkpointing & convergence 
    checkpoint_interval: int   = 0           # save checkpoint every N completed traj (0=disabled)
    convergence_interval: int  = 10          # print live k_on every N% completion
    convergence_check:   bool  = True        # run convergence analysis after BD completes
    convergence_tol:     float = 0.05        # relative SE threshold (0.05 = 5%)
    # Outputs
    outputs:             OutputConfig = None

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
            return text.lower() in ('true', '1', 'yes')
        return cast(text)

    cfg = PySTARCConfig(
        pdb              = get('pdb',               cast=str),
        desolvation_alpha= get('desolvation_alpha', cast=float, default=0.07957747),
        debye_length     = get('debye_length',      cast=float, default=7.858),
        max_steps        = get('max_steps',         cast=int,   default=1000000),
        ligand_resname   = get('ligand_resname',    cast=str),
        ligand_charge    = get('ligand_charge',     default=0,    cast=int),
        work_dir         = get('work_dir',          default='bd_sims', cast=str),
        protein_ff       = get('protein_ff',        default='ff14SB', cast=str),
        ligand_ff        = get('ligand_ff',         default='gaff',   cast=str),
        apbs_grid_spacing   = get('apbs_grid_spacing',   default=0.5,  cast=float),
        apbs_coarse_spacing = get('apbs_coarse_spacing', default=2.0,  cast=float),
        n_trajectories   = get('n_trajectories',   default=10000, cast=int),
        n_threads        = get('n_threads',        default=24,    cast=int),
        gpu              = get('gpu',              default=True,  cast=bool),
        seed             = get('seed',             default=1523,  cast=int),
        confidence_interval = get('confidence_interval', default=0.95, cast=float),
        bd_milestone_radius       = get('bd_milestone_radius',       default=30.0, cast=float),
        bd_milestone_radius_inner = get('bd_milestone_radius_inner', default=0.0,  cast=float),
        ghost_atoms      = get('ghost_atoms',      default='auto', cast=str),
        rxns_xml         = get('rxns_xml',         default='',     cast=str),
        receptor_pqr     = get('receptor_pqr',     default='',     cast=str),
        ligand_pqr       = get('ligand_pqr',       default='',     cast=str),
        ion_concentration= get('ion_concentration', default=0.150,  cast=float),
        ion_radius_pos   = get('ion_radius_pos',   default=0.95,   cast=float),
        ion_radius_neg   = get('ion_radius_neg',   default=1.81,   cast=float),
        apbs_cglen       = get('apbs_cglen',        default=0.0,    cast=float),
        apbs_fglen       = get('apbs_fglen',        default=0.0,    cast=float),
        apbs_dime        = get('apbs_dime',          default=129,    cast=int),
        apbs_coarse_dime = get('apbs_coarse_dime',    default=0,      cast=int),
        apbs_fine_dime   = get('apbs_fine_dime',      default=0,      cast=int),
        gpu_force_batch  = get('gpu_force_batch',     default=0,      cast=int),
        pdie             = get('pdie',              default=4.0,    cast=float),
        sdie             = get('sdie',              default=78.0,   cast=float),
        srad             = get('srad',              default=1.5,    cast=float),
        temperature      = get('temperature',       default=298.15, cast=float),
        dt               = get('dt',               default=0.2,    cast=float),
        hydrodynamic_interactions = get('hydrodynamic_interactions', default=False, cast=bool),
        r_hydro_rec      = get('r_hydro_rec',       default=0.0,    cast=float),
        r_hydro_lig      = get('r_hydro_lig',       default=0.0,    cast=float),
        minimum_core_dt  = get('minimum_core_dt',   default=0.0,    cast=float),
        overlap_check    = get('overlap_check',     default=True,   cast=bool),
        multipole_fallback = get('multipole_fallback', default=True, cast=bool),
        lj_forces        = get('lj_forces',          default=False,  cast=bool),
        checkpoint_interval  = get('checkpoint_interval',  default=0,   cast=int),
        convergence_interval = get('convergence_interval', default=10,  cast=int),
        convergence_check    = get('convergence_check',    default=True, cast=bool),
        convergence_tol      = get('convergence_tol',      default=0.05, cast=float),
    )
    # Parse <outputs> block 
    out_el = root.find('outputs')
    if out_el is not None:
        def oget(tag, default=True, cast=bool):
            el = out_el.find(tag)
            if el is None or el.text is None:
                return default
            text = el.text.strip()
            if cast is bool:
                return text.lower() in ('true', '1', 'yes')
            return cast(text)
        cfg.outputs = OutputConfig(
            results_json      = oget('results_json',      True,  bool),
            trajectories_csv  = oget('trajectories_csv',  True,  bool),
            encounters_csv    = oget('encounters_csv',    True,  bool),
            near_misses_csv   = oget('near_misses_csv',   True,  bool),
            full_paths        = oget('full_paths',        True,  bool),
            radial_density    = oget('radial_density',    True,  bool),
            angular_map       = oget('angular_map',       True,  bool),
            fpt_distribution  = oget('fpt_distribution',  True,  bool),
            contact_frequency = oget('contact_frequency', True,  bool),
            milestone_flux    = oget('milestone_flux',    True,  bool),
            p_commit          = oget('p_commit',          True,  bool),
            transition_matrix = oget('transition_matrix', True,  bool),
            energetics        = oget('energetics',        True,  bool),
            pose_clusters     = oget('pose_clusters',     True,  bool),
            save_interval     = oget('save_interval',     10,    int),
        )
    return cfg.validate()

def write_template(path: str | Path = "pystarc_input.xml"):
    """Write a fully commented template input file."""
    template = '''<?xml version="1.0" ?>
<!--
  PySTARC Input File
  ================
  One command:  python run_pystarc.py pystarc_input.xml
  Output:       k_on + 95% CI printed to terminal
-->
<pystarc_input>

    <!-- -- System ---------------------------------------- -->

    <!-- Path to PDB file containing protein + ligand together -->
    <pdb>hostguest.pdb</pdb>

    <!-- Residue name of the ligand in the PDB (3-letter code) -->
    <ligand_resname>BEN</ligand_resname>

    <!-- Net formal charge of the ligand (used by antechamber) -->
    <ligand_charge>1</ligand_charge>

    <!-- Directory where all output files will be written -->
    <work_dir>bd_sims/</work_dir>

    <!-- -- Force field ------------------------------------ -->

    <!-- Protein force field: ff14SB, ff19SB, ff03 -->
    <protein_ff>ff14SB</protein_ff>

    <!-- Ligand force field: gaff, gaff2 -->
    <ligand_ff>gaff</ligand_ff>

    <!-- -- APBS electrostatics ---------------------------- -->

    <!-- Fine grid spacing in Angstrom (default 0.5) -->
    <apbs_grid_spacing>0.5</apbs_grid_spacing>

    <!-- Coarse grid spacing in Angstrom (default 2.0) -->
    <apbs_coarse_spacing>2.0</apbs_coarse_spacing>

    <!-- -- BD simulation ---------------------------------- -->

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

    <!-- -- Ghost atoms ------------------------------------- -->
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

</pystarc_input>
'''
    Path(path).write_text(template)
    print(f"Template written to: {path}")