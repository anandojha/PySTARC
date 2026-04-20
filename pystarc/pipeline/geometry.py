"""
Molecular geometry and diffusion parameters
============================================

Background
-------------------
This module computes the geometric and diffusive properties needed
for the BD simulation from the PQR file data.

1. b-Surface radius
The b-surface is a sphere of radius b centered on the receptor,
from which all BD trajectories are launched.  Its radius is chosen
so that the interaction potential is approximately centrosymmetric:

    b = max_receptor_extent + max_ligand_extent + padding

Typically b ≈ 3-5 × molecular_radius.

2. Escape sphere
The escape sphere at r_esc = 2b is where the outer propagator
decides return vs escape.  Trajectories that reach r_esc either
return to the b-surface (with probability p_return) or are
terminated as escapes.

3. Diffusion coefficients
From the Stokes-Einstein relation:
    D_trans = kBT / (6π η a)    [Å²/ps]
    D_rot   = 3 D_trans / (4a²) [rad²/ps]
    D_rel   = D_trans,1 + D_trans,2  (relative diffusion)

The RMS displacement per step is √(6 D_rel Δt).
For Δt = 1 ps: Δr_rms = √(6 × 0.053) ≈ 0.56 Å.
"""

from __future__ import annotations
from pystarc.hydrodynamics.mc_hydro_radius import mc_hydrodynamic_radius
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class AtomRecord:
    index: int
    name: str
    resname: str
    resid: int
    x: float
    y: float
    z: float
    charge: float
    radius: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def is_ghost(self) -> bool:
        return (
            self.name.strip().upper() == "GHO"
            or self.radius < 1e-6
            or abs(self.charge) < 1e-9
            and self.radius < 1e-6
        )


@dataclass
class MoleculeGeometry:
    n_atoms: int
    n_charged: int
    n_ghost: int
    centroid: np.ndarray
    max_radius: float  # max distance from centroid to atom surface
    hydrodynamic_r: float  # hydrodynamic radius (= max_radius for rigid body)
    ghost_indices: List[int]  # 0-based indices of ghost atoms
    ghost_positions: List[np.ndarray]
    total_charge: float


def parse_pqr(pqr_path: Path) -> List[AtomRecord]:
    """Parse a PQR file and return list of AtomRecord.

    Delegates to the canonical PQR parser in pystarc.structures.pqr_io,
    which handles the full range of PQR format variations (ATOM/HETATM,
    chain column present or absent, 4-char Amber resnames, collapsed
    numeric spacing, trailing element column).

    Legacy fallback: if the canonical parser rejects every line (for
    example a minimal PQR with only nine fields per line, no radius
    column), this function retries with a lenient whitespace parse
    that defaults the missing radius to 1.5 Angstrom, preserving
    prior geometry-module behavior.
    """
    from pystarc.structures.pqr_io import parse_pqr_records
    records = parse_pqr_records(pqr_path)
    if records:
        return [
            AtomRecord(
                index=i,
                name=r.name,
                resname=r.resname,
                resid=r.resid,
                x=r.x,
                y=r.y,
                z=r.z,
                charge=r.charge,
                radius=r.radius,
            )
            for i, r in enumerate(records)
        ]
    # Lenient fallback for legacy PQRs missing the radius column.
    atoms: List[AtomRecord] = []
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                atoms.append(
                    AtomRecord(
                        index=len(atoms),
                        name=parts[2],
                        resname=parts[3],
                        resid=int(parts[4]),
                        x=float(parts[5]),
                        y=float(parts[6]),
                        z=float(parts[7]),
                        charge=float(parts[8]),
                        radius=float(parts[9]) if len(parts) > 9 else 1.5,
                    )
                )
            except (ValueError, IndexError):
                continue
    return atoms


def analyse_molecule(
    pqr_path: Path,
    use_mc_hydro: bool = True,
    grid_spacing: float = None,
    n_mc: int = 1_000_000,
    srad: float = 0.0,
) -> MoleculeGeometry:
    """
    Compute geometric properties of a molecule from its PQR file.

    Hydrodynamic radius uses the Hansen (J. Chem. Phys. 121, 9111, 2004)
    Employs the solvent-excluded surface (SES) with probe_radius=srad
    and grid_spacing=1.0Å.
    Effective radii = atom_radius + srad before voxelisation.
    Parameters
    ----------
    use_mc_hydro : if True (default), use MC algorithm (the reference implementation-exact).
                   if False, use geometric approximation (fast, ~35% error).
    grid_spacing : voxel grid spacing in Å. Defaults to 1.0Å matching reference.
    n_mc         : MC sample count (default 1_000_000, matches the reference implementation).
    srad         : solvent probe radius Å (reference default 1.5; 0.0 for two_spheres).
    """
    atoms = parse_pqr(pqr_path)
    if not atoms:
        raise ValueError(f"No atoms found in {pqr_path}")
    coords = np.array([[a.x, a.y, a.z] for a in atoms])
    radii = np.array([a.radius for a in atoms])
    # b-sphere (max_radius): always geometric - used for BD setup, not Stokes-Einstein
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    max_radius = float(np.max(dists + radii))
    # Grid spacing: uses spacing=1.0Å for large proteins,
    # but adapts for small molecules (bbox/100 capped to [0.02, 1.0]).
    if grid_spacing is None:
        radii_bbox = radii + srad if srad > 0.0 else radii
        bbox = float(
            np.max(
                np.max(coords + radii_bbox[:, None], axis=0)
                - np.min(coords - radii_bbox[:, None], axis=0)
            )
        )
        grid_spacing = max(0.02, min(1.0, bbox / 100.0))
    # Solvent-excluded surface: effective radius = vdW + probe
    radii_hydro = radii + srad if srad > 0.0 else radii
    # Hydrodynamic radius: MC algorithm or geometric fallback
    # Cache result next to PQR file so re-runs skip the expensive MC calculation.
    cache_path = Path(str(pqr_path) + f".r_hydro_s{grid_spacing}_p{srad:.4g}.cache")
    if use_mc_hydro:
        if cache_path.exists():
            try:
                cached = cache_path.read_text().strip().split()
                r_h = float(cached[0])
                centroid = np.array([float(x) for x in cached[1:4]])
                print(f"    r_hydro cache hit: {cache_path.name}")
            except Exception:
                cache_path.unlink(missing_ok=True)
                r_h = None
        else:
            r_h = None

        if r_h is None:
            try:
                r_h, mc_center, _ = mc_hydrodynamic_radius(
                    coords, radii_hydro, spacing=grid_spacing, n_mc=n_mc
                )
                centroid = mc_center
                # Save to cache
                try:
                    line = f"{r_h:.8f} {centroid[0]:.8f} {centroid[1]:.8f} {centroid[2]:.8f}\n"
                    cache_path.write_text(line)
                    print(f"    r_hydro cached: {cache_path.name}")
                except Exception as e:
                    print(f"    r_hydro cache write failed: {e}")
            except Exception:
                r_h = max_radius
    else:
        r_h = max_radius  # geometric approximation
    ghost_idx = [a.index for a in atoms if a.is_ghost]
    ghost_pos = [a.pos for a in atoms if a.is_ghost]
    return MoleculeGeometry(
        n_atoms=len(atoms),
        n_charged=sum(1 for a in atoms if abs(a.charge) > 1e-9),
        n_ghost=len(ghost_idx),
        centroid=centroid,
        max_radius=max_radius,
        hydrodynamic_r=r_h,
        ghost_indices=ghost_idx,
        ghost_positions=ghost_pos,
        total_charge=float(sum(a.charge for a in atoms)),
    )


@dataclass
class SystemGeometry:
    receptor: MoleculeGeometry
    ligand: MoleculeGeometry
    r_start: float  # b-sphere radius (Å)
    r_escape: float  # escape sphere (2 × b-sphere)


def compute_geometry(
    receptor_pqr: Path,
    ligand_pqr: Path,
    bd_milestone_radius: float = 13.0,
    bd_milestone_radius_inner: float = 12.0,
    srad: float = 0.0,
    r_hydro_rec: float = 0.0,
    r_hydro_lig: float = 0.0,
) -> SystemGeometry:
    """
    Compute full system geometry for BD setup.
    b-sphere = bd_milestone_radius (outermost SEEKR milestone, user-defined)
    escape   = 2 × b-sphere  (Luty-McCammon-Zhou convention)
    If r_hydro_rec or r_hydro_lig are > 0, they override the MC-computed
    hydrodynamic radii (matches reference hydro_params.xml values exactly).
    """
    print("\n[5] Computing system geometry ...")
    rec = analyse_molecule(receptor_pqr, srad=srad)
    lig = analyse_molecule(ligand_pqr, srad=srad)
    if r_hydro_rec > 0:
        print(
            f"    r_hydro receptor override: {rec.hydrodynamic_r:.3f} -> {r_hydro_rec:.4f} Å (from XML)"
        )
        rec = MoleculeGeometry(
            n_atoms=rec.n_atoms,
            n_charged=rec.n_charged,
            n_ghost=rec.n_ghost,
            centroid=rec.centroid,
            max_radius=rec.max_radius,
            hydrodynamic_r=r_hydro_rec,
            ghost_indices=rec.ghost_indices,
            ghost_positions=rec.ghost_positions,
            total_charge=rec.total_charge,
        )
    if r_hydro_lig > 0:
        print(
            f"    r_hydro ligand override: {lig.hydrodynamic_r:.3f} -> {r_hydro_lig:.4f} Å (from XML)"
        )
        lig = MoleculeGeometry(
            n_atoms=lig.n_atoms,
            n_charged=lig.n_charged,
            n_ghost=lig.n_ghost,
            centroid=lig.centroid,
            max_radius=lig.max_radius,
            hydrodynamic_r=r_hydro_lig,
            ghost_indices=lig.ghost_indices,
            ghost_positions=lig.ghost_positions,
            total_charge=lig.total_charge,
        )
    r_start = bd_milestone_radius
    r_escape = 2.0 * r_start
    print(
        f"  Receptor : {rec.n_atoms:5d} atoms  q={rec.total_charge:+.2f} e  "
        f"r_hydro={rec.hydrodynamic_r:.3f} Å  "
        f"ghost={rec.n_ghost}"
    )
    print(
        f"  Ligand   : {lig.n_atoms:5d} atoms  q={lig.total_charge:+.2f} e  "
        f"r_hydro={lig.hydrodynamic_r:.3f} Å  "
        f"ghost={lig.n_ghost}"
    )
    print(f"  b-surface (milestone) : {r_start:.1f} Å")
    print(f"  Escape sphere         : {r_escape:.1f} Å  (= 2 × b-surface)")
    return SystemGeometry(
        receptor=rec,
        ligand=lig,
        r_start=r_start,
        r_escape=r_escape,
    )


# Ghost atom / reaction criteria detection
@dataclass
class ReactionPair:
    rec_index: int  # 0-based atom index in receptor
    lig_index: int  # 0-based atom index in ligand
    cutoff: float  # distance cutoff in Å


def _parse_rxns_xml_criteria(rxns_path):
    """
    Parse the rxns XML file and extract reaction pair criteria.
    Supports two formats:
    1. <atom1>rec_idx charge cutoff</atom1> <atom2>lig_idx...</atom2>
    2. <atoms>rec_idx lig_idx</atoms> <distance>cutoff</distance>
    Note: atom indices in the XML are 1-based.
    They are stored as-is and the simulator uses them directly as 0-based
    after subtracting 1 during ContactPair construction.
    """
    pairs = []
    n_needed = -1  # -1 = all pairs (reference default)
    try:
        tree = ET.parse(str(rxns_path))
        root = tree.getroot()
        for reaction in root.iter("reaction"):
            crit = reaction.find("criterion")
            if crit is None:
                continue
            # Read n_needed if present
            nn_node = crit.find("n_needed")
            if nn_node is not None:
                try:
                    n_needed = int(nn_node.text.strip())
                except ValueError:
                    pass
            for pair_node in crit.findall("pair"):
                # Format 1: <atom1>rec_idx charge cutoff</atom1> <atom2>lig_idx...</atom2>
                a1 = pair_node.find("atom1")
                a2 = pair_node.find("atom2")
                if a1 is not None and a2 is not None:
                    try:
                        p1 = a1.text.strip().split()
                        p2 = a2.text.strip().split()
                        rec_idx = int(p1[0]) - 1  # convert 1-based -> 0-based
                        lig_idx = int(p2[0]) - 1
                        cutoff = float(p1[2]) if len(p1) >= 3 else 5.0
                        pairs.append(
                            ReactionPair(
                                rec_index=rec_idx,
                                lig_index=lig_idx,
                                cutoff=cutoff,
                            )
                        )
                    except (ValueError, IndexError):
                        continue
                    continue
                # Format 2: <atoms>rec_idx lig_idx</atoms> <distance>cutoff</distance>
                atoms_node = pair_node.find("atoms")
                distance_node = pair_node.find("distance")
                if atoms_node is not None and distance_node is not None:
                    try:
                        idx = atoms_node.text.strip().split()
                        rec_idx = int(idx[0]) - 1  # convert 1-based -> 0-based
                        lig_idx = int(idx[1]) - 1
                        cutoff = float(distance_node.text.strip())
                        pairs.append(
                            ReactionPair(
                                rec_index=rec_idx,
                                lig_index=lig_idx,
                                cutoff=cutoff,
                            )
                        )
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"  Warning: could not parse rxns XML {rxns_path}: {e}")
    return pairs, n_needed


def auto_detect_reactions(
    geom: "SystemGeometry",
    ghost_atoms: str = "auto",
    rxns_xml: str = "",
    bd_milestone_radius: float = 13.0,
    bd_milestone_radius_inner: float = 12.0,
) -> "List[List[ReactionPair]]":
    """
    Build reaction criteria from GHO ghost atoms.
    1. rxns_xml given  -> parse criteria from the reference implementation rxns file
    2. ghost_atoms manual spec -> parse triplets rec_idx,lig_idx,cutoff
    3. ghost_atoms == 'auto' -> detect GHO atoms in PQR, use bd_milestone_radius as cutoff
    4. No GHO found -> raise clear error (centroid fallback removed - physically wrong)
    """
    # Priority 1: rxns XML
    if rxns_xml and rxns_xml.strip():
        rxns_path = Path(rxns_xml.strip())
        # If relative, try resolving against the PDB parent directory
        if not rxns_path.is_absolute() and not rxns_path.exists():
            # Try relative to cwd - already the common case
            pass
        if rxns_path.exists():
            pairs, n_needed = _parse_rxns_xml_criteria(rxns_path)
            if pairs:
                nn_str = str(n_needed) if n_needed > 0 else f"all ({len(pairs)})"
                print(
                    f"  GHO criteria from rxns XML ({rxns_path.name}): "
                    f"{len(pairs)} pair(s), n_needed={nn_str}"
                )
                for p in pairs:
                    print(
                        f"    rec[{p.rec_index}] -- lig[{p.lig_index}] < {p.cutoff:.1f} A"
                    )
                return [pairs], n_needed
            print(f"  Warning: no pairs in {rxns_path.name}, falling back")
        else:
            print(f"  Warning: rxns_xml not found: {rxns_xml}")
    # Priority 2: manual ghost_atoms spec
    if ghost_atoms.strip().lower() != "auto":
        pairs = []
        for line in ghost_atoms.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(
                    f"ghost_atoms line must be 'rec_idx,lig_idx,cutoff': {line!r}"
                )
            pairs.append(
                ReactionPair(
                    rec_index=int(parts[0]),
                    lig_index=int(parts[1]),
                    cutoff=float(parts[2]),
                )
            )
        return [pairs], -1
    # Priority 3: auto-detect GHO atoms in PQR
    rec_ghosts = geom.receptor.ghost_indices
    lig_ghosts = geom.ligand.ghost_indices
    if rec_ghosts and lig_ghosts:
        # One GHO per molecule - use first GHO of each
        rec_gho = rec_ghosts[0]
        lig_gho = lig_ghosts[0]
        # Reaction criterion: GHO-GHO distance < bd_milestone_radius
        # This is the outermost milestone = b-surface radius
        # q-surface (reaction) = bd_milestone_radius_inner (inner milestone)
        # b-surface (start)    = bd_milestone_radius (outer milestone)
        # Ligand starts at b-surface and reacts when it reaches the q-surface
        rxn_cutoff = (
            bd_milestone_radius_inner
            if bd_milestone_radius_inner > 0
            else bd_milestone_radius
        )
        pairs = [ReactionPair(rec_gho, lig_gho, rxn_cutoff)]
        print(
            f"  GHO reaction criterion: rec[{rec_gho}] -- lig[{lig_gho}] "
            f"< {rxn_cutoff:.1f} A  (q-surface / inner milestone)"
        )
        return [pairs], 1
    # No GHO atoms - raise a clear error (centroid fallback removed)
    # The user must run with GHO-injected PQRs.
    raise RuntimeError(
        "\n\nNo GHO ghost atoms found in receptor.pqr or ligand.pqr.\n"
        "PySTARC requires GHO atoms to define the b-surface reaction criterion.\n"
        "GHO atoms are injected automatically during APBS preparation.\n"
        "This error should not occur in normal usage - please report it."
    )
