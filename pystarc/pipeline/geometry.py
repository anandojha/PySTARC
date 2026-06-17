"""
Molecular geometry and diffusion parameters.

This module computes the geometric and diffusive properties that the
Brownian-dynamics simulation needs, starting from the atom data in a PQR file.

The b-surface is a sphere of radius b centered on the receptor, from which all
Brownian-dynamics trajectories are launched. Its radius is chosen so that the
interaction potential is approximately centrosymmetric:

    b = max_receptor_extent + max_ligand_extent + padding

Here b is typically about 3 to 5 times the molecular radius.

The escape sphere at r_esc = 2b is where the outer propagator decides between
return and escape. A trajectory that reaches r_esc either returns to the
b-surface with probability p_return or is terminated as an escape.

The diffusion coefficients follow from the Stokes-Einstein relation. The
translational coefficient is

    D_trans = kBT / (6π η a)    in Å²/ps,

the rotational coefficient is

    D_rot = 3 D_trans / (4a²)    in rad²/ps,

and the relative translational diffusion coefficient is the sum of the two
translational coefficients,

    D_rel = D_trans,1 + D_trans,2.

The root-mean-square displacement per step is √(6 D_rel Δt). For Δt = 1 ps this
gives Δr_rms = √(6 × 0.053) ≈ 0.56 Å.
"""

from __future__ import annotations
from pystarc.hydrodynamics.mc_hydro_radius import mc_hydrodynamic_radius
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import warnings
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
        # Audit fix on 2026-05-21: the dead trailing clause
        # 'abs(self.charge) < 1e-9 and self.radius < 1e-6' was removed. Python
        # operator precedence parses it as (charge AND radius), and because
        # 'radius < 1e-6' is already the second clause, the AND clause could
        # never fire on its own. The probable original intent was a charge-only
        # fallback, which would need to be written as 'or abs(self.charge) <
        # 1e-9' to actually take effect. That is a behavior change (it would
        # flag any uncharged atom as a ghost) and needs physics validation
        # before it can be reinstated.
        return (
            self.name.strip().upper() == "GHO"
            or self.radius < 1e-6
        )


@dataclass
class MoleculeGeometry:
    n_atoms: int
    n_charged: int
    n_ghost: int
    centroid: np.ndarray
    max_radius: float  # maximum distance from the centroid to an atom surface
    hydrodynamic_r: float  # hydrodynamic radius (equals max_radius for a rigid body)
    ghost_indices: List[int]  # zero-based indices of the ghost atoms
    ghost_positions: List[np.ndarray]
    total_charge: float


def parse_pqr(pqr_path: Path) -> List[AtomRecord]:
    """Parse a PQR file and return a list of AtomRecord.

    The work is delegated to the canonical PQR parser in
    pystarc.structures.pqr_io, which handles the full range of PQR format
    variations. These include ATOM and HETATM records, the chain column being
    present or absent, four-character Amber residue names, collapsed numeric
    spacing, and a trailing element column.

    A legacy fallback is provided as well. If the canonical parser rejects every
    line, for example a minimal PQR with only nine fields per line and no radius
    column, this function retries with a lenient whitespace parse that defaults
    the missing radius to 1.5 Å, preserving the prior behavior of this module.
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
    # Lenient fallback for legacy PQR files that are missing the radius column.
    atoms: List[AtomRecord] = []
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                if len(parts) > 9:
                    radius = float(parts[9])
                else:
                    radius = 1.5
                    warnings.warn(
                        f"PQR line in {pqr_path} has no radius column; "
                        f"defaulting radius to 1.5 A for atom {parts[2]!r}. "
                        f"A wrong radius biases the hydrodynamic radius and "
                        f"hence D_trans and k_on.",
                        stacklevel=2,
                    )
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
                        radius=radius,
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
    Compute the geometric properties of a molecule from its PQR file.

    The hydrodynamic radius follows the method of Hansen (J. Chem. Phys. 121,
    9111, 2004). It uses the solvent-excluded surface with a probe radius equal
    to srad and a grid spacing of 1.0 Å. Each effective radius is the atom radius
    plus srad before voxelisation.

    The parameter use_mc_hydro selects the algorithm. When it is True, which is
    the default, the Monte Carlo algorithm is used and reproduces the reference
    implementation exactly. When it is False, a fast geometric approximation is
    used instead, with an error of roughly 35 percent. The parameter grid_spacing
    is the voxel grid spacing in Å and defaults to 1.0 Å to match the reference.
    The parameter n_mc is the number of Monte Carlo samples and defaults to
    1,000,000, again matching the reference implementation. The parameter srad is
    the solvent probe radius in Å, for which the reference default is 1.5 and the
    two-spheres case uses 0.0.
    """
    atoms = parse_pqr(pqr_path)
    if not atoms:
        raise ValueError(f"No atoms found in {pqr_path}")
    coords = np.array([[a.x, a.y, a.z] for a in atoms])
    radii = np.array([a.radius for a in atoms])
    # The b-sphere radius (max_radius) is always computed geometrically. It is
    # used to set up the Brownian dynamics, not in the Stokes-Einstein relation.
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    max_radius = float(np.max(dists + radii))
    # The grid spacing is 1.0 Å for large proteins, but it adapts for small
    # molecules to bbox/100 clamped to the range 0.02 to 1.0 Å.
    if grid_spacing is None:
        radii_bbox = radii + srad if srad > 0.0 else radii
        bbox = float(
            np.max(
                np.max(coords + radii_bbox[:, None], axis=0)
                - np.min(coords - radii_bbox[:, None], axis=0)
            )
        )
        grid_spacing = max(0.02, min(1.0, bbox / 100.0))
    # For the solvent-excluded surface, the effective radius is the van der Waals
    # radius plus the probe radius.
    radii_hydro = radii + srad if srad > 0.0 else radii
    # Compute the hydrodynamic radius, either with the Monte Carlo algorithm or
    # with the geometric fallback. The result is cached next to the PQR file so
    # that re-runs can skip the expensive Monte Carlo calculation. The cache key
    # records the grid spacing, the probe radius, the Monte Carlo sample count,
    # and a short digest of the atom coordinates and radii, so that a re-run with
    # a different sample count or an edited structure recomputes the radius
    # instead of reusing a value that no longer matches the request.
    import hashlib

    struct_bytes = (
        np.ascontiguousarray(coords, dtype=np.float64).tobytes()
        + np.ascontiguousarray(radii_hydro, dtype=np.float64).tobytes()
    )
    struct_hash = hashlib.sha1(struct_bytes).hexdigest()[:12]
    cache_path = Path(
        str(pqr_path)
        + f".r_hydro_s{grid_spacing}_p{srad:.4g}_n{n_mc}_{struct_hash}.cache"
    )
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
                # Save the result to the cache.
                try:
                    line = f"{r_h:.8f} {centroid[0]:.8f} {centroid[1]:.8f} {centroid[2]:.8f}\n"
                    cache_path.write_text(line)
                    print(f"    r_hydro cached: {cache_path.name}")
                except Exception as e:
                    print(f"    r_hydro cache write failed: {e}")
            except Exception as e:
                warnings.warn(
                    f"Monte Carlo hydrodynamic radius failed for {pqr_path} "
                    f"({type(e).__name__}: {e}); falling back to the geometric "
                    f"max_radius={max_radius:.3f} A. This fallback can bias "
                    f"D_trans and hence k_on.",
                    stacklevel=2,
                )
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
    r_start: float  # b-sphere radius in Å
    r_escape: float  # escape sphere radius, equal to 2 × the b-sphere radius


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
    Compute the full system geometry for the Brownian dynamics setup.

    The b-sphere radius is bd_milestone_radius, the outermost SEEKR milestone,
    which is user-defined. The escape sphere radius is 2 × the b-sphere radius,
    following the Luty-McCammon-Zhou convention. When r_hydro_rec or r_hydro_lig
    are greater than zero, they override the Monte Carlo hydrodynamic radii and
    match the values in the reference hydro_params.xml exactly.
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


# Detection of ghost atoms and reaction criteria.
@dataclass
class ReactionPair:
    rec_index: int  # zero-based atom index in the receptor
    lig_index: int  # zero-based atom index in the ligand
    cutoff: float  # distance cutoff in Å


def _parse_rxns_xml_criteria(rxns_path):
    """
    Parse the rxns XML file and extract the reaction pair criteria.

    Two formats are supported. The first uses an <atom1> element holding the
    receptor index, charge, and cutoff, together with an <atom2> element holding
    the ligand index. The second uses an <atoms> element holding the receptor and
    ligand indices, together with a <distance> element holding the cutoff.

    The atom indices in the XML are one-based. They are converted here by
    subtracting one so that the simulator uses them directly as zero-based indices
    during ContactPair construction.
    """
    pairs = []
    n_needed = -1  # a value of -1 means all pairs, which is the reference default
    declared_n_needed = []  # the explicit n_needed of each reaction, in order
    try:
        tree = ET.parse(str(rxns_path))
        root = tree.getroot()
        for reaction in root.iter("reaction"):
            crit = reaction.find("criterion")
            if crit is None:
                continue
            # Read n_needed for this reaction. It is scoped to the reaction and
            # defaults to -1 so that a reaction without an explicit n_needed does
            # not inherit the value parsed from a previous reaction. The previous
            # code shared a single n_needed across reactions, which mis-parsed
            # multi-reaction files. Single-reaction files are unaffected.
            rxn_n_needed = -1
            nn_node = crit.find("n_needed")
            if nn_node is not None:
                try:
                    rxn_n_needed = int(nn_node.text.strip())
                    declared_n_needed.append(rxn_n_needed)
                except ValueError:
                    pass
            # The flattened path merges all pairs into a single criterion and can
            # carry only one n_needed. Use this reaction's value, which for a
            # single-reaction file reproduces the prior behavior exactly.
            n_needed = rxn_n_needed
            for pair_node in crit.findall("pair"):
                # First format: an <atom1> element with the receptor index,
                # charge, and cutoff, and an <atom2> element with the ligand index.
                a1 = pair_node.find("atom1")
                a2 = pair_node.find("atom2")
                if a1 is not None and a2 is not None:
                    try:
                        p1 = a1.text.strip().split()
                        p2 = a2.text.strip().split()
                        rec_idx = int(p1[0]) - 1  # convert one-based to zero-based
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
                # Second format: an <atoms> element with the receptor and ligand
                # indices, and a <distance> element with the cutoff.
                atoms_node = pair_node.find("atoms")
                distance_node = pair_node.find("distance")
                if atoms_node is not None and distance_node is not None:
                    try:
                        idx = atoms_node.text.strip().split()
                        rec_idx = int(idx[0]) - 1  # convert one-based to zero-based
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
    # The flattened path returns a single n_needed for all merged pairs. If the
    # file declares more than one distinct n_needed across reactions, that
    # information cannot be represented here and the per-reaction parser
    # (_parse_rxns_xml_reaction_groups) should be used instead.
    if len(set(declared_n_needed)) > 1:
        warnings.warn(
            f"rxns XML {rxns_path} declares differing n_needed values "
            f"{declared_n_needed} across reactions. The flattened criteria "
            f"path keeps only the last reaction's value ({n_needed}); use the "
            f"state-machine reaction parser to preserve per-reaction n_needed.",
            stacklevel=2,
        )
    return pairs, n_needed


def auto_detect_reactions(
    geom: "SystemGeometry",
    ghost_atoms: str = "auto",
    rxns_xml: str = "",
    bd_milestone_radius: float = 13.0,
    bd_milestone_radius_inner: float = 12.0,
) -> "List[List[ReactionPair]]":
    """
    Build the reaction criteria from GHO ghost atoms.

    The function tries several sources in order of priority. When rxns_xml is
    given, the criteria are parsed from the reference implementation rxns file.
    When ghost_atoms holds a manual specification, it is parsed as triplets of
    receptor index, ligand index, and cutoff. When ghost_atoms is 'auto', the GHO
    atoms in the PQR are detected and bd_milestone_radius is used as the cutoff.
    If no GHO atoms are found, a clear error is raised, since the centroid
    fallback was removed because it is physically wrong.
    """
    # First priority: the rxns XML file.
    if rxns_xml and rxns_xml.strip():
        rxns_path = Path(rxns_xml.strip())
        # A relative rxns_xml path is interpreted relative to the current
        # working directory, which Path already does. Resolution against the PDB
        # parent directory is intentionally not implemented, so a relative path
        # that does not exist relative to the working directory is reported as
        # not found below.
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
    # Second priority: a manual ghost_atoms specification.
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
    # Third priority: auto-detect the GHO atoms in the PQR.
    rec_ghosts = geom.receptor.ghost_indices
    lig_ghosts = geom.ligand.ghost_indices
    if rec_ghosts and lig_ghosts:
        # There is one GHO atom per molecule, so use the first GHO of each.
        rec_gho = rec_ghosts[0]
        lig_gho = lig_ghosts[0]
        # The reaction fires when the GHO-to-GHO distance falls below
        # bd_milestone_radius. The b-surface is the outer milestone at
        # bd_milestone_radius, where the ligand starts, and the q-surface is the
        # inner milestone at bd_milestone_radius_inner, where the reaction occurs.
        # The ligand starts on the b-surface and reacts when it reaches the
        # q-surface.
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
    # No GHO atoms were found, so raise a clear error. The centroid fallback was
    # removed, and the user must run with GHO-injected PQR files.
    raise RuntimeError(
        "\n\nNo GHO ghost atoms found in receptor.pqr or ligand.pqr.\n"
        "PySTARC requires GHO atoms to define the b-surface reaction criterion.\n"
        "GHO atoms are injected automatically during APBS preparation.\n"
        "This error should not occur in normal usage - please report it."
    )


# Multi-reaction parser with state-machine labels. It is kept distinct from
# _parse_rxns_xml_criteria, which remains the flattened-pairs entry point. This
# parser is called only when state_machine_reactions is True in the config.
@dataclass
class ReactionGroup:
    """
    One reaction with optional state-machine labels.

    The field name is the reaction name read from the <n> element. The field
    state_before is the source state, and the trajectory must be in this state
    for the reaction to fire. The field state_after is the destination state, and
    the trajectory enters this state after the reaction fires. The field pairs is
    the list of ReactionPair objects making up this reaction's criterion. The
    field n_needed is the minimum number of pairs that must fire simultaneously,
    where a value of -1 means all of them.
    """

    name: str
    state_before: Optional[str]
    state_after: Optional[str]
    pairs: List[ReactionPair]
    n_needed: int = -1


def _parse_rxns_xml_reaction_groups(rxns_path):
    """
    Parse the rxns XML while preserving the per-reaction grouping and state
    labels.

    The function returns the tuple (reaction_groups, first_state). The first
    element is a list of ReactionGroup objects. The second is an optional string
    holding the value of the <first_state> element.

    The expected XML format is as follows.

        <roottag>
          <first_state>b_surface</first_state>
          <reactions>
            <reaction>
              <n>name1</n>
              <state_before>src</state_before>
              <state_after>dst</state_after>
              <criterion>
                <n_needed>1</n_needed>
                <pair>
                  <atoms>rec_idx lig_idx</atoms>
                  <distance>cutoff</distance>
                </pair>
              </criterion>
            </reaction>
            ...
          </reactions>
        </roottag>

    The atom indices in the XML are one-based, and this parser converts them to
    zero-based. Reactions without state labels have state_before and state_after
    set to None, in which case the caller can synthesize defaults or fall back to
    the flattened-pairs path.
    """
    groups = []
    first_state = None
    try:
        tree = ET.parse(str(rxns_path))
        root = tree.getroot()
        fs_node = root.find(".//first_state")
        if fs_node is not None and fs_node.text:
            first_state = fs_node.text.strip()
        for reaction in root.iter("reaction"):
            # Read the reaction name from the <n> element.
            n_node = reaction.find("n")
            rxn_name = (
                n_node.text.strip() if (n_node is not None and n_node.text) else ""
            )
            # Read the state labels.
            sb_node = reaction.find("state_before")
            sa_node = reaction.find("state_after")
            state_before = (
                sb_node.text.strip() if (sb_node is not None and sb_node.text) else None
            )
            state_after = (
                sa_node.text.strip() if (sa_node is not None and sa_node.text) else None
            )
            # Read the criterion, which holds the pair list and n_needed.
            crit = reaction.find("criterion")
            if crit is None:
                continue
            rxn_n_needed = -1
            nn_node = crit.find("n_needed")
            if nn_node is not None and nn_node.text:
                try:
                    rxn_n_needed = int(nn_node.text.strip())
                except ValueError:
                    pass
            rxn_pairs = []
            for pair_node in crit.findall("pair"):
                # First format: an <atom1> element with the receptor index and an
                # <atom2> element with the ligand index.
                a1 = pair_node.find("atom1")
                a2 = pair_node.find("atom2")
                if a1 is not None and a2 is not None:
                    try:
                        p1 = a1.text.strip().split()
                        p2 = a2.text.strip().split()
                        rec_idx = int(p1[0]) - 1
                        lig_idx = int(p2[0]) - 1
                        cutoff = float(p1[2]) if len(p1) >= 3 else 5.0
                        rxn_pairs.append(
                            ReactionPair(
                                rec_index=rec_idx,
                                lig_index=lig_idx,
                                cutoff=cutoff,
                            )
                        )
                    except (ValueError, IndexError):
                        continue
                    continue
                # Second format: an <atoms> element with the receptor and ligand
                # indices, and a <distance> element with the cutoff.
                atoms_node = pair_node.find("atoms")
                distance_node = pair_node.find("distance")
                if atoms_node is not None and distance_node is not None:
                    try:
                        idx = atoms_node.text.strip().split()
                        rec_idx = int(idx[0]) - 1
                        lig_idx = int(idx[1]) - 1
                        cutoff = float(distance_node.text.strip())
                        rxn_pairs.append(
                            ReactionPair(
                                rec_index=rec_idx,
                                lig_index=lig_idx,
                                cutoff=cutoff,
                            )
                        )
                    except (ValueError, IndexError):
                        continue
            if rxn_pairs:
                groups.append(
                    ReactionGroup(
                        name=rxn_name,
                        state_before=state_before,
                        state_after=state_after,
                        pairs=rxn_pairs,
                        n_needed=rxn_n_needed,
                    )
                )
    except Exception as e:
        print(f"  Warning: could not parse rxns XML {rxns_path}: {e}")
    return groups, first_state
