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
from pystarc.global_defs.defaults import (
    BD_MILESTONE_RADIUS,
    BD_MILESTONE_RADIUS_INNER,
    QB_FACTOR,
)
from pystarc.global_defs.defaults import GHOST_NAMES
from pystarc.structures.molecules import needed_pairs


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
        return self.name.strip().upper() == "GHO" or self.radius < 1e-6


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
    # The pocket ghost PKT sits inside the receptor surface and is no part of
    # its body. GHO stays in, as it always has.
    body = [a for a in atoms if a.name.strip().upper() not in GHOST_NAMES[1:]]
    coords = np.array([[a.x, a.y, a.z] for a in body])
    radii = np.array([a.radius for a in body])
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
    bd_milestone_radius: float = BD_MILESTONE_RADIUS,
    bd_milestone_radius_inner: float = BD_MILESTONE_RADIUS_INNER,
    srad: float = 0.0,
    r_hydro_rec: float = 0.0,
    r_hydro_lig: float = 0.0,
    r_escape: float = 0.0,
) -> SystemGeometry:
    """
    Compute the full system geometry for the Brownian dynamics setup.

    The b-sphere radius is bd_milestone_radius, the outermost SEEKR milestone,
    which is user-defined. The escape sphere radius is 2 × the b-sphere radius,
    following the Luty-McCammon-Zhou convention, unless r_escape is given as a
    positive value, in which case that value is used. When r_hydro_rec or
    r_hydro_lig are greater than zero, they override the Monte Carlo
    hydrodynamic radii and match the values in the reference hydro_params.xml
    exactly.
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
    if not (r_escape == 0.0 or r_escape > QB_FACTOR * r_start):
        raise ValueError(
            f"r_escape must be 0 (2 x b) or above {QB_FACTOR} x b = "
            f"{QB_FACTOR * r_start:g} Å, got {r_escape}"
        )
    r_escape = r_escape if r_escape > 0.0 else 2.0 * r_start
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
    _esc_note = (
        "= 2 × b-surface"
        if abs(r_escape - 2.0 * r_start) < 1e-9
        else "user-set r_escape"
    )
    print(f"  Escape sphere         : {r_escape:.1f} Å  ({_esc_note})")
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


def _read_reactions(rxns_path):
    """
    Read every reaction of a rxns file, the BrownDye 2 format, strictly.

    Returns (groups, first_state). A pair is <atoms> with the one-based
    receptor and ligand atom numbers (the first number is the receptor atom,
    the second the ligand atom, whatever <molecules> says) and a positive
    <distance>, or <atom1> holding the receptor number, its charge and the
    cutoff with <atom2> holding the ligand number. A missing <n_needed> means
    all pairs, -1 says the same, and any other value must lie between 1 and the
    number of pairs. The name is <name>, else <n>, else reaction_k. Anything
    else raises ValueError naming the file and the reaction.
    """
    try:
        root = ET.parse(str(rxns_path)).getroot()
    except ET.ParseError as e:
        raise ValueError(f"rxns file {rxns_path} is not valid XML: {e}") from None
    fs = (root.findtext(".//first_state") or "").strip() or None
    groups = []
    for k, reaction in enumerate(root.iter("reaction"), start=1):
        name = (
            (reaction.findtext("name") or "").strip()
            or (reaction.findtext("n") or "").strip()
            or f"reaction_{k}"
        )
        where = f"{rxns_path}, reaction {name!r}"
        crit = reaction.find("criterion")
        if crit is None or crit.find("criterion") is not None:
            raise ValueError(f"{where}: needs one <criterion> without nested criteria")
        pairs = []
        for j, pair in enumerate(crit.findall("pair"), start=1):
            try:
                if pair.find("atom1") is not None and pair.find("atom2") is not None:
                    p1 = pair.findtext("atom1").split()
                    idx = [int(p1[0]), int(pair.findtext("atom2").split()[0])]
                    cutoff = float(p1[2])
                else:
                    idx = [int(x) for x in pair.findtext("atoms").split()]
                    cutoff = float(pair.findtext("distance"))
                    if len(idx) != 2:
                        raise ValueError
            except (AttributeError, IndexError, TypeError, ValueError):
                raise ValueError(
                    f"{where}, pair {j}: needs <atoms> with two atom numbers and "
                    "<distance>, or <atom1> with the cutoff and <atom2>"
                ) from None
            if min(idx) < 1 or not cutoff > 0:
                raise ValueError(
                    f"{where}, pair {j}: atom numbers start at 1 and the cutoff "
                    f"must be positive, got {idx} and {cutoff}"
                )
            pairs.append(ReactionPair(idx[0] - 1, idx[1] - 1, cutoff))
        if not pairs:
            raise ValueError(f"{where}: the criterion has no pair")
        nn = (crit.findtext("n_needed") or "").strip()
        try:
            n_needed = int(nn) if nn else -1
        except ValueError:
            raise ValueError(f"{where}: <n_needed> must be a whole number, got {nn!r}") from None
        needed_pairs(n_needed, len(pairs), where)
        groups.append(
            ReactionGroup(
                name=name,
                state_before=(reaction.findtext("state_before") or "").strip() or None,
                state_after=(reaction.findtext("state_after") or "").strip() or None,
                pairs=pairs,
                n_needed=n_needed,
            )
        )
    return groups, fs


def _parse_rxns_xml_criteria(rxns_path, single=False):
    """Pairs of every reaction of a rxns file with the n_needed of a single
    reaction (-1 when the file holds several). With single, a file of several
    reactions raises, since one criterion cannot hold their n_needed."""
    groups, _ = _read_reactions(rxns_path)
    if single and len(groups) > 1:
        raise ValueError(
            f"{rxns_path} holds {len(groups)} reactions. Without "
            "<state_machine_reactions> put all pairs in one reaction with its "
            "n_needed, or set state_machine_reactions true with one pair per "
            "reaction."
        )
    pairs = [p for g in groups for p in g.pairs]
    return pairs, groups[0].n_needed if len(groups) == 1 else -1


def check_rxn_atom_numbers(pairs, receptor_pqr, ligand_pqr) -> List[str]:
    """Warn when the rxns_xml atom numbers and the PQR serials disagree.

    PySTARC reads the atom numbers of rxns_xml as 1-based positions in the PQR.
    BrownDye 2 reads them as PQR serial numbers. The two agree when the serials
    of a PQR run 1..N. When they do not, the same file selects different atoms
    in the two programs. This prints a warning for each molecule whose
    referenced numbers are all also
    serials of its PQR but some of them name a different atom as a serial than
    as a position, and returns the warnings. A ligand split from a complex PQR
    keeps serials that start after the receptor's; numbers written as positions
    by setup.py are then not serials at all and are not ambiguous.
    """
    from pystarc.structures.pqr_io import parse_pqr_records

    msgs = []
    for label, pqr, indices in (
        ("receptor", receptor_pqr, sorted({p.rec_index for p in pairs})),
        ("ligand", ligand_pqr, sorted({p.lig_index for p in pairs})),
    ):
        serials = [a.serial for a in parse_pqr_records(pqr)]
        if serials == list(range(1, len(serials) + 1)):
            continue
        if not all(i + 1 in set(serials) for i in indices):
            continue
        differ = [i for i in indices if 0 <= i < len(serials) and serials[i] != i + 1]
        if not differ:
            continue
        i = differ[0]
        msg = (
            f"  Warning: {Path(pqr).name} is not numbered 1..N, so for "
            f"{len(differ)} of the {len(indices)} {label} atoms in rxns_xml the "
            f"position differs from the PQR serial (number {i + 1} is the atom "
            f"with serial {serials[i]}). PySTARC reads rxns_xml atom numbers as "
            f"1-based positions. A file that numbers atoms by PQR serial, as "
            f"BrownDye 2 files do, must be converted to positions."
        )
        print(msg)
        msgs.append(msg)
    return msgs


def check_rxn_atom_range(pairs, n_rec: int, n_lig: int) -> None:
    """Stop when a reaction pair names an atom past the end of its molecule.

    n_rec and n_lig count the GHO ghost atom appended to each molecule. CuPy
    wraps an index past the end of an array around to its start instead of
    raising, so an out of range pair would test another atom without a word.
    """
    bad = [
        (p.rec_index + 1, p.lig_index + 1)
        for p in pairs
        if not (0 <= p.rec_index < n_rec and 0 <= p.lig_index < n_lig)
    ]
    if bad:
        raise ValueError(
            f"rxns_xml pairs {bad} name atoms outside the receptor (1 to {n_rec}) "
            f"or the ligand (1 to {n_lig}), counting the GHO ghost atom of each"
        )


def auto_detect_reactions(
    geom: "SystemGeometry",
    ghost_atoms: str = "auto",
    rxns_xml: str = "",
    bd_milestone_radius: float = BD_MILESTONE_RADIUS,
    bd_milestone_radius_inner: float = BD_MILESTONE_RADIUS_INNER,
) -> "List[List[ReactionPair]]":
    """
    Build the reaction criterion.

    A given rxns_xml is the criterion, read strictly, with no fallback. Without
    it a manual ghost_atoms specification is read as triplets of receptor index,
    ligand index and cutoff. With ghost_atoms 'auto' the GHO atoms of the PQR
    files react at bd_milestone_radius_inner, which must lie between 0 and
    bd_milestone_radius.
    """
    if rxns_xml and rxns_xml.strip():
        # A relative path is taken from the current working directory.
        rxns_path = Path(rxns_xml.strip()).expanduser()
        if not rxns_path.is_file():
            raise FileNotFoundError(f"rxns_xml not found: {rxns_path} (cwd {Path.cwd()})")
        pairs, n_needed = _parse_rxns_xml_criteria(rxns_path)
        if not pairs:
            raise ValueError(f"rxns file {rxns_path} holds no reaction")
        nn_str = f"all ({len(pairs)})" if n_needed == -1 else str(n_needed)
        print(
            f"  Reaction criterion from {rxns_path.name}: "
            f"{len(pairs)} pair(s), n_needed={nn_str}"
        )
        for p in pairs:
            print(f"    rec[{p.rec_index}] -- lig[{p.lig_index}] < {p.cutoff:.1f} A")
        return [pairs], n_needed
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
        print(f"  Reaction criterion from ghost_atoms: {len(pairs)} pair(s), n_needed=all")
        return [pairs], -1
    # Third priority: auto-detect the GHO atoms in the PQR.
    rec_ghosts = geom.receptor.ghost_indices
    lig_ghosts = geom.ligand.ghost_indices
    if rec_ghosts and lig_ghosts:
        # There is one GHO atom per molecule, so use the first GHO of each.
        rec_gho = rec_ghosts[0]
        lig_gho = lig_ghosts[0]
        # The reaction fires when the GHO-to-GHO distance falls below
        # bd_milestone_radius_inner, inside the b surface where the ligand starts.
        rxn_cutoff = bd_milestone_radius_inner
        if not 0 < rxn_cutoff < bd_milestone_radius:
            raise ValueError(
                "The GHO criterion needs 0 < bd_milestone_radius_inner < "
                f"bd_milestone_radius, got {rxn_cutoff} and {bd_milestone_radius}. "
                "Give <rxns_xml>, <ghost_atoms> or an inner radius."
            )
        pairs = [ReactionPair(rec_gho, lig_gho, rxn_cutoff)]
        print(
            f"  GHO reaction criterion: rec[{rec_gho}] -- lig[{lig_gho}] "
            f"< {rxn_cutoff:.1f} A  (bd_milestone_radius_inner)"
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

    The field name is the reaction name read from the <name> element (or <n>). The field
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
              <name>name1</name>
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
    zero-based. The first number of <atoms> is the receptor atom. Reaction names
    must differ. Reactions without state labels have state_before and state_after
    set to None, in which case the caller can synthesize defaults or fall back to
    the flattened-pairs path.
    """
    groups, first_state = _read_reactions(rxns_path)
    names = [g.name for g in groups]
    if len(set(names)) < len(names):
        raise ValueError(f"rxns file {rxns_path}: reaction names repeat: {names}")
    return groups, first_state
