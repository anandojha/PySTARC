"""
APBS grid generation pipeline.

The Adaptive Poisson-Boltzmann Solver (APBS) solves the linearized
Poisson-Boltzmann equation (LPBE) on a 3D grid:

    ∇·[ε(r)∇φ(r)] - κ̄²(r)φ(r) = -ρ(r)/ε₀

Here ε(r) is the position-dependent dielectric constant, which is
roughly 4 inside the protein (the solute dielectric ε_in) and roughly
78 outside in water (ε_out). The term κ̄² = ε_out × κ² is the modified
Debye-Hückel screening, where κ = 1/λ is the inverse Debye length and
λ ≈ 7.86 Å at 150 mM. Finally, ρ(r) is the fixed charge density taken
from the PQR file.

For each molecule we generate two nested grids. The coarse grid covers
a large domain spanning the Debye screening length. Its job is to
provide accurate boundary conditions (bcfl sdh), and it is not used for
force evaluation at runtime. The fine grid covers a small domain that
resolves the molecular surface and is used for runtime force evaluation.
Its spacing is about 0.5 Å for proteins and about 0.1 Å for small
molecules, and it takes its boundary conditions from the coarse grid
(bcfl map).

Four APBS calculations are run per molecule. The first two are
electrostatic, solving the LPBE with ions on the coarse domain and then
on the fine domain at the molecular surface. The remaining two are Born
calculations run in vacuum (ε = 1, no ions), again on the coarse and
fine domains. The Born grids give the desolvation penalty, that is the
energy cost of moving a charged atom from bulk solvent into the
low-dielectric environment near the other molecule.

The grids are sized differently for the two partners. The receptor
electrostatic grid extends to cover the b-surface radius, which may
require an fglen override for large proteins. The ligand grid is
auto-sized tightly around the molecular extent, which prevents the Born
energy from blowing up where the grid would otherwise sample empty
space. A safety margin of three grid spacings from the boundary is
excluded to avoid APBS boundary condition artifacts.
"""

from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import numpy as np
import subprocess
import shutil
import math


def _run(cmd: str, cwd: Path, step: str):
    print(f"    $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"APBS step '{step}' failed (exit {result.returncode}):\n"
            f"  stderr: {result.stderr[-1000:]}\n"
            f"  stdout: {result.stdout[-500:]}"
        )
    return result


def _is_valid_apbs_dime(d: int) -> bool:
    """
    Check whether d is a valid APBS multigrid dimension.

    APBS multigrid solvers require dime - 1 to be expressible as
    c * 2**(l+1), where c >= 1 is any positive integer and l >= 1 is the
    number of multigrid levels (see apbs/src/generic/mgparm.c, function
    MGparm_check, line 267). This reduces to the requirement that dime be
    odd, be at least 5, and have (dime - 1) divisible by 4.

    APBS recommends, but does not require, c * 2**5 + 1 when dime > 65 for
    best multigrid efficiency. When that is not satisfied APBS auto-adjusts
    rather than failing, so this validator only enforces the hard rule.

    Valid examples are 5, 9, 13, 17, 33, 65, 97, 129, 161, 193, 257, 289,
    321, 385, 449, 513, and 577. Invalid examples are 4, 100, 200, 256,
    300, 400, and 500.
    """
    if d < 5:
        return False
    return (d - 1) % 4 == 0


def _check_tool(name: str):
    if not shutil.which(name):
        raise EnvironmentError(
            f"'{name}' not found in PATH.\n"
            f"Install APBS:  conda install -c conda-forge apbs -y"
        )


def _read_pqr_atoms(pqr_path: Path):
    """Read PQR atoms, skipping GHO ghost atoms.

    Returns a list of (x, y, z, radius) tuples used by the APBS grid
    sizing logic. Parsing is delegated to the canonical PQR reader in
    pystarc.structures.pqr_io.
    """
    from pystarc.structures.pqr_io import parse_pqr_records

    return [
        (r.x, r.y, r.z, r.radius)
        for r in parse_pqr_records(pqr_path)
        if r.name.strip().upper() != "GHO"
    ]


def _compute_grid_params(
    pqr_path: Path,
    srad: float,
    debye_length: float,
    dime: int = 129,
    cglen_override: float = 0.0,
    fglen_override: float = 0.0,
    coarse_dime: int = 0,
    fine_dime: int = 0,
) -> Tuple[dict, dict]:
    """
    Compute two-level APBS mg-manual grid parameters.

    Both levels use the global 'dime' parameter unless coarse_dime or
    fine_dime overrides are passed. The coarse level (level 0) uses
    glen = 2 × fglen with bcfl=sdh. The fine level (level 1) takes its
    glen from the auto-formula or from fglen_override and uses bcfl=map.

    The multigrid scheme requires the coarse grid to strictly enclose the
    fine grid, so cglen must be greater than fglen. APBS's bcfl=map step
    (vpmg.c:bcfl_map) interpolates the coarse-level potential at every
    fine-grid point, and if any fine-grid point lies outside the coarse
    box, APBS aborts with "fillcoChargeMap: fell off of potential map".

    As a design choice, PySTARC uses a fixed two-level focusing scheme,
    whereas BrownDye2 uses adaptive multilevel focusing in which the
    number of levels is chosen by a boundary-potential criterion. At
    physiological ionic strength (Debye length ≈ 8 Å), the boundary
    potential at the coarse box edge (about twice the molecular extent)
    is essentially zero, so two levels are sufficient and give k_on
    values consistent with BrownDye2. For very low ionic strength or very
    highly charged complexes, this design may need revisiting.

    APBS's VREDFRAC = 0.25 (apbs/src/generic/vhal.h) defines the maximum
    allowed grid spacing reduction per level. With the same dime for both
    levels, this means cglen/fglen must lie in (1, 4]. The factor of two
    chosen here sits in the middle of this range and matches BrownDye2's
    per-level scaling.

    Returns the pair (coarse_params, fine_params).
    """
    # Validate the requested dime values. APBS multigrid requires dime to
    # be of the form c * 2**(n+1) + 1. Passing other values causes silent
    # solver failures.
    for d in (dime, coarse_dime, fine_dime):
        if d > 0 and not _is_valid_apbs_dime(d):
            raise ValueError(
                f"Invalid APBS dime={d}. APBS multigrid requires dime to be "
                f"odd, >= 5, with (dime - 1) divisible by 4 (i.e., of the form "
                f"c * 2**(l+1) + 1 with c >= 1 a positive integer and l >= 1). "
                f"Valid values: 5, 9, 13, 17, 33, 65, 97, 129, 161, 193, "
                f"257, 289, 321, 385, 449, 513, 577."
            )
    atoms = _read_pqr_atoms(pqr_path)
    if not atoms:
        raise ValueError(f"No non-GHO atoms in {pqr_path}")
    coords = np.array([[a[0], a[1], a[2]] for a in atoms])
    radii = np.array([a[3] for a in atoms])
    gcent = coords.mean(axis=0).tolist()
    max_atom_radius = float(np.max(radii))
    # Choose the per-level dime, falling back to the global 'dime'
    # parameter when no per-level override is given.
    _fine_dime = fine_dime if fine_dime > 0 else dime
    _coarse_dime = coarse_dime if coarse_dime > 0 else dime
    # Fine level (level 1), which covers the molecular surface.
    if fglen_override > 0.0:
        fglen = fglen_override
    else:
        mol_extent = coords.max(axis=0) - coords.min(axis=0)
        max_extent = float(np.max(mol_extent))
        fglen = (max_extent + 2.0 * (max_atom_radius + srad)) * 1.05
        fglen = max(fglen, 4.0 * (max_atom_radius + srad) * 1.025)
    fine_spacing = fglen / (_fine_dime - 1)
    # Coarse level (level 0), which covers a wider region to supply the
    # boundary conditions. The factor of two ensures cglen > fglen as the
    # multigrid scheme requires, and matches BrownDye2's per-level scaling.
    # Using the same dime for both levels gives a clean integer spacing
    # ratio (coarse_spacing = 2 × fine_spacing), so the bilinear
    # interpolation in bcfl=map is well-conditioned.
    if cglen_override > 0.0:
        cglen = cglen_override
    else:
        cglen = 2.0 * fglen
    # Check that the coarse grid strictly encloses the fine grid. This
    # catches both auto-formula bugs and user-supplied overrides that
    # would otherwise crash APBS later in bcfl_map.
    if cglen <= fglen:
        raise ValueError(
            f"Multigrid invariant violated: cglen={cglen} <= fglen={fglen}. "
            f"APBS bcfl=map requires the coarse grid to strictly enclose "
            f"the fine grid. Pass a larger cglen_override or use the auto "
            f"formula (cglen=2*fglen)."
        )
    coarse_spacing = cglen / (_coarse_dime - 1)

    coarse = {
        "spacing": coarse_spacing,
        "dime": [_coarse_dime, _coarse_dime, _coarse_dime],
        "glen": [cglen, cglen, cglen],
        "gcent": gcent,
        "label": "coarse",
        "bcfl": "sdh",
    }
    fine = {
        "spacing": fine_spacing,
        "dime": [_fine_dime, _fine_dime, _fine_dime],
        "glen": [fglen, fglen, fglen],
        "gcent": gcent,
        "label": "fine",
        "bcfl": "map",  # read the coarse DX as the boundary condition
    }
    return coarse, fine


def _write_apbs_input(
    pqr_path: Path,
    out_dx_name: str,
    params: dict,
    prev_dx_name: str | None,
    work_dir: Path,
    inp_name: str,
    is_born: bool,
    ion_conc: float,
    dielectric_in: float,
    dielectric_out: float,
    srad: float,
    temp: float,
    ion_radius_pos: float = 0.95,
    ion_radius_neg: float = 1.81,
) -> Path:
    """
    Write one APBS mg-manual input file.

    All calculations use chgm spl2, srfm smol, and sdens 10.0. The coarse
    level (level 0) uses bcfl sdh. The fine level (level 1) uses bcfl map
    together with usemap pot, which reads the previous level as its
    boundary condition. Born calculations set sdie = 1.0 (vacuum) and
    include no ions.
    """
    dime_str = " ".join(str(d) for d in params["dime"])
    glen_str = " ".join(f"{v:.4f}" for v in params["glen"])
    gcent_str = " ".join(f"{v:.4f}" for v in params["gcent"])
    sdie = 1.0 if is_born else dielectric_out
    ion_str = (
        ""
        if is_born or ion_conc <= 0
        else (
            f"  ion charge +1 conc {ion_conc:.4f} radius {ion_radius_pos:.5f}\n"
            f"  ion charge -1 conc {ion_conc:.4f} radius {ion_radius_neg:.5f}\n"
        )
    )
    lines = [
        f"read\n",
        f"  mol pqr {pqr_path.name}\n",
    ]
    if prev_dx_name:
        lines.append(f"  pot dx {prev_dx_name}\n")
    lines += [
        f"end\n\n",
        f"elec\n",
        f"  mg-manual\n",
        f"  dime {dime_str}\n",
        f"  glen {glen_str}\n",
        f"  gcent {gcent_str}\n",
        f"  mol 1\n",
        f"  lpbe\n",
    ]
    if params["bcfl"] == "map" and prev_dx_name:
        # bcfl=map requires the previous level's DX file to exist and be
        # readable. Without this check APBS would crash mid-run with a less
        # useful error, so failing early gives a clearer diagnostic.
        prev_dx_path = work_dir / prev_dx_name
        if not prev_dx_path.exists():
            raise FileNotFoundError(
                f"bcfl=map requires {prev_dx_path} to exist; the coarse "
                f"APBS calculation must succeed before the fine one. "
                f"Check the coarse APBS log for errors."
            )
        lines += [
            f"  usemap pot 1\n",
            f"  bcfl map\n",
        ]
    else:
        lines.append(f"  bcfl sdh\n")
    lines += [
        ion_str,
        f"  pdie {dielectric_in:.1f}\n",
        f"  sdie {sdie:.2f}\n",
        f"  srfm smol\n",
        f"  chgm spl2\n",
        f"  sdens 10.0\n",
        f"  srad {srad:.4f}\n",
        f"  temp {temp:.2f}\n",
        f"  calcenergy no\n",
        f"  calcforce no\n",
        f"  write pot dx {out_dx_name}\n",
        f"end\n\nquit\n",
    ]
    inp_path = work_dir / inp_name
    inp_path.write_text("".join(lines))
    return inp_path


def _write_desolvation_grids(
    pqr_path,
    mol_name,
    work_dir,
    coarse,
    fine,
    dielectric_in,
    dielectric_out,
    temp,
    debye_length,
):
    """
    Build the image (Born) desolvation grids from the partner's cavity geometry.

    Desolvation is a self energy set by the partner's low-dielectric VOLUME, not
    by its charges, so it is computed analytically from atomic radii rather than
    from a Poisson-Boltzmann solve of the partner's charge distribution. A PB
    potential decays as a multipole series and changes sign across the surface,
    whereas the desolvation field is strictly positive and decays as 1/r^4. See
    pystarc/pipeline/desolvation_grid.py for the form and references.
    """
    from .desolvation_grid import (
        desolvation_field_on_grid,
        read_pqr_geometry,
        write_dx,
    )

    xyz, rad = read_pqr_geometry(pqr_path)
    written = []
    for level, params in enumerate([coarse, fine]):
        dime = params["dime"]
        glen = params["glen"]
        gcent = params["gcent"]
        spacing = [glen[k] / (dime[k] - 1) for k in range(3)]
        origin = [gcent[k] - glen[k] / 2.0 for k in range(3)]
        field = desolvation_field_on_grid(
            origin,
            spacing,
            dime,
            xyz,
            rad,
            eps_p=dielectric_in,
            eps_s=dielectric_out,
            temp=temp,
            debye_length=debye_length,
        )
        out_path = work_dir / f"{mol_name}{level}_born.dx"
        write_dx(out_path, field, origin, spacing, dime)
        print(
            f"  [4] desolvation - {mol_name} {params['label']}: "
            f"dime={dime[0]} spacing={spacing[0]:.4f}A "
            f"max={float(field.max()):.4g} kBT/e^2"
        )
        written.append(out_path)
    return written


def run_apbs(
    pqr_path: Path,
    mol_name: str,
    work_dir: Path,
    ion_conc: float = 0.150,
    debye_length: float = 7.858,
    dielectric_in: float = 4.0,
    dielectric_out: float = 78.0,
    srad: float = 1.5,
    temp: float = 298.15,
    dime: int = 129,
    ion_radius_pos: float = 0.95,
    ion_radius_neg: float = 1.81,
    cglen_override: float = 0.0,
    fglen_override: float = 0.0,
    coarse_dime: int = 0,
    fine_dime: int = 0,
) -> List[Path]:
    """
    Run APBS for one molecule using two-level mg-manual grids.

    The coarse level (level 0) covers the full screened Yukawa region with
    bcfl sdh, and the fine level (level 1) covers the molecule tightly with
    bcfl map drawn from the coarse level. The function produces four DX
    files. The file {mol}0.dx holds the coarse electrostatic grid covering
    the contact zone, and {mol}1.dx holds the fine electrostatic grid
    covering the molecular surface. Likewise {mol}0_born.dx holds the
    coarse Born desolvation grid and {mol}1_born.dx holds the fine one.

    The force engine uses the finest DX grid that covers each query point.
    At contact, near r ≈ 2.5 Å, it uses the coarse DX (0.16 Å spacing),
    while inside the molecule at r < 2 Å it uses the fine DX (0.032 Å
    spacing).
    """
    _check_tool("apbs")
    # Skip APBS if all DX files already exist, for example when they were
    # symlinked from a parent directory.
    expected_dx = [
        work_dir / f"{mol_name}{i}{s}.dx" for i in [0, 1] for s in ["", "_born"]
    ]
    if all(f.exists() for f in expected_dx):
        print(f"  [4] APBS - {mol_name}: all DX files present, skipping.")
        return expected_dx
    # Electrostatic grid, which uses the fglen override to cover the
    # b-sphere.
    coarse_elec, fine_elec = _compute_grid_params(
        pqr_path,
        srad,
        debye_length,
        dime,
        cglen_override,
        fglen_override,
        coarse_dime,
        fine_dime,
    )
    # Born grid, auto-sized to the molecular extent. The Born energy decays
    # to zero within a few Å of the dielectric boundary, so there is no need
    # to extend it to the b-sphere. Using the large override would create
    # artificial Born gradients far from the surface.
    coarse_born, fine_born = _compute_grid_params(
        pqr_path, srad, debye_length, dime, 0.0, 0.0
    )  # auto-sized grid for the Born calculation
    print(
        f"    Elec grid (coarse): spacing={coarse_elec['spacing']:.4f}Å  "
        f"glen={coarse_elec['glen'][0]:.2f}Å  dime={coarse_elec['dime'][0]}"
    )
    print(
        f"    Elec grid (fine  ): spacing={fine_elec['spacing']:.4f}Å  "
        f"glen={fine_elec['glen'][0]:.2f}Å  dime={fine_elec['dime'][0]}"
    )
    print(
        f"    Born grid (coarse): spacing={coarse_born['spacing']:.4f}Å  "
        f"glen={coarse_born['glen'][0]:.2f}Å  dime={coarse_born['dime'][0]}"
    )
    print(
        f"    Born grid (fine  ): spacing={fine_born['spacing']:.4f}Å  "
        f"glen={fine_born['glen'][0]:.2f}Å  dime={fine_born['dime'][0]}"
    )
    dx_files = []
    for is_born in [False, True]:
        label = "Born desolvation" if is_born else "Electrostatic"
        suffix = "_born" if is_born else ""
        coarse = coarse_born if is_born else coarse_elec
        fine = fine_born if is_born else fine_elec
        if is_born:
            # Desolvation is a cavity self energy, not a PB potential: build it
            # from the partner's radii. APBS is still used for electrostatics.
            dx_files.extend(
                _write_desolvation_grids(
                    pqr_path,
                    mol_name,
                    work_dir,
                    coarse,
                    fine,
                    dielectric_in,
                    dielectric_out,
                    temp,
                    debye_length,
                )
            )
            continue
        print(f"  [4] APBS - {mol_name} {label} (2-level two-level) ...")
        # Coarse level (level 0), which has no previous DX to read.
        inp0 = _write_apbs_input(
            pqr_path=pqr_path,
            out_dx_name=f"{mol_name}0{suffix}",
            params=coarse,
            prev_dx_name=None,
            work_dir=work_dir,
            inp_name=f"{mol_name}_{'born' if is_born else 'elec'}_0.in",
            is_born=is_born,
            ion_conc=ion_conc,
            dielectric_in=dielectric_in,
            dielectric_out=dielectric_out,
            srad=srad,
            temp=temp,
            ion_radius_pos=ion_radius_pos,
            ion_radius_neg=ion_radius_neg,
        )
        _run(
            f"apbs {inp0.name}",
            cwd=work_dir,
            step=f"apbs-{mol_name}-{'born' if is_born else 'elec'}-coarse",
        )
        dx0 = work_dir / f"{mol_name}0{suffix}.dx"
        if not dx0.exists():
            raise RuntimeError(f"Expected {dx0} not found after APBS")
        print(f"    -> {dx0.name}  ({dx0.stat().st_size//1024} KB)  [coarse]")
        dx_files.append(dx0)
        inp0.unlink(missing_ok=True)
        # Fine level (level 1), which reads the coarse DX as its boundary
        # conditions.
        inp1 = _write_apbs_input(
            pqr_path=pqr_path,
            out_dx_name=f"{mol_name}1{suffix}",
            params=fine,
            prev_dx_name=dx0.name,
            work_dir=work_dir,
            inp_name=f"{mol_name}_{'born' if is_born else 'elec'}_1.in",
            is_born=is_born,
            ion_conc=ion_conc,
            dielectric_in=dielectric_in,
            dielectric_out=dielectric_out,
            srad=srad,
            temp=temp,
            ion_radius_pos=ion_radius_pos,
            ion_radius_neg=ion_radius_neg,
        )
        _run(
            f"apbs {inp1.name}",
            cwd=work_dir,
            step=f"apbs-{mol_name}-{'born' if is_born else 'elec'}-fine",
        )
        dx1 = work_dir / f"{mol_name}1{suffix}.dx"
        if not dx1.exists():
            raise RuntimeError(f"Expected {dx1} not found after APBS")
        print(f"    -> {dx1.name}  ({dx1.stat().st_size//1024} KB)  [fine]")
        dx_files.append(dx1)
        inp1.unlink(missing_ok=True)

        for f in work_dir.glob("io.mc"):
            f.unlink(missing_ok=True)
    return dx_files


def run_apbs_both(
    receptor_pqr: Path,
    ligand_pqr: Path,
    work_dir: Path,
    ion_conc: float = 0.150,
    debye_length: float = 7.858,
    dielectric_in: float = 4.0,
    dielectric_out: float = 78.0,
    srad: float = 1.5,
    temp: float = 298.15,
    dime: int = 129,
    ion_radius_pos: float = 0.95,
    ion_radius_neg: float = 1.81,
    cglen_override: float = 0.0,
    fglen_override: float = 0.0,
    coarse_dime: int = 0,
    fine_dime: int = 0,
    # These are kept only for API compatibility.
    fine_spacing: float = 0.5,
    coarse_spacing: float = 2.0,
) -> Tuple[List[Path], List[Path]]:
    """
    Run APBS for both the receptor and the ligand using two-level grids.

    Returns the pair (receptor_dx_files, ligand_dx_files). Each list holds
    the DX files in the order coarse electrostatic, fine electrostatic,
    coarse Born, and fine Born.
    """
    print("\n[4] Running APBS (reference-exact 2-level mg-manual, chgm=spl2) ...")
    rec_dx = run_apbs(
        receptor_pqr,
        "receptor",
        work_dir,
        ion_conc,
        debye_length,
        dielectric_in,
        dielectric_out,
        srad,
        temp,
        dime,
        ion_radius_pos,
        ion_radius_neg,
        cglen_override,
        fglen_override,
        coarse_dime,
        fine_dime,
    )
    # The ligand uses auto-computed grid sizes and the standard dime = 129.
    lig_dx = run_apbs(
        ligand_pqr,
        "ligand",
        work_dir,
        ion_conc,
        debye_length,
        dielectric_in,
        dielectric_out,
        srad,
        temp,
        129,
        ion_radius_pos,
        ion_radius_neg,
        0.0,
        0.0,
    )  # auto-sized grid for the ligand
    total = len(rec_dx) + len(lig_dx)
    print(f"  Total DX files generated: {total}  " f"(8 = 4 receptor + 4 ligand)")
    return rec_dx, lig_dx
