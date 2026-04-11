"""
APBS grid generation pipeline
===============================

Background
-------------------
The Adaptive Poisson-Boltzmann Solver (APBS) solves the linearized
Poisson-Boltzmann equation (LPBE) on a 3D grid:
    ∇·[ε(r)∇φ(r)] - κ̄²(r)φ(r) = -ρ(r)/ε₀
where:
  - ε(r) : position-dependent dielectric constant
    • ε_in ≈ 4 inside the protein (solute dielectric)
    • ε_out ≈ 78 outside (water)
  - κ̄² = ε_out × κ² : modified Debye-Hückel screening
  - κ = 1/λ : inverse Debye length (λ ≈ 7.86 Å at 150 mM)
  - ρ(r) : fixed charge density from the PQR file

Two-Level grid strategy
-----------------------
For each molecule, two nested grids are generated:
1. COARSE grid: Large domain covering the Debye screening length
    - Purpose: provide accurate boundary conditions (bcfl sdh)
    - not used for force evaluation at runtime
2. FINE grid: Small domain resolving the molecular surface
    - Purpose: runtime force evaluation
    - Spacing ~0.5 Å for proteins, ~0.1 Å for small molecules
    - Boundary conditions from the coarse grid (bcfl map)

Four APBS calculations per molecule
------------------------------------
1. Electrostatic (coarse): LPBE with ions, large domain
2. Electrostatic (fine): LPBE with ions, molecular surface
3. Born (coarse): vacuum (ε=1, no ions), large domain
4. Born (fine): vacuum, molecular surface

The Born grids give the desolvation penalty - the energy cost of
moving a charged atom from bulk solvent into the low-dielectric
environment near the other molecule.

Grid sizing
-----------
- Receptor: electrostatic grid extends to cover the b-surface
    radius (may need fglen override for large proteins)
- Ligand: auto-sized tight grid around the molecular extent
    (prevents Born blow-up from sampling empty space)
- Safety margin: 3 grid spacings from the boundary are excluded
    to avoid APBS boundary condition artifacts
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
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"APBS step '{step}' failed (exit {result.returncode}):\n"
            f"  stderr: {result.stderr[-1000:]}\n"
            f"  stdout: {result.stdout[-500:]}"
        )
    return result

def _check_tool(name: str):
    if not shutil.which(name):
        raise EnvironmentError(
            f"'{name}' not found in PATH.\n"
            f"Install APBS:  conda install -c conda-forge apbs -y"
        )

def _read_pqr_atoms(pqr_path: Path):
    """Read PQR atoms, skipping GHO ghost atoms.
    PQR format: ATOM serial name resname chain resseq x y z charge radius
    """
    atoms = []
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                name = parts[2].strip().upper()
                if name == 'GHO':
                    continue
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                # parts[8] = charge, parts[9] = radius
                r = float(parts[9])
                atoms.append((x, y, z, r))
            except (ValueError, IndexError):
                continue
    return atoms

def _compute_grid_params(pqr_path:      Path,
                          srad:          float,
                          debye_length:  float,
                          dime:          int   = 129,
                          cglen_override:float = 0.0,
                          fglen_override:float = 0.0,
                          coarse_dime:   int   = 0,
                          fine_dime:     int   = 0) -> Tuple[dict, dict]:
    """
    Compute two-level APBS mg-manual grid parameters.
      Level 0 (coarse): dime=65, glen=128 -> spacing=2.0 Å
      Level 1 (fine):   dime=161, glen=80  -> spacing=0.5 Å
    PySTARC supports per-level dime via coarse_dime/fine_dime overrides.
    If not set, both levels use the global 'dime' parameter.
    Returns: (coarse_params, fine_params)
    """
    atoms = _read_pqr_atoms(pqr_path)
    if not atoms:
        raise ValueError(f"No non-GHO atoms in {pqr_path}")
    coords  = np.array([[a[0], a[1], a[2]] for a in atoms])
    radii   = np.array([a[3] for a in atoms])
    gcent   = coords.mean(axis=0).tolist()
    max_atom_radius = float(np.max(radii))
    # Per-level dime (the reference uses different values for coarse vs fine)
    _fine_dime   = fine_dime   if fine_dime   > 0 else dime
    _coarse_dime = coarse_dime if coarse_dime > 0 else dime
    # Level 1 (fine): covers molecular surface
    if fglen_override > 0.0:
        fglen = fglen_override
    else:
        mol_extent = coords.max(axis=0) - coords.min(axis=0)
        max_extent = float(np.max(mol_extent))
        fglen = (max_extent + 2.0 * (max_atom_radius + srad)) * 1.05
        fglen = max(fglen, 4.0 * (max_atom_radius + srad) * 1.025)
    fine_spacing = fglen / (_fine_dime - 1)
    # Level 0 (coarse): covers wider region for boundary conditions
    if cglen_override > 0.0:
        cglen = cglen_override
    else:
        cglen = min(2.0 * fglen, float(_coarse_dime - 1))
    coarse_spacing = cglen / (_coarse_dime - 1)

    coarse = {
        'spacing': coarse_spacing,
        'dime':    [_coarse_dime, _coarse_dime, _coarse_dime],
        'glen':    [cglen, cglen, cglen],
        'gcent':   gcent,
        'label':   'coarse',
        'bcfl':    'sdh',
    }
    fine = {
        'spacing': fine_spacing,
        'dime':    [_fine_dime, _fine_dime, _fine_dime],
        'glen':    [fglen, fglen, fglen],
        'gcent':   gcent,
        'label':   'fine',
        'bcfl':    'map',   # reads coarse DX as boundary condition
    }
    return coarse, fine

def _write_apbs_input(pqr_path:      Path,
                       out_dx_name:   str,
                       params:        dict,
                       prev_dx_name:  str | None,
                       work_dir:      Path,
                       inp_name:      str,
                       is_born:       bool,
                       ion_conc:      float,
                       dielectric_in: float,
                       dielectric_out:float,
                       srad:          float,
                       temp:          float,
                       ion_radius_pos:float = 0.95,
                       ion_radius_neg:float = 1.81) -> Path:
    """
    Write one APBS mg-manual input file.
      chgm spl2, srfm smol, sdens 10.0
      Level 0: bcfl sdh
      Level 1: bcfl map + usemap pot (reads previous level as BC)
    Born: sdie=1.0 (vacuum), no ions
    """
    dime_str = " ".join(str(d) for d in params['dime'])
    glen_str = " ".join(f"{v:.4f}" for v in params['glen'])
    gcent_str= " ".join(f"{v:.4f}" for v in params['gcent'])
    sdie     = 1.0 if is_born else dielectric_out
    ion_str  = "" if is_born or ion_conc <= 0 else (
        f"  ion charge +1 conc {ion_conc:.4f} radius {ion_radius_pos:.5f}\n"
        f"  ion charge -1 conc {ion_conc:.4f} radius {ion_radius_neg:.5f}\n"
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
    if params['bcfl'] == 'map' and prev_dx_name:
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

def run_apbs(pqr_path:      Path,
             mol_name:      str,
             work_dir:      Path,
             ion_conc:      float = 0.150,
             debye_length:  float = 7.858,
             dielectric_in: float = 4.0,
             dielectric_out:float = 78.0,
             srad:          float = 1.5,
             temp:          float = 298.15,
             dime:          int   = 129,
             ion_radius_pos:float = 0.95,
             ion_radius_neg:float = 1.81,
             cglen_override:float = 0.0,
             fglen_override:float = 0.0,
             coarse_dime:   int   = 0,
             fine_dime:     int   = 0) -> List[Path]:
    """
    Run APBS for one molecule using two-level two-level mg-manual grids.
    Level 0 (coarse): covers full screened Yukawa region, bcfl sdh
    Level 1 (fine):   covers molecule tightly, bcfl map from coarse
    Produces 4 DX files:
      {mol}0.dx       coarse electrostatic (covers contact zone)
      {mol}1.dx       fine   electrostatic (covers molecule surface)
      {mol}0_born.dx  coarse Born desolvation
      {mol}1_born.dx  fine   Born desolvation
    Force engine uses finest DX grid covering each query point -
    at contact r~2.5A, uses coarse DX (0.16A spacing); inside
    molecule at r<2A, uses fine DX (0.032A spacing).
    """
    _check_tool("apbs")
    # Skip APBS if all DX files already exist (e.g. symlinked from parent)
    expected_dx = [work_dir / f"{mol_name}{i}{s}.dx"
                   for i in [0, 1] for s in ["", "_born"]]
    if all(f.exists() for f in expected_dx):
        print(f"  [4] APBS - {mol_name}: all DX files present, skipping.")
        return expected_dx
    # Electrostatic grid: uses fglen override to cover the b-sphere
    coarse_elec, fine_elec = _compute_grid_params(
        pqr_path, srad, debye_length, dime,
        cglen_override, fglen_override, coarse_dime, fine_dime)
    # Born grid: auto-sized to molecular extent (Born decays to zero
    # within a few Å of the dielectric boundary - no need to extend
    # to the b-sphere). Using the large override creates artificial
    # Born gradients far from the surface.
    coarse_born, fine_born = _compute_grid_params(
        pqr_path, srad, debye_length, dime,
        0.0, 0.0)  # auto for Born
    print(f"    Elec grid (coarse): spacing={coarse_elec['spacing']:.4f}Å  "
          f"glen={coarse_elec['glen'][0]:.2f}Å  dime={coarse_elec['dime'][0]}")
    print(f"    Elec grid (fine  ): spacing={fine_elec['spacing']:.4f}Å  "
          f"glen={fine_elec['glen'][0]:.2f}Å  dime={fine_elec['dime'][0]}")
    print(f"    Born grid (coarse): spacing={coarse_born['spacing']:.4f}Å  "
          f"glen={coarse_born['glen'][0]:.2f}Å  dime={coarse_born['dime'][0]}")
    print(f"    Born grid (fine  ): spacing={fine_born['spacing']:.4f}Å  "
          f"glen={fine_born['glen'][0]:.2f}Å  dime={fine_born['dime'][0]}")
    dx_files = []
    for is_born in [False, True]:
        label  = "Born desolvation" if is_born else "Electrostatic"
        suffix = "_born" if is_born else ""
        coarse = coarse_born if is_born else coarse_elec
        fine   = fine_born   if is_born else fine_elec
        print(f"  [4] APBS - {mol_name} {label} (2-level two-level) ...")
        # Level 0 (coarse) - no previous DX
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
        _run(f"apbs {inp0.name}", cwd=work_dir,
             step=f"apbs-{mol_name}-{'born' if is_born else 'elec'}-coarse")
        dx0 = work_dir / f"{mol_name}0{suffix}.dx"
        if not dx0.exists():
            raise RuntimeError(f"Expected {dx0} not found after APBS")
        print(f"    -> {dx0.name}  ({dx0.stat().st_size//1024} KB)  [coarse]")
        dx_files.append(dx0)
        inp0.unlink(missing_ok=True)
        # Level 1 (fine) - reads coarse DX as boundary conditions
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
        _run(f"apbs {inp1.name}", cwd=work_dir,
             step=f"apbs-{mol_name}-{'born' if is_born else 'elec'}-fine")
        dx1 = work_dir / f"{mol_name}1{suffix}.dx"
        if not dx1.exists():
            raise RuntimeError(f"Expected {dx1} not found after APBS")
        print(f"    -> {dx1.name}  ({dx1.stat().st_size//1024} KB)  [fine]")
        dx_files.append(dx1)
        inp1.unlink(missing_ok=True)

        for f in work_dir.glob("io.mc"):
            f.unlink(missing_ok=True)
    return dx_files

def run_apbs_both(receptor_pqr:   Path,
                  ligand_pqr:     Path,
                  work_dir:       Path,
                  ion_conc:       float = 0.150,
                  debye_length:   float = 7.858,
                  dielectric_in:  float = 4.0,
                  dielectric_out: float = 78.0,
                  srad:           float = 1.5,
                  temp:           float = 298.15,
                  dime:           int   = 129,
                  ion_radius_pos: float = 0.95,
                  ion_radius_neg: float = 1.81,
                  cglen_override: float = 0.0,
                  fglen_override: float = 0.0,
                  coarse_dime:    int   = 0,
                  fine_dime:      int   = 0,
                  # kept for API compatibility
                  fine_spacing:   float = 0.5,
                  coarse_spacing: float = 2.0) -> Tuple[List[Path], List[Path]]:
    """
    Run APBS for both receptor and ligand using two-level two-level grids.
    Returns (receptor_dx_files, ligand_dx_files).
    Each list: [coarse_elec, fine_elec, coarse_born, fine_born]
    """
    print("\n[4] Running APBS (reference-exact 2-level mg-manual, chgm=spl2) ...")
    rec_dx = run_apbs(receptor_pqr, "receptor", work_dir,
                      ion_conc, debye_length, dielectric_in,
                      dielectric_out, srad, temp, dime,
                      ion_radius_pos, ion_radius_neg,
                      cglen_override, fglen_override,
                      coarse_dime, fine_dime)
    # Ligand uses auto-computed grid sizes and standard dime=129.
    lig_dx = run_apbs(ligand_pqr, "ligand", work_dir,
                      ion_conc, debye_length, dielectric_in,
                      dielectric_out, srad, temp, 129,
                      ion_radius_pos, ion_radius_neg,
                      0.0, 0.0)  # auto grid for ligand
    total = len(rec_dx) + len(lig_dx)
    print(f"  Total DX files generated: {total}  "
          f"(8 = 4 receptor + 4 ligand)")
    return rec_dx, lig_dx
