"""
Image (Born) desolvation grid.

The desolvation penalty is a SELF energy: a charge q_i approaching the partner
loses part of its solvation because high dielectric solvent is displaced by the
partner's low dielectric interior. It is a functional of the partner's CAVITY
GEOMETRY (its atomic radii) and is completely independent of the partner's
charges. This is why it cannot come from a Poisson-Boltzmann solve of the
partner's fixed charges: APBS returns the potential of those charges, which
decays as a multipole series and changes sign across the surface, whereas the
desolvation field is strictly positive and decays as 1/r^4.

Form implemented:

    dG_i = alpha * C * D * q_i^2 *
           SUM_j  a_j^3 (1 + k r_ij)^2 exp(-2 k r_ij) / ( r_ij^2 - a_j^2 )^2

    D = (eps_s - eps_p) / [ eps_s * (2*eps_s + eps_p) ]   Kirkwood n=1 image factor
    C = e^2/(4 pi eps_0) divided by kT, i.e. kBT.Angstrom/e^2 (560.46 at 298.15 K)
    k = 1/debye_length, the ionic screening of the induced reaction field
    alpha = empirical prefactor on the self energy, default 1.0

The kernel is the n=1 Kirkwood image term, a^3/r^4 in the far field. The
a^3/(r^2 - a^2)^-2 form is not an exact resummation of the Kirkwood series: no
closed form exists for general eps, and in the conductor limit the closed form
is a^3/[r^2 (r^2 - a^2)]. The n=1 term understates the true series near contact
(0.60x at r = 3 A for a = 1.7) and the (r^2 - a^2)^-2 form overstates it (1.31x
at the same point).

This is a per-atom Kirkwood image response with no mutual polarisation between
atom spheres, not a volume integral over the cavity. The per-sphere volume
integral 4 pi [ a/(2(R^2-a^2)) - ln((R+a)/(R-a))/(4R) ] is a different model,
agreeing only in the far-field a^3/r^4 asymptote.

Everything except alpha and q_i^2 is folded into the stored grid, so the force
kernel computes alpha*q^2*phi and -alpha*q^2*grad(phi).
The stored field is in kBT per e^2, is positive everywhere, and decays as
1/r^4, so the force is repulsive for every atom regardless of charge sign.
"""

from __future__ import annotations
import math
import numpy as np

COULOMB_KCAL = 332.0637  # e^2/(4 pi eps0), kcal/mol.Angstrom/e^2
KB_KCAL = 0.0019872041  # kcal/mol/K
DEN_FLOOR = 0.5  # floor on (r^2 - a^2), Angstrom^2
DEFAULT_DEBYE = 7.858  # Angstrom, 150 mM


def dielectric_factor(eps_p: float = 4.0, eps_s: float = 78.0) -> float:
    return (eps_s - eps_p) / (eps_s * (2.0 * eps_s + eps_p))


def coulomb_kbt(temp: float = 298.15) -> float:
    """e^2/(4 pi eps_0) divided by kT, in kBT.Angstrom/e^2."""
    return COULOMB_KCAL / (KB_KCAL * temp)


def _xp():
    """CuPy if usable, else NumPy."""
    try:
        import cupy as cp

        cp.zeros(1)
        return cp, True
    except Exception:
        return np, False


def desolvation_field_on_grid(
    origin,
    spacing,
    dime,
    atom_xyz: np.ndarray,
    atom_rad: np.ndarray,
    eps_p: float = 4.0,
    eps_s: float = 78.0,
    temp: float = 298.15,
    debye_length: float = DEFAULT_DEBYE,
    cutoff: float = 15.0,
    chunk: int = 4_000_000,
) -> np.ndarray:
    """
    Desolvation field on a regular grid, in kBT per e^2, shape (nx,ny,nz).

    Screened by (1 + k r)^2 exp(-2 k r) with k = 1/debye_length. Contributions
    past `cutoff` are dropped; with screening at 150 mM a 1.7 A atom contributes
    about 3e-5 kBT/e^2 at 15 A, roughly four orders below its contact value.
    """
    xp, on_gpu = _xp()
    nx, ny, nz = (int(d) for d in dime)
    # Prefactor for the Kirkwood n=1 image self energy,
    #     dG = (1/2) * C * D * q^2 * a^3 / r^4 ,  coefficient 1.6616 kBT.A^4/e^2.
    #
    # alpha has a base of 1.0 and carries the model's approximations: the
    # sphere sum overcounts the true
    # solvent-excluded volume by 1.44x, and keeping only the diagonal of the
    # squared molecular field overcounts by a further ~1.37x for a +1 ligand.
    scale = 0.5 * coulomb_kbt(temp) * dielectric_factor(eps_p, eps_s)
    kap = 0.0 if not debye_length else 1.0 / float(debye_length)

    lo = np.array(origin, dtype=np.float64) - cutoff
    hi = (
        lo
        + np.array(spacing, dtype=np.float64) * (np.array([nx, ny, nz]) - 1)
        + 2 * cutoff
    )
    # Drop zero-radius atoms. PySTARC injects ghost atoms at molecule centroids
    # with radius 0. They contribute a^3 = 0, but leaving them in makes the
    # r^-4 kernel evaluate 0/0 at their own grid point and NaN-poison the field.
    keep = np.all((atom_xyz >= lo) & (atom_xyz <= hi), axis=1) & (atom_rad > 0.0)
    ax = xp.asarray(atom_xyz[keep], dtype=xp.float64)
    a2 = xp.asarray((atom_rad[keep] ** 2), dtype=xp.float64)
    a3 = xp.asarray((atom_rad[keep] ** 3), dtype=xp.float64)
    if ax.shape[0] == 0:
        return np.zeros((nx, ny, nz), dtype=np.float64)

    x = xp.asarray(origin[0] + spacing[0] * np.arange(nx), dtype=xp.float64)
    y = xp.asarray(origin[1] + spacing[1] * np.arange(ny), dtype=xp.float64)
    z = xp.asarray(origin[2] + spacing[2] * np.arange(nz), dtype=xp.float64)

    out = xp.zeros(nx * ny * nz, dtype=xp.float64)
    c2 = cutoff * cutoff
    npts = nx * ny * nz
    nyz = ny * nz

    for s in range(0, npts, chunk):
        e = min(s + chunk, npts)
        idx = xp.arange(s, e)
        gx = x[idx // nyz]
        gy = y[(idx % nyz) // nz]
        gz = z[idx % nz]
        acc = xp.zeros(e - s, dtype=xp.float64)
        for j in range(int(ax.shape[0])):
            d2 = (gx - ax[j, 0]) ** 2 + (gy - ax[j, 1]) ** 2 + (gz - ax[j, 2]) ** 2
            d = xp.sqrt(d2)
            # Screened image kernel: a^3 (1 + r/L)^2 exp(-2 r/L) / r^4. Floor r at the atom
            # radius so the interior saturates instead of diverging. That region is
            # inside the excluded volume and is never sampled by a physical trajectory.
            r2 = xp.maximum(d2, xp.maximum(a2[j], DEN_FLOOR))
            screen = (1.0 + kap * d) ** 2 * xp.exp(-2.0 * kap * d)
            acc += xp.where(d2 < c2, a3[j] * screen / (r2 * r2), 0.0)
        out[s:e] = acc

    out = out.reshape(nx, ny, nz) * scale
    return xp.asnumpy(out) if on_gpu else out


def read_pqr_geometry(pqr_path):
    """
    Atom centres and radii from a PQR (charges deliberately ignored).

    PQR has no fixed standard for a trailing element column: some writers emit
    "... x y z q r" and others "... x y z q r element". Detect which by testing
    whether the final field parses as a number, otherwise the radius is read one
    field off and becomes the charge.
    """
    xyz, rad = [], []
    for line in open(pqr_path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        p = line.split()
        try:
            float(p[-1])
            off = 0  # no element column; radius is the last field
        except ValueError:
            off = 1  # trailing element symbol
        rad.append(float(p[-1 - off]))
        xyz.append([float(p[-5 - off]), float(p[-4 - off]), float(p[-3 - off])])
    return np.asarray(xyz, dtype=np.float64), np.asarray(rad, dtype=np.float64)


def probe_contact_value(
    field,
    origin,
    spacing,
    dime,
    atom_xyz,
    atom_rad,
    eps_p=4.0,
    eps_s=78.0,
    temp=298.15,
    debye_length=DEFAULT_DEBYE,
):
    """
    A physically meaningful diagnostic: the field one vdW radius outside the
    outermost atom along +x. field.max() is always the DEN_FLOOR-clamped
    interior plateau and says nothing about what a ligand actually samples.
    """
    j = int(np.argmax(atom_xyz[:, 0]))
    r = float(atom_rad[j]) * 2.0
    pt = atom_xyz[j] + np.array([r, 0.0, 0.0])
    v = desolvation_field_on_grid(
        pt - 1e-3,
        [2e-3] * 3,
        [2, 2, 2],
        atom_xyz,
        atom_rad,
        eps_p,
        eps_s,
        temp,
        debye_length,
    )
    return float(v[0, 0, 0]), r


def write_dx(path, field: np.ndarray, origin, spacing, dime):
    """OpenDX scalar field in the same layout APBS writes."""
    nx, ny, nz = (int(d) for d in dime)
    v = np.asarray(field, dtype=np.float64).reshape(-1)
    with open(path, "w") as f:
        f.write("# Image (Born) desolvation field, kBT per e^2\n")
        f.write(
            "# Cavity self energy from partner radii; independent of partner charges\n"
        )
        f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        f.write("origin %.6e %.6e %.6e\n" % tuple(float(o) for o in origin))
        f.write("delta %.6e 0.000000e+00 0.000000e+00\n" % float(spacing[0]))
        f.write("delta 0.000000e+00 %.6e 0.000000e+00\n" % float(spacing[1]))
        f.write("delta 0.000000e+00 0.000000e+00 %.6e\n" % float(spacing[2]))
        f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        f.write(
            f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n"
        )
        for i in range(0, v.size, 3):
            f.write(" ".join("%.6e" % t for t in v[i : i + 3]) + "\n")
        f.write('attribute "dep" string "positions"\n')
        f.write('object "regular positions regular connections" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')
