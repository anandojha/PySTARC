"""Figures of a finished run (rate, radial, first_hitting, bd_time, steps), drawn from the summary files of its bd_sims folder."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

NAVY, RED, GOLD, BLUE = "#2c3e50", "#c0392b", "#d4a13a", "#4a6fa5"
Z95 = 1.959964
AVOGADRO_CONV = 6.02214076e8

FIGURES = ("rate", "radial", "first_hitting", "bd_time", "steps")

_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.weight": "bold",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.handlelength": 1.4, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3,
    "legend.borderpad": 0.3, "legend.borderaxespad": 0.3,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial:bold",
    "mathtext.it": "Arial:bold:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
}
DEFAULT_DPI = 500
PAGE = (6.8, 4.6)
PLOT_BOX = (0.95, 0.72, 0.2, 0.12)
KON = r"$\mathit{k}_\mathrm{on}$"
KON_UNIT = r"$\mathit{k}_\mathrm{on}\ \mathrm{(M^{-1}\ s^{-1})}$"


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Columns of a CSV as arrays, numeric where every entry parses as a number."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if header is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        col = [row[j] if j < len(row) else "" for row in rows]
        try:
            out[name] = np.array(col, dtype=float)
        except ValueError:
            out[name] = np.array(col, dtype=str)
    return out


def wilson(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimate and 95 % Wilson interval of k successes in n trials."""
    k = np.asarray(k, dtype=float)
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    p = k / n
    denom = 1.0 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2.0 * n)) / denom
    half = Z95 * np.sqrt(p * (1.0 - p) / n + Z95 * Z95 / (4.0 * n * n)) / denom
    return p, centre - half, centre + half


def _decimals(step: float) -> int:
    return max(0, -int(np.floor(np.log10(step) + 1e-12)))


def _ends_even(value: float, step: float) -> bool:
    """True when the number ends in an even digit."""
    digits = f"{abs(value):.{_decimals(step)}f}".replace(".", "")
    return int(digits[-1]) % 2 == 0


def even_ticks(top: float, bottom: float = 0.0, max_n: int = 6) -> np.ndarray:
    """4 to max_n even axis numbers from bottom, ending on the first even number at or above top."""
    span = top - bottom
    if not np.isfinite(span) or span <= 0:
        return np.array([bottom])
    e0 = int(np.floor(np.log10(span)))
    best = None
    for rank, steps in enumerate(((2, 4, 10), (6, 8))):
        for e in range(e0 - 2, e0 + 2):
            for m in steps:
                step = m * 10.0**e
                first = np.floor(bottom / step + 1e-9) * step
                last = np.ceil(top / step - 1e-9) * step
                while not _ends_even(last, step):
                    last += step
                n = int(round((last - first) / step)) + 1
                if 4 <= n <= max_n:
                    ticks = np.round(first + step * np.arange(n), 12) + 0.0
                    key = (round(float(ticks[-1] - ticks[0]), 12), rank, -n)
                    if best is None or key < best[0]:
                        best = (key, ticks)
    if best is not None:
        return best[1]
    return np.round(np.linspace(bottom, top, 5), 12) + 0.0


def _tick_text(step: float) -> Callable:
    dec = _decimals(step)
    return lambda v, _p: f"{v:.{dec}f}".replace("-", "\u2212")


def linear_axis(ax, axis: str, lo: float, hi: float, formatter: Callable | None = None) -> None:
    """Limits and numbers of a linear axis for data from lo to hi."""
    ticks = even_ticks(hi, lo)
    first, last = (ticks[0], ticks[-1]) if len(ticks) > 1 else (lo, max(hi, lo + 1.0))
    room = 0.01 * (last - first)
    (ax.set_xlim if axis == "x" else ax.set_ylim)(first - room, last + room)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    step = ticks[1] - ticks[0] if len(ticks) > 1 else max(abs(hi), 1.0)
    base = formatter if formatter else _tick_text(step)
    target.set_major_formatter(FuncFormatter(lambda v, pos: base(0.0 if abs(v) < 1e-12 else v, pos)))


def log_powers(lo: float, hi: float) -> list[int] | None:
    """Labelled powers of ten for data from lo to hi, from and to even powers; None when fewer than 4."""
    lo_e = int(np.floor(np.log10(lo) + 1e-9))
    lo_e -= lo_e % 2
    hi_e = int(np.ceil(np.log10(hi) - 1e-9))
    hi_e += hi_e % 2
    evens = list(range(lo_e, hi_e + 1, 2))
    if 4 <= len(evens) <= 6:
        return evens
    if len(evens) > 6:
        k = 4
        while (hi_e - lo_e) / k + 1 > 6:
            k += 2
        n = int(np.ceil((hi_e - lo_e) / k)) + 1
        return [hi_e - k * i for i in range(n)][::-1]
    every = list(range(lo_e, hi_e + 1))
    return every if len(every) >= 4 else None


def _power_text(v, _p):
    if v == 0:
        return "0"
    return rf"$\mathregular{{10^{{{int(round(np.log10(abs(v))))}}}}}$"


def log_axis(ax, axis: str, lo: float, hi: float) -> None:
    """Log axis between even powers around lo and hi; linear from 0 when log_powers gives None."""
    powers = log_powers(lo, hi)
    if powers is None:
        linear_axis(ax, axis, 0.0, hi)
        return
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    span = powers[-1] - powers[0]
    (ax.set_xlim if axis == "x" else ax.set_ylim)(10 ** (powers[0] - 0.01 * span), 10 ** (powers[-1] + 0.01 * span))
    majors = [10.0**k for k in powers]
    minors = [m * 10.0**k for k in range(powers[0], powers[-1]) for m in range(1, 10) if m * 10.0**k not in majors]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(majors))
    target.set_minor_locator(FixedLocator(minors))
    target.set_major_formatter(FuncFormatter(_power_text))
    target.set_minor_formatter(NullFormatter())


def page_axes(fig):
    """The plot box of a PAGE figure."""
    w, h = fig.get_size_inches()
    left, bottom, right, top = PLOT_BOX
    return fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h])


def fine_bins(values, lo: float = 0.0, hi: float | None = None) -> np.ndarray:
    """Histogram edges from lo with one round width, about 2.5 sqrt(N) bins, 20 to 60."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([lo, lo + 1.0])
    top = float(v.max()) if hi is None else float(hi)
    span = top - lo
    if span <= 0:
        return np.array([lo, lo + 1.0])
    target = int(np.clip(2.5 * np.sqrt(len(v)), 20, 60))
    e0 = int(np.floor(np.log10(span / target)))
    best = None
    for e in range(e0 - 1, e0 + 2):
        for m in (1.0, 2.0, 2.5, 4.0, 5.0):
            w = m * 10.0**e
            k = int(np.ceil(span / w - 1e-9))
            if best is None or abs(k - target) < best[0]:
                best = (abs(k - target), w, k)
    _, w, k = best
    return lo + w * np.arange(k + 1)


def place_legend(ax, x, ys, extra=None, loc: str = "lower right") -> None:
    """Legend at loc, moved to the best spot when it would cover a plotted point."""
    leg = ax.legend(loc=loc)
    renderer = ax.figure.canvas.get_renderer()
    box = leg.get_window_extent(renderer)
    sets = [np.column_stack([np.asarray(x, float), np.asarray(y, float)]) for y in ys]
    if extra is not None:
        sets.append(np.asarray(extra, float).reshape(-1, 2))
    for xy in sets:
        pts = ax.transData.transform(xy)
        inside = (pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)
        if np.any(inside):
            leg.remove()
            ax.legend(loc="best")
            return


class RunFigures:
    """Figures of one run folder; each draw_* returns None, a note or a skip reason."""

    def __init__(self, run: str | Path):
        self.run = Path(run)
        self.results = {}
        rp = self.run / "results.json"
        if rp.exists():
            self.results = json.load(open(rp))
        self.k_b = self.results.get("k_b")
        self.p_rxn = self.results.get("P_rxn")
        self.k_on = self.results.get("k_on")
        self.r_start = self.results.get("r_start")
        self.r_escape = self.results.get("r_escape")
        self._traj = None

    def _trajectories(self) -> dict[str, np.ndarray]:
        if self._traj is None:
            self._traj = read_csv_columns(self.run / "trajectories.csv")
        return self._traj

    def _rate_scale(self) -> float | None:
        """Factor from P_rxn to k_on."""
        if self.k_b:
            return float(self.k_b) * AVOGADRO_CONV
        if self.k_on and self.p_rxn:
            return float(self.k_on) / float(self.p_rxn)
        return None

    def _encounters(self) -> dict[str, np.ndarray] | None:
        p = self.run / "encounters.csv"
        if not p.exists():
            return None
        enc = read_csv_columns(p)
        return enc if enc and len(enc.get("x", [])) else None

    def draw_rate(self, ax) -> str | None:
        t = self._trajectories()
        outcome = np.char.lower(t["outcome"].astype(str))
        reacted = np.char.startswith(outcome, "react").astype(np.int64)
        n_total = len(reacted)
        if n_total == 0:
            return "skipped: no trajectories"
        cum = np.cumsum(reacted)
        n_points = np.unique(np.concatenate([np.arange(1, min(10, n_total) + 1),
                                             np.logspace(1, np.log10(n_total), 80).astype(int)]))
        n_points = n_points[n_points <= n_total]
        p, lo, hi = wilson(cum[n_points - 1], n_points)
        scale = self._rate_scale()
        unit = scale if scale else 1.0
        label_y = KON_UNIT if scale else r"$\mathit{P}_\mathrm{rxn}$"
        if scale:
            ref = cum[-1] / n_total * unit if cum[-1] > 0 else hi[-1] * unit
            expo = int(np.floor(np.log10(ref))) if ref > 0 else 0
            unit = unit / 10**expo
            label_y = rf"$\mathit{{k}}_\mathrm{{on}}\ \mathrm{{(10^{{{expo}}}\ M^{{-1}}\ s^{{-1}})}}$"
        ax.fill_between(n_points, lo * unit, hi * unit, color=BLUE, alpha=0.22, lw=0, label="95% Wilson interval")
        ax.plot(n_points, p * unit, color=NAVY, lw=1.8, marker="s", ms=3.0, label="Cumulative estimate")
        final = cum[-1] / n_total
        if cum[-1] > 0:
            ax.axhline(final * unit, color=RED, ls="--", lw=1.8, label="Full ensemble estimate")
            data_top = max(p.max(), final) * unit
        else:
            ax.axhline(hi[-1] * unit, color=RED, ls="--", lw=1.8, label="95% upper bound")
            data_top = hi[-1] * unit
        linear_axis(ax, "y", 0.0, data_top)
        log_axis(ax, "x", 1.0, float(n_total))
        ax.set_xlabel("Trajectories")
        ax.set_ylabel(label_y)
        place_legend(ax, n_points, [p * unit, lo * unit])
        return None

    def draw_radial(self, ax) -> str | None:
        p = self.run / "radial_density.csv"
        if not p.exists():
            return "skipped: radial_density.csv missing"
        rad = read_csv_columns(p)
        if "time_ps" not in rad:
            return "skipped: radial_density.csv has no time_ps column"
        # Time per shell volume. The step counts weight short and long steps
        # alike, so only the time gives the density.
        r = rad["r_center"]
        dens = rad["time_ps"] / (4.0 / 3.0 * np.pi * (rad["r_high"] ** 3 - rad["r_low"] ** 3))
        if self.r_escape:
            inside = r <= float(self.r_escape)
            r, dens = r[inside], dens[inside]
        if self.r_start:
            near_b = (r > self.r_start - 5) & (r < self.r_start + 5)
            ref = dens[near_b].mean() if near_b.any() else dens[dens > 0].mean()
        else:
            ref = dens[r > r.max() * 0.6].mean()
        if not ref or not np.isfinite(ref):
            return "skipped: radial density is empty"
        g = dens / ref
        positive = g[g > 0]
        if positive.size == 0:
            return "skipped: radial density is empty"
        g_lo = max(positive.min(), positive.max() * 1e-10)
        ax.plot(r, np.where(g > 0, g, np.nan), color=NAVY, lw=1.8)
        ax.axhline(1.0, color="k", ls=":", lw=1.2)
        log_axis(ax, "y", g_lo, positive.max())
        floor = ax.get_ylim()[0]
        ax.fill_between(r, np.where(g > 0, g, floor), floor, where=g > 0, color=BLUE, alpha=0.15)
        r_max = float(self.r_escape) if self.r_escape else float(r[g > 0].max())
        linear_axis(ax, "x", 0.0, r_max)
        y_lo, y_hi = np.log10(ax.get_ylim())
        y_text = 10 ** (y_lo + 0.03 * (y_hi - y_lo))
        if self.r_start:
            ax.axvline(self.r_start, color=RED, ls="--", lw=1.6)
            ax.text(self.r_start + 1, y_text, r"$\mathit{b}$ surface", color=RED, fontsize=_STYLE["legend.fontsize"], rotation=90, va="bottom")
        if self.r_escape:
            ax.axvline(self.r_escape, color="0.5", ls="--", lw=1.4)
            ax.text(self.r_escape - 1, y_text, "escape surface", color="0.4", fontsize=_STYLE["legend.fontsize"], rotation=90, va="bottom", ha="right")
        ax.set_xlabel(r"Solute–solute centroid distance $\mathit{r}$ (Å)")
        ax.set_ylabel(r"Time density relative to the $\mathit{b}$ surface")
        return None

    def draw_bd_time(self, ax) -> str | None:
        p = self.run / "bd_time_to_reaction.csv"
        if not p.exists():
            return "skipped: bd_time_to_reaction.csv missing"
        times = read_csv_columns(p).get("bd_time_ps", np.array([]))
        if len(times) == 0:
            ax.text(0.5, 0.5, "no reactions", ha="center", va="center", transform=ax.transAxes, color="0.4")
            ax.set_xlabel(r"BD time inside $\mathit{r}_\mathrm{esc}$ (ns)")
            ax.set_ylabel(r"Probability density (ns$^{-1}$)")
            return "note: no reactions"
        ns = times / 1000.0
        med = float(np.median(ns))
        edges = fine_bins(ns)
        dens, edges, _ = ax.hist(ns, bins=edges, density=True, color=BLUE, edgecolor="white", lw=0.4,
                                 label="Reactive trajectories")
        ax.axvline(med, color=RED, ls="--", lw=1.8, label="Median")
        linear_axis(ax, "x", 0.0, float(ns.max()))
        linear_axis(ax, "y", 0.0, float(dens.max()))
        ax.set_xlabel(r"BD time inside $\mathit{r}_\mathrm{esc}$ (ns)")
        ax.set_ylabel(r"Probability density (ns$^{-1}$)")
        centers = 0.5 * (edges[:-1] + edges[1:])
        line = np.column_stack([np.full(25, med), np.linspace(0.0, float(dens.max()), 25)])
        place_legend(ax, centers, [dens, dens / 2.0], extra=line)
        return None

    def draw_steps(self, ax) -> str | None:
        """BD steps per trajectory, escaped and reactive, each a probability density over log10 steps."""
        t = self._trajectories()
        if "n_steps" not in t:
            return "skipped: trajectories.csv has no n_steps column"
        steps = np.asarray(t["n_steps"], float)
        outcome = np.char.lower(t["outcome"].astype(str))
        ok = steps >= 1
        groups = [(np.char.startswith(outcome, "escape") & ok, GOLD, "Escaped trajectories"),
                  (np.char.startswith(outcome, "react") & ok, BLUE, "Reactive trajectories")]
        shown = groups[0][0] | groups[1][0]
        if not shown.any():
            return "skipped: no escaped or reactive trajectories"
        lg = np.log10(np.where(ok, steps, 1.0))
        edges = fine_bins(lg[shown])
        x = 10.0 ** edges
        centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
        dens = []
        for mask, colour, label in groups:
            if mask.any():
                d, _ = np.histogram(lg[mask], bins=edges, density=True)
                ax.bar(x[:-1], d, width=np.diff(x), align="edge", color=colour, alpha=0.7,
                       edgecolor="white", lw=0.4, label=label)
                dens.append(d)
        log_axis(ax, "x", float(steps[shown].min()), float(steps[shown].max()))
        linear_axis(ax, "y", 0.0, float(max(d.max() for d in dens)))
        ax.set_xlabel("BD steps per trajectory")
        ax.set_ylabel("Probability density")
        place_legend(ax, centers, dens, loc="upper left")
        notes = []
        if not groups[1][0].any():
            notes.append("no reactions, escaped only")
        other = int((~shown).sum())
        if other:
            notes.append(f"{other} trajectories that neither reacted nor escaped are not shown")
        return "note: " + "; ".join(notes) if notes else None

    def _hits(self):
        """Ligand centroid positions and unit quaternions at the first hit."""
        enc = self._encounters()
        if enc is None:
            return None
        pos = np.column_stack([enc["x"], enc["y"], enc["z"]]).astype(float)
        quat = np.column_stack([enc["q0"], enc["q1"], enc["q2"], enc["q3"]]).astype(float)
        return pos, quat / np.linalg.norm(quat, axis=1, keepdims=True)

    def _receptor_atoms(self):
        """Receptor heavy atoms (positions, radii, serials) from receptor.pqr."""
        from pystarc.structures.pqr_io import parse_pqr_records
        p = self.run / "receptor.pqr"
        if not p.exists():
            return None
        recs = parse_pqr_records(p)
        keep = [i for i, r in enumerate(recs) if r.name.strip().upper() not in ("GHO", "PKT") and not r.name.strip().upper().startswith("H")]
        pos = np.array([[recs[i].x, recs[i].y, recs[i].z] for i in keep], float)
        rad = np.array([recs[i].radius for i in keep], float)
        serial = np.array([i + 1 for i in keep], int)
        return pos, rad, serial

    @staticmethod
    def _blur(a: np.ndarray, sigma_px: float) -> np.ndarray:
        """Gaussian blur by three box passes along each axis."""
        width = max(1, int(round(np.sqrt(12.0 * sigma_px**2 / 3.0 + 1.0))))
        out = a.astype(float)
        for axis in (0, 1):
            for _ in range(3):
                c = np.cumsum(np.insert(out, 0, 0.0, axis=axis), axis=axis)
                n = out.shape[axis]
                lo = np.clip(np.arange(n) - width // 2, 0, n)
                hi = np.clip(np.arange(n) + width - width // 2, 0, n)
                take = (lambda idx: c[idx, :]) if axis == 0 else (lambda idx: c[:, idx])
                span = (hi - lo).astype(float)
                out = (take(hi) - take(lo)) / (span[:, None] if axis == 0 else span[None, :])
        return out

    @staticmethod
    def _space_fill(ax, uvd, rad, xlim, ylim, npx: int = 1200, smooth_A: float = 0.45,
                    colour=(0.66, 0.74, 0.86), ambient: float = 0.30, occlusion: float = 0.50,
                    lines: bool = False) -> None:
        """Lit molecular surface of the atoms, ray cast and smoothed, clear outside the molecule."""
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        px = (xlim[1] - xlim[0]) / npx
        npy = max(1, int(round((ylim[1] - ylim[0]) / px)))
        nm = int(np.ceil(4.0 / px))
        nx, ny = npx + 2 * nm, npy + 2 * nm
        x_start, y_start = xlim[0] - nm * px, ylim[0] - nm * px
        xs = x_start + px * (np.arange(nx) + 0.5)
        ys = y_start + px * (np.arange(ny) + 0.5)
        zbuf = np.full((ny, nx), -np.inf)
        for (cx, cy, cz), r in zip(uvd, rad):
            i0, i1 = max(0, int((cx - r - x_start) / px)), min(nx, int((cx + r - x_start) / px) + 2)
            j0, j1 = max(0, int((cy - r - y_start) / px)), min(ny, int((cy + r - y_start) / px) + 2)
            if i0 >= i1 or j0 >= j1:
                continue
            dx = xs[i0:i1][None, :] - cx
            dy = ys[j0:j1][:, None] - cy
            h2 = r * r - dx * dx - dy * dy
            z = np.where(h2 > 0, cz + np.sqrt(np.clip(h2, 0.0, None)), -np.inf)
            np.maximum(zbuf[j0:j1, i0:i1], z, out=zbuf[j0:j1, i0:i1])
        hit = np.isfinite(zbuf)
        if not hit.any():
            return
        blur = RunFigures._blur
        zmin = float(zbuf[hit].min())
        m = hit.astype(float)
        sig = smooth_A / px
        wsum = np.maximum(blur(m, sig), 1e-6)
        zs = blur(np.where(hit, zbuf - zmin, 0.0), sig) / wsum
        inside = blur(m, 0.5 / px) > 0.5
        gy, gx = np.gradient(zs, px)
        nrm = np.dstack([-gx, -gy, np.ones_like(zs)])
        nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
        light = np.array([-0.45, 0.55, 0.70])
        light /= np.linalg.norm(light)
        half = light + np.array([0.0, 0.0, 1.0])
        half /= np.linalg.norm(half)
        diffuse = np.clip(nrm @ light, 0.0, 1.0)
        spec = np.clip(nrm @ half, 0.0, 1.0) ** 30
        depth = zs / max(float(zs[inside].max()), 1e-9)
        ins = inside.astype(float)
        occl = np.clip((blur(zs * ins, 2.0 / px) / np.maximum(blur(ins, 2.0 / px), 1e-6) - zs) / 3.0, 0.0, 1.0)
        lum = (ambient + (1.0 - ambient) * diffuse) * (0.86 + 0.14 * depth) * (1.0 - occlusion * occl)
        if lines:
            crease = np.clip((blur(zs * ins, 0.6 / px) / np.maximum(blur(ins, 0.6 / px), 1e-6) - zs - 0.35) / 0.25, 0.0, 1.0)
            edge = np.clip(1.0 - np.abs(blur(ins, 0.08 / px) - 0.5) * 4.0, 0.0, 1.0) * ins
            lum = lum * (1.0 - 0.65 * np.maximum(crease, edge))
        cx_, cy_ = slice(nm, nm + npx), slice(nm, nm + npy)
        img = np.zeros((npy, npx, 4))
        for c_ in range(3):
            img[..., c_] = np.clip(colour[c_] * lum + 0.12 * spec, 0.0, 1.0)[cy_, cx_]
        img[..., 3] = ins[cy_, cx_]
        ax.imshow(img, extent=(xlim[0], xlim[1], ylim[0], ylim[0] + npy * px), origin="lower",
                  interpolation="bilinear", zorder=1)

    @staticmethod
    def _scale_bar(ax, xlim, ylim, choices, below: bool = False) -> None:
        """Scale bar in Å, inside the lower left corner or under the panel."""
        from matplotlib import patheffects
        span = xlim[1] - xlim[0]
        length = next((v for v in choices if v >= 0.18 * span), choices[-1])
        if below:
            per_in = span / (ax.get_position().width * ax.figure.get_size_inches()[0])
            x0, y0 = xlim[0], ylim[0] - 0.2 * per_in
            ax.plot([x0, x0 + length], [y0, y0], color="black", lw=3, solid_capstyle="butt",
                    clip_on=False, zorder=6)
            ax.text(x0 + length + 0.08 * per_in, y0, f"{length:g} Å", ha="left", va="center",
                    fontsize=_STYLE["legend.fontsize"], clip_on=False, zorder=6)
            return
        x0 = xlim[0] + 0.04 * span
        y0 = ylim[0] + 0.04 * (ylim[1] - ylim[0])
        halo = [patheffects.withStroke(linewidth=4, foreground="white")]
        ax.plot([x0, x0 + length], [y0, y0], color="black", lw=3, solid_capstyle="butt", zorder=6,
                path_effects=halo)
        ax.text(x0 + length / 2, y0 + 0.015 * span, f"{length:g} Å", ha="center", va="bottom",
                fontsize=_STYLE["legend.fontsize"], zorder=6, path_effects=halo)

    def draw_first_hitting(self, ax) -> str | None:
        """First hitting point distribution: the receptor seen from the mean hit, with a lens on the hit density."""
        hits = self._hits()
        if hits is None:
            return "skipped: no reactions, encounters.csv empty or missing"
        atoms = self._receptor_atoms()
        if atoms is None:
            return "skipped: receptor.pqr missing"
        pos, _ = hits
        apos, arad, _ = atoms
        m = pos.mean(axis=0)
        e3 = m / np.linalg.norm(m)
        rel = pos - m
        flat = rel - np.outer(rel @ e3, e3)
        e1 = np.linalg.eigh(flat.T @ flat)[1][:, -1]
        e2 = np.cross(e3, e1)
        frame = np.column_stack([e1, e2, e3])
        uvd = apos @ frame
        h_uv = pos @ frame[:, :2]
        c0 = h_uv.mean(axis=0)
        lh = float(np.abs(h_uv - c0).max()) + 1.5
        lx, ly = (c0[0] - lh, c0[0] + lh), (c0[1] - lh, c0[1] + lh)
        bw = 0.7 * h_uv.std(axis=0, ddof=1) * len(h_uv) ** (-1.0 / 6.0)
        pix = 0.2
        nb = int(np.ceil(lh / pix))
        xe = c0[0] + pix * np.arange(-nb, nb + 1)
        ye = c0[1] + pix * np.arange(-nb, nb + 1)
        Xc, Yc = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
        Zc = np.zeros_like(Xc)
        for u_, v_ in h_uv:
            Zc += np.exp(-0.5 * (((Xc - u_) / bw[0]) ** 2 + ((Yc - v_) / bw[1]) ** 2))
        Zc /= len(h_uv) * 2 * np.pi * bw[0] * bw[1]
        P = np.ma.masked_less(Zc, 0.03 * Zc.max())
        cticks = even_ticks(float(Zc.max()), max_n=5)
        vmax = float(cticks[-1])
        dens_cmap = matplotlib.colors.ListedColormap(plt.get_cmap("magma")(np.linspace(0.12, 0.98, 256)))
        line_levels = list(float(Zc.max()) * np.array([0.25, 0.50, 0.75]))
        fig = ax.figure
        w, h = fig.get_size_inches()
        on_page = tuple(np.round((w, h), 3)) == PAGE
        gap, bar_block = 0.3, 1.25
        side = min((w - 0.2 - gap - bar_block) / 2, h - 0.65)
        label = r"Probability density (Å$^{-2}$)"
        tall = side
        lens = None
        if on_page:
            probe = fig.text(0, 0, label, fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")
            need = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + 0.08
            probe.remove()
            tall = min(h - 0.65, max(side, need))
            x0, y0 = 0.1, (h - tall - 0.35) / 2 + 0.35
            ax.set_position([x0 / w, y0 / h, side / w, tall / h])
            lens = fig.add_axes([(x0 + side + gap) / w, y0 / h, side / w, tall / h])
        aspect = tall / side
        ly = (c0[1] - lh * aspect, c0[1] + lh * aspect)
        rmax = float(arad.max())
        cx = 0.5 * (uvd[:, 0].min() + uvd[:, 0].max())
        cy = 0.5 * (uvd[:, 1].min() + uvd[:, 1].max())
        half = 1.03 * (max(np.ptp(uvd[:, 0]), np.ptp(uvd[:, 1])) / 2 + rmax)
        xlim, ylim = (cx - half, cx + half), (cy - half * aspect, cy + half * aspect)
        self._space_fill(ax, uvd, arad, xlim, ylim)
        ax.set_axis_off()
        edge_level = 0.03 * float(Zc.max())
        ax.contourf(Xc, Yc, Zc, levels=[edge_level, float(Zc.max()) * 1.01], colors=["#FB9B4B"], zorder=3)
        ax.contour(Xc, Yc, Zc, levels=[edge_level], colors="black", linewidths=0.8, zorder=4)
        sc = None
        self._scale_bar(ax, xlim, ylim, (5, 10, 20, 50, 100), below=on_page)
        if lens is not None:
            from matplotlib.patches import ConnectionPatch, Rectangle
            ax.add_patch(Rectangle((lx[0], ly[0]), lx[1] - lx[0], ly[1] - ly[0], fill=False, edgecolor="black",
                                   lw=1.4, zorder=5))
            self._space_fill(lens, uvd, arad, lx, ly, ambient=0.40, occlusion=0.25)
            sc = lens.pcolormesh(xe, ye, P, cmap=dens_cmap, vmin=0.0, vmax=vmax, zorder=3)
            from matplotlib import patheffects
            from matplotlib.patches import PathPatch
            halo = [patheffects.withStroke(linewidth=3.2, foreground="white")]
            edge = lens.contour(Xc, Yc, Zc, levels=[0.03 * float(Zc.max())], colors="black", linewidths=1.4, zorder=4)
            edge.set(path_effects=halo)
            clip = PathPatch(edge.get_paths()[0], transform=lens.transData, facecolor="none", edgecolor="none")
            lens.add_patch(clip)
            sc.set_clip_path(clip)
            cs_ = lens.contour(Xc, Yc, Zc, levels=line_levels, colors="black", linewidths=1.4, zorder=4)
            cs_.set(path_effects=halo)
            lens.set_xticks([])
            lens.set_yticks([])
            for sp in lens.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1.4)
            self._scale_bar(lens, lx, ly, (1, 2, 5, 10), below=True)
            for ya, yb in ((ly[1], 1.0), (ly[0], 0.0)):
                fig.add_artist(ConnectionPatch(xyA=(lx[1], ya), coordsA="data", axesA=ax, xyB=(0.0, yb),
                                               coordsB="axes fraction", axesB=lens, color="black", lw=1.0))
            lb = lens.get_position()
            cax = fig.add_axes([lb.x1 + 0.2 / w, lb.y0, 0.16 / w, lb.height])
            bar = fig.colorbar(sc, cax=cax, ticks=cticks)
            bar.ax.yaxis.set_major_formatter(FuncFormatter(_tick_text(cticks[1] - cticks[0] if len(cticks) > 1 else 1.0)))
        else:
            sc = ax.pcolormesh(xe, ye, P, cmap=dens_cmap, vmin=0.0, vmax=vmax, zorder=5)
            bar = fig.colorbar(sc, ax=ax, ticks=cticks)
            bar.ax.yaxis.set_major_formatter(FuncFormatter(_tick_text(cticks[1] - cticks[0] if len(cticks) > 1 else 1.0)))
        bar.set_label(label)
        bar.outline.set_linewidth(1.2)
        if lens is not None:
            fig.canvas.draw()
            r_edge = bar.ax.get_tightbbox(fig.canvas.get_renderer()).x1 / fig.dpi / w
            left = ax.get_position().x0
            shift = (1 - (r_edge - left)) / 2 - left
            for a_ in (ax, lens, cax):
                b_ = a_.get_position()
                a_.set_position([b_.x0 + shift, b_.y0, b_.width, b_.height])
        return None

    def save_all(self, out: str | Path, dpi: int = DEFAULT_DPI, which=FIGURES) -> dict[str, str]:
        """Write fig_<name>.png per figure; returns {name: 'ok' or the note}."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        status: dict[str, str] = {}
        with plt.rc_context(_STYLE):
            for name in which:
                draw: Callable = getattr(self, f"draw_{name}")
                fig = plt.figure(figsize=PAGE)
                ax = page_axes(fig)
                try:
                    note = draw(ax)
                except FileNotFoundError as exc:
                    note = f"skipped: missing {Path(str(exc).split()[-1]).name}"
                if note is None or note.startswith("note"):
                    fig.savefig(out / f"fig_{name}.png", dpi=dpi, facecolor="white")
                plt.close(fig)
                status[name] = "ok" if note is None else note
        return status


def plot_run(run: str | Path, out: str | Path | None = None, dpi: int = DEFAULT_DPI, which=FIGURES) -> dict[str, str]:
    """Draw the figures of a run folder into out (default <run>/figures)."""
    run = Path(run)
    out = Path(out) if out else run / "figures"
    return RunFigures(run).save_all(out, dpi=dpi, which=which)
