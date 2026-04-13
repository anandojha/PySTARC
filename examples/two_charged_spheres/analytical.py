#!/usr/bin/env python3

"""
Analytical verification for charged_spheres
Exact Smoluchowski solution vs PySTARC simulation vs analytical reference.
Run after PySTARC simulation to auto-compare.
"""

from scipy.integrate import quad
import math
import os
import re

# Parameters
q_rec = 1.0
q_lig = -1.0
sdie = 78.0
eps0 = 0.000142
debye = 7.828
mu = 0.243
b = 10.0
a = 2.5
q_esc = 20.0
kT = 1.0
pi = math.pi
CONV = 602.214e6
eps_s = sdie * eps0
V_fac = q_rec * q_lig / (4 * pi * eps_s)
r_h = 1.005  # PySTARC MC hydro radius
D = kT / (6 * pi * mu) * (1 / r_h + 1 / r_h)
# Reference analytical values
a1_ref = 1.01823
a2_ref = 1.01079
qq_ref = -1.00055
D_ref = kT / (6 * pi * mu) * (1 / a1_ref + 1 / a2_ref)


def V_py(r):
    return V_fac * math.exp(-r / debye) / r


def V_ref(r):
    return (
        qq_ref
        * math.exp((a1_ref - r) / debye)
        / (4 * pi * eps_s * r * (a1_ref / debye + 1))
    )


# Exact computation
def igrd(s):
    return math.exp(V_py(1 / s)) / D if s > 0 else 1 / D


I_b, _ = quad(igrd, 0, 1 / b, limit=500)
k_b = 4 * pi / I_b
I_q, _ = quad(igrd, 0, 1 / q_esc, limit=500)
k_q = 4 * pi / I_q
rp = k_b / k_q


def S_igrd(r):
    return math.exp(V_py(r)) / (r**2 * D)


Sbq, _ = quad(S_igrd, b, q_esc, limit=500)
Saq, _ = quad(S_igrd, a, q_esc, limit=500)
Ps = Sbq / Saq
Prxn = Ps / (1 - (1 - Ps) * rp)
kon = CONV * k_b * Prxn


def igrd_ref(s):
    return math.exp(V_ref(1 / s)) / (4 * pi * D_ref) if s > 0 else 1 / (4 * pi * D_ref)


I_ref, _ = quad(igrd_ref, 0, 1 / a, limit=500)
k_ref = CONV / I_ref

# Print results
W = 75


def hdr(t):
    print("=" * W)
    print(f"  {t}")
    print("=" * W)


def row(label, pybd, ref=None, unit=""):
    if ref is not None:
        diff = abs(pybd - ref) / max(abs(ref), 1e-30)
        print(f"  {label:<28s}  {pybd:>14.6f}  {ref:>14.6f}  {diff:>8.2%}")
    else:
        print(f"  {label:<28s}  {pybd:>14.6f}")


hdr("Step 1: Derived quantities")
print(f"  {'':28s}  {'PySTARC':>14s}  {'Reference':>14s}  {'Difference':>8s}")
print(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*8}")
row("r_hydro_rec (Å)", r_h, a1_ref)
row("r_hydro_lig (Å)", r_h, a2_ref)
row("D_rel (Å²/ps)", D, D_ref)
print()
hdr("Step 2: Potential V(r)/kT")
print(f"  {'r (Å)':>8s}  {'PySTARC':>12s}  {'Reference':>12s}  {'Difference':>8s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
for r in [2.5, 3, 5, 7, 10, 15, 20]:
    print(
        f"  {r:8.1f}  {V_py(r):12.6f}  {V_ref(r):12.6f}  {abs(V_py(r)-V_ref(r))/max(abs(V_ref(r)),1e-30):8.2%}"
    )
print()
hdr("Step 3: k_b, RETURN_PROB, P_rxn")
print(f"  k_b             = {k_b:.4f} ų/ps")
print(f"  k_b(q_esc)      = {k_q:.4f} ų/ps")
print(f"  return_prob     = {rp:.6f}")
print(f"  P_single        = {Ps:.6f}  (Smoluchowski, absorbing at a={a} and q={q_esc})")
print(f"  P_rxn (w/return)= {Prxn:.6f}")
print(f"  k_on analytical = {kon:.4e} M⁻¹s⁻¹")
print(f"  Reference k_on  = {k_ref:.4e} M⁻¹s⁻¹")
print(f"  Potential diff   -> k_on diff = {abs(kon-k_ref)/k_ref:.2%}")
print()
hdr("Step 4: Adpative dt at kep separations")
print(
    f"  {'r':>6s}  {'dt_pair':>10s}  {'dt_force':>10s}  {'dt_final':>10s}  {'σ':>8s}  {'σ/r':>8s}"
)
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
for r in [2.6, 3, 4, 5, 7, 10, 15, 19]:
    dt_p = 0.005 * r**2 / D
    ef = math.exp(-r / debye)
    Fr0 = V_fac * ef * (1 / r**2 + 1 / (r * debye))
    Fr1 = -V_fac * ef / debye**2 - 2 * Fr0 / r
    DFr1 = abs(D * Fr1)
    dt_f = 0.01 / DFr1 if (DFr1 > 1e-15 and r < 3 * debye) else dt_p
    io = r > 1.1 * b
    if io:
        db = max(r - b, 1e-3)
        de = max(q_esc - r, 1e-3)
        dt_e = min(db, de) ** 2 / (18 * D)
        dt_fin = min(dt_p, dt_f, dt_e)
    else:
        dt_fin = min(dt_p, dt_f)
    s = math.sqrt(2 * D * dt_fin)
    print(
        f"  {r:6.1f}  {dt_p:10.4f}  {dt_f:10.4f}  {dt_fin:10.4f}  {s:8.4f}  {s/r:8.2%}"
    )
print()
hdr("Step 5: Comparison with simulation")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bd_sims")
sim_prxn = sim_kon = sim_kb = sim_rp = None
if os.path.isdir(log_dir):
    logs = sorted([f for f in os.listdir(log_dir) if f.endswith(".log")])
    if logs:
        txt = open(os.path.join(log_dir, logs[-1])).read()
        m = re.search(r"P_rxn\s+:\s+([\d.]+)", txt)
        sim_prxn = float(m.group(1)) if m else None
        m = re.search(r"k_on\s+:\s+([\d.e+]+)", txt)
        sim_kon = float(m.group(1)) if m else None
        m = re.search(r"k_b:\s+([\d.]+)", txt)
        sim_kb = float(m.group(1)) if m else None
        m = re.search(r"return_prob:\s+([\d.]+)", txt)
        sim_rp = float(m.group(1)) if m else None
print(f"  {'':28s}  {'Analytical':>14s}  {'Simulation':>14s}  {'Ref':>14s}")
print(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*14}")


def cmp(label, exact, sim, ref_str):
    sv = f"{sim:.6f}" if sim is not None else "(run sim)"
    print(f"  {label:<28s}  {exact:14.6f}  {sv:>14s}  {ref_str:>14s}")


cmp("k_b (ų/ps)", k_b, sim_kb, "57.25")
cmp("return_prob", rp, sim_rp, "-")
cmp("P_rxn", Prxn, sim_prxn, "0.49")

if sim_prxn is not None and sim_kon is not None:
    print(f"  {'k_on (M⁻¹s⁻¹)':<28s}  {kon:14.4e}  {sim_kon:14.4e}  {'1.69e+10':>14s}")
    err = abs(sim_prxn - Prxn) / Prxn * 100
    print(f"\n  Simulation vs exact analytical: P_rxn error = {err:.1f}%")
    if err < 5:
        print("  Pass: Excellent agreement (<5%)")
    elif err < 10:
        print("   Marginal (<10%)")
    else:
        print(f"  Fail ({err:.0f}%)")
else:
    print(f"\n  Run simulation first, then re-run this script.")
print()
