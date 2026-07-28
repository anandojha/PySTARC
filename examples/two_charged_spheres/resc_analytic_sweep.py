#!/usr/bin/env python3
"""Analytic escape-radius sweep for the two-charged-sphere control.

Reuses the exact Smoluchowski/return-probability formula from
examples/two_charged_spheres/analytical.py, but sweeps the escape radius q_esc
with the b-sphere (start radius) held fixed at b = 10 A, exactly as the PySTARC
simulation does. The recollision-corrected rate k_on = CONV * k_b * P_rxn is
constructed to be independent of q_esc beyond the interaction range; this checks
that at the level of the analytic theory and quantifies any residual drift.
"""
from scipy.integrate import quad
import math

q_rec, q_lig = 1.0, -1.0
sdie, eps0 = 78.0, 0.000142
debye = 7.828          # Debye length in A
mu = 0.216208
b = 10.0               # b-sphere start radius (fixed, = bd_milestone_radius)
a = 2.5                # reaction contact radius
kT = 1.0
CONV = 602.214e6
eps_s = sdie * eps0
V_fac = q_rec * q_lig / (4 * math.pi * eps_s)
r_h = 1.005            # PySTARC Monte Carlo hydrodynamic radius
D = kT / (6 * math.pi * mu) * (1.0 / r_h + 1.0 / r_h)


def V(r):
    return V_fac * math.exp(-r / debye) / r


def igrd(s):
    return math.exp(V(1.0 / s)) / D if s > 0 else 1.0 / D


def S_igrd(r):
    return math.exp(V(r)) / (r ** 2 * D)


def kon_at(q_esc):
    I_b, _ = quad(igrd, 0, 1.0 / b, limit=500)
    k_b = 4 * math.pi / I_b
    I_q, _ = quad(igrd, 0, 1.0 / q_esc, limit=500)
    k_q = 4 * math.pi / I_q
    rp = k_b / k_q                      # return probability from q back to b
    Sbq, _ = quad(S_igrd, b, q_esc, limit=500)
    Saq, _ = quad(S_igrd, a, q_esc, limit=500)
    Ps = Sbq / Saq                      # survival to b before reaction at a
    Prxn = Ps / (1.0 - (1.0 - Ps) * rp)
    return CONV * k_b * Prxn, k_b, rp, Prxn


print(f"  b (start) = {b} A   a (contact) = {a} A   Debye = {debye} A   D = {D:.5f} A^2/ps")
print(f"  {'q_esc/A':>8s}  {'q_esc/Debye':>11s}  {'k_b':>9s}  {'return_prob':>11s}  {'P_rxn':>8s}  {'k_on (M^-1 s^-1)':>18s}")
print(f"  {'-'*8}  {'-'*11}  {'-'*9}  {'-'*11}  {'-'*8}  {'-'*18}")
ref = None
for q in [15, 20, 25, 30, 40, 60, 80, 120, 160, 240, 400]:
    kon, k_b, rp, Prxn = kon_at(q)
    if ref is None:
        ref = kon
    drift = (kon - ref) / ref
    print(f"  {q:8.1f}  {q/debye:11.2f}  {k_b:9.4f}  {rp:11.5f}  {Prxn:8.5f}  {kon:18.5e}   ({drift:+.2%} vs q=15)")
