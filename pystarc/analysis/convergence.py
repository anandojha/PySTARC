"""
Convergence analysis for Brownian dynamics simulations.

Each Brownian dynamics trajectory is an independent Bernoulli trial. A
trajectory starts from a fresh random position on the b-surface with an
independent random number seed, and it ends either by reacting (a success,
with probability P_rxn) or by escaping (a failure, with probability
1 - P_rxn). Because the trials are independent and identically distributed,
the standard error of the mean is exact and is given by

    SE(P_rxn) = √[P_rxn × (1 - P_rxn) / N]

Here P_rxn is the reaction probability and N is the number of completed
trajectories.

The relative standard error, defined as RSE = SE / P_rxn, directly measures
the precision of the rate constant k_on. It works out to

    RSE = √[(1 - P_rxn) / (N × P_rxn)]

Since RSE is proportional to 1/√N, the number of trajectories needed to reach
a target precision is

    N_needed = (1 - P_rxn) / (P_rxn × RSE_target²)

where RSE_target is the desired relative standard error.

The confidence interval on P_rxn (95% by default) is the Wilson score
interval,

    P ∈ [p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)] / (1 + z²/n)

where p̂ is the observed reaction probability, n is the number of trajectories,
and z is the standard normal quantile (z ≈ 1.96 for 95% coverage). The Wilson
interval is preferred over the normal approximation (p̂ ± 2σ) because it keeps
its coverage even when P_rxn is much smaller than 0.05 or when N is small. The
normal approximation can return negative lower bounds for small P_rxn, which is
unphysical.
"""

from typing import Optional
import math
from pystarc.global_defs.constants import MOLAR_RATE_CONV
import json
import os


def analyse_convergence(
    n_reacted: int,
    n_escaped: int,
    k_b: float,
    tol: float = 0.05,
    conv_factor: float = MOLAR_RATE_CONV,
    work_dir: str = ".",
    confidence: float = 0.95,
    n_max_steps: int = 0,
    nam_omega: float = 0.0,
) -> dict:
    """
    Run the convergence analysis on a completed Brownian dynamics simulation.

    The parameter n_reacted is the total number of trajectories that reacted,
    n_escaped is the total number that escaped, and n_max_steps the number that
    reached the step limit; those count as not reacted. k_b is the encounter rate
    constant in Å³/ps. tol is the relative standard error below which the
    result is considered converged, defaulting to 0.05, that is 5%. conv_factor
    is the factor that converts Å³/ps to M⁻¹ s⁻¹. work_dir is the directory in
    which the convergence report is saved. nam_omega is the Northrup-Allison-McCammon
    escape-return factor k_b / k(q) in [0, 1); when positive, P_rxn, its standard
    error, and the Wilson bounds are transformed to the infinite-q limit.

    The function returns a dictionary holding the convergence results.
    """
    omega = float(nam_omega) if nam_omega else 0.0
    if omega < 0.0 or omega >= 1.0:
        raise ValueError(f"nam_omega must lie in [0, 1), got {omega}")
    N = n_reacted + n_escaped + n_max_steps
    if N == 0:
        return {"converged": False, "reason": "no trajectories were run"}
    P_raw = n_reacted / N

    def _nam_beta(prob: float) -> float:
        if omega <= 0.0:
            return prob
        denom_b = 1.0 - (1.0 - prob) * omega
        return prob / denom_b if denom_b > 0.0 else 1.0

    P = _nam_beta(P_raw)
    k_on = conv_factor * k_b * P
    # Standard error of P_rxn and the relative standard error.
    if P_raw > 0 and P_raw < 1:
        SE_raw = math.sqrt(P_raw * (1 - P_raw) / N)
        if omega > 0.0:
            denom_j = 1.0 - (1.0 - P_raw) * omega
            jac = (1.0 - omega) / (denom_j * denom_j) if denom_j > 0.0 else 0.0
            SE = jac * SE_raw
        else:
            SE = SE_raw
        relative_SE = SE / P if P > 0.0 else float("inf")
    elif P_raw == 0:
        SE = 0.0
        relative_SE = float("inf")
    else:
        SE = 0.0
        relative_SE = 0.0
    SE_kon = conv_factor * k_b * SE
    # Wilson confidence interval at the requested level. N is guaranteed to be
    # nonzero here by the early return above, so no separate N=0 guard is needed.
    # The argument of the square root is clamped to stay non-negative. The 0.95
    # default preserves z = 1.96; other levels invoke the normal quantile so the
    # interval matches the requested confidence.
    if abs(confidence - 0.95) < 1e-9:
        z = 1.96
    else:
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + confidence / 2.0))
    denom = 1 + z**2 / N
    centre = (P_raw + z**2 / (2 * N)) / denom
    # Clamping the argument at zero protects against floating-point roundoff
    # at the P=0 or P=1 boundary, where it can dip slightly negative.
    sqrt_arg = max(P_raw * (1 - P_raw) / N + z**2 / (4 * N**2), 0.0)
    spread = z * math.sqrt(sqrt_arg) / denom
    wilson_lo_raw = max(0.0, centre - spread)
    wilson_hi_raw = min(1.0, centre + spread)
    wilson_lo = _nam_beta(wilson_lo_raw)
    wilson_hi = _nam_beta(wilson_hi_raw)
    wilson_lo_kon = conv_factor * k_b * wilson_lo
    wilson_hi_kon = conv_factor * k_b * wilson_hi
    # Decide whether the run has converged.
    converged = (0 < P_raw < 1) and (relative_SE < tol)
    # Number of trajectories needed to reach a few target tolerances.
    targets = {}
    if 0 < P_raw < 1:
        if omega > 0.0:
            denom_t = 1.0 - (1.0 - P_raw) * omega
            rse_scale = (1.0 - omega) / denom_t if denom_t > 0.0 else 1.0
        else:
            rse_scale = 1.0
        for target_tol in [0.10, 0.05, 0.01]:
            n_needed = int(
                math.ceil(
                    (rse_scale * rse_scale) * (1 - P_raw) / (P_raw * target_tol**2)
                )
            )
            targets[f"{int(target_tol*100)}%"] = n_needed
    result = {
        "N": N,
        "n_reacted": n_reacted,
        "n_escaped": n_escaped,
        "n_max_steps": n_max_steps,
        "P_rxn": P,
        "SE": SE,
        "relative_SE": relative_SE,
        "relative_SE_pct": relative_SE * 100 if P > 0 else float("inf"),
        "k_on": k_on,
        "SE_kon": SE_kon,
        "wilson_CI": [wilson_lo_kon, wilson_hi_kon],
        "wilson_CI_P": [wilson_lo, wilson_hi],
        "converged": converged,
        "confidence_level": confidence,
        "tol": tol,
        "tol_pct": tol * 100,
        "N_needed": targets,
    }
    if omega > 0.0:
        result["P_rxn_raw"] = P_raw
        result["nam_omega"] = omega
    return result


def print_convergence(result: dict) -> str:
    """Print the convergence analysis to the terminal and return it as a string."""
    lines = []
    lines.append("")
    lines.append("  Convergence analysis")
    if "N" not in result:
        lines.append(f"  {result.get('reason', 'no data')}")
        text = "\n".join(lines)
        print(text)
        return text
    lines.append(f"  N trajectories   = {result['N']:,}")
    if result.get("n_max_steps"):
        lines.append(
            f"  Max-steps        = {result['n_max_steps']:,}  (counted as not reacted)"
        )
    lines.append(f"  P_rxn            = {result['P_rxn']:.6f}")
    lines.append(f"  SE(P_rxn)        = {result['SE']:.6f}")
    if result["P_rxn"] > 0:
        lines.append(
            f"  Relative SE      = {result['relative_SE_pct']:.2f}%"
            f"     - k_on known to ±{result['relative_SE_pct']:.2f}%"
        )
    else:
        lines.append(f"  Relative SE      = inf (P_rxn = 0, no reactions)")
    lines.append(
        f"  Wilson {100 * result.get('confidence_level', 0.95):g}% CI    = [{result['wilson_CI'][0]:.4e}, "
        f"{result['wilson_CI'][1]:.4e}] M⁻¹s⁻¹"
    )
    tol_pct = result["tol_pct"]
    if result["converged"]:
        lines.append(
            f"  Converged (relative SE {result['relative_SE_pct']:.2f}% < {tol_pct:g}% threshold)"
        )
    else:
        lines.append(
            f"  Not converged (relative SE {result['relative_SE_pct']:.2f}% > {tol_pct:g}% threshold)"
        )
    if result["N_needed"]:
        lines.append(f"  Trajectories needed")
        for label, n in result["N_needed"].items():
            status = "done" if result["N"] >= n else "need more"
            lines.append(f"    For ±{label} relative SE: {n:,} ({status})")
    text = "\n".join(lines)
    print(text)
    return text


def save_convergence(result: dict, work_dir: str = ".") -> None:
    import json, os

    path = os.path.join(work_dir, "convergence.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Convergence saved -> {path}")
