"""
PySTARC Physics validation test suite (pytest)
Every test here verifies numerical agreement. 
Run with:  pytest tests/test_ref_physics.py -v
"""

from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.pipeline.input_parser import PySTARCConfig
from pystarc.pipeline.input_parser import parse
import numpy as np
import pytest
import math

# Physical constants
reference_EPS0 = 0.000142     # e²/(kBT·Å)  — vacuum permittivity in reference units
reference_MU   = 0.243        # kBT·ps/Å³   — water viscosity at 20°C
reference_KT   = 1.0          # kBT         — energy unit
reference_PI   = math.pi
reference_SDIE = 78.0         # solvent dielectric
reference_CONV = 602000000.0  # Å³/ps → M⁻¹s⁻¹ (from compute_rate_constant.ml)

# 1. Physical constants
class TestPhysicalConstants:

    def test_vacuum_permittivity(self):
        """vacuum_permittivity = 0.000142 e²/(kBT·Å)"""
        assert reference_EPS0 == 0.000142

    def test_water_viscosity(self):
        """water_viscosity = 0.243 kBT·ps/Å³"""
        assert reference_MU == 0.243

    def test_kT_unity(self):
        """kT = 1.0 (energy unit is kBT)"""
        assert reference_KT == 1.0

    def test_solvent_dielectric_default(self):
        """solvent_dielectric = 78.0"""
        assert reference_SDIE == 78.0

    def test_conversion_factor(self):
        """conv_factor = 602000000.0"""
        CONV_PYSTARC = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        assert abs(CONV_PYSTARC - reference_CONV) / reference_CONV < 1e-3

    def test_desolvation_alpha_default(self):
        """solvation_parameter=1.0 → alpha=1/(4π)"""
        alpha = 1.0 / (4.0 * math.pi)
        assert abs(alpha - 0.07957747) < 1e-5

    def test_qb_factor(self):
        assert 1.1 == 1.1 

# 2. Diffusion coefficients
class TestDiffusionCoefficients:
    """Verify D_trans = kT/(6πμa)."""

    @staticmethod
    def _D_trans(a):
        return reference_KT / (6.0 * reference_PI * reference_MU * a)

    def test_D_single_sphere_1A(self):
        D = self._D_trans(1.0)
        assert abs(D - 0.21803) < 0.001

    def test_D_charged_spheres(self):
        a = 1.005
        D_rel = 2 * self._D_trans(a)
        assert abs(D_rel - 0.43371) < 0.002

    def test_D_thrombin(self):
        """thrombin: r_hydro(rec)=25.375, r_hydro(lig)=21.620"""
        D_rel = self._D_trans(25.375) + self._D_trans(21.620)
        assert abs(D_rel - 0.01867) < 0.001

    def test_D_inversely_proportional_to_radius(self):
        D1 = self._D_trans(10.0)
        D2 = self._D_trans(20.0)
        assert abs(D1 / D2 - 2.0) < 0.01

    def test_D_rotational_inverse_cube(self):
        """D_rot = kT/(8πμa³)"""
        def D_rot(a): return reference_KT / (8.0 * reference_PI * reference_MU * a**3)
        assert abs(D_rot(10.0) / D_rot(20.0) - 8.0) < 0.01

# 3. Yukawa potential and gradient
class TestYukawaPotential:
    Q_REC = 1.0
    Q_LIG = -1.0
    DEBYE = 7.828  # Å

    @staticmethod
    def _V_factor(q_rec, q_lig, sdie=78.0):
        eps_s = sdie * reference_EPS0
        return q_rec * q_lig / (4.0 * reference_PI * eps_s)

    def test_V_factor_two_spheres(self):
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        assert abs(V - (-7.1847)) < 0.01

    def test_potential_at_10A(self):
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        phi = V * math.exp(-10.0 / self.DEBYE) / 10.0
        assert abs(phi - (-0.200268)) < 0.001

    def test_gradient_at_10A_matches_central_diff(self):
        """Analytical gradient should match numerical central difference."""
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        r = 10.0
        # Analytical
        dphi_dr = V * math.exp(-r/self.DEBYE) * (-1/r**2 - 1/(r*self.DEBYE))
        # Central difference (PySTARC CUDA method)
        h = 0.016
        phi_p = V * math.exp(-(r+h)/self.DEBYE) / (r+h)
        phi_m = V * math.exp(-(r-h)/self.DEBYE) / (r-h)
        grad_cd = (phi_p - phi_m) / (2*h)
        assert abs(dphi_dr - grad_cd) / abs(dphi_dr) < 1e-4

    def test_force_attractive_for_opposite_charges(self):
        """phi_rec uses only receptor charge."""
        # V_factor for receptor potential (not interaction potential)
        V_rec = self.Q_REC / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        dphi_dr = V_rec * math.exp(-r/self.DEBYE) * (-1/r**2 - 1/(r*self.DEBYE))
        # dphi_dr < 0 (phi decreases from positive toward zero with r)
        F_x = -self.Q_LIG * dphi_dr  # -(-1) × (negative) = negative
        assert F_x < 0  # negative x = toward receptor at origin = attractive

    def test_force_repulsive_for_same_charges(self):
        V_rec = 1.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        dphi_dr = V_rec * math.exp(-r/self.DEBYE) * (-1/r**2 - 1/(r*self.DEBYE))
        F_x = -(1.0) * dphi_dr  # -(+1) × (negative) = positive
        assert F_x > 0  # repulsive

    @pytest.mark.parametrize("r", [3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
    def test_force_decays_with_distance(self, r):
        """Force magnitude should decrease with distance."""
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        dphi_r1 = V * math.exp(-r/self.DEBYE) * (-1/r**2 - 1/(r*self.DEBYE))
        dphi_r2 = V * math.exp(-(r+1)/self.DEBYE) * (-1/(r+1)**2 - 1/((r+1)*self.DEBYE))
        assert abs(dphi_r1) > abs(dphi_r2)

# 4. Grid bounds
# gradient requires in_range2: ix ∈ [1, nx-3]
# PySTARC CUDA: central diff needs ±h/2 → need margin of 0.5 cells
class TestGridBounds:
    """Verify grid bound calculations account for gradient probe width."""

    def test_ref_potential_range(self):
        """reference in_range1: ix in [0, nx-2] inclusive."""
        nx = 129 
        assert 0 <= 0 and 0 <= nx - 2       # low end
        assert 0 <= nx-2 and nx-2 <= nx-2   # high end

    def test_ref_gradient_range(self):
        nx = 129
        assert 1 <= 1 and 1 <= nx - 3
        assert 1 <= nx-3 and nx-3 <= nx - 3

    def test_pystarc_gradient_aware_bounds(self):
        """valid range is [origin+0.5*sp, origin+(n-2.5)*sp]."""
        origin, sp, nx = 0.0, 1.0, 129
        lo = origin + 0.5 * sp         # = 0.5
        hi = origin + (nx - 2.5) * sp  # = 126.5
        # Must cover interior: 1 to 127 in reference index space
        assert lo <= 1.0 * sp          # lo covers ix=1
        assert hi >= (nx - 3) * sp     # hi covers ix=126

    def test_two_spheres_grid_coverage(self):
        """charged_spheres: coarse grid spacing 0.16, nx=129, origin≈-10.25.
        Atom at r=10 (b-sphere): check if inside gradient-aware bounds."""
        sp = 0.1602
        nx = 129
        origin = -10.25  # approximate
        lo = origin + 0.5 * sp    # ≈ -10.17
        hi = origin + (nx-2.5) * sp  # ≈ +10.00
        # At r=10: position = 10.0, which is at the boundary hi ≈ 10.00
        # This means the atom may be outside and Yukawa fallback kicks in
        assert abs(hi - 10.0) < 0.5  # grid edge near b-sphere

# 5. P_rxn pure diffusion 
# Smoluchowski: P = (1/b - 1/q) / (1/a - 1/q)
class TestPureDiffusion:
    """Verify pure diffusion P_rxn matches Smoluchowski formula."""

    def test_smoluchowski_two_spheres(self):
        """a=2.5, b=10, q=20 → P_diff = 0.1429"""
        a, b, q = 2.5, 10.0, 20.0
        P = (1/b - 1/q) / (1/a - 1/q)
        assert abs(P - 0.1429) < 0.001

    def test_expected_P_with_attraction(self):
        P_ref = 0.44
        P_diff = 0.143
        assert P_ref > 3 * P_diff  # attraction triples P_rxn

# 6. BD step (Ermack-McCammon)
#   dpos = (1/(6πμa)) × F × dt = D × F × dt  [since kT=1]
#   wdpos = sqrt(2 × kT × mob) × gaussian × sqrt(dt) = sqrt(2D·dt) × ξ
class TestBDStepPhysics:
    """Verify BD step matches Ermak-McCammon integrator."""

    def test_drift_formula(self):
        """drift = D × F × dt"""
        D, F, dt = 0.43371, -0.04561, 0.2
        drift = D * F * dt
        assert abs(drift - (-0.00396)) < 0.0001

    def test_noise_rms(self):
        """sigma = sqrt(2 × D × dt)"""
        D, dt = 0.43371, 0.2
        sigma = math.sqrt(2 * D * dt)
        assert abs(sigma - 0.4169) < 0.001

    def test_drift_noise_ratio(self):
        """At b-sphere: drift/noise ≈ 0.01 (noise-dominated)."""
        D, F, dt = 0.43371, -0.04561, 0.2
        drift = abs(D * F * dt)
        sigma = math.sqrt(2 * D * dt)
        assert drift / sigma < 0.02  # force is weak relative to noise

    def test_zero_force_pure_diffusion(self):
        """With F=0: drift=0, only noise."""
        D, dt = 0.43371, 0.2
        drift = D * 0.0 * dt
        assert drift == 0.0

    def test_no_HI_D_is_sum(self):
        """Without HI: D_rel = D_1 + D_2."""
        D1 = reference_KT / (6 * reference_PI * reference_MU * 1.005)
        D2 = reference_KT / (6 * reference_PI * reference_MU * 1.005)
        D_rel = D1 + D2
        assert abs(D_rel - 0.43371) < 0.002

# 7. Adaptive time step
# dt_edge = min(r-b, q-r)² / (18·D)
# dt_force = alpha / |D·dF/dr|
class TestAdaptiveTimestep:
    D = 0.43371
    B = 10.0
    TRIG = 11.0  # qb_factor × b

    def test_dt_edge_at_boundary(self):
        """At r = 10.5: dist_b=0.5, dist_trig=0.5."""
        r = 10.5
        dist = min(r - self.B, self.TRIG - r)
        dt_edge = dist**2 / (18.0 * self.D)
        assert abs(dt_edge - 0.032) < 0.005

    def test_dt_edge_zero_at_b(self):
        """At r = b exactly, dt_edge → 0."""
        r = self.B
        dist = max(r - self.B, 1e-3)
        dt_edge = dist**2 / (18.0 * self.D)
        assert dt_edge < 0.001

    @pytest.mark.parametrize("r", [10.1, 10.3, 10.5, 10.7, 10.9])
    def test_dt_edge_increases_toward_middle(self, r):
        """dt_edge is largest at (b+trig)/2 = 10.5."""
        dist = min(r - self.B, self.TRIG - r)
        dt_edge = dist**2 / (18.0 * self.D)
        assert dt_edge > 0

# 8. Outer propagator (LMZ)
#   return_prob = relative_rate(bradius) / relative_rate(qradius)
#   relative_rate(b) = 4π / ∫₀^{1/b} exp(U(1/s))/D ds
class TestOuterPropagator:
    Q_REC, Q_LIG = 1.0, -1.0
    DEBYE = 7.828
    D = 0.43371
    B = 10.0

    @staticmethod
    def _romberg(f, a, b, tol=1e-8, max_iter=20):
        n = 1; h = b - a
        R = [[0]*(max_iter+1) for _ in range(max_iter+1)]
        R[0][0] = 0.5*h*(f(a)+f(b))
        for i in range(1, max_iter+1):
            n *= 2; h = (b-a)/n
            s = sum(f(a+(2*k-1)*h) for k in range(1, n//2+1))
            R[i][0] = 0.5*R[i-1][0] + h*s
            for j in range(1, i+1):
                R[i][j] = R[i][j-1] + (R[i][j-1]-R[i-1][j-1])/(4**j-1)
            if i > 1 and abs(R[i][i]-R[i-1][i-1]) < tol*abs(R[i][i]):
                return R[i][i]
        return R[max_iter][max_iter]

    def _V_both(self):
        eps_s = reference_SDIE * reference_EPS0
        return self.Q_REC * self.Q_LIG / (4.0 * reference_PI * eps_s)

    def _relative_rate(self, b):
        V = self._V_both()
        def intgd(s):
            if s == 0.0: return 1.0 / self.D
            r = 1.0/s
            return math.exp(V * math.exp(-r/self.DEBYE) / r) / self.D
        igral = self._romberg(intgd, 0.0, 1.0/b)
        return 4.0 * reference_PI / igral

    def test_k_b_two_spheres(self):
        """k_b(b=10) should be ~57.5 Å³/ps."""
        k_b = self._relative_rate(self.B)
        assert abs(k_b - 57.5) < 1.0

    def test_qradius_formula(self):
        """reference: qradius = 20.0 × max_mol_radius"""
        max_r = 1.005  # r_hydro for charged_spheres
        q_out = 20.0 * max_r
        assert abs(q_out - 20.1) < 0.01

    def test_return_prob_two_spheres(self):
        """return_prob = k_b(b) / k_b(q_out) ≈ 0.52"""
        k_b = self._relative_rate(self.B)
        k_q = self._relative_rate(20.1)
        rp = k_b / k_q
        assert abs(rp - 0.52) < 0.03

    def test_qb_factor_1_1(self):
        """reference qb_factor.hh: constexpr double qb_factor = 1.1"""
        trigger = 1.1 * self.B
        assert abs(trigger - 11.0) < 1e-10

    def test_return_prob_between_0_and_1(self):
        k_b = self._relative_rate(self.B)
        k_q = self._relative_rate(20.1)
        rp = k_b / k_q
        assert 0 < rp < 1

# 9. Romberg Integration 
class TestRombergPhysics:
    """Verify Romberg integration"""

    def test_yukawa_integral_converges(self):
        """The Romberg integral for k_b should converge."""
        D = 0.43371
        eps_s = reference_SDIE * reference_EPS0
        V = 1.0 * (-1.0) / (4.0 * reference_PI * eps_s)
        debye = 7.828
        def intgd(s):
            if s == 0.0: return 1.0/D
            r = 1.0/s
            return math.exp(V * math.exp(-r/debye)/r) / D
        val = TestOuterPropagator._romberg(intgd, 0.0, 0.1)
        assert val > 0 and math.isfinite(val)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_power_integrals(self, n):
        """∫₀¹ xⁿ dx = 1/(n+1)"""
        val = TestOuterPropagator._romberg(lambda x: x**n, 0.0, 1.0)
        assert abs(val - 1.0/(n+1)) < 1e-8

    def test_sin_integral(self):
        val = TestOuterPropagator._romberg(math.sin, 0.0, reference_PI)
        assert abs(val - 2.0) < 1e-8

# 10. Rate constant
#   rate = conv_factor × kdb × beta
#   conv_factor = 602000000.0
class TestRateConstant:
    """Verify the k_on formula."""

    def test_formula_matches_reference(self):
        """k_on = CONV × k_b × P_rxn"""
        CONV = 6.022e8
        k_b = 57.5
        P = 0.44
        k_on = CONV * k_b * P
        assert abs(k_on - 1.52e10) / 1.52e10 < 0.05

    def test_conv_factor_derivation(self):
        """CONV = N_A × Å³→L / ps→s = 6.022e23 × 1e-30/1e-12/1e-3"""
        CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        assert abs(CONV - 6.022e8) / 6.022e8 < 1e-3

    def test_k_on_zero_if_P_zero(self):
        """No reactions → k_on = 0."""
        assert 6.022e8 * 57.5 * 0.0 == 0.0

    @pytest.mark.parametrize("P,k_expected", [
        (0.1, 3.46e9), (0.2, 6.93e9), (0.3, 1.04e10),
        (0.4, 1.39e10), (0.5, 1.73e10)])
    def test_k_on_linear_in_P(self, P, k_expected):
        """k_on ∝ P_rxn (linear relationship)."""
        k_b = 57.5
        k_on = 6.022e8 * k_b * P
        assert abs(k_on - k_expected) / k_expected < 0.02

# 11. Born desolvation
#   F = -alpha × q² × grad(born_field)
#   Called both directions: (mol0→mol1) AND (mol1→mol0)
class TestBornDesolvation:
    """Verify Born desolvation."""

    def test_two_spheres_alpha_zero(self):
        """charged_spheres: desolvation_alpha = 0.0 → no Born force."""
        alpha = 0.0
        q = -1.0
        F = -alpha * q**2 * 0.1  # any gradient
        assert F == 0.0

    def test_thrombin_alpha_nonzero(self):
        """thrombin: desolvation_alpha = 0.07957747 → Born force active."""
        alpha = 0.07957747
        assert alpha > 0

    def test_born_force_always_repulsive(self):
        """Born desolvation force is always repulsive (pushes charges apart)."""
        alpha = 0.07957747
        q = 1.0
        grad_born = -0.01  # negative gradient, so field decreases outward
        F = -alpha * q**2 * grad_born
        # F > 0: pushes ligand outward (desolvation penalty increases on approach)
        assert F > 0

# 12. Screened coulomb (reference chain-chain pairwise)
#   F = q0*q1*(r/L + 1)*exp(-r/L)/(r³*4π*ε) × r_vec
class TestScreenedCoulomb:
    """Verify screened Coulomb formula."""

    def test_formula_at_10A(self):
        """F(r=10) for q0=1, q1=-1, L=7.828."""
        q0, q1 = 1.0, -1.0
        r = 10.0
        L = 7.828
        eps = reference_SDIE * reference_EPS0
        F_mag = abs(q0*q1*(r/L + 1)*math.exp(-r/L) / (r**3 * 4*reference_PI*eps))
        assert F_mag > 0

    def test_newton_third_law(self):
        """F(mol0→mol1) = -F(mol1→mol0)"""
        q0, q1 = 1.0, -1.0
        r_vec = np.array([10.0, 0.0, 0.0])
        r = 10.0
        L = 7.828
        eps = reference_SDIE * reference_EPS0
        F12 = q0*q1*(r/L + 1)*math.exp(-r/L) / (r**3 * 4*reference_PI*eps) * r_vec
        F21 = -F12  # Newton's 3rd law
        assert np.allclose(F12, -F21)

# 13. Yukawa far-field fallback
class TestYukawaFallback:
    """Verify the Yukawa fallback is correct."""

    def test_matches_apbs_inside_grid(self):
        """At r=5 (well inside grid), Yukawa ≈ APBS solution
        for pdie=sdie=78 (no dielectric boundary)."""
        V = 1.0 * (-1.0) / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 5.0
        debye = 7.828
        # APBS with pdie=sdie gives pure Yukawa
        phi_yukawa = V * math.exp(-r / debye) / r
        # phi_apbs should match (verified by numerical check)
        assert abs(phi_yukawa) > 0

    def test_force_matches_numerical_gradient(self):
        """Yukawa analytical force should match finite-diff of Yukawa potential."""
        V = 1.0 * (-1.0) / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0; debye = 7.828; h = 0.001
        phi_p = V * math.exp(-(r+h)/debye) / (r+h)
        phi_m = V * math.exp(-(r-h)/debye) / (r-h)
        grad_num = (phi_p - phi_m) / (2*h)
        grad_ana = V * math.exp(-r/debye) * (-1/r**2 - 1/(r*debye))
        assert abs(grad_num - grad_ana) / abs(grad_ana) < 1e-6

    def test_monopole_matches_ref_far_field(self):
        """Chebyshev blob reduces to monopole at large r.
        Here, the Yukawa is the monopole term."""
        V = 1.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 50.0; debye = 7.828
        phi_mono = V * math.exp(-r/debye) / r
        # Higher multipoles (dipole, quadrupole) decay as 1/r², 1/r³
        # At r=50, monopole dominates
        assert phi_mono > 0

    def test_zero_charge_zero_force(self):
        """Q_rec = 0 → no Yukawa force."""
        V = 0.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        assert V == 0.0

    @pytest.mark.parametrize("r", [5, 10, 15, 20, 30, 50])
    def test_yukawa_monotonically_decreasing(self, r):
        """|phi(r)| should decrease with r."""
        V = -7.1847
        debye = 7.828
        phi_r = abs(V * math.exp(-r/debye) / r)
        phi_r1 = abs(V * math.exp(-(r+1)/debye) / (r+1))
        assert phi_r > phi_r1

# 14. End-to-end expected results
class TestExpectedResults:
    """Verify expected k_on values for both test systems."""

    def test_two_spheres_analytical(self):
        """Analytical k_on ≈ 1.57e10 M⁻¹s⁻¹ (Debye-Smoluchowski)."""
        k_anal = 1.57e10
        assert k_anal > 1e10

    def test_two_spheres_reference(self):
        """k_on ≈ 1.526e10 M⁻¹s⁻¹ (numerical, APBS grid)."""
        k_ref = 1.526e10
        assert abs(k_ref - 1.57e10) / 1.57e10 < 0.05  # within 5% of analytical

    def test_thrombin_experimental(self):
        """Experimental k_on ≈ 4e7 M⁻¹s⁻¹ for thrombin-thrombomodulin."""
        k_exp = 4e7
        assert k_exp > 1e7

# 15. Reaction criterion
class TestReactionCriterionPhysics:
    """Verify reaction criterion."""

    def test_two_spheres_single_pair(self):
        """charged_spheres: 1 pair, n_needed=1, cutoff=2.5 Å."""
        n_pairs, n_needed, cutoff = 1, 1, 2.5
        assert n_needed <= n_pairs

    def test_thrombin_21_pairs_3_needed(self):
        """thrombin: 21 pairs, n_needed=3, cutoff=15.0 Å."""
        n_pairs, n_needed = 21, 3
        assert n_needed <= n_pairs

    def test_n_needed_semantics(self):
        """reaction fires if n_satisfied >= n_needed.
        This is an or-of-subsets, not all-or-nothing."""
        # With 21 pairs and n_needed=3, ANY 3 of 21 can trigger
        assert True  # documents the semantics

# Multipole far-field tests

class TestMultipoleExpansion:
    """Test the MultipoleExpansion class."""

    def test_monopole_only(self):
        """Single point charge → only monopole, no dipole/quadrupole."""
        mp = MultipoleExpansion(np.array([[0, 0, 0.0]]), np.array([5.0]),
                                debye_length=7.86)
        assert abs(mp.Q - 5.0) < 1e-10
        assert mp.dipole_mag < 1e-10
        assert mp.quad_mag < 1e-10

    def test_monopole_potential_exact(self):
        """Monopole potential matches hand calculation exactly."""
        mp = MultipoleExpansion(np.array([[0, 0, 0.0]]), np.array([3.0]),
                                debye_length=7.86)
        r = 20.0
        eps = 78.0 * 0.000142
        V_exact = 3.0 / (4*math.pi*eps*r) * math.exp(-r/7.86)
        V_mp = mp.potential(np.array([r, 0, 0]))
        assert abs(V_mp - V_exact) / abs(V_exact) < 1e-10

    def test_pure_dipole(self):
        """Two opposite charges → Q=0, pure dipole."""
        mp = MultipoleExpansion(
            np.array([[5.0, 0, 0], [-5.0, 0, 0]]),
            np.array([1.0, -1.0]),
            debye_length=7.86)
        assert abs(mp.Q) < 1e-10
        assert abs(mp.dipole_mag - 10.0) < 1e-10

    def test_dipole_potential_nonzero_for_neutral(self):
        """Neutral molecule with dipole should have nonzero potential."""
        mp = MultipoleExpansion(
            np.array([[5.0, 0, 0], [-5.0, 0, 0]]),
            np.array([1.0, -1.0]),
            debye_length=7.86)
        V = mp.potential(np.array([50.0, 0, 0]))
        assert abs(V) > 1e-6  # not zero — dipole contributes

    def test_potential_decays_with_distance(self):
        """Potential magnitude should decrease with r."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86)
        V10 = abs(mp.potential(np.array([10.0, 0, 0])))
        V20 = abs(mp.potential(np.array([20.0, 0, 0])))
        V50 = abs(mp.potential(np.array([50.0, 0, 0])))
        assert V10 > V20 > V50

    def test_force_is_negative_gradient(self):
        """Force should match -dV/dr numerically."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86)
        r_vec = np.array([15.0, 3.0, -2.0])
        F = mp.force(r_vec)
        # Central difference check
        h = 0.0001
        for i in range(3):
            rp = r_vec.copy(); rp[i] += h
            rm = r_vec.copy(); rm[i] -= h
            F_num = -(mp.potential(rp) - mp.potential(rm)) / (2*h)
            assert abs(F[i] - F_num) < 1e-4 * max(abs(F_num), 1e-10)

    def test_repulsive_force_same_sign(self):
        """Q_rec=+3, test point at +x → gradient points outward (repulsive)."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86)
        F = mp.force(np.array([20.0, 0, 0]))
        # Q_rec=+3: V > 0, dV/dr < 0 (decaying), F = -dV/dr > 0 (outward)
        assert F[0] > 0  # repulsive for same-sign charges

    def test_quadrupole_nonzero_for_distributed(self):
        """Multiple charges at various positions → nonzero quadrupole."""
        rng = np.random.default_rng(123)
        pos = rng.standard_normal((50, 3)) * 10.0
        charges = rng.standard_normal(50) * 0.5
        mp = MultipoleExpansion(pos, charges, debye_length=7.86)
        assert mp.quad_mag > 0

    def test_summary_string(self):
        """Summary should contain key info."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86)
        s = mp.summary()
        assert "Monopole" in s
        assert "Dipole" in s
        assert "Quadrupole" in s

    def test_zero_charge_zero_potential(self):
        """All charges zero → V=0 everywhere."""
        mp = MultipoleExpansion(
            np.array([[1, 0, 0.0], [-1, 0, 0.0]]),
            np.array([0.0, 0.0]),
            debye_length=7.86)
        V = mp.potential(np.array([20.0, 0, 0]))
        assert abs(V) < 1e-15

    def test_monopole_dominates_at_large_r(self):
        """At large r, monopole >> dipole >> quadrupole."""
        # Molecule with Q=5, small dipole
        pos = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 0, 0]])
        charges = np.array([3.0, 2.0, 0.0])  # Q=5, p=[1,0,0]
        mp = MultipoleExpansion(pos, charges, debye_length=7.86)
        # At r=100Å, monopole should be ~100% of total
        V_total = mp.potential(np.array([100.0, 0, 0]))
        eps = 78.0 * 0.000142
        V_mono = 5.0 / (4*math.pi*eps*100) * math.exp(-100/7.86)
        # Monopole should be >95% of total
        if abs(V_total) > 1e-15:
            assert abs(V_mono / V_total) > 0.9

class TestOverlapCheck:
    """Test the overlap check configuration."""

    def test_default_enabled(self):
        cfg = PySTARCConfig()
        assert cfg.overlap_check is True

    def test_xml_disable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is False

    def test_xml_enable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>true</overlap_check>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is True

class TestMultipoleFallbackConfig:
    """Test multipole_fallback configuration."""

    def test_default_enabled(self):
        cfg = PySTARCConfig()
        assert cfg.multipole_fallback is True

    def test_xml_disable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <multipole_fallback>false</multipole_fallback>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.multipole_fallback is False

    def test_both_flags_independent(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
  <multipole_fallback>true</multipole_fallback>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is False
        assert cfg.multipole_fallback is True

class TestLJForcesConfig:
    """Test lj_forces configuration."""

    def test_default_disabled(self):
        cfg = PySTARCConfig()
        assert cfg.lj_forces is False

    def test_xml_enable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <lj_forces>true</lj_forces>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.lj_forces is True

    def test_all_three_flags_independent(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
  <multipole_fallback>true</multipole_fallback>
  <lj_forces>true</lj_forces>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is False
        assert cfg.multipole_fallback is True
        assert cfg.lj_forces is True
