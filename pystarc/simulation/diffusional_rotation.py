"""
Diffusional rotation via quaternion algebra
===========================================

Background
-------------------
In Brownian dynamics, molecules are rigid bodies that both translate
and rotate under thermal fluctuations.  The rotational diffusion
coefficient for a sphere is:
    D_rot = 3 D_trans / (4 a²) = kBT / (8π η a³)
where a is the hydrodynamic radius.

At each BD time step Δt, a random rotation is drawn from the
diffusional distribution:
    θ = √(2 D_rot Δt) × |ξ|

where ξ ~ N(0, I₃) is a 3D Gaussian noise vector.  The rotation
axis is ξ/|ξ| (uniformly distributed on S²) and the rotation
angle is θ.

Quaternion representation
-------------------------
Rotations are stored as unit quaternions q = (w, x, y, z) with
|q| = 1.  A rotation by angle θ about axis n̂ is:
    q = (cos(θ/2), sin(θ/2) × n̂)
Composing two rotations is quaternion multiplication:
    q_new = q_step ⊗ q_old

This is numerically superior to rotation matrices because:
1. Quaternions avoid gimbal lock
2. Renormalization (q - q/|q|) prevents drift
3. Random rotations are trivially generated
4. Exact for any rotation angle (no interpolation table)

Ligand atom positions
---------------------
The ligand atoms in the lab frame are computed from the
reference-frame positions (mol2_pos0) by:
    r_lab = R × r_ref + centroid
where R is the 3×3 rotation matrix extracted from the quaternion.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import math


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product q1 * q2. Format: (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_of_rotvec(omega: np.ndarray) -> np.ndarray:
    """Convert rotation vector omega to unit quaternion (w,x,y,z)."""
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / angle
    half = 0.5 * angle
    return np.array(
        [
            math.cos(half),
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
        ]
    )


def random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random unit quaternion."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return q


def diffusional_rotation(rng: np.random.Generator, tau: float) -> np.ndarray:
    """
    Random rotation quaternion for a body that has diffused for
    dimensionless time tau = t * Dr.
    For tau <= 0.25: single Gaussian step in angle space
    For 0.25 < tau < 4: recursive splitting at known checkpoints
    For tau >= 4: uniformly random rotation (mixing time exceeded)
    Returns unit quaternion (w, x, y, z).
    """
    if tau <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if tau <= 0.25:
        # infinitesimal step: Gaussian in angle space
        sqdt = math.sqrt(2.0 * tau)
        omega = sqdt * rng.standard_normal(3)
        return quat_of_rotvec(omega)
    elif tau < 0.5:
        # split at 0.25
        q0 = diffusional_rotation(rng, 0.25)
        q1 = diffusional_rotation(rng, tau - 0.25)
        return quat_multiply(q1, q0)
    elif tau < 1.0:
        # split at 0.5 using spline-based distribution
        q0 = _spline_rot_0p5(rng)
        q1 = diffusional_rotation(rng, tau - 0.5)
        return quat_multiply(q1, q0)
    elif tau < 2.0:
        # split at 1.0
        q0 = _spline_rot_1p0(rng)
        q1 = diffusional_rotation(rng, tau - 1.0)
        return quat_multiply(q1, q0)
    elif tau < 4.0:
        # split at 2.0
        q0 = _spline_rot_2p0(rng)
        q1 = diffusional_rotation(rng, tau - 2.0)
        return quat_multiply(q1, q0)
    else:
        # past 4 time constants: effectively random
        return random_unit_quat(rng)


# Spline-based rotation distributions at checkpoints
# We approximate them with the exact distribution: rotation angle theta for a
# diffusing sphere has distribution p(theta) propto sin^2(theta/2) * P(theta,t)
# where P is the sum over spherical harmonics.
# For practical purposes we use the Gaussian approximation corrected for SO(3).


def _sample_rotation_angle(rng: np.random.Generator, tau: float) -> float:
    """
    Sample rotation angle theta from the diffusion distribution on SO(3).
    p(theta, t) = sum_{l=0}^{inf} (2l+1) * exp(-l(l+1)*t) * sin(theta/2)
                  * U_l(cos(theta/2))    (Wigner d-functions)
    For moderate tau, we use rejection sampling with a Gaussian proposal.
    """
    sigma = math.sqrt(2.0 * tau)  # std dev of angle
    # Rejection sampling: proposal = half-normal (angles in [0, pi])
    # target = actual diffusion distribution (truncated sum of harmonics)
    max_l = max(10, int(5.0 / tau) + 5)

    def p_target(theta: float) -> float:
        """p(theta) propto sum_l (2l+1) exp(-l(l+1)t) sin(theta)
        (using sin(theta) Jacobian for SO(3))"""
        val = 0.0
        for l in range(max_l + 1):
            val += (2 * l + 1) * math.exp(-l * (l + 1) * tau) * (2 * l + 1)
            # approximate: use trace of rotation matrix = 1 + 2*cos(theta)
            # so weight by sin^2(theta/2) = (1-cos(theta))/2
        # Simplified: p(theta) = sum_l (2l+1)exp(-l(l+1)t) * sin(theta) * U_l(cos theta)
        # Use the known series:
        result = 0.0
        for l in range(max_l + 1):
            # 2*(2l+1)*exp(-l(l+1)*t) * sin((2l+1)*theta/2) / sin(theta/2)
            half = (2 * l + 1) * theta / 2.0
            denom = math.sin(theta / 2.0) if theta > 1e-10 else (2 * l + 1) / 2.0
            if denom > 1e-15:
                result += (
                    2 * (2 * l + 1) * math.exp(-l * (l + 1) * tau) * math.sin(half)
                )
        return max(0.0, result)

    # For large tau use full Gaussian approach
    for _ in range(10000):
        theta = abs(rng.normal(0, sigma))
        theta = theta % (2 * math.pi)
        if theta > math.pi:
            theta = 2 * math.pi - theta
        # accept with probability proportional to sin(theta/2)^2 correction
        # (Gaussian on R^3 vs uniform on SO(3))
        if theta < 1e-10:
            accept = 1.0
        else:
            accept = (math.sin(theta / 2.0) / (theta / 2.0)) ** 2
        if rng.random() < accept:
            return theta
    return sigma  # fallback


def _sample_quat_for_tau(rng: np.random.Generator, tau: float) -> np.ndarray:
    """Sample a quaternion for a specific tau checkpoint."""
    theta = _sample_rotation_angle(rng, tau)
    # random axis
    axis = rng.standard_normal(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis /= norm
    half = theta / 2.0
    return np.array(
        [
            math.cos(half),
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
        ]
    )


# These are the precomputed CDF tables mapping uniform[0,1] -> rotation angle phi
# at tau = 0.5, 1.0, 2.0.
# Format: rprob[i] = P(theta < phis[i]) for diffusional rotation at given tau.

_PHIS = [
    0,
    0.0174533,
    0.0349066,
    0.0523599,
    0.0698132,
    0.0872665,
    0.10472,
    0.122173,
    0.139626,
    0.15708,
    0.174533,
    0.191986,
    0.20944,
    0.226893,
    0.244346,
    0.261799,
    0.279253,
    0.296706,
    0.314159,
    0.331613,
    0.349066,
    0.366519,
    0.383972,
    0.401426,
    0.418879,
    0.436332,
    0.453786,
    0.471239,
    0.488692,
    0.506145,
    0.523599,
    0.541052,
    0.558505,
    0.575959,
    0.593412,
    0.610865,
    0.628319,
    0.645772,
    0.663225,
    0.680678,
    0.698132,
    0.715585,
    0.733038,
    0.750492,
    0.767945,
    0.785398,
    0.802851,
    0.820305,
    0.837758,
    0.855211,
    0.872665,
    0.890118,
    0.907571,
    0.925024,
    0.942478,
    0.959931,
    0.977384,
    0.994838,
    1.01229,
    1.02974,
    1.0472,
    1.06465,
    1.0821,
    1.09956,
    1.11701,
    1.13446,
    1.15192,
    1.16937,
    1.18682,
    1.20428,
    1.22173,
    1.23918,
    1.25664,
    1.27409,
    1.29154,
    1.309,
    1.32645,
    1.3439,
    1.36136,
    1.37881,
    1.39626,
    1.41372,
    1.43117,
    1.44862,
    1.46608,
    1.48353,
    1.50098,
    1.51844,
    1.53589,
    1.55334,
    1.5708,
    1.58825,
    1.6057,
    1.62316,
    1.64061,
    1.65806,
    1.67552,
    1.69297,
    1.71042,
    1.72788,
    1.74533,
    1.76278,
    1.78024,
    1.79769,
    1.81514,
    1.8326,
    1.85005,
    1.8675,
    1.88496,
    1.90241,
    1.91986,
    1.93732,
    1.95477,
    1.97222,
    1.98968,
    2.00713,
    2.02458,
    2.04204,
    2.05949,
    2.07694,
    2.0944,
    2.11185,
    2.1293,
    2.14675,
    2.16421,
    2.18166,
    2.19911,
    2.21657,
    2.23402,
    2.25147,
    2.26893,
    2.28638,
    2.30383,
    2.32129,
    2.33874,
    2.35619,
    2.37365,
    2.3911,
    2.40855,
    2.42601,
    2.44346,
    2.46091,
    2.47837,
    2.49582,
    2.51327,
    2.53073,
    2.54818,
    2.56563,
    2.58309,
    2.60054,
    2.61799,
    2.63545,
    2.6529,
    2.67035,
    2.68781,
    2.70526,
    2.72271,
    2.74017,
    2.75762,
    2.77507,
    2.79253,
    2.80998,
    2.82743,
    2.84489,
    2.86234,
    2.87979,
    2.89725,
    2.9147,
    2.93215,
    2.94961,
    2.96706,
    2.98451,
    3.00197,
    3.01942,
    3.03687,
    3.05433,
    3.07178,
    3.08923,
    3.10669,
    3.12414,
    3.14159,
]

_RPROB_0P5 = [
    0,
    4.80262e-06,
    2.40036e-05,
    6.71701e-05,
    0.000143822,
    0.000263413,
    0.000435311,
    0.000668783,
    0.000972972,
    0.00135688,
    0.00182936,
    0.00239909,
    0.00307454,
    0.00386399,
    0.00477549,
    0.00581687,
    0.00699568,
    0.00831922,
    0.0097945,
    0.0114283,
    0.0132269,
    0.0151965,
    0.0173429,
    0.0196715,
    0.0221874,
    0.0248953,
    0.0277997,
    0.0309044,
    0.0342133,
    0.0377295,
    0.0414558,
    0.0453949,
    0.0495488,
    0.0539193,
    0.0585076,
    0.0633147,
    0.0683411,
    0.0735871,
    0.0790523,
    0.0847363,
    0.0906381,
    0.0967562,
    0.103089,
    0.109635,
    0.11639,
    0.123353,
    0.130521,
    0.13789,
    0.145455,
    0.153214,
    0.161162,
    0.169295,
    0.177606,
    0.186092,
    0.194747,
    0.203565,
    0.212541,
    0.221668,
    0.23094,
    0.240351,
    0.249895,
    0.259563,
    0.269351,
    0.27925,
    0.289254,
    0.299355,
    0.309546,
    0.31982,
    0.33017,
    0.340587,
    0.351065,
    0.361596,
    0.372173,
    0.382789,
    0.393435,
    0.404104,
    0.41479,
    0.425485,
    0.436182,
    0.446874,
    0.457553,
    0.468213,
    0.478848,
    0.48945,
    0.500013,
    0.510531,
    0.520998,
    0.531407,
    0.541753,
    0.55203,
    0.562233,
    0.572355,
    0.582393,
    0.592341,
    0.602194,
    0.611949,
    0.621599,
    0.631142,
    0.640574,
    0.64989,
    0.659087,
    0.668162,
    0.677113,
    0.685934,
    0.694626,
    0.703184,
    0.711606,
    0.719891,
    0.728037,
    0.736041,
    0.743904,
    0.751622,
    0.759196,
    0.766623,
    0.773905,
    0.78104,
    0.788027,
    0.794868,
    0.801561,
    0.808106,
    0.814505,
    0.820758,
    0.826864,
    0.832826,
    0.838644,
    0.844319,
    0.849852,
    0.855244,
    0.860498,
    0.865614,
    0.870594,
    0.875439,
    0.880153,
    0.884736,
    0.88919,
    0.893518,
    0.897722,
    0.901804,
    0.905767,
    0.909612,
    0.913342,
    0.91696,
    0.920468,
    0.923869,
    0.927166,
    0.93036,
    0.933455,
    0.936454,
    0.939359,
    0.942173,
    0.944899,
    0.94754,
    0.950098,
    0.952577,
    0.954978,
    0.957306,
    0.959563,
    0.961752,
    0.963876,
    0.965938,
    0.96794,
    0.969886,
    0.971778,
    0.97362,
    0.975414,
    0.977163,
    0.97887,
    0.980539,
    0.982171,
    0.98377,
    0.985339,
    0.986881,
    0.988398,
    0.989893,
    0.99137,
    0.992831,
    0.994279,
    0.995717,
    0.997148,
    0.998575,
    1,
]

_RPROB_1P0 = [
    0,
    1.92284e-06,
    9.61217e-06,
    2.69056e-05,
    5.76304e-05,
    0.0001056,
    0.000174609,
    0.000268432,
    0.000390813,
    0.000545471,
    0.000736087,
    0.000966308,
    0.00123974,
    0.00155993,
    0.0019304,
    0.00235459,
    0.00283592,
    0.00337772,
    0.00398327,
    0.00465579,
    0.0053984,
    0.00621419,
    0.00710614,
    0.00807717,
    0.00913012,
    0.0102677,
    0.0114927,
    0.0128075,
    0.0142148,
    0.0157168,
    0.0173158,
    0.0190141,
    0.0208138,
    0.0227169,
    0.0247251,
    0.0268404,
    0.0290644,
    0.0313987,
    0.0338447,
    0.0364038,
    0.0390772,
    0.0418661,
    0.0447715,
    0.0477943,
    0.0509353,
    0.0541952,
    0.0575746,
    0.0610739,
    0.0646935,
    0.0684336,
    0.0722943,
    0.0762757,
    0.0803777,
    0.0846001,
    0.0889425,
    0.0934046,
    0.0979859,
    0.102686,
    0.107503,
    0.112438,
    0.117489,
    0.122654,
    0.127934,
    0.133326,
    0.13883,
    0.144444,
    0.150167,
    0.155996,
    0.161931,
    0.16797,
    0.174111,
    0.180352,
    0.186691,
    0.193126,
    0.199655,
    0.206276,
    0.212988,
    0.219786,
    0.226671,
    0.233638,
    0.240685,
    0.247811,
    0.255013,
    0.262288,
    0.269634,
    0.277048,
    0.284528,
    0.29207,
    0.299674,
    0.307334,
    0.31505,
    0.322819,
    0.330637,
    0.338503,
    0.346413,
    0.354365,
    0.362356,
    0.370384,
    0.378445,
    0.386538,
    0.39466,
    0.402808,
    0.410979,
    0.419171,
    0.427382,
    0.435609,
    0.44385,
    0.452102,
    0.460363,
    0.468631,
    0.476903,
    0.485177,
    0.493451,
    0.501723,
    0.509991,
    0.518253,
    0.526506,
    0.53475,
    0.542982,
    0.5512,
    0.559403,
    0.567588,
    0.575756,
    0.583903,
    0.592029,
    0.600131,
    0.60821,
    0.616263,
    0.62429,
    0.632289,
    0.64026,
    0.6482,
    0.656111,
    0.66399,
    0.671837,
    0.679651,
    0.687432,
    0.69518,
    0.702893,
    0.710572,
    0.718216,
    0.725825,
    0.733398,
    0.740937,
    0.74844,
    0.755908,
    0.763341,
    0.770739,
    0.778103,
    0.785432,
    0.792728,
    0.79999,
    0.80722,
    0.814417,
    0.821582,
    0.828717,
    0.835821,
    0.842896,
    0.849942,
    0.856961,
    0.863953,
    0.87092,
    0.877861,
    0.88478,
    0.891675,
    0.89855,
    0.905404,
    0.91224,
    0.919058,
    0.92586,
    0.932647,
    0.93942,
    0.946181,
    0.952932,
    0.959672,
    0.966405,
    0.973132,
    0.979853,
    0.98657,
    0.993285,
    1,
]

_RPROB_2P0 = [
    0,
    9.80583e-07,
    4.90245e-06,
    1.37249e-05,
    2.94049e-05,
    5.3896e-05,
    8.91479e-05,
    0.000137105,
    0.000199705,
    0.00027888,
    0.000376553,
    0.000494639,
    0.000635042,
    0.000799657,
    0.000990368,
    0.00120904,
    0.00145754,
    0.00173771,
    0.00205136,
    0.00240033,
    0.0027864,
    0.00321134,
    0.00367693,
    0.00418491,
    0.00473698,
    0.00533486,
    0.00598023,
    0.00667474,
    0.00742002,
    0.0082177,
    0.00906935,
    0.00997654,
    0.0109408,
    0.0119637,
    0.0130466,
    0.014191,
    0.0153984,
    0.0166702,
    0.0180077,
    0.0194123,
    0.0208853,
    0.0224279,
    0.0240415,
    0.0257272,
    0.0274862,
    0.0293198,
    0.0312289,
    0.0332148,
    0.0352784,
    0.0374208,
    0.039643,
    0.0419459,
    0.0443305,
    0.0467976,
    0.0493482,
    0.051983,
    0.0547028,
    0.0575084,
    0.0604004,
    0.0633796,
    0.0664466,
    0.0696021,
    0.0728465,
    0.0761805,
    0.0796045,
    0.083119,
    0.0867244,
    0.0904212,
    0.0942097,
    0.0980902,
    0.102063,
    0.106129,
    0.110287,
    0.114538,
    0.118882,
    0.12332,
    0.127851,
    0.132476,
    0.137194,
    0.142005,
    0.14691,
    0.151908,
    0.157,
    0.162184,
    0.167462,
    0.172832,
    0.178295,
    0.18385,
    0.189497,
    0.195236,
    0.201066,
    0.206987,
    0.212998,
    0.2191,
    0.225291,
    0.231572,
    0.237941,
    0.244399,
    0.250944,
    0.257576,
    0.264295,
    0.271099,
    0.277989,
    0.284963,
    0.292021,
    0.299162,
    0.306385,
    0.31369,
    0.321076,
    0.328542,
    0.336087,
    0.343711,
    0.351412,
    0.35919,
    0.367044,
    0.374972,
    0.382975,
    0.391051,
    0.399199,
    0.407419,
    0.415708,
    0.424067,
    0.432494,
    0.440988,
    0.449549,
    0.458175,
    0.466865,
    0.475618,
    0.484434,
    0.49331,
    0.502246,
    0.511241,
    0.520294,
    0.529404,
    0.538569,
    0.547789,
    0.557061,
    0.566386,
    0.575762,
    0.585188,
    0.594663,
    0.604185,
    0.613754,
    0.623368,
    0.633026,
    0.642727,
    0.65247,
    0.662253,
    0.672076,
    0.681937,
    0.691835,
    0.701769,
    0.711737,
    0.721739,
    0.731773,
    0.741838,
    0.751933,
    0.762056,
    0.772207,
    0.782384,
    0.792586,
    0.802812,
    0.81306,
    0.82333,
    0.83362,
    0.843929,
    0.854255,
    0.864598,
    0.874956,
    0.885328,
    0.895713,
    0.90611,
    0.916517,
    0.926933,
    0.937357,
    0.947788,
    0.958224,
    0.968665,
    0.979108,
    0.989554,
    1,
]

_PHIS_ARR = np.array(_PHIS)
_RPROB_0P5_ARR = np.array(_RPROB_0P5)
_RPROB_1P0_ARR = np.array(_RPROB_1P0)
_RPROB_2P0_ARR = np.array(_RPROB_2P0)


def _spline_rot_from_table(rng: np.random.Generator, rprob: np.ndarray) -> np.ndarray:
    """
    Sample a rotation quaternion using the precomputed CDF table.
      pc = uniform()
      phi = spline.value(pc)   <- inverse CDF lookup
      quat = random axis * sin(phi/2) + cos(phi/2)
    """
    pc = rng.random()
    # Inverse CDF: find phi such that rprob(phi) = pc
    # Linear interpolation in the table (matching the Spline::value)
    phi = float(np.interp(pc, rprob, _PHIS_ARR))
    # Random axis
    axis = rng.standard_normal(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis /= norm
    half = phi / 2.0
    return np.array(
        [
            math.cos(half),
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
        ]
    )


def _spline_rot_0p5(rng: np.random.Generator) -> np.ndarray:
    """Exact the reference implementation spline rotation at tau=0.5"""
    return _spline_rot_from_table(rng, _RPROB_0P5_ARR)


def _spline_rot_1p0(rng: np.random.Generator) -> np.ndarray:
    """Exact the reference implementation spline rotation at tau=1.0"""
    return _spline_rot_from_table(rng, _RPROB_1P0_ARR)


def _spline_rot_2p0(rng: np.random.Generator) -> np.ndarray:
    """Exact the reference implementation spline rotation at tau=2.0"""
    return _spline_rot_from_table(rng, _RPROB_2P0_ARR)
