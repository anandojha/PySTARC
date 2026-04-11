"""
Quaternion-based rigid-body transforms for PySTARC.
"""

from __future__ import annotations
from typing import Optional  
from typing import Tuple
import numpy as np
import math

# Quaternion
class Quaternion:
    """Unit quaternion  q = (w, x, y, z)."""
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0,
                 y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    # constructors 
    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> "Quaternion":
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-14:
            return cls.identity()
        axis = axis / norm
        s = math.sin(angle / 2.0)
        return cls(math.cos(angle / 2.0),
                   axis[0]*s, axis[1]*s, axis[2]*s)

    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> "Quaternion":
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            return cls(0.25/s,
                       (R[2,1]-R[1,2])*s,
                       (R[0,2]-R[2,0])*s,
                       (R[1,0]-R[0,1])*s)
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return cls((R[2,1]-R[1,2])/s,
                       0.25*s,
                       (R[0,1]+R[1,0])/s,
                       (R[0,2]+R[2,0])/s)
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return cls((R[0,2]-R[2,0])/s,
                       (R[0,1]+R[1,0])/s,
                       0.25*s,
                       (R[1,2]+R[2,1])/s)
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            return cls((R[1,0]-R[0,1])/s,
                       (R[0,2]+R[2,0])/s,
                       (R[1,2]+R[2,1])/s,
                       0.25*s)
        
    # arithmetic 
    def __mul__(self, other: "Quaternion") -> "Quaternion":
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Quaternion":
        n = self.norm()
        if n < 1e-14:
            return Quaternion.identity()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def to_rotation_matrix(self) -> np.ndarray:
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=float)

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        return self.to_rotation_matrix() @ np.asarray(v, dtype=float)

    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

# RigidTransform 
class RigidTransform:
    """A combined rotation (Quaternion) + translation (3-vector)."""

    def __init__(self,
                 rotation: Optional[Quaternion] = None,
                 translation: Optional[np.ndarray] = None):
        self.rotation = rotation or Quaternion.identity()
        self.translation = np.asarray(translation if translation is not None
                                      else np.zeros(3), dtype=float)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls()

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply rotation then translation to (N,3) or (3,) array."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        R = self.rotation.to_rotation_matrix()
        rotated = (R @ pts.T).T
        result = rotated + self.translation
        return result if points.ndim == 2 else result[0]

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """self ∘ other  - apply other first, then self."""
        new_rot = self.rotation * other.rotation
        new_trans = self.rotation.rotate_vector(other.translation) + self.translation
        return RigidTransform(new_rot.normalized(), new_trans)

    def inverse(self) -> "RigidTransform":
        inv_rot = self.rotation.conjugate()
        inv_trans = -inv_rot.rotate_vector(self.translation)
        return RigidTransform(inv_rot.normalized(), inv_trans)

    def __repr__(self) -> str:
        t = self.translation
        return (f"RigidTransform(rot={self.rotation}, "
                f"trans=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}))")

# random rotation helpers 
def random_quaternion(rng: Optional[np.random.Generator] = None) -> Quaternion:
    """Uniform random rotation quaternion (Shoemake 1992)."""
    if rng is None:
        rng = np.random.default_rng()
    u1, u2, u3 = rng.uniform(0, 1, 3)
    w = math.sqrt(1-u1) * math.sin(2*math.pi*u2)
    x = math.sqrt(1-u1) * math.cos(2*math.pi*u2)
    y = math.sqrt(u1)   * math.sin(2*math.pi*u3)
    z = math.sqrt(u1)   * math.cos(2*math.pi*u3)
    return Quaternion(w, x, y, z)

def small_rotation_quaternion(sigma_rad: float,
                              rng: Optional[np.random.Generator] = None
                              ) -> Quaternion:
    """Small random rotation with Gaussian angle ~ N(0, sigma_rad)."""
    if rng is None:
        rng = np.random.default_rng()
    axis = rng.standard_normal(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        return Quaternion.identity()
    axis /= norm
    angle = rng.normal(0.0, sigma_rad)
    return Quaternion.from_axis_angle(axis, angle)