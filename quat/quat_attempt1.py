#%%
from typing import NamedTuple
import numpy as np
import jax
from jax import numpy as jnp
from functools import partial


class Quat(NamedTuple):
    w: float
    x: float
    y: float
    z: float
    def __add__(self, other: "Quat") -> "Quat":
        return Quat(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    def __mul__(self, other: "Quat") -> "Quat":
        return _quat_mul(self, other)
    def __rmul__(self, other: "Quat") -> "Quat":
        return _quat_mul(other, self)
    def conj(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)
    def inv(self) -> "Quat":
        return _quat_inv(self)

def _quat_mul(q1: Quat, q2: Quat) -> Quat:
    a = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    b = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    c = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    d = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    return Quat(a, b, c, d)

def _quat_conj(q: Quat) -> Quat:
    return Quat(q.w, -q.x, -q.y, -q.z)

def _quat_inv(q: Quat) -> Quat:
    return _quat_conj(q) / (q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)

def _quat_div(q1: Quat, q2: Quat) -> Quat:
    return _quat_mul(q1, _quat_inv(q2))

def _quat_norm(q: Quat) -> float:
    return jnp.sqrt(q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)

def _quat_norm2(q: Quat) -> float:
    return q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2

def _quat_rot(q: Quat, v: Quat) -> Quat:
    return _quat_mul(q, _quat_mul(v, _quat_inv(q)))

# ------------------------------------------------------
# follow the convention in so3g
# ------------------------------------------------------

def eular(axis: int, angle: float) -> Quat:
    """
    axis: 0, 1, 2 -> x, y, z
    """
    v = jnp.zeros(4)
    v = v.at[0].set(jnp.cos(angle / 2))
    v = v.at[axis+1].set(jnp.sin(angle / 2))
    return Quat(*v)

def rotation_lonlat(lon: float, lat: float, psi: float = 0) -> Quat:
    """
    return a quaternion that represents the following rotation:
        Qz(lon) Qy(pi/2 - lat) Qz(psi)

    YG: still not sure why the order is not
        Qz(psi) Qy(pi/2 - lat) Qz(lon)

    I suspect it's the difference between rotating the vector or
    rotating the coordinate system.

    """
    return eular(2, lon) * eular(1, np.pi/2 - lat) * eular(2, psi)

def rotation_iso(theta, phi, psi=None):
    """Returns the quaternion that composes the Euler rotations:

        Qz(phi) Qy(theta) Qz(psi)

    Note arguments are in radians.

    """
    return eular(2, phi) * eular(1, theta) * eular(2, psi)

def rotation_xieta(xi, eta, gamma=0):
    """Returns the quaternion that rotates the center of focal plane to
    (xi, eta, gamma).  This is equivalent to composed Euler rotations:

        Qz(phi) Qy(theta) Qz (psi)

    where

        xi = - sin(theta) * sin(phi)
        eta = - sin(theta) * cos(phi)
        gamma = psi + phi

    from this, we get

     => tan(phi) = xi / eta
        1 + tan^2(phi) = 1 / cos^2(phi)
        sin(theta) = +- eta * sqrt(1 + tan^2(phi))
                   = +- sqrt(eta^2 + xi^2)

     => phi = atan2(xi, eta)
        theta = asin(sqrt(eta^2 + xi^2))
        psi = gamma - phi

    theta is the angular distance from the z-axis

    """
    phi = np.arctan2(xi, eta)
    theta = np.arcsin(np.sqrt(eta**2 + xi**2))
    psi = gamma - phi
    return rotation_iso(theta, phi, psi)

def decompose_iso(q: Quat):
    """
    one can show that (see sympy section decompose_iso below)

        tan(psi) = [ w*x+y*z / w*y-x*z ]
        tan(phi) = [ y*z-w*x / w*y+x*z ]

        [ x^2+y^2 / w^2+z^2 ]
      = 2*sin(theta/2)^2 / (cos(theta) + 1)
      = sin(theta/2)^2 / cos(theta/2)^2

        tan(theta/2)^2 = (x^2+y^2) / (w^2+z^2)
     => theta = 2*atan2(sqrt(x^2+y^2), sqrt(w^2+z^2))

    """
    a, b, c, d = q.w, q.x, q.y, q.z
    psi = jnp.arctan2(a*b+c*d, a*c-b*d)
    phi = jnp.arctan2(c*d-a*b, a*c+b*d)
    theta = 2 * np.arctan2((b**2 + c**2)**0.5, (a**2 + d**2)**0.5)
    return theta, phi, psi

def decompose_lonlat(q: Quat):
    """
    decompose the quaternion into lon, lat, psi

    note for longitude, latitude, psi, the translation rule is that

        lon = phi
        lat = pi/2 - theta
        psi = psi

    """
    theta, phi, psi = decompose_iso(q)
    return phi, (np.pi/2-theta), psi


@partial(jnp.vectorize, signature='(q)->(q)')
def decompose_xieta(q: Quat):
    """
    decompose the quaternion into xi, eta, gamma

    note for xi, eta, gamma, the translation rule is that

        xi = - sin(theta) * sin(phi)
        eta = - sin(theta) * cos(phi)
        gamma = psi + phi

    """
    theta, phi, psi = decompose_iso(q)
    xi = -jnp.sin(theta) * jnp.sin(phi)
    eta = -jnp.sin(theta) * jnp.cos(phi)
    gamma = psi + phi
    return xi, eta, gamma


#------------------------------------------------------
# #%% decompose_iso
# from sympy import *

# theta, phi, psi = symbols('theta, phi, psi')

# q1 = Quat(cos(phi/2), 0, 0, sin(phi/2))
# q2 = Quat(cos(theta/2), 0, sin(theta/2), 0)
# q3 = Quat(cos(psi/2), 0, 0, sin(psi/2))
# r = q1 * q2 * q3
# a, b, c, d = r.w, r.x, r.y, r.z
# print(simplify((a*b+c*d)/(a*c-b*d)))
# print(simplify((c*d-a*b)/(a*c+b*d)))
# print(simplify((b**2+c**2)/(a**2+d**2)))
#------------------------------------------------------
