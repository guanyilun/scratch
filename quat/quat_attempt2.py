"""second attempt to implement quaternion in jax, see attempt1.py for more documentation"""

from typing import NamedTuple
import numpy as np
import jax
from jax import numpy as jnp
from functools import partial


class Quat(NamedTuple):
    v: jax.Array
    def __mul__(self, other):
        if isinstance(other, Quat):
            return Quat(_quat_mul(self.v, other.v))
        else:
            return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, Quat):
            return Quat(_quat_mul(other.v, self.v))
        else:
            return NotImplemented
    def __invert__(self):
        return Quat(_quat_inv(self.v))
    def conj(self):
        return Quat(_quat_conj(self.v))

class QuatArray(NamedTuple):
    v: jax.Array
    def __mul__(self, other):
        if isinstance(other, QuatArray):
            return QuatArray(_quat_mul_v(self.v, other.v))
        elif isinstance(other, Quat):
            return QuatArray(_quat_mul_vs(self.v, other.v))
        else:
            return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, Quat):
            return QuatArray(_quat_mul_sv(other.v, self.v))
        elif isinstance(other, QuatArray):
            return QuatArray(_quat_mul_v(other.v, self.v))
        else:
            return NotImplemented
    def __invert__(self):
        return QuatArray(_quat_inv_v(self.v))
    def conj(self):
        return QuatArray(_quat_conj_v(self.v))

@jax.jit
def _quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    a = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    b = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    c = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    d = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return jnp.array([a, b, c, d])
_quat_mul_v  = jax.jit(jax.vmap(_quat_mul, in_axes=(0, 0)))
_quat_mul_sv = jax.jit(jax.vmap(_quat_mul, in_axes=(None, 0)))
_quat_mul_vs = jax.jit(jax.vmap(_quat_mul, in_axes=(0, None)))

@jax.jit
def _quat_conj(q: jax.Array) -> jax.Array:
    return jnp.array([q[0], -q[1], -q[2], -q[3]])
_quat_conj_v = jax.jit(jax.vmap(_quat_conj, in_axes=(0,)))

@jax.jit
def _quat_inv(q: jax.Array) -> jax.Array:
    return _quat_conj(q) / (jnp.sum(q**2))
_quat_inv_v = jax.jit(jax.vmap(_quat_inv, in_axes=(0,)))

def _quat_to_mat_col3(q: jax.Array) -> jax.Array:
    x = 2*(q[1]*q[3] + q[0]*q[2])
    y = 2*(q[2]*q[3] - q[0]*q[1])
    z = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return jnp.array([x, y, z])

def quat_to_mat_col3(q: Quat) -> jax.Array:
    return _quat_to_mat_col3(q.v)

def _quat_from_rot(angle, direction):
    a = jnp.cos(angle / 2)
    b = jnp.sin(angle / 2) * direction[0]
    c = jnp.sin(angle / 2) * direction[1]
    d = jnp.sin(angle / 2) * direction[2]
    return jnp.array([a, b, c, d])

def quat_from_rot(angle, direction):
    return Quat(_quat_from_rot(angle, direction))

# ------------------------------------------------------
# follow the convention in so3g
# ------------------------------------------------------

@jax.jit
def _euler(axis: int, angle: float) -> jax.Array:
    v = jnp.zeros(4)
    v = v.at[0].set(jnp.cos(angle / 2))
    v = v.at[axis+1].set(jnp.sin(angle / 2))
    return v
_euler_v = jax.jit(jax.vmap(_euler, in_axes=(None, 0)))

def euler(axis: int, angle: jax.Array) -> Quat:
    if isinstance(angle, jnp.ndarray):
        return QuatArray(_euler_v(axis, angle))
    else:
        return Quat(_euler(axis, angle))

def rotation_lonlat(lon: jax.Array, lat: jax.Array, psi: jax.Array | None = None) -> Quat | QuatArray:
    res = euler(2, lon) * euler(1, np.pi/2 - lat)
    if psi is not None:
        res = res * euler(2, psi)
    return res

def rotation_iso(theta, phi, psi=None):
    if psi is None:
        return euler(2, phi) * euler(1, theta)
    return euler(2, phi) * euler(1, theta) * euler(2, psi)

def rotation_xieta(xi, eta, gamma=0):
    phi = jnp.arctan2(xi, eta)
    theta = jnp.arcsin(jnp.sqrt(eta**2 + xi**2))
    psi = gamma - phi
    return rotation_iso(theta, phi, psi)

def decompose_iso(q: jax.Array):
    a = q.v[..., 0]
    b = q.v[..., 1]
    c = q.v[..., 2]
    d = q.v[..., 3]
    psi = jnp.arctan2(a*b+c*d, a*c-b*d)
    phi = jnp.arctan2(c*d-a*b, a*c+b*d)
    theta = 2 * jnp.arctan2((b**2 + c**2)**0.5, (a**2 + d**2)**0.5)
    return theta, phi, psi

def decompose_lonlat(q: jax.Array):
    theta, phi, psi = decompose_iso(q)
    return phi, (np.pi/2-theta), psi

def decompose_xieta(q: jax.Array):
    theta, phi, psi = decompose_iso(q)
    xi = -jnp.sin(theta) * jnp.sin(phi)
    eta = -jnp.sin(theta) * jnp.cos(phi)
    gamma = psi + phi
    return xi, eta, gamma

# #%%
# alpha = jnp.ones(100)
# q1 = eular(2, alpha)
# q2 = eular(1, alpha)
# q3 = eular(2, 0)

# q = rotation_xieta(alpha, alpha*0, alpha*0)
# # %%
# decompose_iso(q2)
# %%
