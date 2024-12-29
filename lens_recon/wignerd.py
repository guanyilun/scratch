import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
from functools import partial

@partial(jax.jit, inline=True)
def alpha_f(l: jax.Array, s1: int, s2: int) -> jax.Array:
    return jnp.where((l <= jnp.abs(s1)) | (l <= jnp.abs(s2)), 0.0, 
                     jnp.sqrt((l**2 - s1**2)*(l**2 - s2**2))/l)

def wigd_init(s1: int, s2: int, cos_theta: jax.Array) -> jax.Array:
    # Handle sign based on s1 + s2 parity
    s12sign = jnp.where(((s1 + s2) % 2) == 1, -1.0, 1.0)
    A = 1.0

    # Swap if needed
    s1, s2 = jnp.where((c:=jnp.abs(s1) > jnp.abs(s2)), s2, s1), jnp.where(c, s1, s2)
    A = jnp.where(c, A * s12sign, A)

    # Handle negative s2
    s1, s2 = jnp.where((c:=s2<0), -s1, s1), jnp.where(c, -s2, s2)
    A = jnp.where(c, A * s12sign, A)

    # Calculate prefactor A
    # dynamically allocated array doesn't work with jax
    # so I had to use fori_loop
    # i = jnp.arange(1, s2 - abs_s1 + 1)
    # A = A * jnp.prod(jnp.sqrt((s2 + abs_s1 + i) / i))
    abs_s1 = jnp.abs(s1)
    A = jax.lax.fori_loop(
        1,  # start
        s2 - abs_s1 + 1,  # end
        lambda i, v: v*jnp.sqrt((s2+abs_s1+i)/i), 
        A
    )
    # Calculate wigd
    return s2, A * ((1.0 + cos_theta)/2.0)**((s2 + s1)/2.0) * ((1.0 - cos_theta)/2.0)**((s2 - s1)/2.0)

def wigd_rec(l, s1, s2, cos_theta, wigd_hi, wigd_lo):
    """Recursive step for Wigner d."""
    alpha_hi = alpha_f(l + 1, s1, s2)
    alpha_lo = alpha_f(l, s1, s2)

    beta = jnp.where((s1 == 0) | (s2 == 0), 0.0, (s1 * s2)/(l * (l + 1)))
    wigd_new = ((2*l + 1)*(cos_theta - beta) * wigd_hi - alpha_lo*wigd_lo) / alpha_hi
    return wigd_new, wigd_hi

@partial(jax.jit, static_argnums=(4, 5, 6))
def cf_from_cl(s1, s2, cl, cos_theta, lmax=None, lmin=0, prefactor=False):
    """Calculate ∑ₗ cl d_{s1,s2}^l"""
    if lmax is None: lmax = len(cl) - 1
    l0, wigd_hi = wigd_init(s1, s2, cos_theta)
    wigd_lo = jnp.zeros_like(cos_theta)

    fac = (2*l0+1)/(4*np.pi) if prefactor else 1.0
    cf = jnp.where((l0 >= lmin) & (l0 <= lmax), cl[l0] * wigd_hi * fac, jnp.zeros_like(cos_theta))

    def loop_body(carry):
        l, wigd_hi, wigd_lo, cf = carry
        wigd_hi, wigd_lo = wigd_rec(l, s1, s2, cos_theta, wigd_hi, wigd_lo)
        l = l + 1
        fac = (2*l+1)/(4*jnp.pi) if prefactor else 1.0
        cf = cf + jnp.where(
            (l >= lmin),
            cl[l] * wigd_hi * fac,
            0.0
        )
        return (l, wigd_hi, wigd_lo, cf)

    def loop_cond(carry):
        l, _, _, _ = carry
        return l < lmax
    
    init_state = (l0, wigd_hi, wigd_lo, cf)
    _, _, _, cf = jax.lax.while_loop(loop_cond, loop_body, init_state)

    return cf

@partial(jax.jit, static_argnums=(5,))
def cl_from_cf(s1, s2, cf, cos_theta, weights, lmax):
    """Calculate ∫ dcosθ cf d_{s1,s2}^l(θ)"""
    l0, wigd_hi = wigd_init(s1, s2, cos_theta)
    wigd_hi = wigd_hi * weights
    wigd_lo = jnp.zeros_like(cos_theta)
    cl = jnp.zeros(lmax + 1, dtype=jnp.float64)

    cl = cl.at[l0].set(
        jnp.where(l0 <= lmax, jnp.sum(cf * wigd_hi), 0.0)
    )

    def loop_body(carry):
        l, wigd_hi, wigd_lo, cl = carry
        wigd_hi, wigd_lo = wigd_rec(l, s1, s2, cos_theta, wigd_hi, wigd_lo)
        l += 1
        cl = cl.at[l].set(jnp.sum(cf * wigd_hi))
        return (l, wigd_hi, wigd_lo, cl)

    def loop_cond(carry):
        l, _, _, _ = carry
        return l < lmax

    init_state = (l0, wigd_hi, wigd_lo, cl)
    _, _, _, cl = jax.lax.while_loop(loop_cond, loop_body, init_state)
    return cl

# Example usages
if __name__ == "__main__":
    s1, s2 = 1, 2
    assert np.allclose(
        alpha_f(np.arange(5, 10), s1, s2), 
        [4.48998886, 5.57773351, 6.63940002, 7.68521307, 8.72062972])

    cos_theta = np.array([-0.75, -0.5 ,-0.25, 0.25, 0.5, 0.75])
    l0, wigd_hi = wigd_init(s1, s2, np.array(cos_theta))
    print(f"{l0=}, {wigd_hi=}")
    assert np.allclose(
        wigd_hi,
        [0.08267973, 0.21650635, 0.36309219, 0.60515365, 0.64951905, 0.5787581])

    wigd_lo = np.zeros_like(wigd_hi)
    wigd_hi, _ = wigd_rec(10, s1, s2, cos_theta, wigd_hi, wigd_lo)
    print(f"{wigd_hi=}")
    assert np.allclose(
        wigd_hi,
        [-0.12382019723570781, -0.21871634356469039, -0.1898342901845391, 0.2734900790794207, 0.6101034846804522, 0.8257122028677083])
    
    lmax = 10
    cl = np.arange(lmax + 1)
    cos_theta = np.linspace(-1, 1, 10)
    cf = cf_from_cl(s1, s2, cl, cos_theta, lmax)
    print(f"{cf=}")
    assert np.allclose(
        cf,
        [0.0, -1.4236208075434558, 1.3172886463419415, -0.621077274210937, -0.3104460696913065, 1.8022470909608022, -1.902863895245635, 3.139384174670357, -2.7838894965822467, 0.0])

    weights = np.ones_like(cos_theta)
    cl_recon = cl_from_cf(s1, s2, cf, cos_theta, weights, lmax)
    print(f"{cl_recon=}")
    assert np.allclose(
        cl_recon,
        [0.0, 0.0, 0.08558917001267696, -0.8918690803928095, -1.3643483186273897, -0.7240572707990893, -0.6074739860830027, 1.172257045428006, 0.5797840225510533, 2.2034783702814273, 1.1270394245456616])


# %%
