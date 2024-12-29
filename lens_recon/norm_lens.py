#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import numpy as np

from glquad import GLQuad

# @partial(jax.jit, static_argnums=(0,1,2))
def qtt(lmax, rlmin, rlmax, ucl, ocl):
    ilmax = len(ucl) - 1
    
    # initialize gl quadrature
    glq = GLQuad(int((ilmax*2 + lmax + 1)/2))  # Assuming GLQuad is imported
    
    # common factors
    ell = jnp.arange(0, len(ucl))
    llp1 = ell * (ell + 1)
    div_dl = 1/ocl
    cl_div_dl = ucl/ocl
    
    # get zeta terms: convert to angular space
    zeta_00 = glq.cf_from_cl(0, 0, div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_01_p = glq.cf_from_cl(0, 1, jnp.sqrt(llp1) * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_01_m = glq.cf_from_cl(0, -1, jnp.sqrt(llp1) * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_11_p = glq.cf_from_cl(1, 1, llp1 * ucl * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_11_m = glq.cf_from_cl(1, -1, llp1 * ucl * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    
    # back to ell space
    nlpp_term_1 = glq.cl_from_cf(-1, -1, zeta_00*zeta_11_p - zeta_01_p**2, lmax)
    nlpp_term_2 = glq.cl_from_cf(1, -1, zeta_00*zeta_11_m - zeta_01_p*zeta_01_m, lmax)
    
    return 1/(np.pi * llp1 * (nlpp_term_1 + nlpp_term_2))

#%%
if __name__ == '__main__':
    cltt = np.arange(1, 102, dtype=np.float64)
    nltt = np.zeros_like(cltt)
    ucl = cltt
    ocl = cltt + nltt
    lmax_p = 100
    rtt = qtt(100, 1, 100, ucl, ocl)
    print(f"{rtt=}")

    import pytempura as tp
    print(tp.norm_lens.qtt(lmax_p, 1, lmax_p, ucl, ucl, ocl)[0])


# %%
