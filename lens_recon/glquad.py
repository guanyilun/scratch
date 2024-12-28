#%%
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from dataclasses import dataclass
from wignerd import cf_from_cl, cl_from_cf
from fastgl import FastGL

@dataclass
class GLQuad:
    x: np.ndarray
    w: np.ndarray

    def __init__(self, n):
        gl = FastGL(n)
        self.x = gl.x
        self.w = gl.w

    def __repr__(self):
        return f"GLQuad(n={len(self.x)})"

    def cf_from_cl(self, s1, s2, cl, lmax=None, lmin=0, prefactor=False):
        if lmax is None: lmax = len(cl) - 1
        return cf_from_cl(s1, s2, cl, self.x, lmax, prefactor, lmin)

    def cl_from_cf(self, s1, s2, cf, lmax):
        return cl_from_cf(s1, s2, cf, self.x, self.w, lmax)
    
#%%
if __name__ == '__main__':
    gl = GLQuad(15)
    cl = np.arange(1, 12)
    cf = gl.cf_from_cl(1, 2, cl)
    print(cf)
    cl_new = gl.cl_from_cf(3, 4, cf, 10)
    print(cl_new)

# %%
