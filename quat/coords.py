#%%
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

import quat_attempt2 as quat

ERA_EPOCH = 946684800 + 3600 * 12
ERA_POLY = np.array([6.300387486754831, 4.894961212823756])
DEG = np.pi / 180


def naive_azel2bore(t, az, el, roll=0, site=None):
    J = (t - ERA_EPOCH) / 86400
    era = jnp.polyval(ERA_POLY, J)
    lst = era + site.lon * DEG

    Q = (
        quat.euler(2, lst) *
        quat.euler(1, np.pi/2 - site.lat * DEG) *
        quat.euler(2, np.pi) *
        quat.euler(2, -az) *
        quat.euler(1, np.pi/2 - el) *
        quat.euler(2, roll)
    )
    return Q

