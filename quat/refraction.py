# see if I can implement atmosphere refraction calculation in erfa using jax
#%%
import numpy as np
import jax
from jax import numpy as jnp
import jax.numpy as jnp
from functools import partial
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def refraction_coeffs(pressure, humidity, temperature):
    """mm frequency only"""
    logging.info("Calculating refraction coefficients...")
    T  = np.clip(temperature, -150, 200)
    P  = np.clip(pressure, 0, 10000)
    H  = np.clip(humidity, 0, 1)
    ps = (10**((0.7859+0.03477*T) / (1+0.00412*T))) * (1+P*(4.5e-6+6e-10*T**2))
    if P > 0: pw = H*ps / (1-(1-H)*ps/P)
    else: pw = 0
    T += 273.15
    gamma = (77.6890e-6*P - (6.3938e-6-0.375463/T)*pw)/T
    beta = 4.4474e-6*T*(1 - 0.0074*pw)
    return gamma*(1-beta), -gamma*(beta-gamma/2)

@partial(jax.jit, static_argnums=(1,2,3))
def refraction_angle(el, pressure, humidity, temperature):
    el = jnp.deg2rad(el)
    tan_z = jnp.tan(jnp.pi/2 - el)
    A, B = refraction_coeffs(pressure, humidity, temperature)
    ref_angle = tan_z * (A + B * tan_z**2)
    return jnp.rad2deg(ref_angle)


