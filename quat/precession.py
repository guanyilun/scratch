import jax.numpy as jnp
import utils as u

def compute_precession(d):
    t = (d.v1+d.v2-2451545.0) / 36525

    # precession / obliquity corrections
    dpr = -0.29965*u.arcsec
    dob = -0.02524*u.arcsec

    return jnp.array([dpr, dob])*t

def compute_mean_obliquity(d):
    # mean obliquity of the ecliptic
    t = (d.v1+d.v2-2451545.0) / 36525
    eps0 = (84381.448 - 46.8150*t - 0.00059*t**2 + 0.001813*t**3)*u.arcsec
    return eps0



