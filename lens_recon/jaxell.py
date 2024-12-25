"""some pixell functions wrapper in jax for differenciability"""
import jax
import jax.numpy as jnp
import numpy as np
from pixell import enmap
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def fft(imap, wcs):
    shape = imap.shape
    print(f"Compiling fft with shape={shape}, wcs={wcs}")
    fmap = jnp.fft.fft2(imap)
    norm = 1 / np.prod(shape[-2:])**0.5
    norm = norm * enmap.pixsize(shape, wcs)**0.5
    fmap = fmap * norm
    return fmap

@partial(jax.jit, static_argnums=(1,))
def ifft(fmap, wcs):
    shape = fmap.shape
    print(f"Compiling ifft with shape={shape}, wcs={wcs}")
    omap = jnp.fft.ifft2(fmap)
    norm = 1 / np.prod(shape[-2:])**0.5
    norm = norm / enmap.pixsize(shape, wcs)**0.5
    omap = omap * norm
    return omap

@partial(jax.jit, static_argnums=(0, 1))
def modlmap(shape, wcs):
    print(f"Compiling modlmap with shape={shape}, wcs={wcs}")
    return jnp.array(enmap.modlmap(shape, wcs))

@partial(jax.jit, static_argnums=(0, 1))
def lmap(shape, wcs):
    print(f"Compiling lmap with shape={shape}, wcs={wcs}")
    return jnp.array(enmap.lmap(shape, wcs))