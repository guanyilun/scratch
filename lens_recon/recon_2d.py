"""
simple 2d lens reconstruction
"""

import numpy as np
from dataclasses import dataclass
from pixell import enmap, powspec, lensing

deg = np.deg2rad(1)
arcmin = np.deg2rad(1/60)
fwhm = 1.0/(8*np.log(2))**0.5  # beam_sigma = beam_fwhm * fwhm

def gauss_beam(ells, beam_fwhm):
    return np.exp(-ells*(ells+1)*beam_fwhm**2/(16*np.log(2)))

def get_nl(ell, nlev_t, beam_fwhm):
    return (nlev_t * arcmin)**2. / gauss_beam(ell, beam_fwhm*arcmin)**2
    
@dataclass
class Expt:
    """Convenience wrapper class to store experiment parameters
    
    Parameters
    ----------
    name : str
        Experiment name
    beam_fwhm : float
        Beam FWHM in arcmin
    nlev_t : float
        Noise level in temperature in uK-arcmin

    """
    name: str
    beam_fwhm: float
    nlev_t: float
    lmin: int = 30
    lmax: int = 3000
    def get_nl(self, ell):
        return get_nl(ell, self.nlev_t, self.beam_fwhm)
    def get_bl(self, ell):
        return gauss_beam(ell, self.beam_fwhm*arcmin)
    def get_kmask(self, shape, wcs):
        ell = enmap.modlmap(shape, wcs)
        return (ell >= self.lmin) * (ell <= self.lmax)

def sim_lens_map_flat(shape, wcs, ps, expt=None):
    """Simulate a lensed CMB map with a random realization of a lensing potential.

    Parameters
    ----------
    shape : tuple
        Shape of the map to generate with format (ncomp, ny, nx) where ncomp is 
        the number of components (typically 2 for Q/U or 3 for T/Q/U)
    wcs : astropy.wcs.WCS
        World coordinate system defining the pixelization
    ps : array_like
        Power spectrum used to generate random realizations. Should contain
        CMB T/Q/U and lensing potential power spectra.
    expt : Expt, optional
        Experiment object containing beam and noise properties. If provided, the
        lensed map will be convolved with the beam.

    Returns
    -------
    m : ndmap
        The lensed CMB map in real space
    (phi_map, cmb_map_unlensed) : tuple
        The random realizations of the lensing potential and unlensed CMB maps used

    Notes
    -----
    Generates correlated random realizations of CMB fields and a lensing potential,
    applies the lensing deflection, and optionally convolves with a beam.
    """
    assert len(shape) == 3
    maps = enmap.rand_map((shape[0]+1,)+shape[1:], wcs, ps, spin=[0,0,2])
    phi_map, cmb_map_unlensed = maps[0], maps[1:]
    m = lensing.lens_map_flat(cmb_map_unlensed, phi_map)
    if expt is not None:
        l = m.modlmap()
        beam2d = expt.get_bl(l)
        m = enmap.ifft(enmap.fft(m, normalize="phys")*beam2d, normalize="physics").real
    return m, (phi_map, cmb_map_unlensed)

def recon_2d_symlens(shape, wcs, map1, map2, ps_lensinput, expt: Expt, rlmin, rlmax):
    """Perform 2D CMB lensing reconstruction using symlens package.

    This function implements a quadratic estimator for CMB lensing reconstruction
    using the symlens package. It uses the TT (temperature-temperature) estimator
    from Hu & Okamoto 2002.

    Parameters
    ----------
    shape : tuple
        Shape of the map arrays (ny, nx).
    wcs : WCS object
        World Coordinate System object defining the pixelization.
    map1 : ndarray
        First CMB map in Fourier space.
    map2 : ndarray
        Second CMB map in Fourier space.
    ps_lcmb : ndarray
        Power spectrum array containing theoretical CMB spectra.
        Expected ordering: [T, E, B].
    expt : Expt
        Experiment object containing noise properties.
    cmask : ndarray, optional
        CMB mask
    kmask : ndarray, optional
        Fourier space mask. Default is None.

    Returns
    -------
    kappa : ndarray
        Reconstructed CMB lensing convergence map in real space.
    noise_2d : ndarray
        2D noise power spectrum of the reconstruction.

    Notes
    -----
    The reconstruction follows these steps:
    1. Computes the unnormalized quadratic estimator
    2. Calculates the normalization
    3. Applies the normalization to get the final convergence map
    4. Transforms the result back to real space

    References
    ----------
    .. [1] Hu & Okamoto 2002, ApJ 574, 566
    """
    import symlens
    from symlens import utils as su

    kmap1 = enmap.fft(map1, normalize='phys')
    kmap2 = enmap.fft(map2, normalize='phys')
    modlmap = map1.modlmap()

    ell = np.arange(ps_lcmb.shape[-1])
    cltt = ps_lcmb[0, 0]  # phi, T, E, B ordering

    ucltt = su.interp(ell, cltt)(modlmap)  # lensed cl
    tcltt = ucltt + expt.get_nl(modlmap)
    feed_dict = {
        'X': kmap1,
        'Y': kmap2,
        'uC_T_T': ucltt,
        'tC_T_T': tcltt
    }
    kmask = (modlmap >= rlmin) * (modlmap <= rlmax)

    # unnormalized lensing map in fourier space
    ukappa_k = symlens.unnormalized_quadratic_estimator(shape,
                                                        wcs,
                                                        feed_dict,
                                                        "hu_ok",
                                                        "TT",
                                                        xmask=kmask,
                                                        ymask=kmask)

    # normaliztion
    norm_k = symlens.A_l(shape,
                         wcs,
                         feed_dict,
                         "hu_ok",
                         "TT",
                         xmask=kmask,
                         ymask=kmask,
                         kmask=kmask)

    # noise
    noise_2d = symlens.N_l_from_A_l_optimal(shape, wcs, norm_k)

    # normalized Fourier space CMB lensing convergence map
    kappa_k = norm_k * ukappa_k

    # real space CMB lensing convergence map
    kappa = enmap.ifft(kappa_k, normalize='phys').real

    return kappa, (noise_2d,)

def get_cl(map1, map2, ellmin, ellmax, delta_ell, taper=None, taper_order=None):
    import symlens.utils as su

    modlmap = map1.modlmap()
    bin_edges = np.arange(ellmin, ellmax, delta_ell)
    binner = su.bin2D(modlmap, bin_edges)

    kmap1 = enmap.fft(map1, normalize='phys')
    kmap2 = enmap.fft(map2, normalize='phys')

    if taper is None: w = 1
    else: w  = np.mean(taper**taper_order)

    p2d = (kmap1 * kmap2.conj()).real / w
    centers, p1d = binner.bin(p2d)

    return centers, p1d

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    shape, wcs = enmap.geometry(pos=(0,0), res=0.5*arcmin, shape=(500, 500))
    l = enmap.modlmap(shape, wcs)
    ps_lensinput = powspec.read_camb_full_lens("data/cosmo2017_10K_acc3_lenspotentialCls.dat")
    ps_lcmb = powspec.read_spectrum("data/cosmo2017_10K_acc3_lensedCls.dat")
    expt = Expt("SO", 1.4, 6)
    nl = expt.get_nl(np.arange(ps_lcmb.shape[-1])).reshape(1, 1, -1)

    # test sim_lens_map_flat
    m, (phi_map, _) = sim_lens_map_flat((1,)+shape, wcs, ps_lensinput)
    noise_map = enmap.rand_map(shape, wcs, nl)
    m = m[0] # + noise_map

    # test reconstruction
    kappa_recon, _ = recon_2d_symlens(shape, wcs, m, m, ps_lcmb, expt=expt, rlmin=1000, rlmax=3000)
    kappa_in = enmap.ifft(enmap.fft(phi_map, normalize='phys')*l*(l+1)/2, normalize="phys").real

    # test power spectrum
    l, inkappa_x_outkappa = get_cl(kappa_recon, kappa_in, 1000, 3000, 50)
    l, inkappa_x_inkappa = get_cl(kappa_in, kappa_in, 1000, 3000, 50)
    plt.plot(l, np.abs(inkappa_x_outkappa), label='recon x input')
    plt.plot(l, np.abs(inkappa_x_inkappa), label='input x input')
    plt.legend()
    plt.savefig("recon_2d_test.png")