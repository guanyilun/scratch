"""
CMB Map Simulation and Analysis Tools
Provides utilities for generating and analyzing Cosmic Microwave Background (CMB) maps.
"""

# %%
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

# Import pixell modules
from pixell import enmap, curvedsky
from orphics import cosmology

deg = np.deg2rad(1)
arcmin = np.deg2rad(1) / 60

theory = cosmology.default_theory()


class CMBMapGenerator:
    """Generates CMB temperature maps and related components."""
    
    def __init__(self, shape, wcs=None):
        """
        Initialize CMB map generator.
        
        Args:
            N: Number of pixels in each dimension
            pix_size: Pixel size in arcminutes
        """
        self.shape = shape
        if wcs is None:
            self.wcs = enmap.geometry(shape=self.shape, 
                                      box=np.array([[-0.5, -0.5], [0.5, 0.5]])*deg, 
                                      proj="car", res=1/60.0)
        else:
            self.wcs = wcs
        # self.map = enmap.zeros(self.shape, wcs=self.wcs)
        
    def make_cmb_map(self, lmax=3000) -> enmap.enmap:
        """
        Generate a CMB temperature map using pixell.
        
        Args:
            ell: Multipole moments array
            dl_tt: Temperature power spectrum
            
        Returns:
            2D enmap containing CMB temperature map
        """
        ell = np.arange(lmax+1)
        cltt = theory.uCl('TT', ell)
        alm = curvedsky.rand_alm(cltt, lmax=lmax)
        t_map = curvedsky.alm2map(alm, enmap.zeros(self.shape, self.wcs))
        return t_map
    
    def add_point_sources(self, n_sources: int, amplitude: float, 
                         distribution: str = 'poisson') -> enmap.enmap:
        """
        Add point sources to map using specified distribution.
        
        Args:
            n_sources: Number of sources to add
            amplitude: Source amplitude
            distribution: Either 'poisson' or 'exponential'
            
        Returns:
            Map with point sources
        """
        ps_map = enmap.zeros(self.shape, self.wcs)
        Ny, Nx = self.shape

        x = np.random.randint(0, Nx, n_sources)
        y = np.random.randint(0, Ny, n_sources)

        if distribution == 'poisson':
            amps = np.random.poisson(amplitude, n_sources)
        elif distribution == 'exponential':
            amps = np.random.exponential(amplitude, n_sources)
        else:
            raise ValueError("Invalid distribution type")
        np.add.at(ps_map, (x, y), amps)
        return ps_map
    
    def add_sz_clusters(self, n_clusters: int, mean_amplitude: float,
                       beta: float, theta_core: float) -> Tuple[enmap.enmap, npt.NDArray]:
        """
        Add Sunyaev-Zeldovich clusters to map.
        
        Args:
            n_clusters: Number of clusters
            mean_amplitude: Mean cluster amplitude
            beta: Beta model parameter
            theta_core: Core radius
            
        Returns:
            Tuple of (SZ map, cluster catalog)
        """
        sz_map = enmap.zeros(self.shape, self.wcs)
        Ny, Nx = self.shape

        x = np.random.randint(0, Nx, n_clusters)
        y = np.random.randint(0, Ny, n_clusters)
        amps = -np.random.exponential(mean_amplitude, n_clusters)
        sz_cat = np.array([x, y, amps])
        np.add.at(sz_map, (x, y), amps)

        # Apply beta profile
        beta_profile = self._make_beta_profile(beta, theta_core)
        sz_map = enmap.ifft(enmap.fft(sz_map) * enmap.fft(beta_profile)).real

        return sz_map, sz_cat
    
    def _make_beta_profile(self, beta: float, theta_core: float) -> enmap.enmap:
        """Generate beta profile for SZ clusters."""
        modrmap = enmap.fftshift(enmap.modrmap(self.shape, self.wcs))
        beta_profile = (1 + (modrmap/theta_core)**2)**((1 - 3.0*beta)/2.0)
        return beta_profile


def gauss_beam(ell, fwhm):
    """Generate Gaussian beam."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * (ell * (ell + 1) * sigma**2))


class MapProcessor:
    """Processes and analyzes CMB maps."""
    
    @staticmethod
    def convolve_with_beam(map_data: enmap.enmap, beam_fwhm: float) -> enmap.enmap:
        """
        Convolve map with Gaussian beam using pixell.
        
        Args:
            map_data: Input map
            beam_fwhm: Beam FWHM in arcminutes
            
        Returns:
            Convolved map
        """
        l = map_data.modlmap()
        beam = gauss_beam(l, beam_fwhm * arcmin)
        ft_map = enmap.fft(map_data)
        ft_convolved = ft_map * beam
        return enmap.ifft(ft_convolved).real
    
    @staticmethod
    def calculate_power_spectrum(map1: enmap.enmap, map2: Optional[enmap.enmap] = None,
                                delta_ell: float = 100, lmax: float = 5000) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Calculate power spectrum of map(s) using pixell.
        
        Args:
            map1: First map
            map2: Optional second map for cross-spectrum
            delta_ell: Bin width
            ell_max: Maximum multipole
            
        Returns:
            Tuple of (ell array, power spectrum)
        """
        if map2 is None:
            map2 = map1
            
        modlmap = map1.modlmap()
        bin_edges = np.arange(0, lmax+delta_ell, delta_ell)
        bc = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        fmap1 = enmap.fft(map1, normalize='phys')
        fmap2 = enmap.fft(map2, normalize='phys')
        
        idx = np.digitize(modlmap, bin_edges)
        binned = np.bincount(idx.ravel(), weights=(fmap1 * np.conj(fmap2)).ravel().real)[1:-1]
        hits = np.bincount(idx.ravel())[1:-1]
        powspec = binned / hits

        return bc, powspec


def plot_cmb_map(imap: enmap.enmap, cmap: str = 'RdBu_r', title: str = '') -> None:
    """
    Plot CMB map with proper formatting using pixell.
    
    Args:
        map_data: Map to plot
        cmap: Colormap to use
        title: Optional title for plot
    """
    # from pixell import enplot
    # enplot.pshow(imap)
    fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(projection=imap.wcs)
    # ax = fig.add_subplot(1, 1, 1, projection=imap.wcs)
    ax = plt.gca()
    im = ax.imshow(imap, cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(im).set_label('Temperature [K]')
    ax.set_xlabel('Angle [°]')
    ax.set_ylabel('Angle [°]')
    if title:
        ax.set_title(title)
    plt.show()

# %%
if __name__ == '__main__':
    shape, wcs = enmap.geometry(shape=(480, 480), res=1*arcmin, proj='car', pos=(0, 0))
    cmb_gen = CMBMapGenerator(shape, wcs)
    cmb_map = cmb_gen.make_cmb_map()
    plot_cmb_map(cmb_map, title='CMB Temperature Map')

    # %%
    ps_map = cmb_gen.add_point_sources(100, 1.5)
    plot_cmb_map(ps_map, title='CMB Temperature Map with Point Sources')

    # %%
    sz_map, sz_cat = cmb_gen.add_sz_clusters(10, 1e-5, 0.8, 5*arcmin)
    plot_cmb_map(sz_map, title='CMB Temperature Map with Point Sources')

    # %%
    beam_fwhm = 1.0
    convolved_map = MapProcessor.convolve_with_beam(cmb_map, beam_fwhm)
    plot_cmb_map(convolved_map, title='CMB Temperature Map with Beam')

    # %%
    ell, cl = MapProcessor.calculate_power_spectrum(cmb_map)

    plt.figure(figsize=(6, 4))
    plt.plot(ell, cl * (ell * (ell + 1)) / (2 * np.pi), label='CMB')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}$')
    plt.xscale('log')
    plt.yscale('log')