"""
CMB Mapmaking Simulation and Analysis

This script simulates CMB observations with realistic noise properties and
implements different mapmaking techniques to reconstruct the sky signal.

Key components:
1. Sky simulation with CMB, point sources, and SZ clusters
2. Time-Ordered Data (TOD) generation with 1/f noise
3. Map reconstruction using:
   - Binned averaging (simple coadding)
   - Maximum-likelihood estimation with noise modeling
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pixell import enmap
import cmb_utils
from pixell import colorize
colorize.register_color("planck")

# Set random seed for reproducibility
np.random.seed(42)

def simulate_sky_map(shape, wcs, cmb_params, 
                     ps_poisson_params=None, 
                     ps_exp_params=None, 
                     sz_params=None, 
                     beam_fwhm=None):
    """
    Generate a simulated sky map containing CMB, point sources, and SZ clusters.

    Returns:
        tuple: (sky_map, convolved_map) containing:
            - sky_map: Original simulated sky map (μK)
            - convolved_map: Beam-convolved sky map (μK)
    """
    gen = cmb_utils.CMBMapGenerator(shape, wcs)
    cmb_map = gen.make_cmb_map(**cmb_params)

    ps_map = enmap.zeros(shape, wcs)
    if ps_poisson_params is not None:
        ps_map += gen.add_point_sources(**ps_poisson_params, distribution='poisson')
    if ps_exp_params is not None:
        ps_map += gen.add_point_sources(**ps_exp_params, distribution='exponential')

    sz_map = enmap.zeros(shape, wcs)
    if sz_params is not None:
        sz_map, _ = gen.add_sz_clusters(**sz_params)

    combined_map = cmb_map + ps_map + sz_map
    if beam_fwhm is not None:
        convolved_map = cmb_utils.MapProcessor.convolve_with_beam(combined_map, beam_fwhm)
    else:
        convolved_map = combined_map
    return combined_map, convolved_map



#%%
@dataclass
class TOD:
    timestamps: np.ndarray
    data: np.ndarray
    az: np.ndarray | None = None
    el: np.ndarray | None = None

    def __post_init__(self):
        assert len(self.timestamps) == len(self.data)

    @property
    def nsamps(self):
        return len(self.timestamps)
    
    @property
    def dt(self):
        return np.mean(np.diff(self.timestamps))

    @property
    def srate(self):
        return 1 / self.dt


class TODGenerator:
    """Generates TOD with realistic noise using PointingMatrix and NoiseModel."""
    def __init__(self, sky_map, dt):
        self.sky_map = sky_map
        self.dt = dt
        # as we set up a fake scan row by row, nsamples
        # will be the number of pixels
        self.nsamps = np.prod(self.sky_map.shape)
        
    def generate_tod(self, pointing_model, noise_model):
        """Generate TOD with associated pointing and noise model"""
        signal = pointing_model.apply(self.sky_map)
        
        noise_fft = np.fft.fft(np.random.randn(self.nsamps)) \
                  * np.sqrt(noise_model.noise_spec)
        noise = np.fft.ifft(noise_fft).real

        timestamps = np.arange(0, self.nsamps * self.dt, self.dt) 
        return TOD(timestamps, signal+noise)


class NoiseModel:
    """Models 1/f + white noise and applies inverse noise weighting."""
    def __init__(self, nsamps, dt, fknee=0.1, alpha=3, sigma=40):
        """
        Args:
            fknee: Knee frequency (Hz)
            alpha: 1/f spectral index
            sigma: White noise level (μK·s^0.5)
        """
        self.nsamps = nsamps
        self.dt = dt
        self.fknee = fknee
        self.alpha = alpha
        self.sigma = sigma
        self.noise_spec = self.get_noise_spec()

    def get_noise_spec(self):
        """Generate noise power spectrum"""
        freq = np.abs(np.fft.fftfreq(self.nsamps, self.dt))
        return (1 + (np.maximum(freq, freq[1])/self.fknee)**-self.alpha) * self.sigma**2
    
    def apply_inverse(self, data):
        """Apply inverse noise weighting N^-1 to TOD"""
        ftod = np.fft.fft(data)
        ftod /= self.noise_spec
        return np.fft.ifft(ftod).real


class PointingMatrix:
    """Handles telescope pointing patterns and matrix operations."""
    def __init__(self, shape, wcs, direction=0):
        """
        Args:
            sky_map_shape: Shape of the sky map (ny, nx)
            direction: 0 for horizontal scan, 1 for vertical scan
        """
        self.shape = shape
        self.wcs = wcs
        self.direction = direction
        self.pointing = self._generate_pointing(direction)
        
    def _generate_pointing(self, direction):
        """Generate pointing coordinates (2, nsamples)"""
        pixmap = np.mgrid[:self.shape[0], :self.shape[1]]
        if direction == 0:  # Horizontal
            pixmap[1, 1::2, :] = pixmap[1, 1::2, ::-1]  # Reverse every other row
        else:  # Vertical
            pixmap = np.roll(pixmap, 1, axis=0)
        return pixmap.reshape(2, -1)
    
    def apply(self, map_data):
        """Apply pointing matrix P: map -> TOD"""
        return map_data[self.pointing[0].astype(int), self.pointing[1].astype(int)]
    
    def apply_transpose(self, tod):
        """Apply transpose pointing matrix P^T: TOD -> map"""
        point_flat = np.ravel_multi_index(
            self.pointing.astype(int), 
            self.shape
        )
        return enmap.ndmap(
            np.bincount(point_flat, tod, minlength=np.prod(self.shape)).reshape(self.shape), self.wcs
        )


class Mapmaker:
    """Base class for mapmaking algorithms"""
    def __init__(self, shape, wcs):
        self.shape = shape
        self.wcs = wcs
        self.reset()

    def reset(self):
        """Reset accumulated data"""
        self.hits = enmap.zeros(self.shape, self.wcs)
        self.rhs = enmap.zeros(self.shape, self.wcs)

    def add_data(self, tod, pointing):
        """Add TOD data to map accumulation"""
        raise NotImplementedError

    def get_map(self):
        """Return reconstructed map"""
        raise NotImplementedError


class BinnedMapmaker(Mapmaker):
    """Simple binned averaging mapmaker"""
    def add_data(self, tod, pointing):
        pointing = np.round(pointing.pointing).astype(int)
        np.add.at(self.hits, (pointing[0], pointing[1]), 1)
        np.add.at(self.rhs, (pointing[0], pointing[1]), tod.data)

    def get_map(self):
        """Return binned average map"""
        return np.divide(self.rhs, self.hits, where=self.hits>0, out=np.full_like(self.rhs, np.nan))


class MaximumLikelihoodMapmaker(Mapmaker):
    def __init__(self, shape, wcs):
        super().__init__(shape, wcs)
        self.datasets = []

    def add_data(self, tod, pointing, noise_model):
        """Add TOD with pointing and noise model"""
        self.datasets.append({
            'tod': tod,
            'pointing': pointing,
            'noise_model': noise_model
        })
    
    def _A_operator(self, x):
        x_map = x.reshape(self.shape)
        result = np.zeros_like(x_map)
        for data in self.datasets:
            # P x
            tod = data['pointing'].apply(x_map)
            # N⁻¹ P x
            weighted_tod = data['noise_model'].apply_inverse(tod)
            # P^T N⁻¹ P x
            result += data['pointing'].apply_transpose(weighted_tod)
        return result.ravel()
    
    def _compute_rhs(self):
        rhs = enmap.zeros(self.shape, self.wcs)
        for data in self.datasets:
            # N⁻¹ d
            weighted_tod = data['noise_model'].apply_inverse(data['tod'].data)
            # P^T N⁻¹ d
            rhs += data['pointing'].apply_transpose(weighted_tod)
        return rhs.ravel()

    def get_map(self, niter=50, verbose=True):
        """Solve using conjugate gradient descent"""
        from scipy.sparse.linalg import LinearOperator, cg

        # Set up linear operator
        A = LinearOperator(
            (np.prod(self.shape), np.prod(self.shape)),
            matvec=self._A_operator,
            dtype=np.float64
        )

        # Compute right-hand side
        b = self._compute_rhs()

        # Initial guess
        x0 = np.zeros_like(b)

        # Solve using conjugate gradient
        def callback(xk): 
            residual = np.linalg.norm(A @ xk - b) #Calculate residual
            print(f"{residual=}")
        x, info = cg(A, b, x0=x0, maxiter=niter, tol=1e-6, callback=callback)

        if verbose:
            print(f"Conjugate gradient converged: {info == 0}")
            if info > 0:
                print(f"Convergence not achieved in {niter} iterations")
            elif info < 0:
                print("Illegal input or breakdown")

        return enmap.ndmap(x.reshape(self.shape), self.wcs)

#%%
if __name__ == "__main__":
    shape, wcs = enmap.geometry(pos=(0,0),res=np.deg2rad(0.5/60), shape=(512,512))
    map_sim_params = {
        'shape': shape,
        'wcs': wcs,
        'cmb_params': {
            'lmax': 5000
        },
        'ps_poisson_params': {
            'n_sources': 5000,
            'amplitude': 200,
        },
        'ps_exp_params': {
            'n_sources': 50,
            'amplitude': 100,
        },
        'sz_params': {
            'n_clusters': 500,
            'mean_amplitude': 10,
            'beta': 0.86,
            'theta_core': 1.0 
        },
        'beam_fwhm': 5
    }
    sky_map, sky_map_convolved = simulate_sky_map(**map_sim_params)

    #%%
    from pixell import enplot 
    enplot.pshow(sky_map)

    #%%
    cfg = {
        'dt': 1/200,
        'nsamps': np.prod(sky_map.shape),
        'fknee': 0.1,
        'alpha': 3,
        'sigma': 40.0
    }
    noise = NoiseModel(nsamps=cfg['nsamps'], dt=cfg['dt'])
    tod_gen = TODGenerator(sky_map_convolved, dt=cfg['dt'])
    pointing_lr = PointingMatrix(sky_map.shape, sky_map.wcs, direction=0)
    pointing_ud = PointingMatrix(sky_map.shape, sky_map.wcs, direction=1)
    tod_lr = tod_gen.generate_tod(pointing_model=pointing_lr, noise_model=noise)
    tod_ud = tod_gen.generate_tod(pointing_model=pointing_ud, noise_model=noise)

    mapmaker = MaximumLikelihoodMapmaker(sky_map.shape, sky_map.wcs)
    mapmaker.add_data(tod_lr, pointing_lr, noise)
    mapmaker.add_data(tod_ud, pointing_ud, noise)
    omap = mapmaker.get_map()

    enplot.pshow(sky_map)
    enplot.pshow(omap)

    #%%
    mapmaker = BinnedMapmaker(sky_map.shape, sky_map.wcs)
    mapmaker.add_data(tod_lr, pointing_lr)
    mapmaker.add_data(tod_ud, pointing_ud)
    omap = mapmaker.get_map()

    enplot.pshow(sky_map)
    enplot.pshow(omap)

    #%%
    cr = np.zeros_like(tod_lr.data)
    cr[int(1e5)] = 100000
    tod_lr.data += cr
    # plt.plot((cr)[::10])
    plt.plot((tod_lr.data)[::10])

    #%%
    mapmaker = BinnedMapmaker(sky_map.shape, sky_map.wcs)
    mapmaker.add_data(tod_lr, pointing_lr)
    mapmaker.add_data(tod_ud, pointing_ud)
    omap = mapmaker.get_map()    

    enplot.pshow(sky_map)
    enplot.pshow(omap)

    #%%
    mapmaker = MaximumLikelihoodMapmaker(sky_map.shape, sky_map.wcs)
    mapmaker.add_data(tod_lr, pointing_lr, noise)
    mapmaker.add_data(tod_ud, pointing_ud, noise)
    omap = mapmaker.get_map()

    enplot.pshow(sky_map, grid=False)
    enplot.pshow(omap, grid=False)
