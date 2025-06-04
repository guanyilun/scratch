#%%
import numpy as np
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple
from numpy.typing import NDArray
from scipy.linalg import eigh

from so3g.proj import coords, quat
from pixell import enmap


# useful units
deg = np.deg2rad(1)
arcmin = deg / 60
s = 1 # second
Hz = 1/s


@dataclass
class FocalPlane:
    x: np.ndarray  # in radian
    y: np.ndarray  # in radian

    @classmethod
    def from_radius(cls, radius: float = 0.5*deg, nrows: int = 30):
        """
        Generates a circular focal plane with a given radius and number of rows.

        The focal plane is represented as a set of x and y coordinates
        in radians, arranged in a circular pattern.
        The radius is the distance from the center to the edge of the circle.

        Args:
            radius (float): The radius of the circular focal plane in radians.
            nrows (int): The number of rows to generate. This determines the
                         density of detectors in the focal plane.
        Returns:
            FocalPlane: A new instance of the FocalPlane class with x and y
                        coordinates of the detectors. 
        
        """
        X, Y = np.meshgrid(np.linspace(-1,1, nrows), np.linspace(-1, 1, nrows))
        m = (X**2 + Y**2) < 1
        return cls(X[m]*radius, Y[m]*radius)

    def get_circular_cover(self, n_dummy: int = 50) -> "FocalPlane":
        """
        Generates a circular arrangement of dummy detectors that encloses
        the existing detectors in the focal plane.

        The circle is centered at the mean position of the existing detectors.
        Its radius is determined by the maximum distance from this center
        to any existing detector.

        Args:
            n_dummy (int): The number of dummy detectors for the circular cover.

        Returns:
            FocalPlane: A new FocalPlane instance with the coordinates
                        of the dummy detectors.
        """
        if self.x.size == 0:
            # No detectors to cover, so return an empty cover.
            return FocalPlane(np.array([]), np.array([]))

        # Calculate the mean position (center) of the existing detectors.
        center_x = np.mean(self.x)
        center_y = np.mean(self.y)

        # Calculate the distances of each detector from this calculated center.
        relative_x = self.x - center_x
        relative_y = self.y - center_y
        distances_from_center = np.sqrt(relative_x**2 + relative_y**2)

        # The radius of the cover is the maximum of these distances.
        # If there's only one point, this radius will be 0.
        cover_radius = np.max(distances_from_center) if distances_from_center.size > 0 else 0.0

        # Generate angles for the dummy detectors, evenly spaced.
        angles = np.linspace(0, 2 * np.pi, n_dummy, endpoint=False)

        # Calculate coordinates of dummy detectors on the circle around the calculated center.
        dummy_x = center_x + cover_radius * np.cos(angles)
        dummy_y = center_y + cover_radius * np.sin(angles)

        return FocalPlane(dummy_x, dummy_y)

    @property
    def n_dets(self):
        return len(self.x)


@dataclass
class Scan:
    t: NDArray   # time vector in seconds. shape: (nsamps,)
    az: NDArray  # azimuth vector in radians. shape: (nsamps,)
    el: NDArray  # elevation vector in radians. shape: (nsamps,)
    srate: float

    def get_q_bore(self):
        q_bore = coords.CelestialSightLine.az_el(self.t, self.az, self.el, weather='typical').Q
        return q_bore

    @property
    def nsamps(self):
        return len(self.t)


def generate_ces_scan(
    t0: float,
    az: float,                 # Starting azimuth of the scan range in radians
    el: float,                 # Constant elevation for the scan in radians
    az_throw: float,           # Width of the azimuth scan in radians
    v_az: float = 1 * deg / s, # Azimuth scanning speed in radians per second. Default: 1 deg/s.
    srate: float = 200 * Hz,   # Sample interval in seconds (1 / sample_rate_hz). Default: 0.0025s.
    duration: float = 300.0    # Total duration for the scan in seconds. Default: 10 * 60s.
) -> Scan:
    """
    Generates a Constant Elevation Scan (CES) pattern.

    The scan sweeps back and forth in azimuth at a constant elevation.
    The phase of the scan is continuous with respect to t0.

    Args:
        t0: Absolute start time of the first sample in the returned time vector
            (in seconds, e.g., a Unix timestamp).
        az: The minimum azimuth of the scan range (in radians). This is the
            starting edge of the az_throw.
        el: The constant elevation for the scan (in radians).
        az_throw: The width of the azimuth scan (in radians). The scan will be
                  between 'az' and 'az + az_throw'. Must be non-negative.
        v_az: The scanning speed in azimuth (in radians per second).
              Must be positive if az_throw > 0.
        srate: The time interval between consecutive samples (in Hz).
        duration: The total duration for which to generate the scan pattern
                  (in seconds). Must be positive.

    Returns:
        Scan
    """
    # Input validation
    if az_throw < 0:
        raise ValueError("Azimuth throw (az_throw) must be non-negative.")
    if srate <= 0:
        raise ValueError("Sample interval (srate) must be positive.")
    if duration <= 0:
        # Return empty arrays for non-positive duration, or if no samples generated
        return Scan(np.array([]), np.array([]), np.array([]), srate=0)
    if az_throw > 0 and v_az <= 0:
        raise ValueError("Azimuth speed (v_az) must be positive if az_throw is positive.")

    # Time vector generation
    # np.arange's stop is exclusive.
    t_vector = np.arange(0, duration, 1/srate) + t0
    if t_vector.size == 0:
        return Scan(np.array([]), np.array([]), np.array([]))

    # Azimuth scan parameters
    az_min_rad = az
    az_max_rad = az + az_throw

    # Handle staring scan (az_throw is 0)
    if az_throw == 0:
        az_vector = np.full_like(t_vector, az_min_rad)
        el_vector = np.full_like(t_vector, el)
        return Scan(t=t_vector, az=az_vector, el=el_vector, srate=0)

    # Time for a single sweep (e.g., from az_min_rad to az_max_rad)
    # v_az is positive here due to checks above.
    time_per_sweep = az_throw / v_az

    # Calculate phase based on the absolute time `t_vector`.
    # `phase` goes from 0 towards 2.0 over one full back-and-forth cycle.
    # Using t_vector directly ensures phase continuity if t0 is part of a larger timeline.
    phase = (t_vector / time_per_sweep) % 2.0

    # Correct the phase for the return sweep (when phase > 1.0)
    # to make it a triangular wave between 0.0 and 1.0.
    # This uses the walrus operator (:=) as in your example, for conciseness.
    mask = phase > 1.0
    phase[mask] = 2.0 - phase[mask]

    # Linearly interpolate azimuth based on the corrected phase
    # When phase = 0 (start or end of a full cycle), az_vector = az_min_rad.
    # When phase = 1 (mid-point of a full cycle, end of forward sweep), az_vector = az_max_rad.
    az_vector = az_min_rad * (1.0 - phase) + az_max_rad * phase

    # Elevation is constant
    el_vector = np.full_like(t_vector, el)

    return Scan(t=t_vector, az=az_vector, el=el_vector, srate=srate)


def add_drift(scan: Scan, drift_speed: float = 1 * deg / s, direction=None, seed=None) -> Scan:
    """
    Adds a drift to the scan pattern. Can be used to implement a drift planet scan or used to
    simulate wind drift in the scan.

    Args:
        scan (Scan): The scan object containing time, azimuth, and elevation vectors.
        drift_speed (float): The speed of the drift in radians per second.
        direction (float): The direction of the drift in radians.
        seed (int): Seed for random number generation.

    Returns:
        Scan: A new Scan object with the drift added.
    """
    if seed is not None:
        np.random.seed(seed)

    if direction is None:
        direction = np.random.uniform(0, 2 * np.pi)

    drift_az = drift_speed * np.cos(direction)
    drift_el = drift_speed * np.sin(direction)

    return Scan(
        t=scan.t,
        az=scan.az + drift_az,
        el=scan.el + drift_el
    )


def build_pointing_model(fplane: FocalPlane, scan: Scan):
    """
    Returns: A 3D array (3, n_dets, nsamps) containing RA, Dec, and PA
    """
    q_fp = quat.rotation_xieta(fplane.x, fplane.y)
    q_bore = scan.get_q_bore()

    sky_coords = np.zeros((3, fplane.n_dets, scan.nsamps))
    for i in range(len(fplane.x)):
        sky_coords[0, i, :], sky_coords[1, i, :], sky_coords[2, i, :] = quat.decompose_lonlat(q_bore * q_fp[i])
    return sky_coords


class SkyModel(Protocol):
    def apply(self, sky_map: enmap.enmap) -> enmap.enmap:
        ...

@dataclass
class DummySkyModel(SkyModel):
    """dummy sky model that generates a random map for testing."""
    def apply(self, sky_map: enmap.enmap) -> enmap.enmap:
        return enmap.rand_gauss(sky_map.shape, sky_map.wcs)

@dataclass
class FrozenAtmosphere(SkyModel):
    lknee: float
    alpha: float
    nlev: float
    seed: int | None = None
    lmax: int = 3000

    def apply(self, sky_map: enmap.enmap) -> enmap.enmap:
        shape, wcs = sky_map.shape, sky_map.wcs
        if self.seed is not None:
            np.random.seed(self.seed)

        # modlmap = enmap.modlmap(shape, wcs)
        ell = np.arange(0, self.lmax + 1)
        psd = (ell / self.lknee)**self.alpha * self.nlev**2
        psd[ell == 0] = self.nlev**2
        atm_map = enmap.rand_map(shape, wcs, psd[None, None], scalar=True, spin=[0])
        sky_map += atm_map

        return sky_map


@dataclass
class TOD:
    scan: Scan
    fplane: FocalPlane
    data: NDArray  # shape: (ndets, nsamps)


class TODModel(Protocol):
    def apply(self, tod: TOD) -> TOD:
        """Paints the time-ordered data (TOD) with a model."""
        ...


@dataclass
class CorrelatedOOFModel(TODModel):
    """Adds correlated 1/f noise between detectors."""
    fknee: float           # Knee frequency in Hz
    alpha: float           # Spectral index
    sigma_target: NDArray  # Target std dev per detector. shape: (ndets,)
    n_modes: int = 3       # Number of modes to use for low-rank approximation
    fractions: list[float] = None
    seed: int | None = None

    def apply(self, tod: TOD) -> TOD:
        ndets, nsamps = tod.data.shape

        if self.sigma_target.shape[0] != ndets:
             raise ValueError(f"sigma_target shape {self.sigma_target.shape} does not match TOD detectors {ndets}")

        noise = self._generate_correlated_noise(ndets, nsamps, tod.scan.srate)
        tod.data += noise
        return tod

    def _generate_correlated_noise(self, ndets: int, nsamps: int, srate: float) -> NDArray:
        """Generates noise using low-rank approximation of the covariance matrix."""
        rng = np.random.default_rng(self.seed)

        R_target = generate_low_rank_correlation(
            n_dets=ndets,
            n_modes=self.n_modes,
            fractions=self.fractions if self.fractions is not None else [1.0 / self.n_modes] * self.n_modes
        )

        # Compute latent covariance matrix
        C_latent = np.outer(self.sigma_target, self.sigma_target) * R_target

        # Symmetrize and regularize
        C_latent = (C_latent + C_latent.T) / 2
        C_latent += 1e-12 * np.eye(ndets) * np.median(np.diag(C_latent))

        # Eigendecomposition
        eigvals, eigvecs = eigh(C_latent)
        eigvals = np.clip(eigvals, 0, None)  # Ensure non-negative

        # Determine effective rank (modes with significant eigenvalues)
        rank = np.sum(eigvals > 1e-6 * np.max(eigvals))  # Adjust threshold as needed
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices[:rank]]
        eigvecs = eigvecs[:, sorted_indices[:rank]]

        # Generate r independent 1/f noise streams (vectorized)
        f = np.fft.rfftfreq(nsamps, d=1.0/srate)
        psd = (1 + (np.abs(f)/self.fknee)**self.alpha)
        psd[f == 0] = 1.0

        # Random phases for r modes (instead of ndets)
        phases = rng.uniform(0, 2*np.pi, size=(rank, len(f)))
        phases[:, 0] = 0.0  # DC phase
        if nsamps % 2 == 0 and len(f) > 1:
            phases[:, -1] = rng.choice([0, np.pi], size=rank)

        # Generate r noise streams
        A = np.sqrt(psd)[None, :]  # Shape (1, nfreq)
        complex_spectrum = A * np.exp(1j * phases)
        x_low_rank = np.fft.irfft(complex_spectrum, n=nsamps, axis=1)  # Shape (rank, nsamps)

        noise_latent = (eigvecs * np.sqrt(eigvals)[None, :]) @ x_low_rank  # Shape (ndets, nsamps)

        return noise_latent

def rand_oof(nsamps, srate, fknee, alpha, nlev, seed=None):
    """
    Generates a random time-domain signal with a 1/f^alpha power spectrum.

    Args:
        nsamps (int): Number of samples in the output time-domain signal.
        srate (float): Sampling rate in Hz.
        fknee (float): Knee frequency in Hz. Below this, spectrum is flatter.
        alpha (float): Exponent of the 1/f noise (e.g., -1 for pink noise, -2 for brown/red noise).
                       Note: The formula uses +alpha, so if you want 1/f, alpha should be negative.
                       Or, if you define noise_power ~ (f/fknee)^alpha, then alpha is typically negative.
                       Your current formula noise_spec = nlev**2 * (1 + np.abs(f/fknee) ** alpha)
                       implies alpha > 0 for a spectrum that falls off, or if it's meant to be
                       (f/fknee)^(-alpha), then provide a positive alpha. Let's assume your
                       formula is as intended and alpha is the power law index as written.
        nlev (float): Noise level (amplitude scaling factor).
        seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        numpy.ndarray: A 1D array of length nsamps representing the noise signal.
    """
    if seed is not None:
        np.random.seed(seed)

    f = np.fft.rfftfreq(nsamps, d=1.0/srate)
    
    # Calculate Power Spectral Density (PSD)
    psd = nlev**2 * (1 + (np.abs(f) / fknee)**alpha)
    psd[f==0] = nlev**2  # Ensure DC component is real
    
    # Amplitudes from PSD
    A = np.sqrt(psd)
    
    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, len(f))
    
    # Complex spectrum: A * exp(j*phi)
    complex_spectrum = A * np.exp(1j * phases)
    
    # Ensure DC and Nyquist components are real for irfft
    if f[0] == 0:
        complex_spectrum[0] = A[0] # Real DC component
    if nsamps % 2 == 0 and len(f) > 1: # Nyquist frequency
        complex_spectrum[-1] = A[-1] * np.exp(1j * np.random.choice([0, np.pi]))

    # Inverse FFT to get time-domain signal
    tod = np.fft.irfft(complex_spectrum, n=nsamps)
    
    return tod

def generate_low_rank_correlation(n_dets: int, n_modes: int, fractions: list[float]) -> np.ndarray:
    """
    Generates a low-rank correlation matrix with specified fractions of variance per mode.
    
    Args:
        n_dets: Number of detectors (dimension of the matrix).
        n_modes: Number of dominant modes.
        fractions: List of fractions of total variance for each mode (sum to 1).
    
    Returns:
        R: Correlation matrix (n_dets, n_dets).
    """
    assert abs(sum(fractions) - 1.0) < 1e-10, "Fractions must sum to 1"
    assert len(fractions) == n_modes, "Mismatch between n_modes and fractions length"
    
    # Generate random orthogonal eigenvectors
    U_full = np.random.randn(n_dets, n_dets)
    U_full, _ = np.linalg.qr(U_full)  # Full orthogonal matrix
    
    # Set up eigenvalues: first n_modes have the specified fractions
    eigenvals = np.zeros(n_dets)
    for i in range(n_modes):
        eigenvals[i] = fractions[i] * n_dets
    
    # Remaining eigenvalues are small but non-zero for numerical stability
    remaining_trace = n_dets - np.sum(eigenvals[:n_modes])
    if remaining_trace > 0:
        eigenvals[n_modes:] = remaining_trace / (n_dets - n_modes)
    
    # Construct correlation matrix
    R = U_full @ np.diag(eigenvals) @ U_full.T
    
    return R


def build_tod(
    fplane: FocalPlane,
    scan: Scan,
    sky_models: list[SkyModel],
    tod_models: list[TODModel],
    n_dummy=50,
) -> TOD:
    """
    Builds a time-ordered data (TOD) array based on the focal plane, scan pattern,
    and sky model.

    Args:
        fplane (FocalPlane): The focal plane object containing detector positions.
        scan (Scan): The scan object containing time, azimuth, and elevation vectors.
        sky_models (List[SkyModel]): A list of sky model objects to generate the sky signal.
        tod_models (List[TODModel]): A list of TOD model objects to generate TOD-specific signals (like noise).
        n_dummy (int): Number of dummy detectors for circular cover to determine map bounds.

    Returns:
        TOD: A TOD object containing the generated time-ordered data.
    """
    # dummy detectors to determine map bounds
    dummy_fplane = fplane.get_circular_cover(n_dummy=n_dummy)
    sky_coords = build_pointing_model(dummy_fplane, scan)
    ra_bounds = np.min(sky_coords[0, :, :]), np.max(sky_coords[0, :, :])
    dec_bounds = np.min(sky_coords[1, :, :]), np.max(sky_coords[1, :, :])
    print(f"RA bounds: {np.rad2deg(ra_bounds[0]):.2f} to {np.rad2deg(ra_bounds[1]):.2f} deg")
    print(f"Dec bounds: {np.rad2deg(dec_bounds[0]):.2f} to {np.rad2deg(dec_bounds[1]):.2f} deg")

    # build empty sky in the bounding box
    shape, wcs = enmap.geometry(pos=np.array([[dec_bounds[0], ra_bounds[-1]], [dec_bounds[-1], ra_bounds[0]]]), proj='car', res=0.5*arcmin)
    sky_map = enmap.zeros(shape, wcs)
    print(f"Sky map shape: {sky_map.shape}, wcs: {wcs}")

    # generate the sky model
    for sky_model in sky_models:
        sky_map = sky_model.apply(sky_map)

    if len(sky_models) > 0:
        # get the tod for each detector by reading the sky map at the detector positions
        # only do this if we have a sky model
        sky_coords = build_pointing_model(fplane, scan)
        tod = sky_map.at(pos=[sky_coords[1], sky_coords[0]])  # dec, ra order
    else:
        print("No sky models provided, initializing TOD with zeros.")
        tod = np.zeros((fplane.n_dets, scan.nsamps), dtype=np.float32)

    print(f"Sky map shape: {sky_map.shape}, tod shape: {tod.shape}")

    tod = TOD(scan=scan, data=tod, fplane=fplane)

    # apply TOD models (like noise sims)
    for tod_model in tod_models:
        tod = tod_model.apply(tod)
    return tod


def common_mode_analysis(tod: np.ndarray, 
                         normalize: bool = False,
                         remove_mean: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform common mode analysis on time-ordered data using SVD.
    
    Args:
        tod: Time-ordered data array of shape (n_dets, n_samps)
        normalize: Whether to normalize detector data by their standard deviation
        remove_mean: Whether to remove the mean from each detector's timestream
    
    Returns:
        calibration: Array of shape (n_dets,) - how each detector responds to common mode
        common_mode_timestream: Array of shape (n_samps,) - the common mode signal
        variance_fraction: Fraction of total variance explained by the common mode
    """
    n_dets, n_samps = tod.shape
    
    # Prepare the data
    tod_processed = tod.copy()
    
    # Remove mean from each detector if requested
    if remove_mean:
        tod_processed -= np.mean(tod_processed, axis=1, keepdims=True)
    
    # Normalize by standard deviation if requested
    detector_stds = None
    if normalize:
        detector_stds = np.std(tod_processed, axis=1, keepdims=True)
        # Avoid division by zero
        detector_stds = np.where(detector_stds == 0, 1, detector_stds)
        tod_processed /= detector_stds
    
    # Perform SVD
    # tod_processed = U @ S @ V.T
    # U: (n_dets, n_dets) - left singular vectors (detector patterns)
    # S: (min(n_dets, n_samps),) - singular values
    # V.T: (min(n_dets, n_samps), n_samps) - right singular vectors (time patterns)
    U, S, Vt = np.linalg.svd(tod_processed, full_matrices=False)
    
    # The first singular vector corresponds to the dominant common mode
    calibration_normalized = U[:, 0]  # Shape: (n_dets,)
    common_mode_timestream = S[0] * Vt[0, :]  # Shape: (n_samps,)
    
    # Calculate the fraction of variance explained by the common mode
    total_variance = np.sum(S**2)
    common_mode_variance = S[0]**2
    variance_fraction = common_mode_variance / total_variance
    
    # If we normalized the data, we need to scale back the calibration
    if normalize:
        calibration = calibration_normalized * detector_stds.flatten()
    else:
        calibration = calibration_normalized
    
    # Ensure calibration has a consistent sign convention
    # (e.g., make the mean positive)
    if np.mean(calibration) < 0:
        calibration = -calibration
        common_mode_timestream = -common_mode_timestream
    
    return calibration, common_mode_timestream, variance_fraction


def analyze_multiple_modes(data: np.ndarray, 
                           n_modes: int = 3,
                           normalize: bool = True,
                           remove_mean: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze multiple common modes using SVD.
    
    Args:
        tod: Time-ordered data array of shape (n_dets, n_samps)
        n_modes: Number of modes to analyze
        normalize: Whether to normalize detector data by their standard deviation
        remove_mean: Whether to remove the mean from each detector's timestream
    
    Returns:
        calibrations: Array of shape (n_dets, n_modes) - detector responses to each mode
        mode_timestreams: Array of shape (n_modes, n_samps) - the mode signals
        variance_fractions: Array of shape (n_modes,) - variance fraction for each mode
    """
    n_dets, n_samps = data.shape
    n_modes = min(n_modes, min(n_dets, n_samps))
    
    # Prepare the data (same as single mode analysis)
    data_ = data.copy()
    
    if remove_mean:
        data_ -= np.mean(data_, axis=1, keepdims=True)
    
    detector_stds = None
    if normalize:
        detector_stds = np.std(data_, axis=1, keepdims=True)
        detector_stds = np.where(detector_stds == 0, 1, detector_stds)
        data_ /= detector_stds
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(data_, full_matrices=False)
    
    # Extract the first n_modes
    calibrations_normalized = U[:, :n_modes]  # Shape: (n_dets, n_modes)
    mode_timestreams = S[:n_modes, np.newaxis] * Vt[:n_modes, :]  # Shape: (n_modes, n_samps)
    
    # Calculate variance fractions
    total_variance = np.sum(S**2)
    variance_fractions = S[:n_modes]**2 / total_variance
    
    # Scale back calibrations if we normalized
    if normalize:
        calibrations = calibrations_normalized * detector_stds
    else:
        calibrations = calibrations_normalized
    
    # Ensure consistent sign convention for each mode
    for i in range(n_modes):
        if np.mean(calibrations[:, i]) < 0:
            calibrations[:, i] = -calibrations[:, i]
            mode_timestreams[i, :] = -mode_timestreams[i, :]
    
    return calibrations, mode_timestreams, variance_fractions

def remove_common_mode(data: np.ndarray, 
                       calibration: Optional[np.ndarray] = None,
                       common_mode: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Remove the common mode from the TOD.
    
    Args:
        tod: Original time-ordered data array of shape (n_dets, n_samps)
        calibration: Detector calibrations from common_mode_analysis()
        common_mode: Common mode timestream from common_mode_analysis()
    
    Returns:
        tod_cleaned: TOD with common mode removed
    """
    if calibration is None or common_mode is None:
        calibration, common_mode, _ = common_mode_analysis(data)
    
    # Reconstruct and subtract the common mode
    common_mode_contribution = calibration[:, np.newaxis] * common_mode[np.newaxis, :]
    tod_cleaned = tod - common_mode_contribution
    
    return tod_cleaned


if __name__ == '__main__':
    # 1. Create a Focal Plane
    fp_radius = 0.8 * deg
    fp_nrows = 10 # Results in a decent number of detectors for a small radius
    focal_plane = FocalPlane.from_radius(radius=fp_radius, nrows=fp_nrows)
    print(f"\n1. FocalPlane created with {focal_plane.n_dets} detectors.")
    print(f"   Detector x positions (first 5, radians): {focal_plane.x[:5]}")
    print(f"   Detector y positions (first 5, radians): {focal_plane.y[:5]}")

    # 2. Generate a Scan Strategy (Constant Elevation Scan)
    sim_t0 = 1672531200.0     # Example start time (Unix timestamp, e.g., Jan 1, 2023)
    sim_az_start_deg = 45.0   # Scan centered around this azimuth
    sim_el_deg = 60.0         # Elevation of the scan
    sim_az_throw_deg = 10.0   # Width of AZ scan (total sweep)
    sim_v_az_deg_s = 1.0     # Scan speed in AZ
    sim_srate_hz = 200.0       # Sampling rate
    sim_duration_s = 120     # Duration of the observation

    scan_pattern = generate_ces_scan(
        t0=sim_t0,
        az=sim_az_start_deg * deg, # Min azimuth
        el=sim_el_deg * deg,
        az_throw=sim_az_throw_deg * deg,
        v_az=sim_v_az_deg_s * deg / s,
        srate=sim_srate_hz * Hz,
        duration=sim_duration_s * s
    )
    print(f"\n2. Scan strategy generated:")
    print(f"   Time vector shape: {scan_pattern.t.shape}")
    print(f"   Azimuth vector shape: {scan_pattern.az.shape}")
    print(f"   Elevation vector shape: {scan_pattern.el.shape}")
    print(f"   Number of samples: {scan_pattern.nsamps}")


    # 3. Build Pointing Model (get sky coordinates for each detector)
    print("\n3. Building pointing model...")
    # Using default site="so" and weather="typical"
    sky_coords = build_pointing_model(focal_plane, scan_pattern)
    print(f"   Sky coordinates array shape: {sky_coords.shape} (expected: 3, n_dets, nsamps)")
    print(f"   Example RA for det 0, samp 0: {np.rad2deg(sky_coords[0,0,0]):.4f} deg")
    print(f"   Example Dec for det 0, samp 0: {np.rad2deg(sky_coords[1,0,0]):.4f} deg")
    print(f"   Example PA for det 0, samp 0: {np.rad2deg(sky_coords[2,0,0]):.4f} deg")


    # 4. Generate 1/f Noise for one detector
    print("\n4. Generating 1/f noise for one detector...")
    noise_fknee = 1 * Hz   # Knee frequency for noise
    noise_alpha = -2.0     # Spectral index for 1/f type noise
    noise_nlev = 1e-4      # Noise level (arbitrary units for this test)
    noise_seed = 123       # Seed for reproducibility

    sample_noise_tod = rand_oof(
        nsamps=scan_pattern.nsamps,
        srate=sim_srate_hz, # Pass srate directly
        fknee=noise_fknee,
        alpha=noise_alpha,
        nlev=noise_nlev,
        seed=noise_seed
    )
    print(f"   Generated noise TOD shape: {sample_noise_tod.shape}")
    print(f"   Noise TOD mean: {np.mean(sample_noise_tod):.2e}, std: {np.std(sample_noise_tod):.2e}")

    import matplotlib.pyplot as plt
    print("\n5. Plotting results")

    plt.figure(figsize=(12, 10))

    # Plot 1: Sky path of the first detector
    plt.subplot(2, 1, 1)

    # Plotting celestial longitude (RA-like) vs latitude (Dec-like)
    for i in range(0, focal_plane.n_dets, 10):
        plt.plot(np.rad2deg(sky_coords[0, i, :]), np.rad2deg(sky_coords[1, i, :]), 
                    marker='.', linestyle='-', markersize=3, label=f'Detector {i} Path', alpha=0.5)

    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title(f"Sky Path")
    # Standard astronomical convention: RA increases to the left
    plt.gca().invert_xaxis() # Uncomment if RA is plotted and this convention is desired
    plt.grid(True)

    # Plot 2: Sample 1/f noise TOD
    plt.subplot(2, 1, 2)
    time_axis = (scan_pattern.t - sim_t0) if scan_pattern.t.size > 0 else np.arange(sample_noise_tod.size) / sim_srate_hz
    plt.plot(time_axis, sample_noise_tod)
    plt.xlabel(f"Time (s) since t0 ({sim_t0:.0f})")
    plt.ylabel("Noise Amplitude")
    plt.title(f"Sample 1/f Noise TOD (fknee={noise_fknee:.2f} Hz, alpha={noise_alpha:.1f})")
    plt.grid(True)
            
    plt.tight_layout()
    plt.show()

    # 6. Build TOD
    print("\n6. Building TOD...")
    # Create a dummy sky model
    # dummy_sky_model = DummySkyModel()
    sky_models = [
        # FrozenAtmosphere(lknee=500, alpha=-2, nlev=1, seed=125, lmax=6000),
    ]
    tod_models = [
        CorrelatedOOFModel(
            fknee=1 * Hz,
            alpha=-2,
            sigma_target=np.ones(focal_plane.n_dets),
            # n_modes=1,
            # fractions=[1],
            n_modes=2,
            fractions=[0.99, 0.01],
            seed=123,
        )
    ]
    # Build the TOD
    tod = build_tod(
        fplane=focal_plane,
        scan=scan_pattern,
        sky_models=sky_models,
        tod_models=tod_models,
        n_dummy=20,
    )
    print(f"   TOD shape: {tod.data.shape}")

    # calibrate
    gain, _, frac = common_mode_analysis(tod.data)
    print(f"   Common mode gain: {gain[:5]}")
    print(f"   Common mode variance fraction: {frac:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot((tod.data/gain[:, None])[::10, :].T, alpha=0.1);


# %%
