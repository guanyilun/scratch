import numpy as np
from numpy.typing import NDArray
from typing import Protocol, Optional, Tuple
from dataclasses import dataclass
from matplotlib import pyplot as plt
import alphashape
from shapely.geometry import Point, Polygon
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

        # Calculate coordinates of dummy detectors on the circle around
        # the calculated center.
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
    time_per_sweep = az_throw / v_az

    # Calculate azimuth: every two sweeps give one cycle
    phase = (t_vector / time_per_sweep) % 2.0
    mask = phase > 1.0
    phase[mask] = 2.0 - phase[mask]
    az_vector = az_min_rad * (1.0 - phase) + az_max_rad * phase

    # Elevation is constant
    el_vector = np.full_like(t_vector, el)

    return Scan(t=t_vector, az=az_vector, el=el_vector, srate=srate)


@dataclass
class TOD:
    scan: Scan
    fplane: FocalPlane
    data: NDArray      # shape: (ndets, nsamps)
    pointing: NDArray  # shape: (3, ndets, nsamps)


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


@dataclass
class Footprint:
    """
    A class to represent the footprint of a scan in RA and Dec coordinates.
    This is used to visualize the scan pattern on the sky.
    """
    geometry: Polygon
    bounds: Optional[Tuple[float, float, float, float]] = None


def get_scan_footprint(fplane: FocalPlane, scan: Scan,
                       n_dummy: int = 50, n_levels: int = 20, interior_pts: int = 500) -> Footprint:
    """
    Returns the shape of the focal plane in RA and Dec coordinates using boundary detection.

    Args:
        fplane (FocalPlane): The focal plane object containing detector positions.
        scan (Scan): The scan object containing time, azimuth, and elevation vectors.

    Returns:
        Footprint
    """
    fp = fplane.get_circular_cover(n_dummy=n_dummy)
    sky_coords = build_pointing_model(fp, scan)  # shape: (3, n_dets, nsamps)

    # Extract ra and dec (in radians)
    ra = sky_coords[0, :, :].ravel()
    dec = sky_coords[1, :, :].ravel()

    # If no points, return None
    if ra.size == 0:
        return None

    data = np.column_stack((ra, dec))

    max_dec = np.max(dec)
    min_dec = np.min(dec)

    # Define epsilon in radians
    epsilon = 10 * arcmin

    # Get top and bottom slices
    top_mask = (dec >= max_dec - epsilon) & (dec <= max_dec + epsilon)
    bot_mask = (dec >= min_dec - epsilon) & (dec <= min_dec + epsilon)

    top = data[top_mask]
    bot = data[bot_mask]

    # Initialize boundary points
    boundary_points = []

    # Top boundary: leftmost and rightmost points
    if top.size > 0:
        tl_index = np.argmin(top[:,0])
        tr_index = np.argmax(top[:,0])
        boundary_points.append(top[tl_index])
        boundary_points.append(top[tr_index])

    # Bottom boundary: leftmost and rightmost points
    if bot.size > 0:
        bl_index = np.argmin(bot[:,0])
        br_index = np.argmax(bot[:,0])
        boundary_points.append(bot[bl_index])
        boundary_points.append(bot[br_index])

    # Sample at 20 dec levels
    dec_samples = np.linspace(min_dec + epsilon, max_dec - epsilon, n_levels)
    for dec_sample in dec_samples:
        sample_mask = (dec >= dec_sample - epsilon) & (dec <= dec_sample + epsilon)
        sample_slice = data[sample_mask]
        if sample_slice.size > 0:
            left_index = np.argmin(sample_slice[:,0])
            right_index = np.argmax(sample_slice[:,0])
            boundary_points.append(sample_slice[left_index])
            boundary_points.append(sample_slice[right_index])

    # Convert boundary_points to numpy array
    boundary_points = np.array(boundary_points)

    # Add interior points (500 points randomly selected)
    if data.shape[0] > interior_pts:
        interior_indices = np.random.choice(data.shape[0], size=interior_pts, replace=False)
        interior_points = data[interior_indices]
        all_points = np.vstack((boundary_points, interior_points))
    else:
        all_points = data

    # Compute alpha shape
    alpha_shape = alphashape.alphashape(all_points, 2)
    bounds = [np.min(ra), np.min(dec), np.max(ra), np.max(dec)]
    return Footprint(geometry=alpha_shape, bounds=bounds)


class SkyModel(Protocol):
    def apply(self, sky_map: enmap.enmap) -> enmap.enmap:
        """Paints the sky with a model."""
        ...

class TODModel(Protocol):
    def apply(self, tod: TOD) -> TOD:
        """Paints the time-ordered data (TOD) with a model."""
        ...


@dataclass
class CorrelatedOOF(TODModel):
    """Adds correlated 1/f noise between detectors using specified modes."""
    fknee: float | list[float]    # Knee frequency in Hz (scalar or per-mode)
    alpha: float | list[float]    # Spectral index (scalar or per-mode)
    nlevs: NDArray                # Noise levels of each mode. shape: (n_modes,)
    gains: NDArray                # Mode gains: (n_modes, n_dets)
    seed: int | None = None

    def apply(self, tod: TOD) -> TOD:
        ndets, nsamps = tod.data.shape
        if self.gains.shape[1] != ndets:
            raise ValueError(f"gains shape {self.gains.shape} does not match TOD detectors {ndets}")
        if self.gains.shape[0] != len(self.nlevs):
            raise ValueError(f"gains and nlevs must have same number of modes")

        noise = self._generate_correlated_noise(ndets, nsamps, tod.scan.srate)
        tod.data += noise
        return tod

    def _generate_correlated_noise(self, ndets: int, nsamps: int, srate: float) -> NDArray:
        n_modes = len(self.nlevs)
        rng = np.random.default_rng(self.seed)

        # Generate seeds for each mode if needed
        seeds = rng.integers(0, 2**32, size=n_modes) if self.seed is not None else [None] * n_modes

        # Generate independent 1/f noise streams for each mode
        mode_noise = np.zeros((n_modes, nsamps))
        for i in range(n_modes):
            # Get mode-specific parameters
            fknee_i = self.fknee[i] if isinstance(self.fknee, (list, np.ndarray)) else self.fknee
            alpha_i = self.alpha[i] if isinstance(self.alpha, (list, np.ndarray)) else self.alpha

            # Generate 1/f noise with unit variance
            noise_stream = rand_oof(
                nsamps=nsamps,
                srate=srate,
                fknee=fknee_i,
                alpha=alpha_i,
                nlev=self.nlevs[i],
                seed=seeds[i]
            )

            mode_noise[i, :] = noise_stream

        # Project modes to detectors: (n_modes, nsamps) @ (n_modes, n_dets).T -> (n_dets, nsamps)
        return self.gains.T @ mode_noise


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


@dataclass
class CosmicRaySimulator(TODModel):
    """
    A simple cosmic ray simulator that adds spikes to the TOD.
    """
    n_per_sample: float = 0.01  # Average number of spikes per sample
    radius_scale: float = 4 * arcmin
    radius_shape: float = 1
    radius_max_factor: float = 10
    amp_shape: float = 1
    amp_scale: float = 1
    amp_max_factor: float = 10

    def radial_profile(self, r, radius=5*arcmin):
        return np.exp(-r / radius)

    def temporal_profile(self):
        return np.array([1, 0.2, -0.1, 0])

    def apply(self, tod: TOD) -> TOD:
        n_hits_expect = tod.scan.nsamps * self.n_per_sample
        n_hits = int(np.random.poisson(n_hits_expect))
        if n_hits == 0:
            print("No cosmic ray hits generated.")
            return tod
        print(f"Generating {n_hits} cosmic ray hits.")

        # randomly choose detector indices and sample indices for hits
        det_indices = np.random.randint(0, tod.fplane.n_dets, size=n_hits)
        samp_indices = np.random.randint(0, tod.scan.nsamps, size=n_hits)

        # what radius and amplitudes to use for each hit?
        radius_raw = np.random.pareto(self.radius_shape, size=n_hits) + 1
        radius_raw = np.clip(radius_raw, 0, self.radius_max_factor - 1)
        radius = radius_raw * self.radius_scale

        amps_row = np.random.pareto(self.amp_shape, size=n_hits) + 1
        amps_row = np.clip(amps_row, 0, self.amp_max_factor - 1)
        amps = amps_row * self.amp_scale

        fp = tod.fplane  # alias
        for i in range(n_hits):
            det_idx = det_indices[i]
            samp_idx = samp_indices[i]
            x_c = tod.fplane.x[det_idx]
            y_c = tod.fplane.y[det_idx]

            # calculate the distance from the detector to the spike
            r = ((fp.x - x_c)**2 + (fp.y - y_c)**2)**0.5

            # calculate the expected amplitude for each detector
            A = amps[i] * self.radial_profile(r, radius=radius[i])

            # with temporal profile
            snip = A[:, None] * self.temporal_profile()[None, :]

            samp_idx_end = min(tod.scan.nsamps, samp_idx + snip.shape[1])

            # add a spike at the given detector and sample index
            tod.data[:, samp_idx:samp_idx_end] += snip[:, :samp_idx_end-samp_idx]
        return tod


@dataclass
class PointSourceSimulator(TODModel):
    n_srcs: int = 1
    beam_sigma: float = 2 * arcmin
    flux_limits: Tuple[float, float] = (1e-6, 1e-3)  # Flux limits in Jy

    def apply(self, tod: TOD) -> TOD:
        footprint = get_scan_footprint(tod.fplane, tod.scan)
        srcs = random_point_in_footprint(footprint, n_points=self.n_srcs)
        print(f"Simulating {self.n_srcs} point sources within footprint")

        ra_, dec_ = tod.pointing[0], tod.pointing[1]  # shape: (n_dets, nsamps)
        for src in srcs:
            print(f"\tAdding point source at RA: {np.rad2deg(src[0]):.2f} deg, Dec: {np.rad2deg(src[1]):.2f} deg")
            r = np.sqrt((ra_ - src[0])**2 + (dec_ - src[1])**2)
            beam = np.exp(-0.5 * (r / self.beam_sigma)**2)
            flux = 10**np.random.uniform(np.log10(self.flux_limits[0]), np.log10(self.flux_limits[1]))
            tod.data += flux * beam
        return tod


def random_point_in_footprint(footprint: Footprint, n_points: int = 1):
    """
    Generates a random point within the given footprint.

    Args:
        footprint (Footprint): The footprint object containing the geometry.

    Returns:
        np.ndarray: An array of shape (n_samples, 2) containing random RA and Dec points.

    """
    min_ra, min_dec, max_ra, max_dec = footprint.bounds
    points = []
    while len(points) < n_points:
        # Generate random RA and Dec within the bounds
        ra = np.random.uniform(min_ra, max_ra)
        dec = np.random.uniform(min_dec, max_dec)

        # Check if the point is inside the footprint
        if footprint.geometry.contains(Point(ra, dec)):
            points.append(([ra, dec]))

    return np.array(points)


# main interface

def build_tod(
    fplane: FocalPlane,
    scan: Scan,
    sky_models: list[SkyModel] = [],
    tod_models: list[TODModel] = [],
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

    # full pointing model for the focal plane
    sky_coords = build_pointing_model(fplane, scan)

    if len(sky_models) > 0:
        # get the tod for each detector by reading the sky map at the detector positions
        # only do this if we have a sky model
        tod = sky_map.at(pos=[sky_coords[1], sky_coords[0]])  # dec, ra order
    else:
        print("No sky models provided, initializing TOD with zeros.")
        tod = np.zeros((fplane.n_dets, scan.nsamps), dtype=np.float32)

    print(f"Sky map shape: {sky_map.shape}, tod shape: {tod.shape}")

    tod = TOD(scan=scan, data=tod, fplane=fplane, pointing=sky_coords)

    # apply TOD models (like noise sims)
    for tod_model in tod_models:
        tod = tod_model.apply(tod)
    return tod


if __name__ == "__main__":
    fp_radius = 0.8 * deg
    fp_nrows = 30
    fplane = FocalPlane.from_radius(radius=fp_radius, nrows=fp_nrows)
    print(f"FocalPlane created with {fplane.n_dets} detectors.")

    plt.scatter(np.rad2deg(fplane.x), np.rad2deg(fplane.y), s=20)
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.axis('equal')

    sim_t0 = 1672531200.0       # Example start time (Unix timestamp, e.g., Jan 1, 2023)
    sim_az_start = 45.0 * deg   # Scan centered around this azimuth
    sim_el = 60.0 * deg         # Elevation of the scan
    sim_az_throw = 10.0 * deg   # Width of AZ scan (total sweep)
    sim_v_az = 1.0 * deg / s    # Scan speed in Az in deg per sec
    sim_srate = 200.0 * Hz      # Sampling rate
    sim_duration = 120 * s      # Duration of the observation

    scan_pattern = generate_ces_scan(
        t0=sim_t0,
        az=sim_az_start,
        el=sim_el,
        az_throw=sim_az_throw,
        v_az=sim_v_az,
        srate=sim_srate,
        duration=sim_duration
    )

    plt.plot(scan_pattern.t, np.rad2deg(scan_pattern.az))
    plt.ylabel("Azimuth [deg]")
    plt.xlabel("Unix Time [s]")

    np.random.seed(42)

    nmodes = 1
    gains = np.ones((nmodes, fplane.n_dets))
    gains[0, :] = np.clip((np.random.randn(fplane.n_dets) * 0.2 + 1), 0, 5)
    nlevs = np.ones((nmodes,))

    tod = build_tod(
        fplane=fplane,
        scan=scan_pattern,
        tod_models=[
            CorrelatedOOF(
                fknee=100,
                alpha=-2,
                nlevs=nlevs,
                gains=gains,
            ),
            PointSourceSimulator(
                n_srcs=1,
                beam_sigma=4 * arcmin,
                flux_limits=(1e-2, 1e1)
            ),
            CosmicRaySimulator(
                n_per_sample=0.001,
                amp_scale=0.3,
            ),
        ],
    )