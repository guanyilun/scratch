import ephem
from datetime import datetime
import numpy as np
import contextlib
import numba
from typing import NamedTuple
import healpy as hp

from pixell import reproject

deg = np.deg2rad(1)
arcmin = np.deg2rad(1/60)

# defaults
NSIDE = 128
SRATE = 1

######################
#   Site related     #
######################

class Site(ephem.Observer):
    def __init__(self, lat, lon, elev):
        super().__init__()
        self.lat = str(lat)  # assume deg
        self.lon = str(lon)  # assume deg
        self.elev = elev     # assume meter
    def set_date_from_timestamp(self, timestamp):
        """assume that the input is ctime (int)"""
        d = datetime.utcfromtimestamp(timestamp)
        t = d.year, d.month, d.day, d.hour, d.minute, d.second+d.microsecond*1e-6
        self.date = ephem.date(t)

sites = {
    'ACT': Site(lat=-22.958, lon=-67.786, elev=5200),
}

@contextlib.contextmanager
def temp_site(site):
    """preserve original site.date and restore it after the context,
    to make operation pure."""
    date = site.date
    yield site
    site.date = date

######################
#   Scan related     #
######################

class Scan(NamedTuple):
    t0: float
    t1: float
    az: float
    el: float
    throw: float
    velocity: float


# @numba.jit
def _get_azel(t0, t1, az0, el0, throw, velocity, srate=SRATE):
    """Get the az, el of a scan at a given sampling rate. velocity is in deg/s"""
    nsamps = int(np.floor((t1 - t0) * srate))
    ts = np.zeros(nsamps, dtype='f8') + t0
    el = np.zeros(nsamps, dtype='f8') + el0
    az = np.zeros(nsamps, dtype='f8')

    az[0] = az0
    v = velocity

    for i in np.arange(1, nsamps):
        ts[i] = ts[i-1] + 1 / srate
        az[i] = az[i-1] + v * (1/srate)

        if az[i] > az0 + throw/2:
            v *= -1
            az[i] = az0 + throw/2
        elif az[i] < az0 - throw/2:
            v *= -1
            az[i] = az0 - throw/2

    return ts, np.deg2rad(az), np.deg2rad(el)

def get_azel(scan, srate=SRATE):
    return _get_azel(scan.t0, scan.t1, scan.az, scan.el, scan.throw, scan.velocity, srate=srate)

def get_radec(scan, site, srate=SRATE):
    """This function projects a CES scan onto the sky, returning
    the ra and dec of the scan.

    Parameters
    ----------
    scan : Scan
        The scan to project
    site : ephem.Observer
        The site to project the scan at
    samp_rate : float
        The sampling rate of the scan in seconds
    """
    ts, az, el = get_azel(scan, srate=srate)
    ra = np.zeros_like(ts)
    dec = np.zeros_like(ts)

    # use a context manager to preserve the original site.date for stability
    with temp_site(site) as s:
        for i in range(len(ts)):
            s.set_date_from_timestamp(ts[i])
            ra[i], dec[i] = s.radec_of(az[i], el[i])
    return ts, ra, dec

######################
# Instrument Related #
######################

class FocalPlane(NamedTuple):
    ndet: int
    radius: float  # in radian

class Convolver:
    def __init__(self, fp, nside=NSIDE):
        # hard to be exact on ndet, but approximately we could use area_per_det to approximate
        area_per_det = np.pi*fp.radius**2 / fp.ndet  # area in arcmin^2

        pixarea = hp.nside2pixarea(nside, degrees=False)
        ndet_per_pixel = pixarea / area_per_det

        if ndet_per_pixel < 1:
            # use pixel area as resolution
            resol = np.sqrt(pixarea)  # approximation
            self.boost = 1
        else:
            # use pixel area as resolution
            resol = hp.nside2resol(nside, arcmin=False)
            self.boost = ndet_per_pixel

        X, Y = np.meshgrid(np.arange(-fp.radius, fp.radius+resol, resol),
                           np.arange(-fp.radius, fp.radius+resol, resol))
        xind, yind = np.where((X**2 + Y**2) <= fp.radius**2)
        self.dra = np.deg2rad(X[xind, yind]/60)[:,None] # * np.cos(dec[None,:])
        self.ddec = np.deg2rad(Y[xind, yind]/60)[:,None]

    def convolve_focalplane(self, ra, dec): 
        ra_full = ra[None,:] + self.dra * np.cos(dec[None,:])
        dec_full = dec[None,:] + self.ddec
        return ra_full, dec_full

DEFAULT_FP = FocalPlane(100, np.deg2rad(3))
DEFAULT_CONVOLVER = Convolver(DEFAULT_FP, nside=NSIDE)

def scan2hitcount(scan, site=None, nside=NSIDE, srate=SRATE, hitcount=None, fp_convolver=None):
    if site is None: site = sites['ACT']
    _, ra, dec = get_radec(scan, site, srate=srate)
    if fp_convolver is None: fp_convolver = DEFAULT_CONVOLVER
    ra, dec = fp_convolver.convolve_focalplane(ra, dec)
    # generate hitcount
    if hitcount is None: hitcount = np.zeros(hp.nside2npix(nside))
    # convert ra, dec to pix idx
    all_pixs = hp.ang2pix(nside, np.ravel(np.pi/2-dec)%np.pi, np.ravel(ra)%(2*np.pi))  # theta <- 90 - dec
    # count hits
    uniq_pixs, counts = np.unique(all_pixs, return_counts=True)
    if fp_convolver is not None: counts = counts*fp_convolver.boost
    hitcount[uniq_pixs] += counts
    return hitcount


# utility functions
def project_hitcount(hitcount_hp, car_geometry):
    # project healpix to CAR pixelization
    hitcount_car = reproject.healpix2map(hitcount_hp, *car_geometry, method='spline')
    return np.array(hitcount_car)
