"""Generate CMBLensed_fullsky_alms"""

import os, os.path as op
import argparse
import numpy as np
import healpy as hp
from pixell import enmap, utils, lensing, powspec, curvedsky

parser = argparse.ArgumentParser(description='Generate lensed CMB')
parser.add_argument("--sim_num",    type=int, help="the number of simulation")
parser.add_argument("--odir",       type=str, help="Output directory")
parser.add_argument("--lmax",       type=int, default=3000, help="Max multipole for lensing")
parser.add_argument("--lmax-write", type=int, default=3000, help="Max multipole to write")
parser.add_argument("--pix-size",   type=float, default=1, help="Pixel width in arcmin")
parser.add_argument("--phi_ps",     type=str, default="data/cosmo2017_10K_acc3_lenspotentialCls.dat", help="Input phi cl file")
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
cmb_dir = args.odir

print("read phi ps: %s", args.phi_ps)
ps_lensinput = powspec.read_camb_full_lens(args.phi_ps)

#make phi totally uncorrelated with both T and E.  This is necessary due to the way that separate phi and CMB seeds were put forward in an update to the pixell library around mid-Nov 2018
ps_lensinput[0, 1:, :] = 0.
ps_lensinput[1:, 0, :] = 0.

# get clpp
ls = np.arange(ps_lensinput.shape[-1])
clpp = ps_lensinput[0, 0]

isim = args.sim_num

print(f'doing sim {isim}, calling lensing.rand_map')
cmb_seed = (isim, 0, 0, 0)
phi_seed = (isim, 0, 2, 0)

# generate and write lensed CMB alm
shape, wcs = enmap.fullsky_geometry(args.pix_size*utils.arcmin)
l_tqu_map, = lensing.rand_map((3,)+shape, wcs, ps_lensinput,
                              lmax=args.lmax,
                              output="l",
                              phi_seed=phi_seed,
                              seed=cmb_seed,
                              verbose=True)
    
print('calling curvedsky.map2alm')
alm = curvedsky.map2alm(l_tqu_map, lmax=args.lmax_write, spin=[0,2])

filename = cmb_dir + f"/CMBLensed_fullsky_alm_{isim:03d}.fits"
print(f'writing to disk, filename is {filename}')
hp.write_alm(filename, np.complex64(alm), overwrite=True)
del alm
    
# generate phi and kappa alm, write kappa_alm
phi_alm = curvedsky.rand_alm(clpp, lmax=args.lmax, seed=phi_seed)

fac_lens = ls*(ls+1)/2
kappa_alm = hp.almxfl(phi_alm, fac_lens)

hp.write_alm(cmb_dir + f'/kappa_fullsky_alm_{isim:03d}.fits', kappa_alm, overwrite=True)

del phi_alm, kappa_alm