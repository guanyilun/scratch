"""example of lens reconstruction on curved sky"""

#%%
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from pixell import enmap, powspec, lensing
import pytempura as tp
from falafel import qe
from recon_2d import Expt

def isotropic_filter(alm,tcls,lmin,lmax,ignore_te=True,
                     ucls=None, wiener=False):
    """Taken from falafel.utils.py"""
    if ucls is None:
        ucltt, uclte, uclee, uclbb = None, None, None, None
    else:
        ucltt, uclte, uclee, uclbb = ucls['TT'], ucls['TE'], ucls['EE'], ucls['BB']

    tcltt, tclte, tclee, tclbb = tcls['TT'], tcls['TE'], tcls['EE'], tcls['BB']

    if ignore_te:
        filt_T, filt_E, filt_B = tcltt*0, tclee*0, tclbb*0
        with np.errstate(divide='ignore', invalid='ignore'):
            filt_T[2:] = 1./tcltt[2:]
            filt_E[2:] = 1./tclee[2:]
            filt_B[2:] = 1./tclbb[2:]
            if wiener:
                if ucls is None:
                    print("ERROR: Must provide ucls if Wiener filtering.")
                    return [None, None, None]
                filt_T[2:] *= ucltt[2:]
                filt_E[2:] *= uclee[2:]
                filt_B[2:] *= uclbb[2:]
        talm = qe.filter_alms(alm[0],filt_T,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[1],filt_E,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_B,lmin=lmin,lmax=lmax)

    else:
        filt_TT, filt_TE, filt_ET, filt_EE = tcltt*0, tclte*0, tclte*0, tclee*0
        filt_BB = tclbb*0

        with np.errstate(divide='ignore', invalid='ignore'):
            # det of TT + EE block (aka prefactor of its inverse)
            te_det = 1. / (tcltt[2:]*tclee[2:] - tclte[2:]**2.)

            filt_TT[2:] = te_det
            filt_EE[2:] = te_det
            filt_TE[2:] = te_det
            filt_ET[2:] = te_det
            filt_BB[2:] = 1. / tclbb[2:]

            if wiener:
                if ucls is None:
                    print("ERROR: Must provide ucls if Wiener filtering.")
                    return [None, None, None]
                filt_TT[2:] *= (ucltt[2:]*tclee[2:] - uclte[2:]*tclte[2:])
                filt_EE[2:] *= (uclee[2:]*tcltt[2:] - uclte[2:]*tclte[2:])
                filt_BB[2:] *= uclbb[2:]
                # these two are no longer symmetric 
                filt_TE[2:] *= (uclte[2:]*tcltt[2:] - ucltt[2:]*tclte[2:])
                filt_ET[2:] *= (uclte[2:]*tclee[2:] - uclee[2:]*tclte[2:])
            else:
                filt_TT[2:] *= tclee[2:]
                filt_EE[2:] *= tcltt[2:]
                filt_TE[2:] *= -tclte[2:]
                filt_ET[2:] *= -tclte[2:]

        talm = qe.filter_alms(alm[0],filt_TT,lmin=lmin,lmax=lmax) + \
               qe.filter_alms(alm[1],filt_TE,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[0],filt_ET,lmin=lmin,lmax=lmax) + \
               qe.filter_alms(alm[1],filt_EE,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_BB,lmin=lmin,lmax=lmax)
        
    return [talm,ealm,balm]

rlmin, rlmax = 100, 3000  # CMB multipole range for reconstruction
estimators = ['TT']

# Expt configuration
expt = Expt('SO', 1.4, 6, 100, 3000)

# Get theory spectra
ps_lensinput = powspec.read_camb_full_lens("data/cosmo2017_10K_acc3_lenspotentialCls.dat")
ls = np.arange(ps_lensinput.shape[-1])
ucls = {
    'TT': ps_lensinput[1,1],
    'EE': ps_lensinput[2,2],
    'BB': ps_lensinput[3,3],
    'TE': ps_lensinput[1,2]
}
nls = {
    'TT': (nl := expt.get_nl(ls)),
    'TE': nl*0,
    'EE': nl*2,
    'BB': nl*2
}
tcls = { spec: ucls[spec] + nls[spec] for spec in ['TT','TE','EE','BB'] }

# Get CMB alms
alm = hp.read_alm("out/CMBLensed_fullsky_alm_000.fits", hdu=(1,2,3))

# Get normalizations
Als = tp.get_norms(estimators, ucls, ucls, tcls,
                   lmin=expt.lmin, lmax=expt.lmax)

# alternative call: lcl expects shape=(nspec=4, lmax+1)
# all_estimators = ['TT','TE','EE','TB','EB','MV']
# QDO = [k in estimators for k in all_estimators]
# Ag, Ac, Wg, Wc = tp.norm_lens.qall(QDO, lmax, rlmin, rlmax, lcl, lcl, ocl)

# Filter
Xdat = isotropic_filter(alm,tcls,expt.lmin,expt.lmax)

# decide on a geometry for the intermediate operations
res = 1.0 # resol in arcmin
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
px = qe.pixelization(shape, wcs)

# Reconstruct
recon = qe.qe_all(px, ucls, 3000,
                  fTalm=Xdat[0],fEalm=Xdat[1], fBalm=Xdat[2],
                  estimators=estimators,
                  xfTalm=Xdat[0],xfEalm=Xdat[1],xfBalm=Xdat[2])
    
# Get input kappa alms
ikalm = hp.read_alm("out/kappa_fullsky_alm_000.fits")

# Cross-correlate and plot
kalms = {}
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
plt.figure()
for est in estimators:
    kalms[est] = lensing.phi_to_kappa(hp.almxfl(recon[est][0].astype(np.complex128),Als[est][0])) # ignore curl
    plt.plot(ells,(ells*(ells+1.)/2.)**2. * Als[est][0],ls='--', label='N0')
    ccls = hp.alm2cl(kalms[est],ikalm)
    rcls = hp.alm2cl(kalms[est],kalms[est])
    plt.plot(ells,rcls,label='r x r')
    plt.plot(ells,ccls,label = 'r x i')
    plt.plot(ells,icls, label = 'i x i')

plt.yscale('log')
plt.xlabel(r'$L$')
plt.ylabel(r'$C_L^{\kappa\kappa} (L(L+1)/2)^2$')
plt.legend()