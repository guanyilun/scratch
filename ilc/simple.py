#%%
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import warnings
from scipy import interpolate
from contextlib import contextmanager
import itertools

@contextmanager
def nowarn():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

class constants:
    h = 6.62607015e-34  # Planck constant
    c = 299792458  # speed of light
    k_B = 1.380649e-23  # Boltzmann constant
    T_cmb = 2.7255  # CMB temperature
    Jy = 1e-26  # Jansky in SI units: 1e-26 W/m^2/Hz

class conversions:
    def SI_to_Jy(nu, I):
        return I / constants.Jy  # W/m^2/Hz to Jy/sr
    def Jy_to_SI(nu, I):
        return I * constants.Jy
    def SI_to_KCMB(nu, I):
        return I / spectra.dB_dT(nu, constants.T_cmb)
    def KCMB_to_SI(nu, I):
        return I * spectra.dB_dT(nu, constants.T_cmb)
    def KCMB_to_Jy(nu, I):
        return conversions.SI_to_Jy(nu, conversions.KCMB_to_SI(nu, I))  # K_CMB to Jy/sr
    def Jy_to_KCMB(nu, I):
        return conversions.SI_to_KCMB(nu, conversions.Jy_to_SI(nu, I))
    def SI_to_KRJ(nu, I):
        return I / spectra.dB_dTRJ(nu)
    def KRJ_to_SI(nu, I):
        return I * spectra.dB_dTRJ(nu)
    def KRJ_to_Jy(nu, I):
        return conversions.SI_to_Jy(nu, conversions.KRJ_to_SI(nu, I))  # K_RJ to Jy/sr
    def Jy_to_KRJ(nu, I):
        return conversions.SI_to_KRJ(nu, conversions.Jy_to_SI(nu, I))
    def Jy_to_KCMB(nu, I):
        return conversions.SI_to_KCMB(nu, conversions.Jy_to_SI(nu, I))  # Jy/sr to K_CMB
    def Jy_to_KRJ(nu, I):
        return conversions.SI_to_KRJ(nu, conversions.Jy_to_SI(nu, I))
    def KCMB_to_KRJ(nu, I):
        return conversions.SI_to_KRJ(nu, conversions.KCMB_to_SI(nu, I))  # K_CMB to K_RJ
    def KRJ_to_KCMB(nu, I):
        return conversions.SI_to_KCMB(nu, conversions.KRJ_to_SI(nu, I))


class spectra:
    def blackbody(nu, T):
        x = constants.h * nu / constants.k_B / T
        return 2 * constants.h * nu**3 / constants.c**2 / jnp.expm1(x)

    def dB_dT(nu, T):
        x = constants.h * nu / constants.k_B / T
        return spectra.blackbody(nu, T) * x * jnp.exp(x) / (T * (jnp.exp(x) - 1))

    def dB_dTRJ(nu):
        return 2 * nu**2 * constants.k_B / constants.c**2

    cmb_iso = partial(blackbody, T=constants.T_cmb)
    cmb_ani = partial(dB_dT, T=constants.T_cmb)
    ksz = cmb_ani
    
    def tsz(nu):
        x = constants.h * nu / constants.k_B / constants.T_cmb
        I = x * (jnp.exp(x)+1) / jnp.expm1(x) + 4
        return I * constants.T_cmb * spectra.dB_dT(nu, constants.T_cmb)

    def cib(nu, beta, Td):
        return spectra.blackbody(nu, Td) * nu**beta

    cib_poisson = partial(cib, beta=2.1, Td=9.7)
    cib_clustered = partial(cib, beta=2.1, Td=9.7)

    def radio_poisson(nu, alpha_s=-0.5):
        return nu**-alpha_s

    def galactic_dust(nu, alpha_d=3.8):
        return nu**alpha_d
    
class angular_spectra:
    def _build_tsz(template):
        l, dl = np.loadtxt(template, unpack=True, dtype=np.float64)
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        cl_f = interpolate.interp1d(l, cl, kind='linear', fill_value='extrapolate') 
        def tsz_f(nu1, nu2, l, a_tsz=1.5, nu_ref=150e9):
            return a_tsz * cl_f(l) * spectra.tsz(nu1) * spectra.tsz(nu2) / spectra.tsz(nu_ref)**2
        return tsz_f
    tsz = _build_tsz("data/tSZ_template.txt")

    def _build_ksz(template):
        l, dl = np.loadtxt(template, unpack=True, dtype=jnp.float64)
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        cl_f = interpolate.interp1d(l, cl, kind='linear', fill_value='extrapolate')
        def ksz_f(nu1, nu2, l, a_ksz=1.5):
            return a_ksz * cl_f(l)
        return ksz_f
    ksz = _build_ksz("data/kSZ_template.txt")

    def _build_tsz_x_cib(template):
        l, dl = np.loadtxt(template, unpack=True, dtype=jnp.float64)
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        cl_f = interpolate.interp1d(l, cl, kind='linear', fill_value='extrapolate') 
        def tsz_x_cib_f(nu1, nu2, l, xi=0.2, a_tsz=4.0, a_cib=5.7, beta_cib=2.1, Td=9.7, nu_ref=150e9):
            return \
            -2*xi*jnp.sqrt(a_tsz * a_cib)*cl_f(l) * \
                (spectra.tsz(nu1)*spectra.cib_poisson(nu2, beta=beta_cib, Td=Td) + \
                 spectra.tsz(nu2)*spectra.cib_poisson(nu1, beta=beta_cib, Td=Td)) / \
            (2 * spectra.tsz(nu_ref) * spectra.cib_poisson(nu_ref, beta=beta_cib, Td=Td))
        return tsz_x_cib_f
    tsz_x_cib = _build_tsz_x_cib("data/minus_tSZ_CIB_template.txt")

    def cib_poisson(nu1, nu2, l, a_cib=7.0, Td=9.7, beta=2.1, nu_ref=150e9):
        dl = a_cib * (l/3000)**2
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        return cl * spectra.cib(nu1, beta=beta, Td=Td) * spectra.cib(nu2, beta=beta, Td=Td) / spectra.cib(nu_ref, beta=beta, Td=Td)**2

    def cib_clustered(nu1, nu2, l, a_cib=5.7, n=1.2, Td=9.7, beta=2.1, nu_ref=150e9):
        dl = a_cib * (l/3000)**(2-n)
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        return cl * spectra.cib(nu1, beta=beta, Td=Td) * spectra.cib(nu2, beta=beta, Td=Td) / spectra.cib(nu_ref, beta=beta, Td=Td)**2

    def cib(nu1, nu2, l, cibp_cfg={}, cibc_cfg={}):
        return angular_spectra.cib_poisson(nu1, nu2, l, **cibp_cfg) + angular_spectra.cib_clustered(nu1, nu2, l, **cibc_cfg)

    def radio_poisson(nu1, nu2, l, a_s=3.2, alpha_s=-0.5, nu_ref=150e9):
        dl = a_s * (l/3000)**2
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        return cl * spectra.radio_poisson(nu1, alpha_s=alpha_s) * spectra.radio_poisson(nu2, alpha_s=alpha_s) / spectra.radio_poisson(nu_ref, alpha_s=alpha_s)**2

    def galactic_dust(nu1, nu2, l, a_d=1.7, alpha_d=3.8, nu_ref=150e9):
        dl = a_d * (l/3000)**2
        with nowarn():
            cl = dl * 2 * np.pi / l / (l+1)
        return cl * spectra.galactic_dust(nu1, alpha_d=alpha_d) * spectra.galactic_dust(nu2, alpha_d=alpha_d) / spectra.galactic_dust(nu_ref, alpha_d=alpha_d)**2

def cmb_ilc_weights_minvar(freqs, l, components):
    c = np.zeros((len(l), len(freqs), len(freqs)))
    for (i, j) in itertools.product(range(len(freqs)), range(len(freqs))):
        nu1, nu2 = freqs[i], freqs[j]
        for comp in components:
            c[:, i, j] += comp(nu1, nu2, l)
    cinv = np.linalg.inv(c)
    w = cinv.sum(axis=2) / cinv.sum(axis=(1, 2))[:, None]
    return w

     
#%%
if __name__ == '__main__':
    from pysm3 import units as u
    from matplotlib import pyplot as plt

    def test_eq(a, b, name="Test"):
        assert np.allclose(a, b), f"{name} failed: {a} != {b}"
        print(f"{name} Passed: {a} == {b}")

    # =======================
    # Test unit conversions
    # =======================

    c = (1 * u.Jy / u.sr).to(u.K_CMB, equivalencies=u.cmb_equivalencies(150 * u.GHz)).value
    test_eq(c, conversions.Jy_to_KCMB(150e9, 1), "Jy to K_CMB")

    c = (1 * u.Jy / u.sr).to(u.K_RJ, equivalencies=u.cmb_equivalencies(150 * u.GHz)).value
    test_eq(c, conversions.Jy_to_KRJ(150e9, 1), "Jy to K_RJ")

    c = (1 * u.K_CMB).to(u.Jy / u.sr, equivalencies=u.cmb_equivalencies(150 * u.GHz)).value
    test_eq(c, conversions.KCMB_to_Jy(150e9, 1), "K_CMB to Jy")
    
    c = (1 * u.K_RJ).to(u.Jy / u.sr, equivalencies=u.cmb_equivalencies(150 * u.GHz)).value
    test_eq(c, conversions.KRJ_to_Jy(150e9, 1), "K_RJ to Jy")

    c = (1 * u.K_CMB).to(u.K_RJ, equivalencies=u.cmb_equivalencies(150 * u.GHz)).value
    test_eq(c, conversions.KCMB_to_KRJ(150e9, 1), "K_CMB to K_RJ")

    c = (1 * u.K_RJ).to(u.K_CMB, equivalencies=u.cmb_equivalencies(150 * u.GHz)).value
    test_eq(c, conversions.KRJ_to_KCMB(150e9, 1), "K_RJ to K_CMB")

    # %%
    # =======================
    # test spectra
    # =======================

    nu = np.arange(10, 1000, 10) * 1e9
    nu_ref = 150e9
    with plt.rc_context({
        'font.size': 14,
        'figure.dpi': 200,
    }):
        plt.plot(nu/1e9, spectra.tsz(nu)/spectra.tsz(nu_ref), label="tSZ")
        plt.plot(nu/1e9, spectra.cib_poisson(nu)/spectra.cib_poisson(nu_ref), label="CIB Poisson")
        plt.plot(nu/1e9, spectra.cib_clustered(nu)/spectra.cib_clustered(nu_ref), label="CIB Clustered")
        plt.plot(nu/1e9, spectra.radio_poisson(nu)/spectra.radio_poisson(nu_ref), label="Radio Poisson")
        plt.plot(nu/1e9, spectra.galactic_dust(nu)/spectra.galactic_dust(nu_ref), label="Galactic Dust")
        plt.plot(nu/1e9, spectra.cmb_iso(nu)/spectra.cmb_iso(nu_ref), label="CMB Iso")
        plt.plot(nu/1e9, spectra.cmb_ani(nu)/spectra.cmb_ani(nu_ref), label="CMB Ani")
        plt.plot(nu/1e9, spectra.ksz(nu)/spectra.ksz(nu_ref), label="kSZ")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Spectra")
        plt.title("Spectra normalized to 150 GHz")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    # %%
    # =======================
    # test angular spectra
    # =======================
    from matplotlib import pyplot as plt
    from orphics import cosmology
    theory = cosmology.default_theory()

    nu1, nu2 = 150e9, 150e9
    l = jnp.arange(500, 6000)
    tsz = angular_spectra.tsz(nu1, nu2, l)
    ksz = angular_spectra.ksz(nu1, nu2, l)
    tsz_x_cib = angular_spectra.tsz_x_cib(nu1, nu2, l)
    cib_poisson = angular_spectra.cib_poisson(nu1, nu2, l)
    cib_clustered = angular_spectra.cib_clustered(nu1, nu2, l)
    radio_poisson = angular_spectra.radio_poisson(nu1, nu2, l)
    cltt = theory.lCl('TT', l)
    pre = l**2/(2*np.pi)

    with plt.rc_context({
        'font.size': 14,
        'figure.dpi': 200,
    }):
        plt.figure(figsize=(6,4))
        plt.plot(l, tsz*pre, label="tsz")
        plt.plot(l, ksz*pre, label="ksz")
        plt.plot(l, tsz_x_cib*pre, label="tsz_x_cib")
        plt.plot(l, cib_poisson*pre, label="cib_poisson")
        plt.plot(l, cib_clustered*pre, label="cib_clustered")
        plt.plot(l, radio_poisson*pre, label="radio_poisson") 
        plt.plot(l, cltt*pre, label="TT")
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$\ell^2 C_\ell / 2\pi$")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        plt.title("Angular Spectra at 150 GHz")
    
    # %%
    # =======================
    # test ILC
    # =======================
    freqs = np.array([93, 145, 225, 278]) * 1e9
    l = np.arange(500, 6000, 50)
    components = [
        angular_spectra.tsz, angular_spectra.ksz, angular_spectra.tsz_x_cib, 
        angular_spectra.cib_poisson, angular_spectra.cib_clustered, 
        angular_spectra.radio_poisson, angular_spectra.galactic_dust
    ]
    w = cmb_ilc_weights_minvar(freqs, l, components)

    with plt.rc_context({
        'font.size': 14,
        'figure.dpi': 200,
    }):
        plt.figure(figsize=(6,4))
        for i, freq in enumerate(freqs):
            plt.plot(l, w[:, i], label=f"{freq/1e9} GHz")
        plt.xlabel(r"$\ell$")
        plt.ylabel("Weights")
        plt.xscale('log')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        plt.title("ILC Weights")

# %%
