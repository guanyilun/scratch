"""remote dipole calculation"""

from dataclasses import dataclass
import numpy as np
from scipy import interpolate, integrate, special


class constants:
    h = 6.62607015e-34  # Planck constant
    c = 299792458       # speed of light
    k_B = 1.380649e-23  # Boltzmann constant
    T_cmb = 2.7255      # CMB temperature
    Jy = 1e-26          # Jansky in SI: 1e-26 W/m^2/Hz


@dataclass
class Cosmology:
    Omega_b: float = 0.04897 # Baryon density
    Omega_c: float = 0.2607  # Cold dark matter density
    # Omega_m: float = 0.3111  # Matter density
    Omega_k: float = 0.0     # Curvature density (flat universe)
    Omega_Lambda: float = 0.6889  # Dark energy
    w: float = -1.0          # Dark energy equation of state
    wa: float = 0.0          # Dark energy evolution parameter
    h: float = 0.6766        # H0 = 100h km/s/Mpc
    
    As: float = 2.105e-9     # Primordial power spectrum amplitude
    ns: float = 0.9665       # Scalar spectral index
    tau: float = 0.0561      # Optical depth
    z_recomb: float = 1089.80

    def __post_init__(self):
        """Calculate derived parameters after initialization"""
        self.H0 = 100.0 * self.h  # km/s/Mpc

    @property
    def Omega_m(self) -> float:
        """Total matter density parameter"""
        return self.Omega_b + self.Omega_c
       
    @property
    def Omega_r(self) -> float:
        """Dark energy density parameter"""
        return 1.0 - self.Omega_k - self.Omega_m - self.Omega_Lambda
    
    @property
    def H0_Mpc(self) -> float:
        """Hubble parameter today in Mpc^-1"""
        return self.H0 / (constants.c/1000)
    
    def z_to_a(self, z: float) -> float:
        """Scale factor as a function of redshift"""
        return 1.0 / (1.0 + z)
    
    def a_to_z(self, a: float) -> float:
        """Redshift as a function of scale factor"""
        return 1.0/a - 1.0
     
    def E(self, a: float) -> float:
        """Reduced Hubble parameter H(a)/H0"""
        exp_de = 3.0 * (1.0 + self.w + self.wa * (1.0 - a))
        E2 = self.Omega_m / a**3 \
             + self.Omega_Lambda / a**exp_de \
             + self.Omega_k / a**2 \
             + self.Omega_r / a**4
        return np.sqrt(E2)
    
    def H(self, a: float) -> float:
        """Hubble parameter at scale factor a in 1/Mpc"""
        return self.H0_Mpc * self.E(a)
    
    def dE_da(self, a: float) -> float:
        """Derivative of reduced Hubble parameter with respect to scale factor"""
        exp_de = 3.0 * (1.0 + self.w + self.wa * (1.0 - a))
        d = - 3.0*(self.Omega_b + self.Omega_c) / a**4 \
            - 2.0*self.Omega_k / a**3 \
            + (-3.0*self.wa*a + exp_de) * self.Omega_Lambda / a**(exp_de + 1) \
            - 4.0*self.Omega_r / a**5
        return d / (2.0 * self.E(a))
    
    def dH_dt(self, a: float) -> float:
        """Time derivative of Hubble parameter"""
        return a * self.H(a) * self.H0_Mpc * self.dE_da(a)
    
    def a_to_chi(self, a: float) -> float:
        """Comoving distance to scale factor a"""
        integrand = lambda a: 1.0 / (a**2 * self.H(a))
        result = integrate.quad(integrand, a, 1.0)[0]
        return result
    
    def z_to_chi(self, z: float) -> float:
        """Comoving distance to redshift z"""
        return self.a_to_chi(self.z_to_a(z))
    
    def _get_chi_interpolators(self, n_points: int = 1000):
        """Set up interpolation functions for faster calculations"""
        self.a_array = np.logspace(-4, 0, n_points)
        self.z_array = self.a_to_z(self.a_array)
        
        self.chi_array = np.array([self.a_to_chi(a) for a in self.a_array])
        
        chi_of_a = interpolate.interp1d(self.a_array, self.chi_array, 
                                kind='cubic', bounds_error=False, fill_value='extrapolate')
        a_of_chi = interpolate.interp1d(self.chi_array, self.a_array,
                                kind='cubic', bounds_error=False, fill_value='extrapolate')
        return (chi_of_a, a_of_chi)
    a_to_chi_interp, chi_to_a_interp = _get_chi_interpolators()
    
    def growth_factor(self, a: float) -> float:
        """Linear growth factor D(a)/a"""
        integrand = lambda a: 1.0 / (a * self.E(a))**3
        integral = integrate.quad(integrand, 0, a)[0]
        return 5.0/2.0 * self.Omega_m * self.E(a) * integral

    def Dpsi(self, a):
        """
        ~ Potential "growth function" from linear theory plus approximations
        Phi = Phi_prim * T(k) * (9/10 * D_1(a)/a) per Dodelson Eq. 7.5
        Dpsi is ("9/10" * D_1(a)/a)
        Dodelson Eq. 7.5, Eq. 7.32
        """
        y = a / self.a_eq
        fancy_9_10 = (16.0*np.sqrt(1.0 + y) + 9.0*y**3 + 2.0*y**2 - 8.0*y - 16.0) / (10.0*y**3)
        Dpsi = fancy_9_10 * self.growth_factor(a)
        return Dpsi

    def Dv(self, a):
        """
        Velocity growth function on superhorizon scales
        Dodelson 7.15 minus 7.16, v is 5.78
        v_i ~ - Dv d_i psi
        grad*v_i ~ k^2 * Dv * psi
        """
        y = a / self.a_eq
        Dv = 2.0 * (a**2) * self.H(a) \
                    / (self.Omega_m * self.H0**2) * y / (4.0 + 3.0*y) \
                * (self.Dpsi(a) + a * self.derv_Dpsi(a))
        return Dv

    def derv_Dpsi(self, a):
        """
        Derivative of the growth function with respect to the scale factor
        """
        y = a / self.a_eq
        E_a = self.E(a)
        dE_da = self.dE_da(a)
        fancy_9_10 = (16.0*np.sqrt(1.0 + y) + 9.0*y**3 + 2.0*y**2 - 8.0*y - 16.0) / (10.0*y**3)
        P1 = ((8.0/np.sqrt(1.0 + y) + 27.0*y**2 + 4.0*y - 8.0) / (self.a_eq*10.0*y**3)) * self.growth_factor(a)
        P2 = self.Dpsi(a) * (-4/a + dE_da/E_a)
        P3 = fancy_9_10 * (5.0/2.0) * self.Omega_m / (a**4 * E_a**2)

        derv_Dpsi = P1 + P2 + P3
        return derv_Dpsi

    def Tpsi(self, k):
        """
        'Best fitting' transfer function From Eq. 7.70 in Dodelson
        Assumes: no baryons, nonlinear effects, phi=psi always (9/10 -> 0.86), normal (no?) DE effects
        """
        fac = np.exp(-self.Omega_b*(1+np.sqrt(2*self.h)/self.Omega_m))
        k_eq = self.a_eq * self.H(self.a_eq)
        x = k / k_eq / fac
        x[np.where(x<1.0e-10)] = 1
        T = (np.log(1 + 0.171 * x) / (0.171 * x)) * (1 + 0.284 * x
        + (1.18 * x)**2 + (0.399 * x)**3 + (0.49 * x)**4)**(-0.25)

        return T

    @property
    def a_eq(self):
        """Scale factor at matter radiation equality"""
        return self.Omega_r / self.Omega_m

    def G_psz(self, a, ze, k, comps=['sw', 'dopp', 'isw']):
        """pSZ Integral kernel for SW effect"""
        chi_dec = self.z_to_chi(self.z_recomb)
        chi_e = self.z_to_chi_interp(ze)
        G_psz = 0

        if 'sw' in comps:
            G_SW_psz = -4 * np.pi * (2 * self.Dpsi(a)
                                    [0] - 3 / 2) * special.spherical_jn(2, k * (chi_dec - chi_e))
            G_psz += G_SW_psz

        if 'dopp' in comps:
            G_Dopp_psz = (4 * np.pi / 5) * k * self.Dv(a)[0] * (
                3 * special.spherical_jn(3, k * (chi_dec - chi_e)) - 2 * special.spherical_jn(1, k * (chi_dec - chi_e)))
            G_psz += G_Dopp_psz

        if 'isw' in comps:
            chi_a = self.a_to_chi_interp(a)
            delta_chi = chi_a - chi_e
            delta_chi[-1] = 0

            s2 = k[..., np.newaxis] * delta_chi
            integrand = special.spherical_jn(
                2, s2) * self.derv_Dpsi_inter(a)
            g_isw_psz = -8 * np.pi * integrate.simps(integrand, a)
            G_psz += g_isw_psz
            
        return G_psz

class Transfers:
    def psz_at_z(cosmo, ze, k, L):
        """PSZ transfer function for a given redshift"""
        if ze == 0.0:
            transfer_psz = np.zeros((len(k), len(L)), dtype=np.complex64)
            Ker = cosmo.G_psz(k, 0.0)
            Tk = cosmo.Tpsi(k)

            for l_id, l in enumerate(L):
                if l == 2:
                    c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                        np.sqrt(special.factorial(l + 2) /
                                special.factorial(l - 2))
                    transfer_psz[:, l_id] = c * (1. / 15.) * Tk * Ker
                else:
                    transfer_psz[:, l_id] = 0.0 + 1j * 0.0

        else:
            transfer_psz = np.zeros((len(k), len(L)), dtype=np.complex64)
            Ker = cosmo.G_psz(k, ze)
            Tk = cosmo.Tpsi(k)
            Chie = cosmo.a_to_chi_interp(cosmo.z_to_a(ze))

            for l_id, l in enumerate(L):
                if l < 2:
                    transfer_psz[:, l_id] = 0.0 + 1j * 0.0
                else:
                    c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                        np.sqrt(special.factorial(l + 2) /
                                special.factorial(l - 2))
                    transfer_psz[:, l] = c * (special.spherical_jn(l, k * Chie) / ((k * Chie)**2)) * Tk * Ker

            return transfer_psz

if __name__ == '__main__':
    import numpy as np
    from astropy.cosmology import Planck18 as ACosmology

    cosmo = Cosmology()

    astropy_cosmo = ACosmology

    # Define test function
    z_test = 2.0
    a_test = cosmo.z_to_a(z_test)
    
    H_test = cosmo.H(a_test)
    H_astropy = astropy_cosmo.H(z_test)/(constants.c/1000)
    print(f"Hubble parameter H at a={a_test}: Cosmology={H_test}, Astropy={H_astropy}")

    chi_z_test = cosmo.z_to_chi(z_test)
    chi_z_astropy = astropy_cosmo.comoving_distance(z_test).value  # in Mpc
    print(f"Comoving distance to z={z_test}: Cosmology={chi_z_test}, Astropy={chi_z_astropy}")

    growth_a_test = cosmo.growth_factor(a_test)
    print(f"Growth factor D(a)/a at a={a_test}: Cosmology={growth_a_test}")