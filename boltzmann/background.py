"""
Cosmological Background Evolution Calculator

This module implements calculations for cosmological background evolution,
including Hubble parameter, energy densities, and neutrino-related quantities.
"""

import numpy as np
from scipy.special import zeta, legendre, roots_legendre
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Optional

# Physical Constants
class constants:
    zeta_3: float = zeta(3)  # Riemann ζ(3) for phase space integrals
    km_s_Mpc_100: float = 2.1331196424576403e-33  # eV
    G_natural: float = 6.708830858490363e-57
    m_H: float = 9.382720881604904e8

class CosmoParams:
    """
    Cosmological parameters container
    
    Attributes:
        h (float): Dimensionless Hubble parameter (H0/100 km/s/Mpc)
        Omega_r (float): Radiation density parameter
        Omega_b (float): Baryon density parameter
        Omega_c (float): Cold dark matter density parameter
        N_nu (float): Effective number of neutrino species
        sum_m_nu (float): Sum of neutrino masses in eV
    """
    def __init__(self, h, Omega_r, Omega_b, Omega_c, N_nu, sum_m_nu, nq=15):
        self.h = h
        self.Omega_r = Omega_r
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.N_nu = N_nu
        self.sum_m_nu = sum_m_nu

        self.nq = nq

        self._validate()

        # some useful precomputations
        self.H0 = self.h * constants.km_s_Mpc_100
        self.rho_crit = (3 / (8 * np.pi)) * self.H0**2 / constants.G_natural
        self.T_gm = (15 / np.pi**2 * self.rho_crit * self.Omega_r)**(1/4)

        # neutrino
        self.T_nu = (self.N_nu/3)**(1/4)*(4/11)**(1/3) * (15/(np.pi**2)*self.rho_crit*self.Omega_r)**(1/4)
        nu_factor = (90 * constants.zeta_3 / (11 * np.pi**4)) * \
                   (self.Omega_r * self.h**2 / self.T_gm) * \
                   ((self.N_nu/3)**(3/4))  # N_nu/3 because we assume 1 massive neutrino
        self.Omega_nu = self.sum_m_nu * nu_factor / self.h**2
        self.Omega_lambda = 1 - (self.Omega_r*(1+(2/3)*(7/8)*(4/11)**(4/3)*self.N_nu) + \
                                 self.Omega_b + self.Omega_c + self.Omega_nu)
        self.neutrino = NeutrinoDistribution(self.T_nu)

        # misc: gauss-legendre quadrature weights 
        self._quad_pts, self._quad_wts = roots_legendre(self.nq)

    def _validate(self):
        """Validate parameters after initialization"""
        if any(param < 0 for param in [self.h, self.Omega_r, self.Omega_b, self.Omega_c, self.N_nu]):
            raise ValueError("Cosmological parameters must be non-negative")
        if self.sum_m_nu < 0:
            raise ValueError("Sum of neutrino masses must be non-negative")

    def H_a(self, a):
        """Hubble parameter at scale factor a"""
        rho_nu_0, _ = self.rho_P_nu_0(a)
        return self.H0 * ((self.Omega_c+self.Omega_b)*(a**-3) + \
                          self.Omega_r*(a**4)*(a**-4)*(1+(2/3)*(7/8)*(4/11)**(4/3)*self.N_nu) + \
                          rho_nu_0 / self.rho_crit + \
                          self.Omega_lambda)**0.5

    def H_conformal_a(self, a):
        return a * self.H_a(a)

    def H_x(self, x):
        return self.H_a(np.exp(x))

    def H_conformal_x(self, x):
        return np.exp(x) * self.H_a(np.exp(x))

    def eta_x(self, x):
        """Conformal time at log(a) = x"""
        logamin, logamax = -13.75, np.log10(np.exp(x))
        def I_eta(y):
            return 1.0 / (xq2q(y, logamin, logamax) * self.H_conformal_a(xq2q(y, logamin, logamax))) \
                   / dxdq(xq2q(y, logamin, logamax), logamin, logamax)
        return np.sum(I_eta(self._quad_pts) * self._quad_wts)

    def rho_P_nu_0(self, a):
        """
        Calculate massive neutrino metric perturbations.

        Returns:
            tuple: (rho, P) density and pressure

        """
        # Define integration limits
        log_q_min = np.log10(self.T_nu/30)
        log_q_max = np.log10(self.T_nu*30)
    
        m = self.sum_m_nu  # Using sum_m_nu instead of Σm_ν
    
        def eps_x(x, am):
            """Calculate epsilon(x) function"""
            q = xq2q(x, log_q_min, log_q_max)
            return np.sqrt(q**2 + am**2)
    
        def I_rho(x):
            """Integrand for density"""
            q = xq2q(x, log_q_min, log_q_max)
            dxdq_val = dxdq(q, log_q_min, log_q_max)
            return (q**2 * eps_x(x, a*m) * self.neutrino.f0(q) / dxdq_val)
    
        def I_P(x):
            """Integrand for pressure"""
            q = xq2q(x, log_q_min, log_q_max)
            dxdq_val = dxdq(q, log_q_min, log_q_max)
            return (q**4 / eps_x(x, a*m)) * self.neutrino.f0(q) / dxdq_val
    
        # Calculate integrals using quadrature
        rho = 4*np.pi * a**(-4) * np.sum(I_rho(self._quad_pts) * self._quad_wts)
        P = 4*np.pi/3 * a**(-4) * np.sum(I_P(self._quad_pts) * self._quad_wts)
    
        return rho, P

@dataclass
class NeutrinoDistribution:
    T_nu: float
    def f0(self, q: np.ndarray) -> np.ndarray:
        """Fermi-Dirac distribution for massless neutrinos"""
        gs = 2
        return gs / (2*np.pi)**3 / (np.exp(q/self.T_nu) + 1)
    def dlnf0_dlnq(self, q: np.ndarray) -> np.ndarray:
        """Logarithmic derivative of the distribution function"""
        return -q/self.T_nu / (1 + np.exp(-q/self.T_nu))


class BackgroundEvolution:
    """
    Handles the evolution of background cosmological quantities
    
    Args:
        params: Cosmological parameters
        x_grid: Grid in log(a) for interpolation
        nq: Number of quadrature points for momentum integration
    """
    
    def __init__(self, 
                 params: CosmoParams, 
                 x_grid: Optional[np.ndarray] = None):
        self.params = params
        
        if x_grid is None:
            x_grid = np.arange(-20.0, 0.01, 0.01)
        self.x_grid = x_grid
        
        # Initialize splines
        # Calculate values on grid
        H_conf_values = [self.params.H_conformal_x(x) for x in self.x_grid]
        eta_values = [self.params.eta_x(x) for x in self.x_grid]
        rho_nu_values = [self.params.rho_P_nu_0(np.exp(x))[0] for x in self.x_grid]
        
        # Create base splines
        self.H_conf = CubicSpline(self.x_grid, H_conf_values)
        self.eta = CubicSpline(self.x_grid, eta_values)
        self.rho_nu = CubicSpline(self.x_grid, rho_nu_values)
        
        # Create derivative splines
        self.H_conf_p = CubicSpline((x:=self.H_conf.x), self.H_conf(x, 1))
        self.H_conf_pp = CubicSpline((x:=self.H_conf.x), self.H_conf(x, 2))
        self.eta_p = CubicSpline((x:=self.eta.x), self.eta(x, 1))
        self.eta_pp = CubicSpline((x:=self.eta.x), self.eta(x, 2))
    
# utils
def to_ui(lq, lqmi, lqma):
    """Maps lq to the range [-1, 1] based on lqmi and lqma."""
    return -1 + (2 / (lqma - lqmi)) * (lq - lqmi)

def from_ui(x, lqmi, lqma):
    """Maps x from the range [-1, 1] back to the original range based on lqmi and lqma."""
    return lqmi + ((lqma - lqmi) / 2) * (x + 1)

def dxdq(q, logqmin, logqmax):
    """Calculates dx/dq, where x is the scaled coordinate in [-1, 1] and q is the original coordinate."""
    return (1 + to_ui(1 + logqmin, logqmin, logqmax)) / (q * np.log(10))

def xq2q(x, logqmin, logqmax):
    """Maps x (in the range [-1, 1]) back to q using from_ui."""
    return 10.0 ** from_ui(x, logqmin, logqmax)