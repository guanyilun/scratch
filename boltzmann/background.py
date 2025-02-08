"""
Cosmological Background Evolution Calculator

This module implements calculations for cosmological background evolution,
including Hubble parameter, energy densities, and neutrino-related quantities.
"""

#%%
import numpy as np
from scipy.special import zeta, roots_legendre
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from dataclasses import dataclass
from typing import Optional, Any
from numba import njit


class constants:
    zeta_3: float = zeta(3)  # Riemann ζ(3) for phase space integrals
    km_s_Mpc_100: float = 0.00033356409519815205  # Mpc^-1
    G_natural: float = 2.7435866787007285e-115  # Mpc^-2
    eV_natural: float = 1.5637383059878979e29  # [eV -> Mpc^-1]

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
        m_axion (float): Axion mass in eV
        f_axion (float): Axion fraction of dark matter
    """
    def __init__(self, h, Omega_r, Omega_b, Omega_c, N_nu, sum_m_nu, m_axion, f_axion, nq=15, x_grid=None):
        self.h = h
        self.Omega_r = Omega_r
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.N_nu = N_nu
        self.sum_m_nu = sum_m_nu * constants.eV_natural
        self.m_axion = m_axion * constants.eV_natural
        self.f_axion = f_axion

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

        # Axion parameters
        self.maxion_twiddle = self.m_axion / self.H0  # dimensionless axion mass
        self.Omega_axion = self.f_axion * (self.Omega_c + self.Omega_b)
        self.a_osc = None  # Will be set during evolution
        
        # Total matter and lambda
        self.Omega_m = self.Omega_b + self.Omega_c + self.Omega_axion
        self.Omega_lambda = 1 - (self.Omega_r*(1+(2/3)*(7/8)*(4/11)**(4/3)*self.N_nu) + \
                                 self.Omega_m + self.Omega_nu)

        self.quad_pts, self.quad_wts = roots_legendre(nq)
        self.neutrino = NeutrinoDistribution(self.T_nu, self.sum_m_nu, self.quad_pts, self.quad_wts)

        if x_grid is None:
            x_grid = np.arange(-20.0, 0.01, 0.01)
        self.x_grid = x_grid

        self._validate()

    def _validate(self):
        """Validate parameters after initialization"""
        if any(param < 0 for param in [self.h, self.Omega_r, self.Omega_b, self.Omega_c, self.N_nu]):
            raise ValueError("Cosmological parameters must be non-negative")
        if self.sum_m_nu < 0:
            raise ValueError("Sum of neutrino masses must be non-negative")
        if self.m_axion < 0:
            raise ValueError("Axion mass must be non-negative")
        if not 0 <= self.f_axion <= 1:
            raise ValueError("Axion fraction must be between 0 and 1")

    def H_a(self, a, phi_twiddle=None, phi_dot_twiddle=None):
        """Hubble parameter at scale factor a"""
        rho_nu_0, _ = self.neutrino.rho_P_nu_0(a)
        
        # Add axion contribution if field values provided
        Omega_axion = 0
        if phi_twiddle is not None and phi_dot_twiddle is not None:
            Omega_axion = self._compute_omega_axion(a, phi_twiddle, phi_dot_twiddle)

        return self.H0 * ((self.Omega_c+self.Omega_b)*(a**-3) + \
                          self.Omega_r*(a**-4)*(1+(2/3)*(7/8)*(4/11)**(4/3)*self.N_nu) + \
                          rho_nu_0 / self.rho_crit + \
                          Omega_axion + self.Omega_lambda)**0.5

    def _axion_derivs(self, x, v):
        """Compute derivatives for axion field equations"""
        a = np.exp(x)
        m_a = self.maxion_twiddle
        
        H = self.H_a(a, v[0], v[1])
        h_conf = a * H / constants.km_s_Mpc_100
        
        dphi_dloga = v[1] * self.h / h_conf
        dphi_dot_dloga = -2.0 * v[1] - m_a**2 * a**2 * self.h * v[0] / h_conf
        
        return [dphi_dloga, dphi_dot_dloga]

    def _compute_omega_axion(self, a, phi_twiddle, phi_dot_twiddle):
        """Compute axion energy density"""
        # axionCAMB convention - v1, v2 carry an extra h: phi / m_pl = sqrt(6) v[0]/ h
        return ((phi_dot_twiddle/a)**2 + (self.maxion_twiddle*phi_twiddle)**2)/self.h**2
        # my convention: phi / m_pl = sqrt(6) v[0]
        # return ((phi_dot_twiddle/a)**2 + (self.maxion_twiddle*phi_twiddle)**2)

    def _get_axion_fraction(self, vtwiddle_init):
        """
        Get resulting axion fraction for a given initial field value
        Returns:
            tuple: (f_out, a_osc) - achieved fraction and oscillation scale factor
        """
        dfac = 10.0
        y0 = [vtwiddle_init, 0.0]

        def detect_transition(t, y):
            a = np.exp(t)
            H = self.H_a(a, y[0], y[1])
            return self.maxion_twiddle - dfac * H / self.H0
        detect_transition.terminal = True
        detect_transition.direction = 1

        sol = solve_ivp(
            self._axion_derivs,
            (self.x_grid[0], self.x_grid[-1]),
            y0,
            events=detect_transition,
            rtol=1e-8,
            atol=1e-10
        )

        # Get values at final time
        a_final = np.exp(sol.t[-1])
        phi_twiddle_final = sol.y[0][-1]
        phi_dot_twiddle_final = sol.y[1][-1]

        # Compute axion fraction
        Omega_axion = self._compute_omega_axion(a_final, phi_twiddle_final, phi_dot_twiddle_final)
        Omega_m = (self.Omega_b + self.Omega_c) * a_final**(-3)
        f_out = Omega_axion / (Omega_axion + Omega_m)

        # Get oscillation scale factor
        a_osc = np.exp(sol.t_events[0][0]) if len(sol.t_events[0]) > 0 else a_final

        return f_out, a_osc


    def _compute_initial_conditions(self):
        """Compute initial conditions for axion evolution"""
        # Initial scale factor estimates
        a_matter = (self.Omega_m / self.maxion_twiddle**2)**(1/3)
        a_rad = (self.Omega_r / self.maxion_twiddle**2)**(1/4)
        a_m = self.Omega_r / self.Omega_m
        a_rel = 10.0
        a_lambda = (self.Omega_r / self.Omega_lambda)**(1/4)
        a_scalar = (self.Omega_axion / self.maxion_twiddle**2)**(1/3)
        
        # Find safe initial a
        a_init = min(a_rel, a_lambda, a_m, a_matter, a_rad, a_scalar) * 1e-7
        
        # Get initial brackets for field value
        vtwiddle_estimates = [
            np.sqrt(self.f_axion),
            np.sqrt(self.f_axion * self.Omega_m / 
                   (np.sqrt(self.maxion_twiddle) * self.Omega_r**0.75)),
            np.sqrt(self.f_axion * self.Omega_m / 
                   (np.sqrt(self.maxion_twiddle) * self.Omega_r))
        ]
        vtwiddle_min = min(vtwiddle_estimates) / 100
        vtwiddle_max = max(vtwiddle_estimates) * 100

        # Use root finder to determine correct initial conditions
        def target(log_vtwiddle):
            f_out, _ = self._get_axion_fraction(np.exp(log_vtwiddle))
            return np.log(f_out/self.f_axion)

        result = root_scalar(
            target,
            bracket=(np.log(vtwiddle_min), np.log(vtwiddle_max)),
            method='brentq'
        )
        vtwiddle_init = np.exp(result.root)
        
        return a_init, vtwiddle_init

    def solve_axion_evolution(self):
        """Solve axion field evolution"""
        
        # Get initial conditions
        a_init, vtwiddle_init = self._compute_initial_conditions()
        y0 = [vtwiddle_init, 0.0]  # Start at rest
        
        # fluid approximation
        # dfac = 10.0  # Transition parameter for m=dfac*H
        # def detect_transition(t, y):
        #     """Detect when m = dfac*H"""
        #     a = np.exp(t)
        #     H = self.H_a(a, y[0], y[1])
        #     return self.maxion_twiddle - dfac * H / self.H0
        # detect_transition.terminal = True
        # detect_transition.direction = 1
        
        # Solve until transition
        sol = solve_ivp(
            self._axion_derivs, 
            (np.log(a_init), self.x_grid[-1]),
            y0,
            method='DOP853',  # 8th order Runge-Kutta
            t_eval=self.x_grid,
            # events=detect_transition,
            rtol=1e-5,
        )
        phi_twiddle = sol.y[0]      # \tilde{v}_1
        phi_dot_twiddle = sol.y[1]  # \tilde{v}_2
        a_grid = np.exp(self.x_grid)
        h_conf = a_grid * self.H_a(a_grid, phi_twiddle, phi_dot_twiddle) / constants.km_s_Mpc_100
        c_ad2 = 1 + (2*self.maxion_twiddle**2*a_grid**2*phi_twiddle * self.h) / (3*h_conf*phi_dot_twiddle)
        w_ax = (phi_dot_twiddle**2/a_grid**2 - self.maxion_twiddle**2*phi_twiddle**2) / \
               (phi_dot_twiddle**2/a_grid**2 + self.maxion_twiddle**2*phi_twiddle**2)

        # fluid approximation: not needed for now

        # phi_twiddle = np.zeros_like(self.x_grid)
        # phi_dot_twiddle = np.zeros_like(self.x_grid)
        # Find transition point
        # transition_idx = len(self.x_grid)
        # if len(sol.t_events[0]) > 0:
        #     self.a_osc = np.exp(sol.t_events[0][0])
        #     transition_idx = np.searchsorted(self.x_grid, np.log(self.a_osc))
        #     print(f"Axion oscillation begins at a = {self.a_osc:.2e}")
        # else:
        #     print("Warning: No oscillation transition detected")
        #     self.a_osc = np.exp(self.x_grid[-1])
        
        # # Fill pre-oscillation evolution
        # phi_twiddle[:transition_idx] = sol.y[0][:transition_idx]
        # phi_dot_twiddle[:transition_idx] = sol.y[1][:transition_idx]
        
        # # Compute quantities at oscillation transition
        # if transition_idx < len(self.x_grid):
        #     # Get values just before transition
        #     phi_osc = phi_twiddle[transition_idx-1]
        #     phi_dot_osc = phi_dot_twiddle[transition_idx-1]
            
        #     # Compute energy density at oscillation (needed for WKB)
        #     Omega_ax_osc = self._compute_omega_axion(self.a_osc, phi_osc, phi_dot_osc)
            
        #     # Monitor oscillation parameters
        #     H_osc = self.H_a(self.a_osc, phi_osc, phi_dot_osc)
        #     print(f"At oscillation:")
        #     print(f"  H/H0 = {H_osc/self.H0:.2e}")
        #     print(f"  m/(dfac*H) = {self.maxion_twiddle/(dfac*H_osc/self.H0):.2f}")
        #     print(f"  phi = {phi_osc:.2e}")
        #     print(f"  phi_dot = {phi_dot_osc:.2e}")
            
        #     # Fill post-oscillation evolution using WKB approximation
        #     a_values = np.exp(self.x_grid[transition_idx:])
            
        #     # Density scales as a^-3 after oscillation
        #     Omega_ax_a = Omega_ax_osc * (a_values/self.a_osc)**(-3)
                
        #     # Reconstruct field values from density
        #     # Use WKB solution phi ∝ a^(-3/2) cos(m*t)
        #     # We average over oscillations for background quantities
        #     omega = self.maxion_twiddle * a_values  # Physical frequency
        #     phi_amp = np.sqrt(2*Omega_ax_a/(omega**2)) / self.h  # Amplitude from energy
                
        #     phi_twiddle[transition_idx:] = phi_amp  # Store amplitude
        #     phi_dot_twiddle[transition_idx:] = omega * phi_amp  # Store average derivative
        #     print(f"Final axion fraction = {self.f_axion:.3f}")
        return {
            'x': x_grid,
            'a': a_grid,
            'v1': phi_twiddle,
            'v2': phi_dot_twiddle,
            'c_ad2': c_ad2,
            'w_ax': w_ax
        }

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
        return np.sum(I_eta(self.quad_pts) * self.quad_wts)


@dataclass
class NeutrinoDistribution:
    T_nu: float
    sum_m_nu: float
    quad_pts: Any
    quad_wts: Any

    def __post_init__(self):
        log_q_min = np.log10(self.T_nu/30)
        log_q_max = np.log10(self.T_nu*30)
        
        q = xq2q(self.quad_pts, log_q_min, log_q_max)
        dxdq_val = dxdq(q, log_q_min, log_q_max)
        self.I_rho_part = (q**2 * self.f0(q) / dxdq_val)
        self.I_P_part = q**4 * self.f0(q) / dxdq_val 
        self.q = q

    def f0(self, q: np.ndarray) -> np.ndarray:
        """Fermi-Dirac distribution for massless neutrinos"""
        gs = 2
        return gs / (2*np.pi)**3 / (np.exp(q/self.T_nu) + 1)

    def dlnf0_dlnq(self, q: np.ndarray) -> np.ndarray:
        """Logarithmic derivative of the distribution function"""
        return -q/self.T_nu / (1 + np.exp(-q/self.T_nu))

    def rho_P_nu_0(self, a):
        """
        Calculate massive neutrino metric perturbations.

        Returns:
            tuple: (rho, P) density and pressure

        """
        if isinstance(a, np.ndarray):
            return _rho_P_nu_0_vec(a, self.sum_m_nu, self.q, 
                            self.I_rho_part, self.I_P_part, 
                            self.quad_wts)
        else:
            return _rho_P_nu_0(a, self.sum_m_nu, self.q, 
                            self.I_rho_part, self.I_P_part, 
                            self.quad_wts)       


# optimized function
@njit
def _rho_P_nu_0(a, sum_m_nu, q, I_rho_part, I_P_part, quad_wts):
    """
    JIT-compiled routine to compute the neutrino density and pressure.
    
    Parameters:
      a         : scale factor (float)
      sum_m_nu  : neutrino mass sum (float)
      q         : pre-computed quadrature q-points (1D np.array)
      I_rho_part: precomputed array for the density integrand (q**2*f0/dxdq)
      I_P_part  : precomputed array for the pressure integrand (q**4*f0/dxdq)
      quad_wts  : quadrature weights (1D np.array)
      
    Returns:
      (rho, P) : tuple with density and pressure.
    """
    am = a * sum_m_nu
    n = q.shape[0]
    s_rho = 0.0
    s_P   = 0.0
    for i in range(n):
        eps = np.sqrt(q[i]*q[i] + am*am)
        s_rho += I_rho_part[i] * eps * quad_wts[i]
        s_P   += I_P_part[i] / eps * quad_wts[i]
    rho = 4.0 * np.pi * a**(-4) * s_rho
    P   = (4.0 * np.pi / 3.0) * a**(-4) * s_P
    return rho, P


@njit
def _rho_P_nu_0_vec(a, sum_m_nu, q, I_rho_part, I_P_part, quad_wts):
    """
    Vectorized version of rhoP_nu_0 over array 'a' of any shape.
    
    Parameters:
      a         : array of scale factors (any shape)
      sum_m_nu  : neutrino mass sum (float)
      q         : pre-computed quadrature q-points (1D np.array)
      I_rho_part: precomputed density integrand (q²*f0/dxdq)
      I_P_part  : precomputed pressure integrand (q⁴*f0/dxdq)
      quad_wts  : quadrature weights (1D np.array)
      
    Returns:
      (rho, P) : tuple of density and pressure arrays with the same shape as 'a'
    """
    # Flatten the input array 'a' to handle any shape
    a_flat = a.ravel()
    n = len(a_flat)
    
    # Initialize output arrays
    rho_flat = np.empty(n, dtype=np.float64)
    P_flat = np.empty(n, dtype=np.float64)
    
    # Compute rho and P for each element in the flattened array
    for i in range(n):
        a_i = a_flat[i]
        rho, P = _rho_P_nu_0(a_i, sum_m_nu, q, I_rho_part, I_P_part, quad_wts)
        rho_flat[i] = rho
        P_flat[i] = P
    
    # Reshape the output arrays to match the input shape of 'a'
    rho = rho_flat.reshape(a.shape)
    P = P_flat.reshape(a.shape)
    
    return rho, P

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
        # Solve axion evolution first
        _, self.phi, self.phi_dot = self.params.solve_axion_evolution(self.x_grid)
        
        # Calculate Hubble values including axion contribution
        H_conf_values = []
        for i, x in enumerate(self.x_grid):
            a = np.exp(x)
            H_conf_values.append(a * self.params.H_a(a, self.phi[i], self.phi_dot[i]))
        
        eta_values = [self.params.eta_x(x) for x in self.x_grid]
        
        # Create base splines
        # Use natural boundary conditions (second derivative = 0)
        self.H_conf = CubicSpline(self.x_grid, H_conf_values, bc_type='natural')
        self.eta = CubicSpline(self.x_grid, eta_values, bc_type='natural')
        
        # Create derivative splines
        self.H_conf_p = CubicSpline(self.x_grid, np.array([
            self.H_conf(x, 1) for x in self.x_grid
        ]), bc_type='natural')
        
        self.H_conf_pp = CubicSpline(self.x_grid, np.array([
            self.H_conf(x, 2) for x in self.x_grid
        ]), bc_type='natural')
        
        self.eta_p = CubicSpline(self.x_grid, np.array([
            self.eta(x, 1) for x in self.x_grid
        ]), bc_type='natural')
        
        # Calculate and spline axion energy density
        rho_axion = []
        for i, x in enumerate(self.x_grid):
            a = np.exp(x)
            rho = ((self.phi_dot[i]/a)**2 + 
                  (self.params.maxion_twiddle*self.phi[i]*a)**2) * self.params.h**2
            rho_axion.append(rho)
        self.rho_axion = CubicSpline(self.x_grid, rho_axion)
        
        # Store useful quantities for perturbation calculations
        if self.params.a_osc is not None:
            print(f"Axion oscillation begins at a = {self.params.a_osc:.2e}")
            print(f"Axion fraction achieved: {self.params.f_axion:.3f}")

    
# utils
@njit
def to_ui(lq, lqmi, lqma):
    """Maps lq to the range [-1, 1] based on lqmi and lqma."""
    return -1 + (2 / (lqma - lqmi)) * (lq - lqmi)

@njit
def from_ui(x, lqmi, lqma):
    """Maps x from the range [-1, 1] back to the original range based on lqmi and lqma."""
    return lqmi + ((lqma - lqmi) / 2) * (x + 1)

@njit
def dxdq(q, logqmin, logqmax):
    """Calculates dx/dq, where x is the scaled coordinate in [-1, 1] and q is the original coordinate."""
    return (1 + to_ui(1 + logqmin, logqmin, logqmax)) / (q * np.log(10))

@njit
def xq2q(x, logqmin, logqmax):
    """Maps x (in the range [-1, 1]) back to q using from_ui."""
    return 10.0 ** from_ui(x, logqmin, logqmax)


if __name__ == '__main__':
    import numpy as np

    # Create cosmological parameters with axion
    x_grid = np.linspace(-14, 0, 10000)  # matches ntable=5000 in Fortran
    params = CosmoParams(
        h=0.67,
        Omega_r=5e-5,
        Omega_b=0.022,
        Omega_c=0.119,
        N_nu=3.046,
        sum_m_nu=0.06,    # eV
        m_axion=1e-32,    # eV
        f_axion=0.01,      # 10% of dark matter is axions
        x_grid=x_grid
    )
    #%%
    res = params.solve_axion_evolution()

    # %%
    # import matplotlib.pyplot as plt
    # plt.plot(res['x'], res['v1'], label='v1')
    # plt.plot(res['x'], res['v2'], label='v2')
    # plt.legend()
    # plt.plot(res['x'], res['w_ax'])
    # plt.xlim(left=-4, right=0)
    # plt.plot(res['x'][:8000], res['c_ad2'][:8000])