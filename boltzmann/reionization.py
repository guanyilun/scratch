from typing import Protocol, Any, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline


# Unit conversion constants
H0_NATURAL_UNIT_CONVERSION = 6.582119569509066e-16
KELVIN_NATURAL_UNIT_CONVERSION = 11604.51812155008

class constants_nu:
    sigma_T = 1.7084774406497884e-15
    m_H = 9.382720881604904e8
    m_e = 510998.9499961642

@dataclass
class IonizationHistory:
    """Python equivalent of Julia's IonizationHistory struct"""
    tau_0: float  # Optical depth
    X_e: Any      # Electron fraction interpolation
    tau: Any      # Optical depth interpolation
    tau_p: Any    # First derivative of optical depth
    tau_pp: Any  # Second derivative of optical depth
    g_tilde: Any     # Visibility function interpolation
    g_tilde_p: Any    # First derivative of visibility function
    g_tilde_pp: Any  # Second derivative of visibility function
    Tmat: Any  # Matter temperature interpolation
    csb2: Any  # Sound speed squared interpolation


@dataclass
class RECFAST:
    """Python equivalent of Julia's RECFAST struct"""
    bg: Any  # Background evolution
    
    # Fundamental constants in SI units
    C: float = field(default=2.99792458e8)
    k_B: float = field(default=1.380658e-23)
    h_P: float = field(default=6.6260755e-34)
    m_e: float = field(default=9.1093897e-31)
    m_H: float = field(default=1.673575e-27)  # av. H atom
    not4: float = field(default=3.9715e0)  # mass He/H atom
    sigma: float = field(default=6.6524616e-29)
    a: float = field(default=7.565914e-16)
    G: float = field(default=6.6742e-11)

    # Additional constants
    Lambda: float = field(default=8.2245809e0)
    Lambda_He: float = field(default=51.3e0)
    L_H_ion: float = field(default=1.096787737e7)
    L_H_alpha: float = field(default=8.225916453e6)
    L_He1_ion: float = field(default=1.98310772e7)
    L_He2_ion: float = field(default=4.389088863e7)
    L_He_2s: float = field(default=1.66277434e7)
    L_He_2p: float = field(default=1.71134891e7)

    # C2 photon rates and atomic levels
    A2P_s: float = field(default=1.798287e9)
    A2P_t: float = field(default=177.58e0)
    L_He_2Pt: float = field(default=1.690871466e7)
    L_He_2St: float = field(default=1.5985597526e7)
    L_He2St_ion: float = field(default=3.8454693845e6)
    sigma_He_2Ps: float = field(default=1.436289e-22)
    sigma_He_2Pt: float = field(default=1.484872e-22)

    # Gaussian fit parameters
    AGauss1: float = field(default=-0.14e0)
    AGauss2: float = field(default=0.079e0)
    zGauss1: float = field(default=7.28e0)
    zGauss2: float = field(default=6.73e0)
    wGauss1: float = field(default=0.18e0)
    wGauss2: float = field(default=0.33e0)

    # PPB fitting parameters for Hydrogen
    a_PPB: float = field(default=4.309)
    b_PPB: float = field(default=-0.6166)
    c_PPB: float = field(default=0.6703)
    d_PPB: float = field(default=0.5300)

    # VF fitting parameters for Helium
    a_VF: float = field(default=10**(-16.744))
    b_VF: float = field(default=0.711)
    T_0: float = field(default=10**(0.477121))
    T_1: float = field(default=10**5.114)

    # HeI triplet fitting parameters
    a_trip: float = field(default=10**(-16.306))
    b_trip: float = field(default=0.761)

    # Matter departs from radiation when t(Th) > H_frac * t(H)
    H_frac: float = field(default=1e-3)

    # Switches
    Hswitch: int = field(default=1)
    Heswitch: int = field(default=6)

    # Cosmology parameters
    Yp: float = field(default=0.24)
    OmegaB: float = field(default=0.046)

    def __post_init__(self):
        # Set H₀ from background
        self.HO = self.bg.params.H0 / H0_NATURAL_UNIT_CONVERSION
        
        # Set OmegaG and Tnow
        self.OmegaG = 5.0469e-5
        self.Tnow = (15 / (np.pi**2) * self.bg.params.rho_crit * self.OmegaG)**(1/4) * KELVIN_NATURAL_UNIT_CONVERSION

        # Helium abundance parameters
        self.mu_H = 1 / (1 - self.Yp)  # Mass per H atom
        self.mu_T = self.not4 / (self.not4 - (self.not4 - 1) * self.Yp)  # Mass per atom
        self.fHe = self.Yp / (self.not4 * (1 - self.Yp))  # n_He_tot / n_H_tot

        # Additional derived constants
        self.Nnow = 3 * self.HO * self.HO * self.OmegaB / (8 * np.pi * self.G * self.mu_H * self.m_H)
        self.fu = 1.14 if self.Hswitch == 0 else 1.125
        self.b_He = 0.86  # He fudge factor
        self.tol = 1e-8

        # Constants that don't need to be calculated later
        self.Lalpha = 1 / self.L_H_alpha
        self.Lalpha_He = 1 / self.L_He_2p
        self.DeltaB = self.h_P * self.C * (self.L_H_ion - self.L_H_alpha)
        self.CDB = self.DeltaB / self.k_B
        self.DeltaB_He = self.h_P * self.C * (self.L_He1_ion - self.L_He_2s)  # 2s, not 2p
        self.CDB_He = self.DeltaB_He / self.k_B
        self.CB1 = self.h_P * self.C * self.L_H_ion / self.k_B
        self.CB1_He1 = self.h_P * self.C * self.L_He1_ion / self.k_B  # ionization for HeI
        self.CB1_He2 = self.h_P * self.C * self.L_He2_ion / self.k_B  # ionization for HeII
        self.CR = 2 * np.pi * (self.m_e / self.h_P) * (self.k_B / self.h_P)
        self.CK = self.Lalpha**3 / (8 * np.pi)
        self.CK_He = self.Lalpha_He**3 / (8 * np.pi)
        self.CL = self.C * self.h_P / (self.k_B * self.Lalpha)
        self.CL_He = self.C * self.h_P / (self.k_B / self.L_He_2s)  # comes from det.bal. of 2s-1s
        self.CT = (8/3) * (self.sigma / (self.m_e * self.C)) * self.a
        self.Bfact = self.h_P * self.C * (self.L_He_2p - self.L_He_2s) / self.k_B

    # def recfast_init(self, z: float) -> Tuple[float, float, float]:
    #     """Initialize RECFAST calculations at a given redshift"""
    #     if z > 8000.:
    #         x_H0 = 1.
    #         x_He0 = 1.
    #         x0 = 1. + 2 * self.fHe
    #     elif z > 3500.:
    #         x_H0 = 1.
    #         x_He0 = 1.
    #         rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
    #                     self.CB1_He2 / (self.Tnow * (1 + z))) / self.Nnow
    #         rhs = rhs * 1.  # ratio of g's is 1 for He++ <-> He+
    #         x0 = 0.5 * (np.sqrt((rhs - 1 - self.fHe)**2 + 
    #                            4 * (1 + 2 * self.fHe) * rhs) - (rhs - 1 - self.fHe))
    #     elif z > 2000.:
    #         x_H0 = 1.
    #         rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
    #                     self.CB1_He1 / (self.Tnow * (1 + z))) / self.Nnow
    #         rhs = 4 * rhs  # ratio of g's is 4 for He+ <-> He0
    #         x_He0 = 0.5 * (np.sqrt((rhs - 1)**2 + 4 * (1 + self.fHe) * rhs) - (rhs - 1))
    #         x0 = x_He0
    #         x_He0 = (x0 - 1.) / self.fHe
    #     else:
    #         rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
    #                     self.CB1 / (self.Tnow * (1 + z))) / self.Nnow
    #         x_H0 = 0.5 * (np.sqrt(rhs**2 + 4 * rhs) - rhs)
    #         x_He0 = 0.
    #         x0 = x_H0
    #     return x_H0, x_He0, x0

    def ion_recfast(self, y: np.ndarray, z: float) -> np.ndarray:
        """Calculate RECFAST derivatives at a particular redshift"""
        x_H = y[0]
        x_He = y[1]
        x = x_H + self.fHe * x_He
        Tmat = y[2]

        n = self.Nnow * (1 + z)**3
        n_He = self.fHe * self.Nnow * (1 + z)**3
        Trad = self.Tnow * (1 + z)

        # Scale factor and Hubble parameter
        a = 1 / (1 + z)
        x_a = self.a2x(a)
        Hz = self.bg.H_conf(x_a) / a / H0_NATURAL_UNIT_CONVERSION
        dHdz = (-self.bg.H_conf_p(x_a) + self.bg.H_conf(x_a)) / H0_NATURAL_UNIT_CONVERSION

        # Get radiative rates using PPQ fit
        Rdown = 1e-19 * self.a_PPB * (Tmat / 1e4)**self.b_PPB / (
            1. + self.c_PPB * (Tmat / 1e4)**self.d_PPB)
        Rup = Rdown * (self.CR * Tmat)**(1.5) * np.exp(-self.CDB / Tmat)

        # Calculate He using Verner & Ferland type formula
        sq_0 = np.sqrt(Tmat / self.T_0)
        sq_1 = np.sqrt(Tmat / self.T_1)
        
        Rdown_He = self.a_VF / (sq_0 * (1 + sq_0)**(1 - self.b_VF))
        Rdown_He = Rdown_He / (1 + sq_1)**(1 + self.b_VF)
        Rup_He = Rdown_He * (self.CR * Tmat)**(1.5) * np.exp(-self.CDB_He / Tmat)
        Rup_He = 4. * Rup_He  # statistical weights factor for HeI

        # Handle potential overflow
        He_Boltz = np.exp(min(680., self.Bfact / Tmat))

        # Deal with H and its fudges
        if self.Hswitch == 0:
            K = self.CK / Hz
        else:
            # Fit a double Gaussian correction function
            K = self.CK / Hz * (1.0 + 
                self.AGauss1 * np.exp(-((np.log(1 + z) - self.zGauss1) / self.wGauss1)**2) +
                self.AGauss2 * np.exp(-((np.log(1 + z) - self.zGauss2) / self.wGauss2)**2))

        # Add the HeI part
        Rdown_trip = self.a_trip / (sq_0 * (1 + sq_0)**(1 - self.b_trip))
        Rdown_trip = Rdown_trip / ((1 + sq_1)**(1 + self.b_trip))
        Rup_trip = Rdown_trip * np.exp(-self.h_P * self.C * self.L_He2St_ion / (self.k_B * Tmat))
        Rup_trip = Rup_trip * ((self.CR * Tmat)**1.5) * (4/3)

        # Handle He flag conditions
        if (x_He < 5.e-9) or (x_He > 0.980):
            Heflag = 0
        else:
            Heflag = self.Heswitch

        if Heflag == 0:
            K_He = self.CK_He / Hz
        else:
            tauHe_s = self.A2P_s * self.CK_He * 3 * n_He * (1 - x_He) / Hz
            pHe_s = (1 - np.exp(-tauHe_s)) / tauHe_s
            K_He = 1 / (self.A2P_s * pHe_s * 3 * n_He * (1 - x_He))

            if ((Heflag == 2) or (Heflag >= 5)) and (x_H < 0.9999999):
                # Use fitting formula for continuum opacity of H
                Doppler = 2 * self.k_B * Tmat / (self.m_H * self.not4 * self.C * self.C)
                Doppler = self.C * self.L_He_2p * np.sqrt(Doppler)
                gamma_2Ps = 3 * self.A2P_s * self.fHe * (1 - x_He) * self.C * self.C / (
                    np.sqrt(np.pi) * self.sigma_He_2Ps * 8 * np.pi * Doppler * (1 - x_H)
                ) / ((self.C * self.L_He_2p)**2)
                
                pb = 0.36
                qb = self.b_He
                AHcon = self.A2P_s / (1 + pb * (gamma_2Ps**qb))
                K_He = 1 / ((self.A2P_s * pHe_s + AHcon) * 3 * n_He * (1 - x_He))

        # Calculate the derivatives
        f1, f2, f3 = 0., 0., 0.

        # Calculate Thomson and Hubble times
        timeTh = (1 / (self.CT * Trad**4)) * (1 + x + self.fHe) / x
        timeH = 2 / (3 * self.HO * (1 + z)**1.5)

        # Calculate derivatives based on conditions
        if x_H > 0.99:
            f1 = 0.
        elif x_H > 0.985:
            f1 = (x * x_H * n * Rdown - Rup * (1 - x_H) * np.exp(-self.CL / Tmat)) / (Hz * (1 + z))
        else:
            f1 = ((x * x_H * n * Rdown - Rup * (1.0 - x_H) * np.exp(-self.CL / Tmat)) *
                  (1.0 + K * self.Lambda * n * (1.0 - x_H))) / (
                      Hz * (1.0 + z) * (1.0 / self.fu + K * self.Lambda * n * (1.0 - x_H) / self.fu +
                                       K * Rup * n * (1.0 - x_H)))

        if x_He < 1e-15:
            f2 = 0.
        else:
            f2 = ((x * x_He * n * Rdown_He - Rup_He * (1 - x_He) * np.exp(-self.CL_He / Tmat)) *
                  (1 + K_He * self.Lambda_He * n_He * (1 - x_He) * He_Boltz)) / (
                      Hz * (1 + z) * (1 + K_He * (self.Lambda_He + Rup_He) * n_He * (1 - x_He) * He_Boltz))

        # Calculate matter temperature evolution
        if timeTh < self.H_frac * timeH:
            epsilon = Hz * (1 + x + self.fHe) / (self.CT * Trad**3 * x)
            f3 = (self.Tnow + epsilon * ((1 + self.fHe) / (1 + self.fHe + x)) *
                  ((f1 + self.fHe * f2) / x) - epsilon * dHdz / Hz + 3 * epsilon / (1 + z))
        else:
            f3 = (self.CT * (Trad**4) * x / (1 + x + self.fHe) * (Tmat - Trad) / 
                  (Hz * (1 + z)) + 2 * Tmat / (1 + z))

        return np.array([f1, f2, f3])

    @staticmethod
    def a2x(a: float) -> float:
        """Convert scale factor to x coordinate"""
        return np.log(a)

    def Xe_He_evolution(self, z: float, sol: Any) -> float:
        """Calculate He evolution at a given redshift"""
        y = sol(z)[0]  # First component of solution
        rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
                    self.CB1 / (self.Tnow * (1 + z))) / self.Nnow
        x_H0 = 0.5 * (np.sqrt(rhs**2 + 4 * rhs) - rhs)
        return x_H0 + self.fHe * y

    def Xe_H_He_evolution(self, z: float, sol: Any) -> float:
        """Calculate H and He evolution at a given redshift"""
        return sol(z)[0] + self.fHe * sol(z)[1]

    def late_Tmat(self, Tm: float, params: Tuple, z: float) -> float:
        """Calculate late-time matter temperature evolution"""
        x = params[1]  # Unpack the ionization fraction from params
        a = 1 / (1 + z)
        x_a = self.a2x(a)
        Hz = self.bg.H_conf(x_a) / a / H0_NATURAL_UNIT_CONVERSION
        Trad = self.Tnow * (1 + z)
        dTm = (self.CT * Trad**4 * x / (1 + x + self.fHe) * 
               (Tm - Trad) / (Hz * (1 + z)) + 2 * Tm / (1 + z))
        return dTm

    def Tmat_early(self, z: float) -> float:
        """Calculate matter temperature for z > 3500"""
        return self.Tnow * (1 + z)

    def Xe_early(self, z: float) -> float:
        """Calculate Xe until joint H/He recombination"""
        x0 = 1.0
        if z > 8000.:
            x0 = 1 + 2 * self.fHe
        elif z > 5000.:
            rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
                        self.CB1_He2 / (self.Tnow * (1 + z))) / self.Nnow
            rhs = rhs * 1.  # ratio of g's is 1 for He++ <-> He+
            x0 = 0.5 * (np.sqrt((rhs - 1 - self.fHe)**2 + 
                               4 * (1 + 2 * self.fHe) * rhs) - (rhs - 1 - self.fHe))
        elif z > 3500.:
            x0 = 1 + self.fHe
        else:
            rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
                        self.CB1_He1 / (self.Tnow * (1 + z))) / self.Nnow
            rhs = rhs * 4  # ratio of g's is 4 for He+ <-> He0
            x_He0 = 0.5 * (np.sqrt((rhs - 1)**2 + 4 * (1 + self.fHe) * rhs) - (rhs - 1))
            x0 = x_He0
        return x0

    def end_of_saha_condition(self, z: float) -> float:
        """Determine redshift at which we have to stop He Saha"""
        rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
                    self.CB1_He1 / (self.Tnow * (1 + z))) / self.Nnow
        rhs = rhs * 4  # ratio of g's is 4 for He+ <-> He0
        x_He0 = 0.5 * (np.sqrt((rhs - 1)**2 + 4 * (1 + self.fHe) * rhs) - (rhs - 1))
        x0 = x_He0
        x_He0 = (x0 - 1) / self.fHe
        return x_He0 - 0.99

    def ion_recfast_H_Saha(self, u: np.ndarray, z: float) -> np.ndarray:
        """Calculate RECFAST derivatives for H Saha equilibrium"""
        rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
                    self.CB1 / (self.Tnow * (1 + z))) / self.Nnow
        x_H0 = 0.5 * (np.sqrt(rhs**2 + 4 * rhs) - rhs)
        u_full = np.array([x_H0, u[0], u[1]])
        du = self.ion_recfast(u_full, z)
        return np.array([du[1], du[2]])

    def init_He_evolution(self, z0: float) -> np.ndarray:
        """Initialize He evolution at a given redshift"""
        x_H0 = 1.
        rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z0)) - 
                     self.CB1_He1 / (self.Tnow * (1 + z0))) / self.Nnow
        rhs = rhs * 4.  # ratio of g's is 4 for He+ <-> He0
        x_He0 = 0.5 * (np.sqrt((rhs - 1)**2 + 4 * (1 + self.fHe) * rhs) - (rhs - 1))  # He Saha
        x0 = x_He0
        x_He0 = (x0 - 1) / self.fHe
        return np.array([x_He0, self.Tmat_early(z0)])

    def x_H0_H_Saha(self, z: float) -> float:
        """Calculate H ionization fraction in Saha equilibrium"""
        rhs = np.exp(1.5 * np.log(self.CR * self.Tnow / (1 + z)) - 
                     self.CB1 / (self.Tnow * (1 + z))) / self.Nnow
        x_H0 = 0.5 * (np.sqrt(rhs**2 + 4 * rhs) - rhs)
        return x_H0

    def end_He_evo_condition(self, z: float) -> float:
        """Condition for end of He evolution"""
        return self.x_H0_H_Saha(z) - 0.985


    def find_root_with_scan(self, func, z_start, z_end, z_scan_points=50):
        """Find root by scanning the interval first to find sign changes"""
        # Ensure the scanned interval is in ascending order:
        if z_start > z_end:
            z_scan = np.linspace(z_end, z_start, z_scan_points)[::-1]
        else:
            z_scan = np.linspace(z_start, z_end, z_scan_points)
    
        f_scan = [func(z) for z in z_scan]
    
        # Find where function changes sign
        sign_changes = np.where(np.diff(np.signbit(f_scan)))[0]
    
        if len(sign_changes) == 0:
            # No sign change found, return a reasonable default
            return (z_start + z_end) / 2
    
        # Use the first sign change point
        idx = sign_changes[0]
        z_left = z_scan[idx]
        z_right = z_scan[idx + 1]
    
        # Ensure that the bracket is ordered properly for root_scalar:
        a, b = sorted((z_left, z_right))
        return root_scalar(func, bracket=(a, b), method='brentq').root

    def recfastsolve(self, zinitial: float = 10000., zfinal: float = 0., 
                     method: str = 'DOP853', rtol: float = None, atol: float=1e-10) -> Any:
        """Solve RECFAST equations"""
        if rtol is None:
            rtol = self.tol

        z_epoch_He_Saha_begin = min(zinitial, 3500.)

        # Find evolution start points using scanning method
        z_He_evo_start = self.find_root_with_scan(
            self.end_of_saha_condition, z_epoch_He_Saha_begin, zfinal)

        z_H_He_evo_start = self.find_root_with_scan(
            self.end_He_evo_condition, z_He_evo_start, zfinal)

        # Evolve Helium and Tmat
        y2 = self.init_He_evolution(z_He_evo_start)

        def he_deriv(z, y):
            return self.ion_recfast_H_Saha(y, z)

        sol_He = solve_ivp(
            he_deriv,
            t_span=(z_He_evo_start, z_H_He_evo_start),
            y0=y2,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True
        )

        # Evolve Hydrogen, Helium, and Tmat
        z3 = z_H_He_evo_start
        y3 = np.array([
            self.x_H0_H_Saha(z3),
            sol_He.sol(z3)[0],
            sol_He.sol(z3)[1]
        ])

        def h_he_deriv(z, y):
            return self.ion_recfast(y, z)

        sol_H_He = solve_ivp(
            h_he_deriv,
            t_span=(z3, zfinal),
            y0=y3,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True
        )

        return RecfastSolution(
            recfast=self,
            zinitial=zinitial,
            zfinal=zfinal,
            z_He_evo_start=z_He_evo_start,
            z_H_He_evo_start=z_H_He_evo_start,
            sol_He=sol_He.sol,
            sol_H_He=sol_H_He.sol
        )

    def get_ionization_history(self) -> IonizationHistory:
        """Create ionization history from RECFAST calculations"""
        x_grid = self.bg.x_grid
    
        # Solve RECFAST equations
        rhist = self.recfastsolve()
        trhist = rhist.tanh_reio_solve()
    
        xinitial_RECFAST = z2x(rhist.zinitial)
        Xe_initial = rhist.Xe_RECFAST(rhist.zinitial)

        # Define functions for interpolation
        def Xe_function(x):
            if x < xinitial_RECFAST:
                return Xe_initial
            return trhist.Xe_TanhReio(x2z(x))

        def Trad_function(x):
            return self.Tnow * (1 + x2z(x))

        def Tmat_function(x):
            if x < xinitial_RECFAST:
                return Trad_function(x)
            return trhist.Tmat_TanhReio(x2z(x))

        # Calculate optical depth and visibility functions
        tau, tau_prime = self.tau_functions(x_grid, Xe_function)
        g_tilde = lambda x: -tau_prime(x) * np.exp(-tau(x))

        # Create interpolation objects for all functions
        Xe_spl = CubicSpline(x_grid, [Xe_function(x) for x in x_grid])
        tau_spl = CubicSpline(x_grid, [tau(x) for x in x_grid])
        g_tilde_spl = CubicSpline(x_grid, [g_tilde(x) for x in x_grid])
        Tmat_spl = CubicSpline(x_grid, [Tmat_function(x) for x in x_grid])

        # Calculate sound speed
        csb2_pre = self.k_B / (self.m_H * self.C**2) * (1/self.mu_T + (1-self.Yp) * Xe_spl(x_grid))
        Tmat_values = Tmat_spl(x_grid)
        dTmat_values = Tmat_spl(x_grid, 1)
        csb2_values = csb2_pre * (Tmat_values - 1/3 * dTmat_values)
        csb2_spl = CubicSpline(x_grid, csb2_values)

        def _make_deriv(spl, order=1):
            return CubicSpline(spl.x, spl(spl.x, order))

        # Create and return IonizationHistory object
        return IonizationHistory(
            tau_0=tau(0.0),
            X_e=Xe_spl,
            tau=tau_spl,
            tau_p=_make_deriv(tau_spl, 1),
            tau_pp=_make_deriv(tau_spl, 2),
            g_tilde=g_tilde_spl,
            g_tilde_p=_make_deriv(g_tilde_spl, 1),
            g_tilde_pp=_make_deriv(g_tilde_spl, 2),
            Tmat=Tmat_spl,
            csb2=csb2_spl
        )
        
    def n_b(self, a):
        return self.bg.params.Omega_b * self.bg.params.rho_crit / (constants_nu.m_H * a**3)

    def n_H(self, a):
        return self.n_b(a) * (1-self.Yp)

    def tau_prime(self, x: float, Xe_f: Any) -> float:
        """Calculate τ' at a given x"""
        a = x2a(x)
        return -Xe_f(x) * self.n_H(a) * constants_nu.sigma_T * a / self.bg.H_conf(x)

    def tau_functions(self, x: np.ndarray, Xe_f: Any):
        """Calculate optical depth and its derivative"""
        # Ensure x is increasing
        if x[1] <= x[0]:
            raise ValueError("x must be increasing")

        # Calculate τ' values
        tau_primes = np.array([self.tau_prime(xi, Xe_f) for xi in x])
    
        # Do reverse cumulative integration
        rx = x[::-1]  # Reverse x
        reversed_tau_primes = tau_primes[::-1]  # Reverse τ' values
        tau_integrated = np.cumsum(np.diff(rx, prepend=rx[0]) * reversed_tau_primes)[::-1]

        # Create interpolation function for τ
        tau_interp = interp1d(x, tau_integrated, kind='cubic', fill_value="extrapolate")
        tau_prime_interp = interp1d(x, tau_primes, kind='cubic', fill_value="extrapolate")
        return tau_interp, tau_prime_interp


@dataclass
class RecfastSolution:
    """Container for RECFAST solution"""
    recfast: 'RECFAST'
    zinitial: float
    zfinal: float
    z_He_evo_start: float
    z_H_He_evo_start: float
    sol_He: Any
    sol_H_He: Any

    def Xe_RECFAST(self, z: float) -> float:
        """Calculate ionization fraction at a given redshift"""
        r = self.recfast

        if z > 8000.:
            return 1 + 2 * r.fHe
        elif z > 5000.:
            rhs = np.exp(1.5 * np.log(r.CR * r.Tnow / (1 + z)) - 
                        r.CB1_He2 / (r.Tnow * (1 + z))) / r.Nnow
            rhs = rhs * 1.  # ratio of g's is 1 for He++ <-> He+
            return 0.5 * (np.sqrt((rhs - 1 - r.fHe)**2 + 
                                4 * (1 + 2 * r.fHe) * rhs) - (rhs - 1 - r.fHe))
        elif z > 3500.:
            return 1 + r.fHe
        elif z > self.z_He_evo_start:
            rhs = np.exp(1.5 * np.log(r.CR * r.Tnow / (1 + z)) - 
                        r.CB1_He1 / (r.Tnow * (1 + z))) / r.Nnow
            rhs = rhs * 4  # ratio of g's is 4 for He+ <-> He0
            x_He0 = 0.5 * (np.sqrt((rhs - 1)**2 + 4 * (1 + r.fHe) * rhs) - (rhs - 1))
            return x_He0
        elif z > self.z_H_He_evo_start:
            return r.Xe_He_evolution(z, self.sol_He)
        else:
            return r.Xe_H_He_evolution(z, self.sol_H_He)

    def Tmat_RECFAST(self, z: float) -> float:
        """Calculate matter temperature at a given redshift"""
        r = self.recfast
        if z > self.z_He_evo_start:
            return r.Tmat_early(z)
        elif z > self.z_H_He_evo_start:
            return self.sol_He(z)[1]
        else:
            return self.sol_H_He(z)[2]

    def reionization_Xe(self, z: float) -> float:
        """Calculate ionization fraction including reionization effects"""
        r = self.recfast
        X_fin = 1 + r.Yp / (r.not4 * (1 - r.Yp))  # ionization frac today

        # Reionization parameters (TO REPLACE)
        zre, alpha, delta_H = 7.6711, 1.5, 0.5
        zHe, delta_He = 3.5, 0.5
        f_He = X_fin - 1

        x_orig = self.Xe_RECFAST(z)
        
        # H reionization
        x_reio_H = ((X_fin - x_orig) / 2 * 
                    (1 + np.tanh(((1 + zre)**alpha - (1 + z)**alpha) / 
                                (alpha * (1 + zre)**(alpha - 1)) / delta_H)) + x_orig)
        
        # He reionization
        x_reio_He = f_He / 2 * (1 + np.tanh((zHe - z) / delta_He))
        
        return x_reio_H + x_reio_He

    def reionization_Tmat_ode(self, Tm: float, z: float) -> float:
        """Calculate matter temperature evolution during reionization"""
        r = self.recfast
        x_reio = self.reionization_Xe(z)

        a = 1 / (1 + z)
        x_a = r.a2x(a)
        Hz = r.bg.H_conf(x_a) / a / H0_NATURAL_UNIT_CONVERSION
        Trad = r.Tnow * (1 + z)
        
        dTm = (r.CT * Trad**4 * x_reio / (1 + x_reio + r.fHe) *
               (Tm - Trad) / (Hz * (1 + z)) + 2 * Tm / (1 + z))
        
        return dTm

    def tanh_reio_solve(self, zre_ini: float = 50.0) -> "TanhReionizationHistory":
        """Solve for reionization history using tanh model"""
        recfast = self.recfast

        def reio_deriv(z: float, y: np.ndarray) -> np.ndarray:
            """ODE for matter temperature during reionization"""
            return np.array([self.reionization_Tmat_ode(y[0], z)])

        # Solve the reionization ODE
        sol_reio = solve_ivp(
            reio_deriv,
            t_span=(zre_ini, self.zfinal),
            y0=np.array([self.Tmat_RECFAST(zre_ini)]),
            method='RK45',
            rtol=recfast.tol,
            dense_output=True
        )

        return TanhReionizationHistory(
            zre_ini=zre_ini,
            ionization_history=self,
            sol_reionization_Tmat=sol_reio.sol
        )


@dataclass
class TanhReionizationHistory:
    """Class for handling reionization history with tanh model"""
    zre_ini: float
    ionization_history: RecfastSolution
    sol_reionization_Tmat: Any

    def Xe_TanhReio(self, z: float) -> float:
        """Calculate ionization fraction with reionization effects"""
        if z > self.zre_ini:
            return self.ionization_history.Xe_RECFAST(z)
        else:
            return self.ionization_history.reionization_Xe(z)

    def Tmat_TanhReio(self, z: float) -> float:
        """Calculate matter temperature with reionization effects"""
        if z > self.zre_ini:
            return self.ionization_history.Tmat_RECFAST(z)
        else:
            return self.sol_reionization_Tmat(z)[0]




import numpy as np

def a2z(a: float) -> float:
    """Convert scale factor to redshift."""
    return 1.0 / a - 1.0

def z2a(z: float) -> float:
    """Convert redshift to scale factor."""
    return 1.0 / (z + 1.0)

def a2x(a: float) -> float:
    """Convert scale factor to x (log of scale factor)."""
    return np.log(a)

def x2a(x: float) -> float:
    """Convert x (log of scale factor) to scale factor."""
    return np.exp(x)

def z2x(z: float) -> float:
    """Convert redshift to x (log of scale factor)."""
    return a2x(z2a(z))

def x2z(x: float) -> float:
    """Convert x (log of scale factor) to redshift."""
    return a2z(x2a(x))
