import numpy as np
from numba import njit, types
from numba.experimental import jitclass
import math

# Define jitclass spec
wigner_spec = [
    ('j2', types.float64),
    ('j3', types.float64),
    ('m2', types.float64),
    ('m3', types.float64),
    ('n_min', types.int64),
    ('n_max', types.int64),
]

@jitclass(wigner_spec)
class WignerF:
    def __init__(self, j2, j3, m2, m3):
        self.j2 = float(j2)
        self.j3 = float(j3)
        self.m2 = float(m2)
        self.m3 = float(m3)
        self.n_min = max(int(abs(j2 - j3)), int(abs(m2 + m3)))
        self.n_max = int(j2 + j3)

@njit
def selection_rules(w):
    small = 0.1
    return (
        abs(w.m2) <= abs(w.j2) + small and
        abs(w.m3) <= abs(w.j3) + small and
        abs((w.m2 - w.j2) - round(w.m2 - w.j2)) < 1e-10 and
        abs((w.m3 - w.j3) - round(w.m3 - w.j3)) < 1e-10
    )

@njit
def A_coeff(w, j):
    term1 = j**2 - (w.j2 - w.j3)**2
    term2 = (w.j2 + w.j3 + 1)**2 - j**2
    term3 = j**2 - (w.m2 + w.m3)**2
    product = term1 * term2 * term3
    if product < 0:
        return 0.0
    return math.sqrt(product)

@njit
def B_coeff(w, j):
    return (2*j + 1) * (
        (w.m2 + w.m3) * (w.j2*(w.j2+1) - w.j3*(w.j3+1)) -
        (w.m2 - w.m3) * j*(j+1)
    )

@njit
def X_psi(w, j):
    return j * A_coeff(w, j+1)

@njit
def Y_psi(w, j):
    return B_coeff(w, j)

@njit
def Z_psi(w, j):
    return (j+1) * A_coeff(w, j)

@njit
def f_jmax_sgn(w):
    return 1 if int(w.j2 - w.j3 + w.m2 + w.m3) % 2 == 0 else -1

@njit
def r_psi_backward(w, nmid, psi):
    n_stop = nmid
    for n in range(w.n_max, nmid-1, -1):
        idx = n - w.n_min
        if n == w.n_max:
            denom = Y_psi(w, n)
            if abs(denom) < 1e-15:
                psi[idx] = 1.0
                return n
            psi[idx] = -Z_psi(w, n) / denom
        else:
            denom = Y_psi(w, n) + X_psi(w, n) * psi[idx+1]
            if abs(denom) < 1e-15:
                psi[idx] = 1.0
                return n
            psi[idx] = -Z_psi(w, n) / denom
        
        if not math.isfinite(psi[idx]):
            psi[idx] = 1.0
            return n
        if abs(psi[idx]) >= 1.0:
            return n
    return n_stop

@njit
def s_psi_forward(w, nmid, psi):
    n_stop = nmid
    for n in range(w.n_min, nmid+1):
        idx = n - w.n_min
        if n == w.n_min:
            denom = Y_psi(w, n)
            if abs(denom) < 1e-15:
                psi[idx] = 1.0
                return n
            psi[idx] = -X_psi(w, n) / denom
        else:
            denom = Y_psi(w, n) + Z_psi(w, n) * psi[idx-1]
            if abs(denom) < 1e-15:
                psi[idx] = 1.0
                return n
            psi[idx] = -X_psi(w, n) / denom
        
        if not math.isfinite(psi[idx]):
            psi[idx] = 1.0
            return n
        if abs(psi[idx]) >= 1.0:
            return n
    return n_stop

@njit
def psi_aux_plus(w, n_minus, nc, psi):
    start_index = n_minus
    if n_minus == w.n_min:
        psi[0] = 1.0
        if w.n_min == 0:
            denom = A_coeff(w, 1)
            if abs(denom) < 1e-15:
                psi[1] = 0.0
            else:
                psi[1] = -(w.m3 - w.m2 + 2*B_coeff(w, w.n_min)) / denom
        else:
            denom = w.n_min * A_coeff(w, w.n_min+1)
            if abs(denom) < 1e-15:
                psi[1] = 0.0
            else:
                psi[1] = -B_coeff(w, w.n_min) / denom
        start_index = w.n_min + 1
    
    for n in range(start_index, nc):
        idx = n - w.n_min
        Xn = X_psi(w, n)
        if abs(Xn) < 1e-15:
            psi[idx+1] = 0.0
        else:
            term = Y_psi(w, n)*psi[idx] + Z_psi(w, n)*psi[idx-1]
            psi[idx+1] = -term / Xn

@njit
def normalization(w, psi):
    norm = 0.0
    for i, j in enumerate(range(w.n_min, w.n_max+1)):
        norm += (2*j + 1) * psi[i]**2
    return math.sqrt(abs(norm))

@njit
def f_to_min_m0(w, nmid, psi):
    n = nmid + 1
    while n <= w.n_max - 1:
        Xn = X_psi(w, n)
        if abs(Xn) < 1e-15:
            psi_val = 0.0
        else:
            term1 = Y_psi(w, n) * psi[n - w.n_min]
            term2 = Z_psi(w, n) * psi[n-1 - w.n_min]
            psi_val = -(term1 + term2) / Xn
        psi[n+1 - w.n_min] = psi_val
        n += 2

@njit
def f_to_max_m0(w, nmid, psi):
    n = nmid - 1
    while n >= w.n_min + 1:
        Zn = Z_psi(w, n)
        if abs(Zn) < 1e-15:
            psi_val = 0.0
        else:
            term1 = Y_psi(w, n) * psi[n - w.n_min]
            term2 = X_psi(w, n) * psi[n+1 - w.n_min]
            psi_val = -(term1 + term2) / Zn
        psi[n-1 - w.n_min] = psi_val
        n -= 2

@njit
def classical_wigner3j_m0(w, w3j):
    w3j[:] = 0.0
    nmid = (w.n_min + w.n_max) // 2
    if (nmid - w.n_min) % 2 != 0:
        nmid += 1
    idx_mid = nmid - w.n_min
    if idx_mid < len(w3j):
        w3j[idx_mid] = 1.0
    
    # Only run if within bounds
    if nmid < w.n_max:
        f_to_min_m0(w, nmid, w3j)
    if nmid > w.n_min:
        f_to_max_m0(w, nmid, w3j)
    
    norm_val = normalization(w, w3j)
    if norm_val > 1e-15:
        norm_val = 1.0 / norm_val
        if (w3j[-1] > 0) != (f_jmax_sgn(w) > 0):
            norm_val *= -1
        w3j[:] *= norm_val

@njit
def wigner3j_f_core(w, w3j):
    length = w.n_max - w.n_min + 1
    
    # Handle special cases
    if length == 1:
        w3j[0] = f_jmax_sgn(w) / math.sqrt(w.n_min + w.j2 + w.j3 + 1)
        return
    elif length == 2:
        w3j[:] = 1.0
        s_psi_forward(w, w.n_min, w3j)
        norm_val = normalization(w, w3j)
        if norm_val > 1e-15:
            norm_val = 1.0 / norm_val
            if (w3j[1] > 0) != (f_jmax_sgn(w) > 0):
                norm_val *= -1
            w3j[:] *= norm_val
        return
    
    # General solution
    w3j[:] = 0.0
    nmid = (w.n_min + w.n_max + 1) // 2
    
    # Special case for m2=m3=0
    if w.m2 == 0 and w.m3 == 0:
        adjusted_mid = (w.n_min + w.n_max) // 2
        if (adjusted_mid - w.n_min) % 2 != 0:
            adjusted_mid += 1
        # Check if adjusted_mid is within valid range
        if w.n_min <= adjusted_mid <= w.n_max:
            classical_wigner3j_m0(w, w3j)
            return
    
    # Initialize recurrence array
    temp_psi = np.ones_like(w3j)
    
    # Run recurrences
    n_plus = r_psi_backward(w, nmid, temp_psi)
    n_minus = s_psi_forward(w, nmid, temp_psi)
    
    # Propagate ratios
    for k in range(n_plus+1, w.n_max+1):
        idx_curr = k - w.n_min
        idx_prev = (k-1) - w.n_min
        if 0 <= idx_curr < len(temp_psi) and 0 <= idx_prev < len(temp_psi):
            temp_psi[idx_curr] *= temp_psi[idx_prev]
    
    for k in range(n_minus-1, w.n_min-1, -1):
        idx_curr = k - w.n_min
        idx_next = (k+1) - w.n_min
        if 0 <= idx_curr < len(temp_psi) and 0 <= idx_next < len(temp_psi):
            temp_psi[idx_curr] *= temp_psi[idx_next]
    
    # Classical recurrence in middle
    psi_n_minus = temp_psi[n_minus - w.n_min] if n_minus >= w.n_min and n_minus <= w.n_max else 1.0
    psi_n_plus = temp_psi[n_plus - w.n_min] if n_plus >= w.n_min and n_plus <= w.n_max else 1.0
    
    psi_aux_plus(w, n_minus, n_plus, temp_psi)
    
    # Rescale segments
    if n_minus > w.n_min:
        current_val = temp_psi[n_minus - w.n_min]
        if abs(current_val) > 1e-15:
            scale_mid = psi_n_minus / current_val
            for j in range(w.n_min, n_minus):
                idx = j - w.n_min
                if 0 <= idx < len(temp_psi):
                    temp_psi[idx] *= scale_mid
    
    if n_plus < w.n_max:
        current_val = temp_psi[n_plus - w.n_min]
        if abs(psi_n_plus) > 1e-15:
            scale_end = current_val / psi_n_plus
            for j in range(n_plus+1, w.n_max+1):
                idx = j - w.n_min
                if 0 <= idx < len(temp_psi):
                    temp_psi[idx] *= scale_end
    
    # Normalize
    norm_val = normalization(w, temp_psi)
    if norm_val > 1e-15:
        norm_val = 1.0 / norm_val
        if (temp_psi[-1] > 0) != (f_jmax_sgn(w) > 0):
            norm_val *= -1
        w3j[:] = temp_psi * norm_val

def wigner3j_f(j2, j3, m2, m3, dtype=np.float64):
    w = WignerF(j2, j3, m2, m3)
    if not selection_rules(w):
        return np.array([], dtype=dtype), np.array([], dtype=dtype)
    
    length = w.n_max - w.n_min + 1
    if length <= 0:
        return np.array([], dtype=dtype), np.array([], dtype=dtype)
    
    w3j = np.zeros(length, dtype=dtype)
    wigner3j_f_core(w, w3j)
    
    j1_vals = np.arange(w.n_min, w.n_max+1, dtype=dtype)
    return j1_vals, w3j
