import numpy as np
import pytest
from background import (
    CosmoParams,
    NeutrinoDistribution,
    BackgroundEvolution,
    to_ui,
    from_ui,
    xq2q
)

# Test data - standard ΛCDM parameters
@pytest.fixture
def standard_params():
    return {
        'h': 0.67,
        'Omega_r': 5.4e-5,
        'Omega_b': 0.022,
        'Omega_c': 0.12,
        'N_nu': 3.046,
        'sum_m_nu': 0.06,  # eV
        'nq': 100  # number of quadrature points
    }

# Test CosmoParams initialization and validation
def test_cosmo_params_initialization(standard_params):
    params = CosmoParams(**standard_params)
    assert params.h == standard_params['h']
    assert params.Omega_r == standard_params['Omega_r']
    assert params.Omega_b == standard_params['Omega_b']
    assert params.Omega_c == standard_params['Omega_c']
    assert params.N_nu == standard_params['N_nu']
    assert params.sum_m_nu == standard_params['sum_m_nu']

def test_cosmo_params_validation():
    with pytest.raises(ValueError):
        CosmoParams(h=-1, Omega_r=0.1, Omega_b=0.1, Omega_c=0.1, 
                   N_nu=3, sum_m_nu=0.06, nq=100)
    
    with pytest.raises(ValueError):
        CosmoParams(h=0.67, Omega_r=0.1, Omega_b=0.1, Omega_c=0.1, 
                   N_nu=3, sum_m_nu=-0.06, nq=100)

# Test Hubble parameter calculations
def test_hubble_parameter(standard_params):
    params = CosmoParams(**standard_params)
    
    # Test at a=1 (today)
    H_today = params.H_a(1.0)
    assert H_today == pytest.approx(params.H0, rel=1e-5)

# Test NeutrinoDistribution
def test_neutrino_distribution():
    T_nu = 1.0  # test temperature
    dist = NeutrinoDistribution(T_nu)
    
    # Test momentum values
    q_test = np.array([0.1, 1.0, 10.0])
    
    # Test f0 is positive and decreasing with momentum
    f0_values = dist.f0(q_test)
    assert np.all(f0_values > 0)
    assert np.all(np.diff(f0_values) < 0)
    
    # Test dlnf0_dlnq
    dlnf0_values = dist.dlnf0_dlnq(q_test)
    assert np.all(dlnf0_values < 0)  # should be negative for all q

# Test utility functions
def test_coordinate_transformations():
    logqmin, logqmax = -5, 5
    
    # Test to_ui and from_ui are inverses
    x_test = np.array([-1.0, 0.0, 1.0])
    q_test = np.array([1e-5, 1.0, 1e5])
    
    for x in x_test:
        q = from_ui(x, logqmin, logqmax)
        x_recovered = to_ui(q, logqmin, logqmax)
        assert x == pytest.approx(x_recovered)
    
    for q in q_test:
        x = to_ui(np.log10(q), logqmin, logqmax)
        q_recovered = xq2q(x, logqmin, logqmax)
        assert q == pytest.approx(q_recovered)

# Test BackgroundEvolution
def test_background_evolution(standard_params):
    params = CosmoParams(**standard_params)
    x_grid = np.linspace(-10, 0, 100)
    be = BackgroundEvolution(params, x_grid)
    
    # Test interpolation is continuous
    x_test = -5.0
    
    # Test H_conf and its derivatives
    H_conf = be.H_conf(x_test)
    H_conf_deriv = be.H_conf_p(x_test)
    H_conf_second = be.H_conf_pp(x_test)
    
    assert isinstance(H_conf[()], float)
    assert isinstance(H_conf_deriv[()], float)
    assert isinstance(H_conf_second[()], float)
    
    # Test eta and its derivatives
    eta = be.eta(x_test)
    eta_deriv = be.eta_p(x_test)
    eta_second = be.eta_pp(x_test)
    
    assert isinstance(eta[()], float)
    assert isinstance(eta_deriv[()], float)
    assert isinstance(eta_second[()], float)

# Test total energy density consistency
def test_energy_density_consistency(standard_params):
    params = CosmoParams(**standard_params)
    
    # Test that total density is close to critical density at a=1
    rho_nu_0, _ = params.rho_P_nu_0(1.0)
    total_omega = (
        params.Omega_b + 
        params.Omega_c + 
        params.Omega_r * (1 + (2/3)*(7/8)*(4/11)**(4/3)*params.N_nu) +
        rho_nu_0/params.rho_crit +
        params.Omega_lambda
    )
    assert total_omega == pytest.approx(1.0, rel=1e-3)