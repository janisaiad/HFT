import pytest
import numpy as np
import polars as pl
from hft.models.hawkes import Hawkes

def test_hawkes_initialization():
    N = 2
    mu = np.array([0.1, 0.2])
    def phi(t):
        return np.array([[0.5 * np.exp(-t), 0.2 * np.exp(-t)],
                        [0.3 * np.exp(-t), 0.4 * np.exp(-t)]])
    def psi(t):
        return np.array([0.1 * np.exp(-t), 0.2 * np.exp(-t)])
    
    hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N)
    
    assert hawkes.dim == N
    assert np.array_equal(hawkes.mu, mu)
    assert len(hawkes.list_of_events) == N
    assert all(len(events) == 0 for events in hawkes.list_of_events)

def test_add_event():
    hawkes = Hawkes(phi=lambda t: np.array([[0.1, 0], [0, 0.1]]), 
                    psi=lambda t: np.array([0.1, 0.1]), 
                    mu=np.array([0.1, 0.1]), N=2)
    
    hawkes.add_event(1.0, 0)
    hawkes.add_event(2.0, 1)
    
    assert hawkes.list_of_events[0] == [1.0]
    assert hawkes.list_of_events[1] == [2.0]

def test_get_intensity():
    N = 2
    mu = np.array([0.1, 0.1])
    def phi(t):
        return np.array([[0.5 * np.exp(-t), 0.2 * np.exp(-t)],
                        [0.3 * np.exp(-t), 0.4 * np.exp(-t)]])
    def psi(t):
        return np.array([0.1 * np.exp(-t), 0.1 * np.exp(-t)])
    
    hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N)
    hawkes.add_event(1.0, 0)
    
    intensity = hawkes.get_intensity(2.0)
    expected = mu + np.array([0.5, 0.3]) * np.exp(-1.0)
    assert np.allclose(intensity, expected)

def test_verify_l1_norm_phi():
    N = 2
    mu = np.array([0.1, 0.1])
    def phi(t):
        return np.array([[0.2 * np.exp(-t), 0.1 * np.exp(-t)],
                        [0.1 * np.exp(-t), 0.2 * np.exp(-t)]])
    def psi(t):
        return np.array([0.1 * np.exp(-t), 0.1 * np.exp(-t)])
    
    hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N)
    assert hawkes.verify_l1_norm_phi() == True

def test_get_average_intensity():
    N = 2
    mu = np.array([0.1, 0.1])
    def phi(t):
        return np.array([[0.2 * np.exp(-t), 0.1 * np.exp(-t)],
                        [0.1 * np.exp(-t), 0.2 * np.exp(-t)]])
    def psi(t):
        return np.array([0.1 * np.exp(-t), 0.1 * np.exp(-t)])
    
    hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N)
    avg_intensity = hawkes.get_average_intensity(1.0)
    assert len(avg_intensity) == N
    assert all(i > 0 for i in avg_intensity)


def test_convolution_product():
    N = 2
    mu = np.array([0.1, 0.1])
    def phi(t):
        return np.array([[0.5 * np.exp(-t), 0.2 * np.exp(-t)],
                        [0.3 * np.exp(-t), 0.4 * np.exp(-t)]])
    def psi(t):
        return np.array([0.1 * np.exp(-t), 0.1 * np.exp(-t)])
    
    hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N)
    
    conv_func = hawkes.convolution_product(lambda t: phi(t)[0,0])
    t = 1.0
    result = conv_func(t)
    assert isinstance(result, float)
    assert result > 0


    conv_matrix = hawkes.convolution_product_matrix(phi)
    assert conv_matrix(t).shape == (N, N)
    assert np.all(conv_matrix(t) >= 0)

    conv_result1 = hawkes.get_convolution_product(t)
    conv_result2 = hawkes.get_convolution_product_matrix(t)
    assert np.array_equal(conv_result1, conv_result2)
    assert conv_result1.shape == (N, N)
    
    
def test_parquet_reading_and_wiener_hopf():
    
    N = 8  # 8 event types for market data
    mu = np.array([0.1] * N)  # Base intensities
    
    def phi(t):  # Placeholder kernel function
        return np.array([[0.1 * np.exp(-t)] * N] * N)
    
    def psi(t):  # Placeholder response function 
        return np.array([0.1 * np.exp(-t)] * N)
    
    hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N, stepsize=100)
    
    # Test parquet reading
    df = hawkes.get_parquet("data/DB_MBP_10/data/hawkes_dataset/GBDC/GBDC_2023-07-17.parquet")
    assert isinstance(df, pl.DataFrame)
    
    # Test g estimation
    g_results = hawkes.get_g_from_parquet(df)
    assert isinstance(g_results, dict)
    assert "g_estimates" in g_results
    assert "g_integrals" in g_results
    assert "ug_integrals" in g_results
    assert "t_grid" in g_results
    assert "lambda_i" in g_results
    
    # Check dimensions
    assert g_results["g_estimates"].shape == (N, N, hawkes.stepsize)
    assert g_results["g_integrals"].shape == (N, N, hawkes.stepsize)
    assert g_results["ug_integrals"].shape == (N, N, hawkes.stepsize)
    assert len(g_results["t_grid"]) == hawkes.stepsize
    assert len(g_results["lambda_i"]) == N
    
    # Test Wiener-Hopf system construction
    A, b = hawkes.compute_wiener_hopf_linear_kernel(g_results)
    expected_size = N * N * hawkes.stepsize
    assert A.shape == (expected_size, expected_size)
    assert b.shape == (expected_size,)
    
    # Test phi estimation
    phi_values = hawkes.solve_phi_from_wiener_hopf(g_results)
    assert phi_values.shape == (N, N, hawkes.stepsize)
    assert np.all(np.isfinite(phi_values))  # Check no NaN or inf values
    
    # Test reconstructed kernel function
    t_test = 0.5
    phi_t = hawkes.phi(t_test)
    assert phi_t.shape == (N, N)
    assert np.all(np.isfinite(phi_t))
