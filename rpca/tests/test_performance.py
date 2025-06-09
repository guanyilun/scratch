import numpy as np
import time
from rpca import RobustPCA

def test_rpca_performance():
    # Generate a low-rank matrix with sparse noise
    np.random.seed(42)
    n_samples, n_features = 1000, 500
    rank = 10
    
    # Create low rank component
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(n_features, rank)
    L = U @ V.T
    
    # Add sparse noise
    S = np.zeros((n_samples, n_features))
    mask = np.random.rand(n_samples, n_features) < 0.1  # 10% sparse noise
    S[mask] = np.random.randn(mask.sum()) * 5
    
    # Combine components
    X = L + S
    
    # Time the RPCA fit
    rpca = RobustPCA(n_components=rank, verbose=True)
    
    start_time = time.time()
    rpca.fit(X)
    end_time = time.time()
    
    fit_time = end_time - start_time
    print(f"\nFit time: {fit_time:.2f} seconds")
    
    # Verify accuracy
    l_err = np.linalg.norm(rpca.low_rank_ - L) / np.linalg.norm(L)
    s_err = np.linalg.norm(rpca.sparse_ - S) / np.linalg.norm(S)
    
    print(f"Relative error (low rank): {l_err:.4f}")
    print(f"Relative error (sparse): {s_err:.4f}")
    print(f"Number of iterations: {rpca.n_iter_}")
    
    # Basic accuracy checks
    # Basic accuracy checks
    assert l_err < 0.1, "Low rank reconstruction error too high"
    assert s_err < 0.2, "Sparse reconstruction error too high"
    # Check that the algorithm converged by verifying final errors are small
    final_error = rpca.errors[-1]
    assert final_error < 0.01, f"Failed to converge sufficiently, final error: {final_error}"
if __name__ == '__main__':
    test_rpca_performance()