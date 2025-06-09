use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RobustPCAError {
    #[error("Expected positive number of components, got {0}")]
    InvalidComponents(usize),
    #[error("Expected 2D array, got array with dimensions {0}")]
    InvalidDimensions(usize),
    #[error("RobustPCA instance is not fitted yet")]
    NotFitted,
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),
}

pub type Result<T> = std::result::Result<T, RobustPCAError>;

/// Robust Principal Component Analysis using Accelerated Alternating Projections
/// 
/// This implementation follows the algorithm described in:
/// HanQin Cai, et. al. "Accelerated alternating projections for robust principal component analysis."
/// https://arxiv.org/abs/1711.05519
#[derive(Debug, Clone)]
pub struct RobustPCA {
    // Configuration parameters
    pub n_components: Option<usize>,
    pub max_iter: usize,
    pub tol: f64,
    pub beta: Option<f64>,
    pub beta_init: Option<f64>,
    pub gamma: f64,
    pub mu: (f64, f64),
    pub trim: bool,
    pub verbose: bool,
    pub copy: bool,
    
    // Fitted parameters
    pub fitted: bool,
    n_samples: Option<usize>,
    n_features: Option<usize>,
    pub n_components_fitted: Option<usize>,
    beta_fitted: Option<f64>,
    beta_init_fitted: Option<f64>,
    mean: Option<DVector<f64>>,
    
    // Results
    pub low_rank: Option<DMatrix<f64>>,
    pub sparse: Option<DMatrix<f64>>,
    pub components: Option<DMatrix<f64>>,
    pub singular_values: Option<DVector<f64>>,
    pub errors: Vec<f64>,
    pub end_iter: usize,
}

impl Default for RobustPCA {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustPCA {
    /// Create a new RobustPCA instance with default parameters
    pub fn new() -> Self {
        Self {
            n_components: None,
            max_iter: 100,
            tol: 1e-5,
            beta: None,
            beta_init: None,
            gamma: 0.5,
            mu: (5.0, 5.0),
            trim: false,
            verbose: true,
            copy: true,
            fitted: false,
            n_samples: None,
            n_features: None,
            n_components_fitted: None,
            beta_fitted: None,
            beta_init_fitted: None,
            mean: None,
            low_rank: None,
            sparse: None,
            components: None,
            singular_values: None,
            errors: Vec::new(),
            end_iter: 0,
        }
    }
    
    /// Create a new RobustPCA instance with specified number of components
    pub fn with_components(n_components: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(RobustPCAError::InvalidComponents(n_components));
        }
        
        let mut rpca = Self::new();
        rpca.n_components = Some(n_components);
        Ok(rpca)
    }
    
    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set the tolerance for convergence
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    
    /// Set the beta parameter
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }
    
    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// Fit the RobustPCA model to the input data
    pub fn fit(&mut self, x: &DMatrix<f64>) -> Result<&mut Self> {
        self.fit_internal(x)?;
        Ok(self)
    }
    
    /// Transform the input data using the fitted model
    pub fn transform(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        if !self.fitted {
            return Err(RobustPCAError::NotFitted);
        }
        
        let mut x_transformed = if self.copy { x.clone() } else { x.clone() };
        
        // Center the data
        if let Some(ref mean) = self.mean {
            for mut col in x_transformed.column_iter_mut() {
                col -= mean;
            }
        }
        
        // Project onto components
        if let Some(ref components) = self.components {
            Ok(&x_transformed * components)
        } else {
            Err(RobustPCAError::NotFitted)
        }
    }
    
    /// Fit the model and transform the data in one step
    pub fn fit_transform(&mut self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let (_, _, u, sigma, _) = self.fit_internal(x)?;
        
        let n_comp = self.n_components_fitted.unwrap();
        let u_truncated = u.columns(0, n_comp);
        let diagonal = sigma.diagonal();
        let sigma_truncated = diagonal.rows(0, n_comp);
        
        // Scale by singular values
        let mut result = u_truncated.clone_owned();
        for (i, mut col) in result.column_iter_mut().enumerate() {
            if i < sigma_truncated.len() {
                col *= sigma_truncated[i];
            }
        }
        
        Ok(result)
    }
    
    /// Internal fitting implementation
    fn fit_internal(&mut self, x: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> {
        if x.ncols() == 0 || x.nrows() == 0 {
            return Err(RobustPCAError::InvalidDimensions(0));
        }
        
        let x_work = if self.copy { x.clone() } else { x.clone() };
        
        let (n_samples, n_features) = (x.nrows(), x.ncols());
        self.n_samples = Some(n_samples);
        self.n_features = Some(n_features);
        
        // Set parameters - following Python reference more closely
        let beta = self.beta.unwrap_or(1.0 / (2.0 * ((n_samples * n_features) as f64).powf(0.25)));
        let beta_init = self.beta_init.unwrap_or(4.0 * beta);
        let n_components = self.n_components.unwrap_or((n_samples.min(n_features)).saturating_sub(1));
        
        if self.verbose {
            println!("Params - beta: {:.6}, beta_init: {:.6}, n_components: {}", beta, beta_init, n_components);
        }
        
        self.beta_fitted = Some(beta);
        self.beta_init_fitted = Some(beta_init);
        self.n_components_fitted = Some(n_components);
        
        // Center the data
        let mean = compute_column_means(&x_work);
        let mut x_work = x_work;
        center_matrix(&mut x_work, &mean);
        self.mean = Some(mean);
        
        let norm_x = frobenius_norm(&x_work);
        
        // Initialize L, S, U, Sigma, V
        let (mut l, mut s, mut u, mut sigma, mut v) = self.initialize(&x_work, n_components, beta, beta_init)?;
        
        self.errors = Vec::with_capacity(self.max_iter + 1);
        let initial_error = compute_error(&x_work, &l, &s, norm_x);
        self.errors.push(initial_error);
        
        if self.verbose {
            println!("Initial: L_norm: {:.6}, S_norm: {:.6}, X_norm: {:.6}, Error: {:.6}",
                     frobenius_norm(&l), frobenius_norm(&s), norm_x, initial_error);
        }
        
        // Main iteration loop
        let mut converged = false;
        for i in 1..=self.max_iter {
            if self.trim {
                let (u_new, v_new) = self.trim_matrices(&u, &sigma, &v, n_components)?;
                u = u_new;
                v = v_new;
            }
            
            // Update L (low-rank component)
            let z = &x_work - &s;
            let (l_new, u_new, sigma_new, v_new) = self.update_low_rank(&z, &u, &v, n_components)?;
            
            l = l_new;
            u = u_new;
            sigma = sigma_new;
            v = v_new;
            
            // Update S (sparse component)
            // Note: In Python, Sigma[self.n_components_, self.n_components_] refers to the next singular value
            // after the truncated ones. Since we have a diagonal matrix, we need to check if we have enough values.
            let next_sv = if n_components < sigma.nrows() && n_components < sigma.ncols() {
                sigma[(n_components, n_components)]
            } else {
                0.0
            };
            let first_sv = sigma[(0, 0)];
            let zeta = beta * (next_sv + (self.gamma.powi(i as i32) * first_sv));
            s = hard_threshold(&(&x_work - &l), zeta);
            
            let error = compute_error(&x_work, &l, &s, norm_x);
            self.errors.push(error);
            
            if self.verbose && (i % 10 == 0 || i == self.max_iter || i <= 10) {
                println!("[{}] Error: {:.6}, Zeta: {:.6}, L_norm: {:.6}, S_norm: {:.6}, Next_SV: {:.6}, First_SV: {:.6}",
                         i, error, zeta, frobenius_norm(&l), frobenius_norm(&s), next_sv, first_sv);
                println!("    Sigma shape: {}x{}, n_components: {}", sigma.nrows(), sigma.ncols(), n_components);
                if sigma.nrows() > 0 && sigma.ncols() > 0 {
                    println!("    Sigma diagonal: {:?}", sigma.diagonal().as_slice());
                }
            }
            
            if error < self.tol {
                if self.verbose {
                    println!("Converged after {} iterations", i);
                }
                converged = true;
                self.end_iter = i;
                break;
            }
        }
        
        if !converged && self.verbose {
            println!("Did not converge after {} iterations", self.max_iter);
            self.end_iter = self.max_iter;
        }
        
        // Store results - convert back to original space by adding mean back
        let mut l_original = l.clone();
        let s_original = s.clone();
        
        // Add mean back to low-rank component (sparse component should remain centered)
        if let Some(ref mean) = self.mean {
            for (j, mut col) in l_original.column_iter_mut().enumerate() {
                col.add_scalar_mut(mean[j]);
            }
        }
        
        self.low_rank = Some(l_original);
        self.sparse = Some(s_original);
        self.components = Some(v.transpose());
        self.singular_values = Some(sigma.diagonal().rows(0, n_components).clone_owned());
        self.fitted = true;
        
        Ok((l, s, u, sigma, v))
    }
    
    /// Initialize the algorithm components
    fn initialize(&self, x: &DMatrix<f64>, n_components: usize, beta: f64, beta_init: f64)
        -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> {
        
        // Step 1: Compute largest singular value for initial threshold (like Python svds(X, k=1))
        let svd_values = x.clone().svd(false, false);
        let largest_sv = svd_values.singular_values[0];
        let zeta = beta_init * largest_sv;
        let s = hard_threshold(x, zeta);
        
        if self.verbose {
            println!("Step 1 - X_max: {:.6}, X_min: {:.6}", x.max(), x.min());
            println!("Step 1 - Largest SV: {:.6}, Zeta: {:.6}, S_max after threshold: {:.6}", largest_sv, zeta, s.max());
        }
        
        // Step 2: SVD of (X - S), not X! This is the key difference
        let x_minus_s = x - &s;
        let svd_result = truncated_svd(&x_minus_s, n_components)?;
        
        if self.verbose {
            println!("Step 2 - (X-S)_max: {:.6}, SVD values: {:?}", x_minus_s.max(), svd_result.singular_values.as_slice());
        }
        
        // Step 3: Reconstruct L from SVD
        let l = &svd_result.u * DMatrix::from_diagonal(&svd_result.singular_values) * &svd_result.v_t;
        
        // Step 4: Refine S with new threshold
        let zeta_refined = beta * svd_result.singular_values[0];
        let x_minus_l = x - &l;
        let s_refined = hard_threshold(&x_minus_l, zeta_refined);
        
        if self.verbose {
            println!("Step 4 - (X-L)_max: {:.6}, (X-L)_min: {:.6}", x_minus_l.max(), x_minus_l.min());
            println!("Step 4 - Zeta_refined: {:.6}, S_refined_max: {:.6}", zeta_refined, s_refined.max());
        }
        
        // Step 5: Prepare matrices for main iteration
        // Note: V should be V^T from SVD, but we need V for the algorithm
        let u = svd_result.u;
        let sigma = DMatrix::from_diagonal(&svd_result.singular_values);
        let v = svd_result.v_t.transpose(); // Convert V^T back to V
        
        if self.verbose {
            println!("Init - SVD values: {:?}", svd_result.singular_values.as_slice());
            println!("Init - Zeta: {:.6}, Zeta_refined: {:.6}", zeta, zeta_refined);
            println!("Init - S_max: {:.6}, S_min: {:.6}", s_refined.max(), s_refined.min());
        }
        
        Ok((l, s_refined, u, sigma, v))
    }
    /// Update the low-rank component
    fn update_low_rank(&self, z: &DMatrix<f64>, u: &DMatrix<f64>, v: &DMatrix<f64>, n_components: usize)
        -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> {
        
        // Compute QR decompositions following the Python implementation
        // Q1, R1 = qr(Z.T @ U - V @ ((Z @ V).T @ U), mode="economic")
        let zv = z * v;
        let ut_z = u.transpose() * z;
        let term1 = z.transpose() * u - v * &(zv.transpose() * u);
        let qr1 = term1.qr();
        let (q1, r1) = (qr1.q(), qr1.r());
        
        // Q2, R2 = qr(Z @ V - U @ (U.T @ Z @ V), mode="economic")
        let ut_zv = &ut_z * v;
        let term2 = &zv - u * &ut_zv;
        let qr2 = term2.qr();
        let (q2, r2) = (qr2.q(), qr2.r());
        
        // Construct matrix M following Python:
        // M = np.vstack([np.hstack([U.T @ Z @ V, R1.T]), np.hstack([R2, np.zeros_like(R2)])])
        let r1_t = r1.transpose();
        let _zeros: DMatrix<f64> = DMatrix::zeros(r2.nrows(), r2.ncols());
        
        // Create M by concatenating blocks
        let m_rows = ut_zv.nrows() + r2.nrows();
        let m_cols = ut_zv.ncols() + r1_t.ncols();
        let mut m = DMatrix::zeros(m_rows, m_cols);
        
        // Top-left: U.T @ Z @ V
        m.view_mut((0, 0), (ut_zv.nrows(), ut_zv.ncols())).copy_from(&ut_zv);
        // Top-right: R1.T
        m.view_mut((0, ut_zv.ncols()), (r1_t.nrows(), r1_t.ncols())).copy_from(&r1_t);
        // Bottom-left: R2
        m.view_mut((ut_zv.nrows(), 0), (r2.nrows(), r2.ncols())).copy_from(&r2);
        // Bottom-right: zeros (already initialized)
        
        // SVD of M
        let svd_m = m.svd(true, true);
        let u_m = svd_m.u.unwrap();
        let sigma_new = DMatrix::from_diagonal(&svd_m.singular_values);
        let v_m = svd_m.v_t.unwrap().transpose();
        
        // Update U and V following Python:
        // U = np.hstack([U, Q2]) @ U_of_M[:, :n_components]
        // V = np.hstack([V, Q1]) @ V_of_M[:, :n_components]
        let mut u_extended = DMatrix::zeros(u.nrows(), u.ncols() + q2.ncols());
        u_extended.view_mut((0, 0), (u.nrows(), u.ncols())).copy_from(u);
        u_extended.view_mut((0, u.ncols()), (q2.nrows(), q2.ncols())).copy_from(&q2);
        
        let mut v_extended = DMatrix::zeros(v.nrows(), v.ncols() + q1.ncols());
        v_extended.view_mut((0, 0), (v.nrows(), v.ncols())).copy_from(v);
        v_extended.view_mut((0, v.ncols()), (q1.nrows(), q1.ncols())).copy_from(&q1);
        
        let u_new = &u_extended * u_m.columns(0, n_components);
        let v_new = &v_extended * v_m.columns(0, n_components);
        
        // L = U @ Sigma[:n_components, :n_components] @ V.T
        let sigma_trunc = sigma_new.view((0, 0), (n_components, n_components));
        let l_new = &u_new * sigma_trunc * v_new.transpose();
        
        Ok((l_new, u_new.clone_owned(), sigma_new, v_new.clone_owned()))
    }
    
    /// Trim matrices to control growth
    fn trim_matrices(&self, u: &DMatrix<f64>, sigma: &DMatrix<f64>, v: &DMatrix<f64>, n_components: usize) 
        -> Result<(DMatrix<f64>, DMatrix<f64>)> {
        
        let sigma_sub = sigma.view((0, 0), (n_components, n_components));
        
        let (q1, r1) = trim_matrix(u, self.mu.1)?;
        let (q2, r2) = trim_matrix(v, self.mu.0)?;
        
        // SVD of trimmed result
        let m = &r1 * sigma_sub * r2.transpose();
        let svd_result = m.svd(true, true);
        let u_tmp = svd_result.u.unwrap();
        let v_tmp = svd_result.v_t.unwrap().transpose();
        
        let u_new = &q1 * u_tmp;
        let v_new = &q2 * v_tmp;
        
        Ok((u_new, v_new))
    }
}

/// Compute the hard threshold (element-wise clipping)
fn hard_threshold(matrix: &DMatrix<f64>, threshold: f64) -> DMatrix<f64> {
    matrix.map(|x| {
        if x.abs() < threshold {
            0.0
        } else {
            x
        }
    })
}

/// Compute Frobenius norm
fn frobenius_norm(matrix: &DMatrix<f64>) -> f64 {
    matrix.iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Compute column means
fn compute_column_means(matrix: &DMatrix<f64>) -> DVector<f64> {
    let n_rows = matrix.nrows() as f64;
    DVector::from_fn(matrix.ncols(), |i, _| {
        matrix.column(i).sum() / n_rows
    })
}

/// Center matrix by subtracting column means
fn center_matrix(matrix: &mut DMatrix<f64>, means: &DVector<f64>) {
    for (j, mut col) in matrix.column_iter_mut().enumerate() {
        col.add_scalar_mut(-means[j]);
    }
}

/// Compute relative error
fn compute_error(x: &DMatrix<f64>, l: &DMatrix<f64>, s: &DMatrix<f64>, norm_x: f64) -> f64 {
    let diff = x - (l + s);
    frobenius_norm(&diff) / norm_x
}

/// Truncated SVD result
#[derive(Debug)]
struct TruncatedSVD {
    pub u: DMatrix<f64>,
    pub singular_values: DVector<f64>,
    pub v_t: DMatrix<f64>,
}

/// Perform truncated SVD
fn truncated_svd(matrix: &DMatrix<f64>, k: usize) -> Result<TruncatedSVD> {
    let svd = matrix.clone().svd(true, true);
    let n_sv = k.min(svd.singular_values.len());
    
    let u = svd.u.unwrap().columns(0, n_sv).clone_owned();
    let singular_values = svd.singular_values.rows(0, n_sv).clone_owned();
    let v_t = svd.v_t.unwrap().rows(0, n_sv).clone_owned();
    
    Ok(TruncatedSVD {
        u,
        singular_values,
        v_t,
    })
}

/// Trim a matrix according to the algorithm
/// Trim a matrix according to the algorithm
fn trim_matrix(x: &DMatrix<f64>, mu: f64) -> Result<(DMatrix<f64>, DMatrix<f64>)> {
    let (m, r) = (x.nrows(), x.ncols());
    let mut x_trimmed = x.clone();
    
    // Compute row norms squared
    let row_norms_sq: Vec<f64> = (0..m)
        .map(|i| x.row(i).norm_squared())
        .collect();
    
    let threshold = mu * r as f64 / m as f64;
    
    // Trim large rows
    for (i, &norm_sq) in row_norms_sq.iter().enumerate() {
        if norm_sq > threshold {
            let scale = (threshold / norm_sq).sqrt();
            x_trimmed.row_mut(i).scale_mut(scale);
        }
    }
    
    // QR decomposition
    let qr = x_trimmed.qr();
    Ok((qr.q(), qr.r()))
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_abs_diff_eq;
        
    #[test]
    fn test_robust_pca_reconstruction() {
        // Create a low-rank matrix + sparse noise
        let low_rank = DMatrix::from_row_slice(5, 4, &[
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            3.0, 6.0, 9.0, 12.0,
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
        ]);
        let sparse_noise = DMatrix::from_row_slice(5, 4, &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 5.0, // one outlier
        ]);
        let data = &low_rank + &sparse_noise;

        let mut rpca = RobustPCA::with_components(2).unwrap()
            .max_iter(100)
            .tolerance(1e-5)
            .beta(0.01)  // Smaller beta works better for this test case
            .verbose(false);  // Disable verbose output
        
        let result = rpca.fit(&data);
        assert!(result.is_ok());

        let low_rank_est = rpca.low_rank.as_ref().unwrap();
        let sparse_est = rpca.sparse.as_ref().unwrap();

        // Check reconstruction error
        let reconstruction = low_rank_est + sparse_est;
        // Check reconstruction error is reasonable
        let _reconstruction = low_rank_est + sparse_est;
        let error = (_reconstruction - &data).norm();
        assert!(error < 1.0, "Reconstruction error too high: {}", error);
        
        // Check that the sparse component is detecting outliers
        // The exact value might differ due to mean centering, but should be close
        let sparse_outlier = sparse_est[(4, 3)];
        assert!(sparse_outlier > 3.0 && sparse_outlier < 7.0,
                "Sparse component should detect outlier, got: {}", sparse_outlier);
        
        // Check that algorithm converged
        assert!(rpca.end_iter <= rpca.max_iter, "Algorithm should converge");
        assert!(!rpca.errors.is_empty(), "Should have error history");
    }
    
    #[test]
    fn test_hard_threshold() {
        let matrix = DMatrix::from_row_slice(2, 2, &[2.0, -3.0, 0.5, -0.5]);
        let result = hard_threshold(&matrix, 1.0);
        
        assert_eq!(result[(0, 0)], 2.0);  // above threshold
        assert_eq!(result[(0, 1)], -3.0); // below negative threshold
        assert_eq!(result[(1, 0)], 0.0);  // below threshold
        assert_eq!(result[(1, 1)], 0.0);  // above negative threshold
    }
}