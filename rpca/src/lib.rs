use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2};
use nalgebra::DMatrix;
use pyo3::types::PyDict;
use pyo3::Python;
use pyo3::PyObject;

// Re-export the main RobustPCA implementation
mod robust_pca;
use robust_pca::RobustPCA as RustRobustPCA;

/// Python wrapper for the Rust RobustPCA implementation
#[pyclass]
struct RobustPCA {
    inner: RustRobustPCA,
}

#[pymethods]
impl RobustPCA {
    /// Create a new RobustPCA instance
    #[new]
    #[pyo3(signature = (n_components=None, max_iter=100, tol=1e-5, beta=None, beta_init=None, gamma=0.5, mu=(5.0, 5.0), trim=false, verbose=true, copy=true))]
    fn new(
        n_components: Option<usize>,
        max_iter: usize,
        tol: f64,
        beta: Option<f64>,
        beta_init: Option<f64>,
        gamma: f64,
        mu: (f64, f64),
        trim: bool,
        verbose: bool,
        copy: bool,
    ) -> PyResult<Self> {
        let mut inner = if let Some(n_comp) = n_components {
            RustRobustPCA::with_components(n_comp)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            RustRobustPCA::new()
        };
        
        inner.max_iter = max_iter;
        inner.tol = tol;
        inner.beta = beta;
        inner.beta_init = beta_init;
        inner.gamma = gamma;
        inner.mu = mu;
        inner.trim = trim;
        inner.verbose = verbose;
        inner.copy = copy;
        
        Ok(Self { inner })
    }
    
    /// Fit the RobustPCA model to the input data
    /// Fit the RobustPCA model to the input data
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_matrix = numpy_to_nalgebra(x)?;
        self.inner.fit(&x_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
    
    /// Transform data using the fitted model
    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
        if !self.inner.fitted {
            return Err(PyValueError::new_err("Model not fitted yet"));
        }
        
        let x_matrix = numpy_to_nalgebra(x)?;
        if let Some(ref components) = self.inner.components {
            // x_matrix: (n_samples, n_features)
            // components: (n_components, n_features)
            // We need: x_matrix * components^T
            let result = &x_matrix * components.transpose();
            Ok(nalgebra_to_numpy(py, &result))
        } else {
            Err(PyValueError::new_err("Model not fitted yet"))
        }
    }
    
    /// Fit the model and transform the data in one step
    fn fit_transform<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
        let x_matrix = numpy_to_nalgebra(x)?;
        let result = self.inner.fit_transform(&x_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(nalgebra_to_numpy(py, &result))
    }
    
    /// Get the low-rank component
    #[getter]
    fn low_rank_(&self, py: Python) -> PyResult<Option<PyObject>> {
        Ok(self.inner.low_rank.as_ref().map(|m| nalgebra_to_numpy(py, m).into()))
    }
    
    /// Get the sparse component
    #[getter]
    fn sparse_(&self, py: Python) -> PyResult<Option<PyObject>> {
        Ok(self.inner.sparse.as_ref().map(|m| nalgebra_to_numpy(py, m).into()))
    }
    
    /// Get the principal components
    #[getter]
    fn components_(&self, py: Python) -> PyResult<Option<PyObject>> {
        Ok(self.inner.components.as_ref().map(|m| nalgebra_to_numpy(py, m).into()))
    }

    /// Transform data back to its original space
    fn inverse_transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
        if !self.inner.fitted {
            return Err(PyValueError::new_err("Model not fitted yet"));
        }
        
        let x_matrix = numpy_to_nalgebra(x)?;
        if let Some(ref components) = self.inner.components {
            let result = &x_matrix * components;
            Ok(nalgebra_to_numpy(py, &result))
        } else {
            Err(PyValueError::new_err("Model not fitted yet"))
        }
    }
    
    /// Get the singular values
    #[getter]
    fn singular_values<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<f64>>> {
        match &self.inner.singular_values {
            Some(vector) => {
                let matrix = DMatrix::from_column_slice(vector.len(), 1, vector.as_slice());
                Ok(Some(nalgebra_to_numpy(py, &matrix)))
            },
            None => Ok(None),
        }
    }
    
    /// Get the convergence errors
    #[getter]
    fn errors(&self) -> Vec<f64> {
        self.inner.errors.clone()
    }
    
    /// Get the number of iterations performed
    #[getter]
    fn n_iter_(&self) -> usize {
        self.inner.end_iter
    }
    
    /// Get the number of components
    #[getter]
    fn n_components(&self) -> Option<usize> {
        self.inner.n_components
    }
    
    /// Get the fitted number of components
    #[getter]
    fn n_components_(&self) -> Option<usize> {
        self.inner.n_components_fitted
    }
    
    /// Check if the model has been fitted
    #[getter]
    fn is_fitted(&self) -> bool {
        self.inner.fitted
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "RobustPCA(n_components={:?}, max_iter={}, tol={}, fitted={})",
            self.inner.n_components,
            self.inner.max_iter,
            self.inner.tol,
            self.inner.fitted
        )
    }
    
    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Convert numpy array to nalgebra matrix
fn numpy_to_nalgebra(array: PyReadonlyArray2<f64>) -> PyResult<DMatrix<f64>> {
    let array = array.as_array();
    let (rows, cols) = array.dim();
    
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(array[[i, j]]);
        }
    }
    
    Ok(DMatrix::from_row_slice(rows, cols, &data))
}

/// Convert nalgebra matrix to numpy array
fn nalgebra_to_numpy<'py>(py: Python<'py>, matrix: &DMatrix<f64>) -> &'py PyArray2<f64> {
    PyArray2::from_vec2(py, &matrix.row_iter()
        .map(|row| row.iter().cloned().collect())
        .collect::<Vec<Vec<f64>>>())
        .unwrap()
}

/// Create a RobustPCA instance with specified components
#[pyfunction]
#[pyo3(signature = (n_components, **kwargs))]
fn robust_pca_with_components(n_components: usize, kwargs: Option<&PyDict>) -> PyResult<RobustPCA> {
    let mut max_iter = 100;
    let mut tol = 1e-5;
    let mut beta = None;
    let mut beta_init = None;
    let mut gamma = 0.5;
    let mut mu = (5.0, 5.0);
    let mut trim = false;
    let mut verbose = true;
    let mut copy = true;
    
    if let Some(kwargs) = kwargs {
        if let Some(val) = kwargs.get_item("max_iter")? {
            max_iter = val.extract()?;
        }
        if let Some(val) = kwargs.get_item("tol")? {
            tol = val.extract()?;
        }
        if let Some(val) = kwargs.get_item("beta")? {
            beta = Some(val.extract()?);
        }
        if let Some(val) = kwargs.get_item("beta_init")? {
            beta_init = Some(val.extract()?);
        }
        if let Some(val) = kwargs.get_item("gamma")? {
            gamma = val.extract()?;
        }
        if let Some(val) = kwargs.get_item("mu")? {
            mu = val.extract()?;
        }
        if let Some(val) = kwargs.get_item("trim")? {
            trim = val.extract()?;
        }
        if let Some(val) = kwargs.get_item("verbose")? {
            verbose = val.extract()?;
        }
        if let Some(val) = kwargs.get_item("copy")? {
            copy = val.extract()?;
        }
    }
    
    RobustPCA::new(Some(n_components), max_iter, tol, beta, beta_init, gamma, mu, trim, verbose, copy)
}

/// Python module definition
#[pymodule]
fn rpca(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RobustPCA>()?;
    m.add_function(wrap_pyfunction!(robust_pca_with_components, m)?)?;
    
    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Rust RobustPCA")?;
    m.add("__doc__", "Fast Rust implementation of Robust PCA with Python bindings")?;
    
    Ok(())
}
