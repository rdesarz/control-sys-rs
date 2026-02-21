//! State-space model primitives and discretization utilities.
//!
//! This module defines continuous and discrete LTI state-space models:
//!
//! - Continuous: `x_dot = A x + B u`, `y = C x + D u`
//! - Discrete: `x[k+1] = A x[k] + B u[k]`, `y[k] = C x[k] + D u[k]`
//!
//! It also provides multiple continuous-to-discrete conversion methods:
//!
//! - Exact ZOH (augmented matrix exponential)
//! - Tustin (bilinear transform)

extern crate nalgebra as na;

use std::error::Error;
use std::fmt;

/// Error type for state-space model creation and transformations.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// State matrix `A` is not square.
    MatrixANotSquare {
        /// Number of rows in `A`.
        rows: usize,
        /// Number of columns in `A`.
        cols: usize,
    },
    /// State-space matrix dimensions are incompatible.
    DimensionMismatch {
        /// Shape of matrix `A`.
        a: (usize, usize),
        /// Shape of matrix `B`.
        b: (usize, usize),
        /// Shape of matrix `C`.
        c: (usize, usize),
        /// Shape of matrix `D`.
        d: (usize, usize),
    },
    /// Sampling time is invalid.
    InvalidSamplingDt(f64),
    /// Matrix inversion failed.
    SingularMatrix(&'static str),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::MatrixANotSquare { rows, cols } => {
                write!(f, "A must be square, got {rows}x{cols}")
            }
            ModelError::DimensionMismatch { a, b, c, d } => write!(
                f,
                "incompatible dimensions: A={:?}, B={:?}, C={:?}, D={:?}",
                a, b, c, d
            ),
            ModelError::InvalidSamplingDt(dt) => {
                write!(f, "sampling_dt must be finite and > 0.0, got {dt}")
            }
            ModelError::SingularMatrix(ctx) => write!(f, "matrix inversion failed: {ctx}"),
        }
    }
}

impl Error for ModelError {}

/// A trait representing a state-space model in control systems.
///
/// This trait provides methods to access the state-space matrices A, B, C, and D,
/// which are fundamental components of the state-space representation of a system.
///
pub trait StateSpaceModel {
    /// Returns a reference to the state matrix A.
    fn mat_a(&self) -> &na::DMatrix<f64>;

    /// Returns a reference to the input matrix B.
    fn mat_b(&self) -> &na::DMatrix<f64>;

    /// Returns a reference to the output matrix C.
    fn mat_c(&self) -> &na::DMatrix<f64>;

    /// Returns a reference to the feedthrough matrix D.
    fn mat_d(&self) -> &na::DMatrix<f64>;
}

/// A trait representing a discrete system with a specific sampling time.
///
/// This trait should be implemented by any type that represents a discrete system
/// and provides a method to retrieve the sampling time interval (`dt`).
///
/// # Examples
///
/// ```
/// use control_sys::model::Discrete;
///
/// struct MyDiscreteSystem {
///     sampling_dt: f64,
/// }
///
/// impl Discrete for MyDiscreteSystem {
///     fn sampling_dt(&self) -> f64 {
///         self.sampling_dt
///     }
/// }
///
/// let system = MyDiscreteSystem { sampling_dt: 0.1 };
/// assert_eq!(system.sampling_dt(), 0.1);
/// ```
///
pub trait Discrete {
    /// Returns the sampling time interval (`dt`) of the discrete system.
    fn sampling_dt(&self) -> f64;
}

/// A trait representing a system that has poles in the complex plane.
///
/// # Examples
///
/// ```
/// use nalgebra as na;
/// use control_sys::model::Pole;
///
/// struct MySystem;
///
/// impl Pole for MySystem {
///     fn poles(&self) -> Vec<na::Complex<f64>> {
///         vec![na::Complex::new(1.0, 2.0), na::Complex::new(3.0, 4.0)]
///     }
/// }
///
/// let system = MySystem;
/// let poles = system.poles();
/// assert_eq!(poles, vec![na::Complex::new(1.0, 2.0), na::Complex::new(3.0, 4.0)]);
/// ```
///
pub trait Pole {
    /// Returnes a vector of complex numbers representing the poles of the system.
    fn poles(&self) -> Vec<na::Complex<f64>>;
}

#[derive(Debug, Clone)]
/// A struct representing a continuous state-space model.
///
/// This model is defined by the following matrices:
/// - `mat_a`: The state matrix (A), which defines the system dynamics.
/// - `mat_b`: The input matrix (B), which defines how the input affects the state.
/// - `mat_c`: The output matrix (C), which defines how the state is mapped to the output.
/// - `mat_d`: The feedthrough matrix (D), which defines the direct path from input to output.
pub struct ContinuousStateSpaceModel {
    mat_a: na::DMatrix<f64>,
    mat_b: na::DMatrix<f64>,
    mat_c: na::DMatrix<f64>,
    mat_d: na::DMatrix<f64>,
}

/// Represents a continuous state-space model.
impl ContinuousStateSpaceModel {
    /// Creates a new `ContinuousStateSpaceModel` with full dimension checks.
    ///
    /// The following compatibility constraints are enforced:
    ///
    /// - `A` is square `(n x n)`
    /// - `B` is `(n x m)`
    /// - `C` is `(p x n)`
    /// - `D` is `(p x m)`
    pub fn try_from_matrices(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
    ) -> Result<ContinuousStateSpaceModel, ModelError> {
        validate_state_space_dimensions(mat_a, mat_b, mat_c, mat_d)?;

        Ok(ContinuousStateSpaceModel {
            mat_a: mat_a.clone(),
            mat_b: mat_b.clone(),
            mat_c: mat_c.clone(),
            mat_d: mat_d.clone(),
        })
    }

    /// Creates a new model without runtime checks.
    #[deprecated(note = "Use try_from_matrices for validated construction")]
    pub fn from_matrices(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
    ) -> ContinuousStateSpaceModel {
        Self::try_from_matrices(mat_a, mat_b, mat_c, mat_d).expect("invalid state-space dimensions")
    }

    /// Builds a controllable canonical form state-space model from a transfer function.
    #[cfg(test)]
    fn build_controllable_canonical_form(
        tf: &TransferFunction,
    ) -> Result<ContinuousStateSpaceModel, ModelError> {
        let n_states = tf.denominator_coeffs.len();

        let mut mat_a = na::DMatrix::<f64>::zeros(n_states, n_states);
        mat_a
            .view_range_mut(0..n_states - 1, 1..)
            .copy_from(&na::DMatrix::<f64>::identity(n_states - 1, n_states - 1));
        for (i, value) in tf.denominator_coeffs.iter().rev().enumerate() {
            mat_a[(n_states - 1, i)] = -*value;
        }

        let mut mat_b = na::DMatrix::<f64>::zeros(tf.numerator_coeffs.len(), 1);
        mat_b[(tf.numerator_coeffs.len() - 1, 0)] = 1.0f64;

        let mut mat_c = na::DMatrix::<f64>::zeros(1, n_states);
        for (i, value) in tf.numerator_coeffs.iter().rev().enumerate() {
            if i < n_states {
                mat_c[(0, i)] = *value;
            }
        }

        let mat_d = na::dmatrix![tf.constant];

        Self::try_from_matrices(&mat_a, &mat_b, &mat_c, &mat_d)
    }

    /// Returns the number of states `n` of the model.
    ///
    /// This equals the number of columns of `A` (and rows of `A` since `A` is square).
    pub fn state_space_size(&self) -> usize {
        self.mat_a.ncols()
    }
}

impl StateSpaceModel for ContinuousStateSpaceModel {
    fn mat_a(&self) -> &na::DMatrix<f64> {
        &self.mat_a
    }

    fn mat_b(&self) -> &na::DMatrix<f64> {
        &self.mat_b
    }

    fn mat_c(&self) -> &na::DMatrix<f64> {
        &self.mat_c
    }

    fn mat_d(&self) -> &na::DMatrix<f64> {
        &self.mat_d
    }
}

impl Pole for ContinuousStateSpaceModel {
    fn poles(&self) -> Vec<na::Complex<f64>> {
        self.mat_a.complex_eigenvalues().iter().cloned().collect()
    }
}

#[derive(Debug, Clone)]
/// A struct representing a discrete state-space model.
///
/// This model is defined by the following matrices:
/// - `mat_a`: The state transition matrix.
/// - `mat_b`: The control input matrix.
/// - `mat_c`: The output matrix.
/// - `mat_d`: The feedthrough (or direct transmission) matrix.
///
/// Additionally, the model includes a sampling time `sampling_dt` which represents the time interval between each discrete step.
pub struct DiscreteStateSpaceModel {
    mat_a: na::DMatrix<f64>,
    mat_b: na::DMatrix<f64>,
    mat_c: na::DMatrix<f64>,
    mat_d: na::DMatrix<f64>,
    sampling_dt: f64,
}

impl StateSpaceModel for DiscreteStateSpaceModel {
    fn mat_a(&self) -> &na::DMatrix<f64> {
        &self.mat_a
    }

    fn mat_b(&self) -> &na::DMatrix<f64> {
        &self.mat_b
    }

    fn mat_c(&self) -> &na::DMatrix<f64> {
        &self.mat_c
    }

    fn mat_d(&self) -> &na::DMatrix<f64> {
        &self.mat_d
    }
}

impl DiscreteStateSpaceModel {
    /// Creates a new `DiscreteStateSpaceModel` with matrix and sampling-time checks.
    ///
    /// Matrix compatibility is identical to [`ContinuousStateSpaceModel::try_from_matrices`],
    /// with the additional constraint that `sampling_dt > 0` and finite.
    pub fn try_from_matrices(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
        sampling_dt: f64,
    ) -> Result<DiscreteStateSpaceModel, ModelError> {
        validate_state_space_dimensions(mat_a, mat_b, mat_c, mat_d)?;
        validate_sampling_dt(sampling_dt)?;

        Ok(DiscreteStateSpaceModel {
            mat_a: mat_a.clone(),
            mat_b: mat_b.clone(),
            mat_c: mat_c.clone(),
            mat_d: mat_d.clone(),
            sampling_dt,
        })
    }

    /// Creates a new model without runtime checks.
    #[deprecated(note = "Use try_from_matrices for validated construction")]
    pub fn from_matrices(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
        sampling_dt: f64,
    ) -> DiscreteStateSpaceModel {
        Self::try_from_matrices(mat_a, mat_b, mat_c, mat_d, sampling_dt)
            .expect("invalid state-space dimensions")
    }

    /// Discretizes a continuous model using exact zero-order hold (ZOH).
    ///
    /// For continuous dynamics `x_dot = A x + B u` with piecewise-constant input,
    /// this computes:
    ///
    /// - `A_d = exp(A * dt)`
    /// - `B_d = integral_0^dt exp(A * tau) d tau * B`
    ///
    /// The implementation uses the standard augmented matrix exponential:
    ///
    /// ```text
    /// exp([A*dt  B*dt]
    ///     [  0      0 ]) = [A_d  B_d]
    ///                       [ 0    I ]
    /// ```
    pub fn from_continuous_matrix_zoh(
        mat_ac: &na::DMatrix<f64>,
        mat_bc: &na::DMatrix<f64>,
        mat_cc: &na::DMatrix<f64>,
        mat_dc: &na::DMatrix<f64>,
        sampling_dt: f64,
    ) -> Result<DiscreteStateSpaceModel, ModelError> {
        validate_state_space_dimensions(mat_ac, mat_bc, mat_cc, mat_dc)?;
        validate_sampling_dt(sampling_dt)?;

        let n_states = mat_ac.nrows();
        let n_inputs = mat_bc.ncols();
        let mut aug = na::DMatrix::<f64>::zeros(n_states + n_inputs, n_states + n_inputs);
        aug.view_mut((0, 0), (n_states, n_states))
            .copy_from(&(mat_ac * sampling_dt));
        aug.view_mut((0, n_states), (n_states, n_inputs))
            .copy_from(&(mat_bc * sampling_dt));

        let exp_aug = matrix_exponential(&aug);
        let mat_a = exp_aug.view((0, 0), (n_states, n_states)).into_owned();
        let mat_b = exp_aug
            .view((0, n_states), (n_states, n_inputs))
            .into_owned();

        Self::try_from_matrices(&mat_a, &mat_b, mat_cc, mat_dc, sampling_dt)
    }

    /// Discretizes a continuous model using the bilinear/Tustin transform.
    ///
    /// This corresponds to:
    ///
    /// - `A_d = (I - A*dt/2)^-1 * (I + A*dt/2)`
    /// - `B_d = (I - A*dt/2)^-1 * B*dt`
    ///
    /// It requires inversion of `I - A*dt/2`.
    pub fn from_continuous_matrix_tustin(
        mat_ac: &na::DMatrix<f64>,
        mat_bc: &na::DMatrix<f64>,
        mat_cc: &na::DMatrix<f64>,
        mat_dc: &na::DMatrix<f64>,
        sampling_dt: f64,
    ) -> Result<DiscreteStateSpaceModel, ModelError> {
        validate_state_space_dimensions(mat_ac, mat_bc, mat_cc, mat_dc)?;
        validate_sampling_dt(sampling_dt)?;

        let mat_i = na::DMatrix::<f64>::identity(mat_ac.nrows(), mat_ac.nrows());
        let left = mat_i.clone() - mat_ac.scale(0.5 * sampling_dt);
        let inv_left = left
            .try_inverse()
            .ok_or(ModelError::SingularMatrix("I - A*dt/2 for Tustin"))?;

        let mat_a = &inv_left * (mat_i + mat_ac.scale(0.5 * sampling_dt));
        let mat_b = inv_left * mat_bc.scale(sampling_dt);

        Self::try_from_matrices(&mat_a, &mat_b, mat_cc, mat_dc, sampling_dt)
    }

    /// Discretizes a continuous model using exact ZOH.
    ///
    /// See [`DiscreteStateSpaceModel::from_continuous_matrix_zoh`] for equations.
    pub fn from_continuous_zoh(
        model: &ContinuousStateSpaceModel,
        sampling_dt: f64,
    ) -> Result<DiscreteStateSpaceModel, ModelError> {
        Self::from_continuous_matrix_zoh(
            model.mat_a(),
            model.mat_b(),
            model.mat_c(),
            model.mat_d(),
            sampling_dt,
        )
    }

    /// Discretizes a continuous model using Tustin.
    ///
    /// See [`DiscreteStateSpaceModel::from_continuous_matrix_tustin`] for equations.
    pub fn from_continuous_tustin(
        model: &ContinuousStateSpaceModel,
        sampling_dt: f64,
    ) -> Result<DiscreteStateSpaceModel, ModelError> {
        Self::from_continuous_matrix_tustin(
            model.mat_a(),
            model.mat_b(),
            model.mat_c(),
            model.mat_d(),
            sampling_dt,
        )
    }

}

impl Pole for DiscreteStateSpaceModel {
    fn poles(&self) -> Vec<na::Complex<f64>> {
        self.mat_a.complex_eigenvalues().iter().cloned().collect()
    }
}

impl Discrete for DiscreteStateSpaceModel {
    fn sampling_dt(&self) -> f64 {
        self.sampling_dt
    }
}

#[cfg(test)]
struct TransferFunction {
    numerator_coeffs: Vec<f64>,
    denominator_coeffs: Vec<f64>,
    constant: f64,
}

#[cfg(test)]
impl TransferFunction {
    fn new(
        numerator_coeffs: &[f64],
        denominator_coeffs: &[f64],
        constant: f64,
    ) -> TransferFunction {
        TransferFunction {
            numerator_coeffs: numerator_coeffs.to_vec(),
            denominator_coeffs: denominator_coeffs.to_vec(),
            constant,
        }
    }
}

fn validate_sampling_dt(sampling_dt: f64) -> Result<(), ModelError> {
    if !sampling_dt.is_finite() || sampling_dt <= 0.0 {
        return Err(ModelError::InvalidSamplingDt(sampling_dt));
    }

    Ok(())
}

fn validate_state_space_dimensions(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
    mat_c: &na::DMatrix<f64>,
    mat_d: &na::DMatrix<f64>,
) -> Result<(), ModelError> {
    if !mat_a.is_square() {
        return Err(ModelError::MatrixANotSquare {
            rows: mat_a.nrows(),
            cols: mat_a.ncols(),
        });
    }

    let n_states = mat_a.nrows();
    let n_inputs = mat_b.ncols();
    let n_outputs = mat_c.nrows();

    if mat_b.nrows() != n_states
        || mat_c.ncols() != n_states
        || mat_d.nrows() != n_outputs
        || mat_d.ncols() != n_inputs
    {
        return Err(ModelError::DimensionMismatch {
            a: mat_a.shape(),
            b: mat_b.shape(),
            c: mat_c.shape(),
            d: mat_d.shape(),
        });
    }

    Ok(())
}

fn matrix_exponential(mat: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let n = mat.nrows();
    if n == 0 {
        return na::DMatrix::<f64>::zeros(0, 0);
    }

    // Scaling-and-series approximation:
    // 1) choose a power-of-two scale so ||A/scale|| is small,
    // 2) evaluate exp(A/scale) by truncated Taylor series,
    // 3) square the result repeatedly to recover exp(A).
    let norm_one = max_column_sum_norm(mat);
    let scaling_power = if norm_one <= 0.5 {
        0
    } else {
        (norm_one.log2().ceil().max(0.0) as u32) + 1
    };

    let scale = 2f64.powi(scaling_power as i32);
    let scaled = mat / scale;

    let identity = na::DMatrix::<f64>::identity(n, n);
    let mut result = identity.clone();
    let mut term = identity;

    // exp(M) = I + M + M^2/2! + M^3/3! + ...
    for k in 1..=64 {
        term = (&term * &scaled) / (k as f64);
        result += &term;

        let max_abs = term.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if max_abs < 1e-14 {
            break;
        }
    }

    let mut out = result;
    for _ in 0..scaling_power {
        out = &out * &out;
    }

    out
}

fn max_column_sum_norm(mat: &na::DMatrix<f64>) -> f64 {
    (0..mat.ncols())
        .map(|col| {
            (0..mat.nrows())
                .map(|row| mat[(row, col)].abs())
                .sum::<f64>()
        })
        .fold(0.0f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_matrix(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>, tol: f64) {
        assert_eq!(a.shape(), b.shape());
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() <= tol,
                    "mismatch at ({i},{j}): {} != {}",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    // Verifies checked constructors reject incompatible A/B/C/D dimensions.
    #[test]
    fn test_try_from_matrices_dimension_validation() {
        let err = DiscreteStateSpaceModel::try_from_matrices(
            &na::dmatrix![1.0, 0.0; 0.0, 1.0],
            &na::dmatrix![1.0; 0.0],
            &na::dmatrix![1.0, 0.0],
            &na::dmatrix![0.0, 0.0],
            0.1,
        )
        .unwrap_err();

        assert!(matches!(err, ModelError::DimensionMismatch { .. }));
    }

    // Verifies controllable canonical form matrices for a nominal transfer function.
    #[test]
    fn test_compute_state_space_model_nominal() {
        let tf = TransferFunction::new(&[1.0, 2.0, 3.0], &[1.0, 4.0, 6.0], 8.0);

        let ss_model = ContinuousStateSpaceModel::build_controllable_canonical_form(&tf).unwrap();

        assert_eq!(ss_model.mat_a().shape(), (3, 3));
        assert_eq!(ss_model.mat_a()[(2, 0)], -6.0f64);
        assert_eq!(ss_model.mat_a()[(2, 1)], -4.0f64);
        assert_eq!(ss_model.mat_a()[(2, 2)], -1.0f64);
        assert_eq!(ss_model.mat_a()[(0, 1)], 1.0f64);
        assert_eq!(ss_model.mat_a()[(1, 2)], 1.0f64);

        assert_eq!(ss_model.mat_b().shape(), (3, 1));
        assert_eq!(ss_model.mat_b()[(0, 0)], 0.0f64);
        assert_eq!(ss_model.mat_b()[(1, 0)], 0.0f64);
        assert_eq!(ss_model.mat_b()[(2, 0)], 1.0f64);

        assert_eq!(ss_model.mat_c().shape(), (1, 3));
        assert_eq!(ss_model.mat_c()[(0, 0)], 3.0f64);
        assert_eq!(ss_model.mat_c()[(0, 1)], 2.0f64);
        assert_eq!(ss_model.mat_c()[(0, 2)], 1.0f64);

        assert_eq!(ss_model.mat_d().shape(), (1, 1));
        assert_eq!(ss_model.mat_d()[(0, 0)], 8.0f64);
    }

    // Verifies Tustin discretization against first-order analytic values.
    #[test]
    fn test_discretization_tustin_first_order() {
        let a = na::dmatrix![-2.0];
        let b = na::dmatrix![1.0];
        let c = na::dmatrix![1.0];
        let d = na::dmatrix![0.0];
        let dt = 0.1;

        let model =
            DiscreteStateSpaceModel::from_continuous_matrix_tustin(&a, &b, &c, &d, dt).unwrap();

        approx_eq_matrix(model.mat_a(), &na::dmatrix![0.8181818181818182], 1e-12);
        approx_eq_matrix(model.mat_b(), &na::dmatrix![0.09090909090909091], 1e-12);
    }

    // Verifies ZOH discretization against first-order exact discretization values.
    #[test]
    fn test_discretization_zoh_first_order() {
        let a = na::dmatrix![-2.0];
        let b = na::dmatrix![1.0];
        let c = na::dmatrix![1.0];
        let d = na::dmatrix![0.0];
        let dt = 0.1;

        let model =
            DiscreteStateSpaceModel::from_continuous_matrix_zoh(&a, &b, &c, &d, dt).unwrap();

        let ad = (-2.0f64 * dt).exp();
        let bd = (1.0 - ad) / 2.0;

        approx_eq_matrix(model.mat_a(), &na::dmatrix![ad], 1e-10);
        approx_eq_matrix(model.mat_b(), &na::dmatrix![bd], 1e-10);
    }

    // Verifies pole computation for a model with purely real poles.
    #[test]
    fn test_compute_poles_pure_real() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![2.0, 0.0; 0.0, 1.0],
            &nalgebra::dmatrix![0.0; 0.0],
            &nalgebra::dmatrix![0.0, 0.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        let poles = ss_model.poles();

        assert_eq!(poles.len(), 2);
        assert_eq!(poles[0].re, 2.0);
        assert_eq!(poles[0].im, 0.0);
        assert_eq!(poles[1].re, 1.0);
        assert_eq!(poles[1].im, 0.0);
    }

    // Verifies pole computation for a model with purely imaginary poles.
    #[test]
    fn test_compute_poles_pure_im() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![0.0, -1.0; 1.0, 0.0],
            &nalgebra::dmatrix![0.0; 0.0],
            &nalgebra::dmatrix![0.0, 0.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        let poles = ss_model.poles();

        assert_eq!(poles.len(), 2);
        assert_eq!(poles[0].re, 0.0);
        assert_eq!(poles[0].im, 1.0);
        assert_eq!(poles[1].re, 0.0);
        assert_eq!(poles[1].im, -1.0);
    }

    // Verifies pole computation for a model with complex-conjugate poles.
    #[test]
    fn test_compute_poles_real_and_imaginary_part() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![1.0, -2.0; 2.0, 1.0],
            &nalgebra::dmatrix![0.0; 0.0],
            &nalgebra::dmatrix![0.0, 0.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        let poles = ss_model.poles();

        assert_eq!(poles.len(), 2);
        assert_eq!(poles[0].re, 1.0);
        assert_eq!(poles[0].im, 2.0);
        assert_eq!(poles[1].re, 1.0);
        assert_eq!(poles[1].im, -2.0);
    }
}
