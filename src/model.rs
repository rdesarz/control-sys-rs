extern crate nalgebra as na;

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
    /// Creates a new `ContinuousStateSpaceModel` with the given matrices.
    ///
    /// # Arguments
    ///
    /// * `mat_a` - State matrix (A).
    /// * `mat_b` - Input matrix (B).
    /// * `mat_c` - Output matrix (C).
    /// * `mat_d` - Feedthrough matrix (D).
    ///
    /// # Returns
    ///
    /// A new instance of `ContinuousStateSpaceModel`.
    pub fn from_matrices(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
    ) -> ContinuousStateSpaceModel {
        ContinuousStateSpaceModel {
            mat_a: mat_a.clone(),
            mat_b: mat_b.clone(),
            mat_c: mat_c.clone(),
            mat_d: mat_d.clone(),
        }
    }

    /// Builds a controllable canonical form state-space model from a transfer function.
    ///
    /// # Arguments
    ///
    /// * `tf` - The transfer function to convert.
    ///
    /// # Returns
    ///
    /// A `ContinuousStateSpaceModel` in controllable canonical form.
    fn build_controllable_canonical_form(tf: &TransferFunction) -> ContinuousStateSpaceModel {
        // TODO: Still need to normalize coefficients and check for size
        let n_states = tf.denominator_coeffs.len();

        let mut mat_a = na::DMatrix::<f64>::zeros(n_states, n_states);
        mat_a
            .view_range_mut(0..n_states - 1, 1..)
            .copy_from(&na::DMatrix::<f64>::identity(n_states - 1, n_states - 1));
        for (i, value) in tf.denominator_coeffs.iter().rev().enumerate() {
            mat_a[(n_states - 1, i)] = -value.clone();
        }

        let mut mat_b = na::DMatrix::<f64>::zeros(tf.numerator_coeffs.len(), 1);
        mat_b[(tf.numerator_coeffs.len() - 1, 0)] = 1.0f64;

        let mut mat_c = na::DMatrix::<f64>::zeros(tf.numerator_coeffs.len(), 1);
        for (i, value) in tf.numerator_coeffs.iter().rev().enumerate() {
            mat_c[(i, 0)] = value.clone();
        }

        let mat_d = na::dmatrix![tf.constant];

        ContinuousStateSpaceModel {
            mat_a: mat_a,
            mat_b: mat_b,
            mat_c: mat_c,
            mat_d: mat_d,
        }
    }

    /// Returns the size of the state-space model.
    ///
    /// # Returns
    ///
    /// The number of states in the state-space model.
    pub fn state_space_size(&self) -> usize {
        return self.mat_a.ncols();
    }
}

impl StateSpaceModel for ContinuousStateSpaceModel {
    fn mat_a(&self) -> &na::DMatrix<f64> {
        return &self.mat_a;
    }

    fn mat_b(&self) -> &na::DMatrix<f64> {
        return &self.mat_b;
    }

    fn mat_c(&self) -> &na::DMatrix<f64> {
        return &self.mat_c;
    }

    fn mat_d(&self) -> &na::DMatrix<f64> {
        return &self.mat_d;
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
///
/// # Fields
/// - `mat_a` (`na::DMatrix<f64>`): The state transition matrix.
/// - `mat_b` (`na::DMatrix<f64>`): The control input matrix.
/// - `mat_c` (`na::DMatrix<f64>`): The output matrix.
/// - `mat_d` (`na::DMatrix<f64>`): The feedthrough matrix.
/// - `sampling_dt` (f64): The sampling time interval.
pub struct DiscreteStateSpaceModel {
    mat_a: na::DMatrix<f64>,
    mat_b: na::DMatrix<f64>,
    mat_c: na::DMatrix<f64>,
    mat_d: na::DMatrix<f64>,
    sampling_dt: f64,
}

impl StateSpaceModel for DiscreteStateSpaceModel {
    fn mat_a(&self) -> &na::DMatrix<f64> {
        return &self.mat_a;
    }

    fn mat_b(&self) -> &na::DMatrix<f64> {
        return &self.mat_b;
    }

    fn mat_c(&self) -> &na::DMatrix<f64> {
        return &self.mat_c;
    }

    fn mat_d(&self) -> &na::DMatrix<f64> {
        return &self.mat_d;
    }
}

impl DiscreteStateSpaceModel {
    /// Creates a new `DiscreteStateSpaceModel` with the given state-space matrices and sampling time.
    ///
    /// # Arguments
    ///
    /// * `mat_a` - State transition matrix.
    /// * `mat_b` - Control input matrix.
    /// * `mat_c` - Observation matrix.
    /// * `mat_d` - Feedforward matrix.
    /// * `sampling_dt` - Sampling time interval.
    ///
    /// # Returns
    ///
    /// A new `DiscreteStateSpaceModel` instance.
    pub fn from_matrices(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
        sampling_dt: f64,
    ) -> DiscreteStateSpaceModel {
        DiscreteStateSpaceModel {
            mat_a: mat_a.clone(),
            mat_b: mat_b.clone(),
            mat_c: mat_c.clone(),
            mat_d: mat_d.clone(),
            sampling_dt: sampling_dt,
        }
    }

    /// Converts a continuous state-space model to a discrete state-space model using the forward Euler method.
    ///
    /// # Arguments
    ///
    /// * `mat_ac` - Continuous state transition matrix.
    /// * `mat_bc` - Continuous control input matrix.
    /// * `mat_cc` - Continuous observation matrix.
    /// * `mat_dc` - Continuous feedforward matrix.
    /// * `sampling_dt` - Sampling time interval.
    ///
    /// # Returns
    ///
    /// A new `DiscreteStateSpaceModel` instance.
    pub fn from_continuous_matrix_forward_euler(
        mat_ac: &na::DMatrix<f64>,
        mat_bc: &na::DMatrix<f64>,
        mat_cc: &na::DMatrix<f64>,
        mat_dc: &na::DMatrix<f64>,
        sampling_dt: f64,
    ) -> DiscreteStateSpaceModel {
        let mat_i = na::DMatrix::<f64>::identity(mat_ac.nrows(), mat_ac.nrows());
        let mat_a = (mat_i - mat_ac.scale(sampling_dt)).try_inverse().unwrap();
        let mat_b = &mat_a * mat_bc.scale(sampling_dt);
        let mat_c = mat_cc.clone();
        let mat_d = mat_dc.clone();

        DiscreteStateSpaceModel {
            mat_a: mat_a,
            mat_b: mat_b,
            mat_c: mat_c,
            mat_d: mat_d,
            sampling_dt: sampling_dt,
        }
    }

    /// Converts a continuous state-space model to a discrete state-space model using the forward Euler method.
    ///
    /// # Arguments
    ///
    /// * `model` - A reference to a `ContinuousStateSpaceModel` instance.
    /// * `sampling_dt` - Sampling time interval.
    ///
    /// # Returns
    ///
    /// A new `DiscreteStateSpaceModel` instance.
    pub fn from_continuous_ss_forward_euler(
        model: &ContinuousStateSpaceModel,
        sampling_dt: f64,
    ) -> DiscreteStateSpaceModel {
        Self::from_continuous_matrix_forward_euler(
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
        return self.sampling_dt;
    }
}

struct TransferFunction {
    numerator_coeffs: Vec<f64>,
    denominator_coeffs: Vec<f64>,
    constant: f64,
}

impl TransferFunction {
    fn new(
        numerator_coeffs: &[f64],
        denominator_coeffs: &[f64],
        constant: f64,
    ) -> TransferFunction {
        TransferFunction {
            numerator_coeffs: numerator_coeffs.to_vec(),
            denominator_coeffs: denominator_coeffs.to_vec(),
            constant: constant,
        }
    }
}

#[cfg(test)]
/// This module contains unit tests for the control system models.
///
/// # Tests
///
/// - `test_compute_state_space_model_nominal`: Tests the construction of a continuous state-space model in controllable canonical form from a transfer function and verifies the matrices A, B, C, and D.
/// - `test_compute_poles_pure_real`: Tests the computation of poles for a discrete state-space model with purely real eigenvalues.
/// - `test_compute_poles_pure_im`: Tests the computation of poles for a discrete state-space model with purely imaginary eigenvalues.
/// - `test_compute_poles_real_and_imaginary_part`: Tests the computation of poles for a discrete state-space model with both real and imaginary parts.
/// - `test_compute_controllability_matrix_nominal`: Tests the computation of the controllability matrix for a given state-space model.
/// - `test_controllability_2x2_controllable`: Tests the controllability of a 2x2 discrete state-space model that is controllable.
/// - `test_controllability_3x3_not_controllable`: Tests the controllability of a 3x3 discrete state-space model that is not controllable.
mod tests {
    use super::*;

    #[test]
    fn test_compute_state_space_model_nominal() {
        let tf = TransferFunction::new(&[1.0, 2.0, 3.0], &[1.0, 4.0, 6.0], 8.0);

        let ss_model = ContinuousStateSpaceModel::build_controllable_canonical_form(&tf);

        // Check mat A
        assert_eq!(ss_model.mat_a().shape(), (3, 3));
        assert_eq!(ss_model.mat_a()[(2, 0)], -6.0f64);
        assert_eq!(ss_model.mat_a()[(2, 1)], -4.0f64);
        assert_eq!(ss_model.mat_a()[(2, 2)], -1.0f64);
        assert_eq!(ss_model.mat_a()[(0, 1)], 1.0f64);
        assert_eq!(ss_model.mat_a()[(1, 2)], 1.0f64);

        // Check mat B
        assert_eq!(ss_model.mat_b().shape(), (3, 1));
        assert_eq!(ss_model.mat_b()[(0, 0)], 0.0f64);
        assert_eq!(ss_model.mat_b()[(1, 0)], 0.0f64);
        assert_eq!(ss_model.mat_b()[(2, 0)], 1.0f64);

        // Check mat C
        assert_eq!(ss_model.mat_c().shape(), (3, 1));
        assert_eq!(ss_model.mat_c()[(0, 0)], 3.0f64);
        assert_eq!(ss_model.mat_c()[(1, 0)], 2.0f64);
        assert_eq!(ss_model.mat_c()[(2, 0)], 1.0f64);

        // Check mat D
        assert_eq!(ss_model.mat_d().shape(), (1, 1));
        assert_eq!(ss_model.mat_d()[(0, 0)], 8.0f64);
    }

    #[test]
    fn test_compute_poles_pure_real() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![2.0, 0.0; 0.0, 1.0],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            0.05,
        );

        let poles = ss_model.poles();

        assert_eq!(poles.len(), 2);
        assert_eq!(poles[0].re, 2.0);
        assert_eq!(poles[0].im, 0.0);
        assert_eq!(poles[1].re, 1.0);
        assert_eq!(poles[1].im, 0.0);
    }

    #[test]
    fn test_compute_poles_pure_im() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![0.0, -1.0; 1.0, 0.0],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            0.05,
        );

        let poles = ss_model.poles();

        assert_eq!(poles.len(), 2);
        assert_eq!(poles[0].re, 0.0);
        assert_eq!(poles[0].im, 1.0);
        assert_eq!(poles[1].re, 0.0);
        assert_eq!(poles[1].im, -1.0);
    }

    #[test]
    fn test_compute_poles_real_and_imaginary_part() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![1.0, -2.0; 2.0, 1.0],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
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
