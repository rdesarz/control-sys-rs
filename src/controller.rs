extern crate nalgebra as na;

use crate::analysis::compute_controllability_matrix;
use crate::model::{ModelError, StateSpaceModel};

use std::error::Error;
use std::fmt;

/// Errors related to controller synthesis utilities.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlError {
    /// Model validation error.
    Model(ModelError),
    /// Matrix dimensions are incompatible with the requested operation.
    DimensionMismatch(&'static str),
    /// A matrix inversion failed.
    SingularMatrix(&'static str),
    /// System is not controllable for the requested synthesis.
    NotControllable,
    /// Requested poles are invalid for the model order.
    InvalidDesiredPoles,
    /// Riccati solver failed to converge.
    RiccatiNoConvergence,
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlError::Model(err) => write!(f, "{err}"),
            ControlError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            ControlError::SingularMatrix(msg) => write!(f, "singular matrix: {msg}"),
            ControlError::NotControllable => write!(f, "system is not controllable"),
            ControlError::InvalidDesiredPoles => write!(f, "invalid desired poles"),
            ControlError::RiccatiNoConvergence => write!(f, "DARE solver did not converge"),
        }
    }
}

impl Error for ControlError {}

impl From<ModelError> for ControlError {
    fn from(value: ModelError) -> Self {
        ControlError::Model(value)
    }
}

/// Builds a closed-loop state matrix `A_cl = A - B*K` for `u = -Kx`.
pub fn closed_loop(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
    mat_k: &na::DMatrix<f64>,
) -> Result<na::DMatrix<f64>, ControlError> {
    if !mat_a.is_square() {
        return Err(ControlError::Model(ModelError::MatrixANotSquare {
            rows: mat_a.nrows(),
            cols: mat_a.ncols(),
        }));
    }

    if mat_b.nrows() != mat_a.nrows() {
        return Err(ControlError::DimensionMismatch("B rows must match A rows"));
    }

    if mat_k.nrows() != mat_b.ncols() || mat_k.ncols() != mat_a.ncols() {
        return Err(ControlError::DimensionMismatch(
            "K must have shape (n_inputs, n_states)",
        ));
    }

    Ok(mat_a - mat_b * mat_k)
}

/// SISO pole placement via Ackermann's formula.
///
/// `desired_poles` are real roots for a system of order `n`.
pub fn pole_placement_siso(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
    desired_poles: &[f64],
) -> Result<na::DMatrix<f64>, ControlError> {
    if !mat_a.is_square() {
        return Err(ControlError::Model(ModelError::MatrixANotSquare {
            rows: mat_a.nrows(),
            cols: mat_a.ncols(),
        }));
    }

    let n = mat_a.nrows();
    if mat_b.shape() != (n, 1) || desired_poles.len() != n {
        return Err(ControlError::InvalidDesiredPoles);
    }

    let ctrb = compute_controllability_matrix(mat_a, mat_b)?;
    let ctrb_inv = ctrb.try_inverse().ok_or(ControlError::NotControllable)?;

    let coeffs = characteristic_polynomial_coeffs_from_roots(desired_poles);

    let mut phi_a = mat_a.pow(n as u32);
    for (power, coeff) in coeffs.iter().take(n).enumerate() {
        if power == 0 {
            phi_a += na::DMatrix::<f64>::identity(n, n).scale(*coeff);
        } else {
            phi_a += mat_a.pow(power as u32).scale(*coeff);
        }
    }

    let mut e_n_t = na::DMatrix::<f64>::zeros(1, n);
    e_n_t[(0, n - 1)] = 1.0;

    Ok(e_n_t * ctrb_inv * phi_a)
}

/// Discrete-time LQR using an iterative DARE solve.
pub fn discrete_lqr(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
    mat_q: &na::DMatrix<f64>,
    mat_r: &na::DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<na::DMatrix<f64>, ControlError> {
    if !mat_a.is_square() {
        return Err(ControlError::Model(ModelError::MatrixANotSquare {
            rows: mat_a.nrows(),
            cols: mat_a.ncols(),
        }));
    }

    let n = mat_a.nrows();
    let m = mat_b.ncols();

    if mat_b.nrows() != n
        || mat_q.shape() != (n, n)
        || mat_r.shape() != (m, m)
        || !mat_q.is_square()
        || !mat_r.is_square()
    {
        return Err(ControlError::DimensionMismatch(
            "A(n,n), B(n,m), Q(n,n), R(m,m) required",
        ));
    }

    let mut p = mat_q.clone();

    for _ in 0..max_iter {
        let bt_p = mat_b.transpose() * &p;
        let s = mat_r + &bt_p * mat_b;
        let s_inv = s
            .try_inverse()
            .ok_or(ControlError::SingularMatrix("R + B^T P B"))?;

        let p_next = mat_a.transpose() * &p * mat_a
            - mat_a.transpose() * &p * mat_b * s_inv * bt_p * mat_a
            + mat_q;

        let max_delta = p_next
            .iter()
            .zip(p.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        p = p_next;
        if max_delta < tol {
            let bt_p = mat_b.transpose() * &p;
            let s = mat_r + &bt_p * mat_b;
            let s_inv = s
                .try_inverse()
                .ok_or(ControlError::SingularMatrix("R + B^T P B"))?;
            return Ok(s_inv * bt_p * mat_a);
        }
    }

    Err(ControlError::RiccatiNoConvergence)
}

/// Continuous-time LQR approximation by Tustin discretization + discrete LQR.
///
/// This method is pragmatic and does not solve CARE exactly.
pub fn continuous_lqr(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
    mat_q: &na::DMatrix<f64>,
    mat_r: &na::DMatrix<f64>,
    sampling_dt: f64,
    max_iter: usize,
    tol: f64,
) -> Result<na::DMatrix<f64>, ControlError> {
    if !sampling_dt.is_finite() || sampling_dt <= 0.0 {
        return Err(ControlError::Model(ModelError::InvalidSamplingDt(
            sampling_dt,
        )));
    }

    let n = mat_a.nrows();
    if !mat_a.is_square() || mat_b.nrows() != n {
        return Err(ControlError::DimensionMismatch("invalid A/B dimensions"));
    }

    let i = na::DMatrix::<f64>::identity(n, n);
    let left = i.clone() - mat_a.scale(0.5 * sampling_dt);
    let left_inv = left
        .try_inverse()
        .ok_or(ControlError::SingularMatrix("I - A*dt/2"))?;

    let ad = &left_inv * (i + mat_a.scale(0.5 * sampling_dt));
    let bd = left_inv * mat_b.scale(sampling_dt);

    let qd = mat_q.scale(sampling_dt);
    let rd = mat_r.scale(sampling_dt);

    discrete_lqr(&ad, &bd, &qd, &rd, max_iter, tol)
}

/// Computes SISO reference prefilter `Nbar` for `u = -Kx + Nbar*r`.
pub fn siso_reference_prefilter(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
    mat_c: &na::DMatrix<f64>,
    mat_k: &na::DMatrix<f64>,
) -> Result<f64, ControlError> {
    if mat_b.ncols() != 1 || mat_c.nrows() != 1 || mat_k.nrows() != 1 {
        return Err(ControlError::DimensionMismatch(
            "SISO system required for prefilter",
        ));
    }

    let acl = closed_loop(mat_a, mat_b, mat_k)?;
    let inv_acl = acl
        .try_inverse()
        .ok_or(ControlError::SingularMatrix("A-BK"))?;

    let gain = -(mat_c * inv_acl * mat_b)[(0, 0)];
    if gain.abs() < 1e-12 {
        return Err(ControlError::SingularMatrix("dc-gain is near zero"));
    }

    Ok(1.0 / gain)
}

/// Closed-loop helper for `StateSpaceModel` implementors.
pub fn closed_loop_from_model<T: StateSpaceModel>(
    model: &T,
    mat_k: &na::DMatrix<f64>,
) -> Result<na::DMatrix<f64>, ControlError> {
    closed_loop(model.mat_a(), model.mat_b(), mat_k)
}

fn characteristic_polynomial_coeffs_from_roots(roots: &[f64]) -> Vec<f64> {
    let mut poly = vec![1.0f64];

    for root in roots {
        let mut next = vec![0.0; poly.len() + 1];
        for (i, coeff) in poly.iter().enumerate() {
            next[i] += -root * coeff;
            next[i + 1] += coeff;
        }
        poly = next;
    }

    poly
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

    #[test]
    fn test_pole_placement_siso() {
        let a = na::dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = na::dmatrix![0.0; 1.0];

        let k = pole_placement_siso(&a, &b, &[-2.0, -4.0]).unwrap();
        let acl = closed_loop(&a, &b, &k).unwrap();

        let desired_char_poly = na::dmatrix![0.0, 1.0; -8.0, -6.0];
        approx_eq_matrix(&acl, &desired_char_poly, 1e-10);
    }

    #[test]
    fn test_discrete_lqr_stabilizes_unstable_system() {
        let a = na::dmatrix![1.1];
        let b = na::dmatrix![1.0];
        let q = na::dmatrix![1.0];
        let r = na::dmatrix![1.0];

        let k = discrete_lqr(&a, &b, &q, &r, 5000, 1e-12).unwrap();
        let acl = closed_loop(&a, &b, &k).unwrap();

        assert!(acl[(0, 0)].abs() < 1.0);
    }

    #[test]
    fn test_continuous_lqr_returns_gain() {
        let a = na::dmatrix![0.0, 1.0; 0.0, 0.0];
        let b = na::dmatrix![0.0; 1.0];
        let q = na::dmatrix![10.0, 0.0; 0.0, 1.0];
        let r = na::dmatrix![1.0];

        let k = continuous_lqr(&a, &b, &q, &r, 0.01, 5000, 1e-10).unwrap();
        assert_eq!(k.shape(), (1, 2));
    }
}
