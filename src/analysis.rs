extern crate nalgebra as na;

use crate::model::{ModelError, Pole, StateSpaceModel};

/// Diagnostics for rank-based system properties.
#[derive(Debug, Clone)]
pub struct RankDiagnostics {
    /// Numerical rank of the tested matrix.
    pub rank: usize,
    /// Expected full rank.
    pub expected_rank: usize,
    /// Condition number estimate using singular values.
    pub condition_number: f64,
    /// Whether matrix rank matches expected rank.
    pub is_full_rank: bool,
}

/// Time domain used for stability checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeDomain {
    /// Continuous-time model (`Re(lambda_i) < 0` for stability).
    Continuous,
    /// Discrete-time model (`|lambda_i| < 1` for stability).
    Discrete,
}

/// Consolidated analysis report for an LTI state-space model.
#[derive(Debug, Clone)]
pub struct LtiAnalysisReport {
    /// Eigenvalues of the state matrix.
    pub poles: Vec<na::Complex<f64>>,
    /// Spectral radius `max(|lambda_i|)`.
    pub spectral_radius: f64,
    /// Stability result based on chosen [`TimeDomain`].
    pub is_stable: bool,
    /// Controllability diagnostics.
    pub controllability: RankDiagnostics,
    /// Observability diagnostics.
    pub observability: RankDiagnostics,
}

/// Computes the controllability matrix `[B, AB, ..., A^(n-1)B]`.
pub fn compute_controllability_matrix(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
) -> Result<na::DMatrix<f64>, ModelError> {
    if !mat_a.is_square() {
        return Err(ModelError::MatrixANotSquare {
            rows: mat_a.nrows(),
            cols: mat_a.ncols(),
        });
    }

    if mat_b.nrows() != mat_a.nrows() {
        return Err(ModelError::DimensionMismatch {
            a: mat_a.shape(),
            b: mat_b.shape(),
            c: (0, 0),
            d: (0, 0),
        });
    }

    let n = mat_a.nrows();
    let mut controllability_matrix = mat_b.clone_owned();

    for i in 1..n {
        let column_block = mat_a.pow(i as u32) * mat_b;
        controllability_matrix = na::stack![&controllability_matrix, &column_block];
    }

    Ok(controllability_matrix)
}

/// Checks if a given state-space model is controllable.
pub fn is_ss_controllable<T: StateSpaceModel>(model: &T) -> (bool, na::DMatrix<f64>) {
    let mat_a = model.mat_a();
    let mat_b = model.mat_b();

    let mat_contr = compute_controllability_matrix(mat_a, mat_b)
        .expect("State-space dimensions are invalid for controllability computation");

    (mat_contr.rank(1e-3) == mat_a.nrows(), mat_contr)
}

/// Computes the observability matrix `[C; CA; ...; CA^(n-1)]`.
pub fn compute_observability_matrix(
    mat_a: &na::DMatrix<f64>,
    mat_c: &na::DMatrix<f64>,
) -> Result<na::DMatrix<f64>, ModelError> {
    if !mat_a.is_square() {
        return Err(ModelError::MatrixANotSquare {
            rows: mat_a.nrows(),
            cols: mat_a.ncols(),
        });
    }

    if mat_c.ncols() != mat_a.nrows() {
        return Err(ModelError::DimensionMismatch {
            a: mat_a.shape(),
            b: (0, 0),
            c: mat_c.shape(),
            d: (0, 0),
        });
    }

    let n = mat_a.nrows();
    let mut observability_mat = mat_c.clone_owned();

    for i in 1..n {
        let column_block = mat_c * mat_a.pow(i as u32);
        observability_mat = na::stack![&observability_mat; &column_block];
    }

    Ok(observability_mat)
}

/// Checks if a given state-space model is observable.
pub fn is_ss_observable<T: StateSpaceModel>(model: &T) -> (bool, na::DMatrix<f64>) {
    let mat_a = model.mat_a();
    let mat_c = model.mat_c();

    let mat_obs = compute_observability_matrix(mat_a, mat_c)
        .expect("State-space dimensions are invalid for observability computation");

    (mat_obs.rank(1e-3) == mat_a.nrows(), mat_obs)
}

/// Computes controllability rank diagnostics.
pub fn controllability_diagnostics<T: StateSpaceModel>(
    model: &T,
    rank_tol: f64,
) -> Result<RankDiagnostics, ModelError> {
    let mat = compute_controllability_matrix(model.mat_a(), model.mat_b())?;
    let rank = mat.rank(rank_tol);
    let expected_rank = model.mat_a().nrows();
    let condition_number = condition_number(&mat, 1e-12);

    Ok(RankDiagnostics {
        rank,
        expected_rank,
        condition_number,
        is_full_rank: rank == expected_rank,
    })
}

/// Computes observability rank diagnostics.
pub fn observability_diagnostics<T: StateSpaceModel>(
    model: &T,
    rank_tol: f64,
) -> Result<RankDiagnostics, ModelError> {
    let mat = compute_observability_matrix(model.mat_a(), model.mat_c())?;
    let rank = mat.rank(rank_tol);
    let expected_rank = model.mat_a().nrows();
    let condition_number = condition_number(&mat, 1e-12);

    Ok(RankDiagnostics {
        rank,
        expected_rank,
        condition_number,
        is_full_rank: rank == expected_rank,
    })
}

/// Returns spectral radius `max(|lambda_i|)`.
pub fn spectral_radius<T: Pole>(model: &T) -> f64 {
    model
        .poles()
        .iter()
        .map(|pole| pole.norm())
        .fold(0.0, f64::max)
}

/// Returns true if all discrete poles are strictly inside the unit circle.
pub fn is_discrete_stable<T: Pole>(model: &T) -> bool {
    model.poles().iter().all(|pole| pole.norm() < 1.0)
}

/// Returns true if all continuous poles have negative real part.
pub fn is_continuous_stable<T: Pole>(model: &T) -> bool {
    model.poles().iter().all(|pole| pole.re < 0.0)
}

/// Performs a full first-pass LTI analysis.
///
/// This combines:
///
/// - Pole extraction
/// - Spectral radius
/// - Stability check for the selected [`TimeDomain`]
/// - Controllability diagnostics
/// - Observability diagnostics
pub fn analyze_lti<T: StateSpaceModel + Pole>(
    model: &T,
    domain: TimeDomain,
    rank_tol: f64,
) -> Result<LtiAnalysisReport, ModelError> {
    let poles = model.poles();
    let spectral_radius = poles.iter().map(|pole| pole.norm()).fold(0.0f64, f64::max);
    let is_stable = match domain {
        TimeDomain::Continuous => poles.iter().all(|pole| pole.re < 0.0),
        TimeDomain::Discrete => poles.iter().all(|pole| pole.norm() < 1.0),
    };

    let controllability = controllability_diagnostics(model, rank_tol)?;
    let observability = observability_diagnostics(model, rank_tol)?;

    Ok(LtiAnalysisReport {
        poles,
        spectral_radius,
        is_stable,
        controllability,
        observability,
    })
}

fn condition_number(mat: &na::DMatrix<f64>, min_sv_tol: f64) -> f64 {
    let svd = mat.clone().svd(false, false);
    let mut max_sv = 0.0f64;
    let mut min_sv = f64::INFINITY;

    for sv in svd.singular_values.iter() {
        max_sv = max_sv.max(*sv);
        if *sv > min_sv_tol {
            min_sv = min_sv.min(*sv);
        }
    }

    if min_sv.is_infinite() || min_sv <= 0.0 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::DiscreteStateSpaceModel;

    // Verifies controllability matrix construction for a nominal 2x2 system.
    #[test]
    fn test_compute_controllability_matrix_2x2() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0;
                                       2.0, 1.0];
        let mat_b = nalgebra::dmatrix![1.0;
                                       2.0];

        let result = compute_controllability_matrix(&mat_a, &mat_b).unwrap();

        assert_eq!(result.nrows(), mat_b.nrows());
        assert_eq!(result.ncols(), mat_b.ncols() * mat_a.ncols());
        assert_eq!(result, na::stack![&mat_b, &mat_a * &mat_b]);
    }

    // Verifies a non-square A matrix is rejected for controllability computation.
    #[test]
    fn test_compute_controllability_matrix_mat_a_not_square() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0];
        let mat_b = nalgebra::dmatrix![1.0;
                                       2.0];

        let result = compute_controllability_matrix(&mat_a, &mat_b);

        assert_eq!(
            result,
            Err(ModelError::MatrixANotSquare { rows: 1, cols: 2 })
        );
    }

    // Verifies a known controllable 2x2 model is detected as controllable.
    #[test]
    fn test_controllability_2x2_controllable() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![1.0, -2.0;
                            2.0, 1.0],
            &nalgebra::dmatrix![1.0;
                            2.0],
            &nalgebra::dmatrix![1.0, 0.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        let (result, _) = is_ss_controllable(&ss_model);

        assert!(result);
    }

    // Verifies a known 3x3 model is detected as not controllable.
    #[test]
    fn test_controllability_3x3_not_controllable() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &na::dmatrix![
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0;
            0.0, 0.0, 0.0],
            &na::dmatrix![
            1.0;
            0.0;
            0.0],
            &nalgebra::dmatrix![1.0, 0.0, 0.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        assert!(!is_ss_controllable(&ss_model).0);
    }

    // Verifies observability matrix construction for a nominal 2x2 system.
    #[test]
    fn test_compute_observability_matrix_2x2() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0;
                                       2.0, 1.0];
        let mat_c = nalgebra::dmatrix![1.0, 2.0];

        let result = compute_observability_matrix(&mat_a, &mat_c).unwrap();

        assert_eq!(result.nrows(), mat_a.ncols());
        assert_eq!(result.ncols(), mat_c.ncols());
        assert_eq!(result, na::stack![&mat_c; &mat_c * &mat_a]);
    }

    // Verifies a non-square A matrix is rejected for observability computation.
    #[test]
    fn test_compute_observability_matrix_mat_a_not_square() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0];
        let mat_c = nalgebra::dmatrix![1.0, 2.0];

        let result = compute_observability_matrix(&mat_a, &mat_c);

        assert_eq!(
            result,
            Err(ModelError::MatrixANotSquare { rows: 1, cols: 2 })
        );
    }

    // Verifies a known observable 2x2 model is detected as observable.
    #[test]
    fn test_is_observable_2x2_observable_system() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![1.0, -2.0;
                            2.0, 1.0],
            &nalgebra::dmatrix![0.0;
                            0.0],
            &nalgebra::dmatrix![1.0, 2.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        let (result, _) = is_ss_observable(&ss_model);

        assert!(result);
    }

    // Verifies a known 3x3 model is detected as not observable.
    #[test]
    fn test_observability_3x3_not_observable() {
        #[allow(deprecated)]
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &na::dmatrix![
            1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 0.0],
            &na::dmatrix![0.0;
                          0.0;
                          0.0],
            &nalgebra::dmatrix![1.0, 0.0, 0.0],
            &nalgebra::dmatrix![0.0],
            0.05,
        );

        let (observable, obs_mat) = is_ss_observable(&ss_model);

        assert!(!observable);
        assert_eq!(
            obs_mat,
            na::dmatrix![
            1.0, 0.0, 0.0;
            1.0, 0.0, 0.0;
            1.0, 0.0, 0.0]
        );
    }

    // Verifies discrete stability helpers on a stable single-state model.
    #[test]
    fn test_stability_checks() {
        #[allow(deprecated)]
        let discrete_stable = DiscreteStateSpaceModel::from_matrices(
            &na::dmatrix![0.8],
            &na::dmatrix![1.0],
            &na::dmatrix![1.0],
            &na::dmatrix![0.0],
            0.1,
        );

        assert!(is_discrete_stable(&discrete_stable));
        assert!(spectral_radius(&discrete_stable) < 1.0);
    }

    // Verifies the consolidated LTI report fields are populated consistently.
    #[test]
    fn test_analyze_lti_discrete_report() {
        #[allow(deprecated)]
        let model = DiscreteStateSpaceModel::from_matrices(
            &na::dmatrix![0.8, 0.0; 0.0, 0.7],
            &na::dmatrix![1.0; 0.0],
            &na::dmatrix![1.0, 0.0],
            &na::dmatrix![0.0],
            0.1,
        );

        let report = analyze_lti(&model, TimeDomain::Discrete, 1e-6).unwrap();

        assert!(report.is_stable);
        assert!(report.spectral_radius < 1.0);
        assert_eq!(report.controllability.expected_rank, 2);
        assert_eq!(report.observability.expected_rank, 2);
    }
}
