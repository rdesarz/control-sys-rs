extern crate nalgebra as na;

use crate::control::model::StateSpaceModel;

/// Computes the controllability matrix for a given state-space representation.
///
/// # Arguments
///
/// * `mat_a` - The state matrix (A) of the system.
/// * `mat_b` - The input matrix (B) of the system.
///
/// # Returns
///
/// * `Result<na::DMatrix<f64>, &'static str>` - The controllability matrix if successful, or an error message if the A matrix is not square.
///
/// # Errors
///
/// This function will return an error if the A matrix is not square.
///
/// # Panics
///
/// This function does not panic.
///
/// # Examples
///
/// ```
/// use control_sys_rs::control::analysis;
///
/// let mat_a = nalgebra::dmatrix![1.0, -2.0;
///                                2.0, 1.0];
/// let mat_b = nalgebra::dmatrix![1.0;
///                                2.0];
/// match analysis::compute_controllability_matrix(&mat_a, &mat_b) {
///     Ok(result) => { println!("Controllability matrix is {}", result); },
///     Err(msg) => { println!("{}", msg)}
/// }
/// ```
///
pub fn compute_controllability_matrix(
    mat_a: &na::DMatrix<f64>,
    mat_b: &na::DMatrix<f64>,
) -> Result<na::DMatrix<f64>, &'static str> {
    if !mat_a.is_square() {
        return Err("Error when computing controllability matrix. The A matrix is not square.");
    }

    let n = mat_a.nrows(); // Number of states
    let mut controllability_matrix = mat_b.clone_owned();

    // Generate [B, AB, A^2B, ...]
    for i in 1..n {
        let column_block = mat_a.pow(i as u32) * mat_b; // Compute A^i * B
        controllability_matrix = na::stack![controllability_matrix, column_block];
    }

    Ok(controllability_matrix)
}

/// Checks if a given state-space model is controllable.
///
/// # Arguments
///
/// * `model` - A reference to a state-space model that implements the `StateSpaceModel` trait.
///
/// # Returns
///
/// * `bool` - `true` if the system is controllable, `false` otherwise.
///
/// # Panics
///
/// This function will panic if the computation of the controllability matrix fails.
///
/// # Examples
///
/// ```
/// use control_sys_rs::control::analysis;
/// use control_sys_rs::control::model;
///
/// let ss_model = model::DiscreteStateSpaceModel::from_matrices(
/// &nalgebra::dmatrix![1.0, -2.0;
///                     2.0, 1.0],
/// &nalgebra::dmatrix![1.0;
///                     2.0],
/// &nalgebra::dmatrix![],
/// &nalgebra::dmatrix![],
/// 0.05,
/// );
/// let (is_controllable, controllability_matrix) = analysis::is_ss_controllable(&ss_model);
/// ```
pub fn is_ss_controllable<T: StateSpaceModel>(model: &T) -> (bool, na::DMatrix<f64>) {
    let mat_a = model.mat_a();
    let mat_b = model.mat_b();

    // We know that mat_a is square so we simply unwrap() the result
    let mat_contr = compute_controllability_matrix(&mat_a, &mat_b)
        .expect("State matrix of the model is not square");

    // Since the input is a state space model, we expect A to be square
    return (mat_contr.rank(1e-3) == mat_a.nrows(), mat_contr);
}

/// Computes the observability matrix for a given state-space representation.
///
/// # Arguments
///
/// * `mat_a` - The state matrix (A) of the system.
/// * `mat_c` - The input matrix (B) of the system.
///
/// # Returns
///
/// * `Result<na::DMatrix<f64>, &'static str>` - The observability matrix if successful, or an error message if the A matrix is not square.
///
/// # Errors
///
/// This function will return an error if the A matrix is not square.
///
/// # Panics
///
/// This function does not panic.
///
/// # Examples
///
/// ```
/// use control_sys_rs::control::analysis;
///
/// let mat_a = nalgebra::dmatrix![1.0, -2.0;
///                                2.0, 1.0];
/// let mat_c = nalgebra::dmatrix![1.0, 2.0];
///
/// match analysis::compute_observability_matrix(&mat_a, &mat_c) {
///     Ok(result) => { println!("Observability matrix is {}", result); },
///     Err(msg) => { println!("{}", msg)}
/// }
/// ```
pub fn compute_observability_matrix(
    mat_a: &na::DMatrix<f64>,
    mat_c: &na::DMatrix<f64>,
) -> Result<na::DMatrix<f64>, &'static str> {
    if !mat_a.is_square() {
        return Err("Error when computing observability matrix. The A matrix is not square.");
    }

    let n = mat_a.nrows(); // Number of states
    let mut observability_mat = mat_c.clone_owned();

    // Generate [C; CA; CA^2;...]
    for i in 1..n {
        let column_block = mat_c * mat_a.pow(i as u32);
        observability_mat = na::stack![observability_mat; column_block];
    }

    Ok(observability_mat)
}

/// Checks if a given state-space model is observable.
///
/// # Arguments
///
/// * `model` - A reference to a state-space model that implements the `StateSpaceModel` trait.
///
/// # Returns
///
/// * A tuple with a `bool` which is `true` if the system is controllable, `false` otherwise and the observability matrix.
///
/// # Panics
///
/// This function will panic if the computation of the controllability matrix fails.
///
/// # Examples
///
/// ```
/// use control_sys_rs::control::analysis;
/// use control_sys_rs::control::model;
///
/// let ss_model = model::DiscreteStateSpaceModel::from_matrices(
/// &nalgebra::dmatrix![1.0, -2.0;
///                     2.0, 1.0],
/// &nalgebra::dmatrix![],
/// &nalgebra::dmatrix![1.0, 2.0],
/// &nalgebra::dmatrix![],
/// 0.05,
/// );
/// let (is_observable, observability_matrix) = analysis::is_ss_observable(&ss_model);
/// ```
pub fn is_ss_observable<T: StateSpaceModel>(model: &T) -> (bool, na::DMatrix<f64>) {
    let mat_a = model.mat_a();
    let mat_c = model.mat_c();

    // StateSpaceModel performs check of the matrices so computing observability matrix should not fail
    let mat_obs = compute_observability_matrix(&mat_a, &mat_c)
        .expect("State matrix of the model is not square");

    // Since the input is a state space model, we expect A to be square
    return (mat_obs.rank(1e-3) == mat_a.nrows(), mat_obs);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::control::model::DiscreteStateSpaceModel;

    #[test]
    fn test_compute_controllability_matrix_2x2() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0;
                                       2.0, 1.0];
        let mat_b = nalgebra::dmatrix![1.0; 
                                       2.0];

        let result = compute_controllability_matrix(&mat_a, &mat_b).unwrap();

        assert_eq!(result.nrows(), mat_b.ncols() * mat_a.nrows());
        assert_eq!(result.ncols(), mat_a.ncols());
        assert_eq!(result, na::stack![mat_b, mat_a * &mat_b])
    }

    #[test]
    fn test_compute_controllability_matrix_mat_a_not_square() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0];
        let mat_b = nalgebra::dmatrix![1.0; 
                                       2.0];

        let result = compute_controllability_matrix(&mat_a, &mat_b);

        assert_eq!(
            result,
            Err("Error when computing controllability matrix. The A matrix is not square.")
        );
    }

    #[test]
    fn test_controllability_2x2_controllable() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![1.0, -2.0; 
                            2.0, 1.0],
            &nalgebra::dmatrix![1.0;
                            2.0],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            0.05,
        );

        let (result, _) = is_ss_controllable(&ss_model);

        assert!(result);
    }

    #[test]
    fn test_controllability_3x3_not_controllable() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &na::dmatrix![
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0;
            0.0, 0.0, 0.0],
            &na::dmatrix![
            1.0;
            0.0;
            0.0],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![],
            0.05,
        );

        assert!(!is_ss_controllable(&ss_model).0);
    }

    #[test]
    fn test_compute_observability_matrix_2x2() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0;
                                       2.0, 1.0];
        let mat_c = nalgebra::dmatrix![1.0, 2.0];

        let result = compute_observability_matrix(&mat_a, &mat_c).unwrap();

        assert_eq!(result.nrows(), mat_a.ncols());
        assert_eq!(result.ncols(), mat_c.ncols());
        assert_eq!(result, na::stack![&mat_c; &mat_c * mat_a]);
    }

    #[test]
    fn test_compute_observability_matrix_mat_a_not_square() {
        let mat_a = nalgebra::dmatrix![1.0, -2.0];
        let mat_b = nalgebra::dmatrix![1.0; 
                                       2.0];

        let result = compute_observability_matrix(&mat_a, &mat_b);

        assert_eq!(
            result,
            Err("Error when computing observability matrix. The A matrix is not square.")
        );
    }

    #[test]
    fn test_is_observable_2x2_observable_system() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &nalgebra::dmatrix![1.0, -2.0; 
                            2.0, 1.0],
            &nalgebra::dmatrix![],
            &nalgebra::dmatrix![1.0, 2.0],
            &nalgebra::dmatrix![],
            0.05,
        );

        let (result, _) = is_ss_observable(&ss_model);

        assert!(result);
    }

    #[test]
    fn test_observability_3x3_not_observable() {
        let ss_model = DiscreteStateSpaceModel::from_matrices(
            &na::dmatrix![
            1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 0.0],
            &na::dmatrix![],
            &nalgebra::dmatrix![1.0, 0.0, 0.0],
            &nalgebra::dmatrix![],
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
}
