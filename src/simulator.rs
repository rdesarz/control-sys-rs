use nalgebra as na;

use crate::model::{
    ContinuousStateSpaceModel, Discrete, DiscreteStateSpaceModel, ModelError, StateSpaceModel,
};

use std::error::Error;
use std::fmt;

/// Simulation-related runtime errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SimulationError {
    /// Input dimensions do not match the model dimensions.
    DimensionMismatch {
        /// Expected number of input rows (`B.ncols()`).
        expected_u_rows: usize,
        /// Provided number of input rows.
        actual_u_rows: usize,
        /// Expected number of states (`A.nrows()`).
        expected_x0_rows: usize,
        /// Provided number of initial-state rows.
        actual_x0_rows: usize,
    },
    /// Invalid duration.
    InvalidDuration(f64),
    /// Invalid model state.
    Model(ModelError),
}

impl fmt::Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulationError::DimensionMismatch {
                expected_u_rows,
                actual_u_rows,
                expected_x0_rows,
                actual_x0_rows,
            } => write!(
                f,
                "dimension mismatch: u rows expected {expected_u_rows} got {actual_u_rows}, x0 rows expected {expected_x0_rows} got {actual_x0_rows}"
            ),
            SimulationError::InvalidDuration(duration) => {
                write!(f, "duration must be finite and >= 0.0, got {duration}")
            }
            SimulationError::Model(err) => write!(f, "{err}"),
        }
    }
}

impl Error for SimulationError {}

impl From<ModelError> for SimulationError {
    fn from(value: ModelError) -> Self {
        SimulationError::Model(value)
    }
}

/// Simulates a discrete state-space model for an arbitrary input sequence.
///
/// The model equations are:
/// `x[k+1] = A x[k] + B u[k]`
/// `y[k]   = C x[k] + D u[k]`
pub fn simulate(
    model: &(impl StateSpaceModel + Discrete),
    mat_u: &na::DMatrix<f64>,
    x0: &na::DVector<f64>,
) -> Result<(na::DMatrix<f64>, na::DMatrix<f64>), SimulationError> {
    let n_state = model.mat_a().nrows();
    let n_input = model.mat_b().ncols();
    let n_output = model.mat_c().nrows();
    let sim_time = mat_u.ncols();

    if mat_u.nrows() != n_input || x0.nrows() != n_state {
        return Err(SimulationError::DimensionMismatch {
            expected_u_rows: n_input,
            actual_u_rows: mat_u.nrows(),
            expected_x0_rows: n_state,
            actual_x0_rows: x0.nrows(),
        });
    }

    let mut mat_x = na::DMatrix::<f64>::zeros(n_state, sim_time + 1);
    let mut mat_y = na::DMatrix::<f64>::zeros(n_output, sim_time);
    mat_x.column_mut(0).copy_from(x0);

    for i in 0..sim_time {
        let xk = mat_x.column(i).into_owned();
        let uk = mat_u.column(i).into_owned();

        let yk = model.mat_c() * &xk + model.mat_d() * &uk;
        mat_y.column_mut(i).copy_from(&yk);

        let x_next = model.mat_a() * xk + model.mat_b() * uk;
        mat_x.column_mut(i + 1).copy_from(&x_next);
    }

    Ok((mat_y, mat_x))
}

/// Generates a unit-step response of a discrete state-space model.
pub fn step_for_discrete_ss(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> Result<(na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>), SimulationError> {
    if !duration.is_finite() || duration < 0.0 {
        return Err(SimulationError::InvalidDuration(duration));
    }

    let n_samples = (duration / model.sampling_dt()).floor().max(0.0) as usize;
    let n_inputs = model.mat_b().ncols();
    let initial_state = na::DVector::<f64>::zeros(model.mat_a().nrows());
    let input = na::DMatrix::from_element(n_inputs, n_samples, 1.0f64);

    let (response, states) = simulate(model, &input, &initial_state)?;

    Ok((response, input, states))
}

/// Generates a unit-step response for a continuous model after ZOH discretization.
pub fn step_for_continuous_ss(
    model: &ContinuousStateSpaceModel,
    sampling_dt: f64,
    duration: f64,
) -> Result<(na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>), SimulationError> {
    let discrete_model = DiscreteStateSpaceModel::from_continuous_zoh(model, sampling_dt)?;

    step_for_discrete_ss(&discrete_model, duration)
}

/// Generates an impulse response for a discrete model.
pub fn impulse_for_discrete_ss(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> Result<(na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>), SimulationError> {
    if !duration.is_finite() || duration < 0.0 {
        return Err(SimulationError::InvalidDuration(duration));
    }

    let n_samples = (duration / model.sampling_dt()).floor().max(0.0) as usize;
    let n_inputs = model.mat_b().ncols();
    let mut input = na::DMatrix::zeros(n_inputs, n_samples);
    if n_samples > 0 {
        input.column_mut(0).fill(1.0);
    }

    let initial_state = na::DVector::<f64>::zeros(model.mat_a().nrows());
    let (response, states) = simulate(model, &input, &initial_state)?;

    Ok((response, input, states))
}

/// Generates a ramp response for a discrete model.
pub fn ramp_for_discrete_ss(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> Result<(na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>), SimulationError> {
    if !duration.is_finite() || duration < 0.0 {
        return Err(SimulationError::InvalidDuration(duration));
    }

    let n_samples = (duration / model.sampling_dt()).floor().max(0.0) as usize;
    let n_inputs = model.mat_b().ncols();
    let mut input = na::DMatrix::zeros(n_inputs, n_samples);

    for i in 0..n_samples {
        let value = i as f64 * model.sampling_dt();
        for input_row in 0..n_inputs {
            input[(input_row, i)] = value;
        }
    }

    let initial_state = na::DVector::<f64>::zeros(model.mat_a().nrows());
    let (response, states) = simulate(model, &input, &initial_state)?;

    Ok((response, input, states))
}

/// Generates a simulation time vector with length `n_samples`.
pub fn time_vector(
    sampling_dt: f64,
    n_samples: usize,
) -> Result<na::DVector<f64>, SimulationError> {
    if !sampling_dt.is_finite() || sampling_dt <= 0.0 {
        return Err(SimulationError::Model(ModelError::InvalidSamplingDt(
            sampling_dt,
        )));
    }

    Ok(na::DVector::from_iterator(
        n_samples,
        (0..n_samples).map(|i| i as f64 * sampling_dt),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::DiscreteStateSpaceModel;

    fn first_order_siso_model_with_d() -> DiscreteStateSpaceModel {
        DiscreteStateSpaceModel::try_from_matrices(
            &na::dmatrix![0.5],
            &na::dmatrix![1.0],
            &na::dmatrix![1.0],
            &na::dmatrix![2.0],
            0.1,
        )
        .unwrap()
    }

    #[test]
    fn test_simulate_includes_feedthrough_term() {
        let model = first_order_siso_model_with_d();
        let u = na::dmatrix![1.0, 1.0, 1.0];
        let x0 = na::dvector![0.0];

        let (y, _x) = simulate(&model, &u, &x0).unwrap();

        assert!((y[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((y[(0, 1)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_step_supports_mimo_inputs() {
        let model = DiscreteStateSpaceModel::try_from_matrices(
            &na::dmatrix![1.0, 0.0; 0.0, 1.0],
            &na::dmatrix![1.0, 0.0; 0.0, 1.0],
            &na::dmatrix![1.0, 0.0; 0.0, 1.0],
            &na::dmatrix![0.0, 0.0; 0.0, 0.0],
            0.1,
        )
        .unwrap();

        let (_y, u, _x) = step_for_discrete_ss(&model, 1.0).unwrap();

        assert_eq!(u.nrows(), 2);
        assert_eq!(u.ncols(), 10);
    }

    #[test]
    fn test_zero_duration_response() {
        let model = first_order_siso_model_with_d();
        let (y, u, x) = step_for_discrete_ss(&model, 0.0).unwrap();

        assert_eq!(y.shape(), (1, 0));
        assert_eq!(u.shape(), (1, 0));
        assert_eq!(x.shape(), (1, 1));
    }

    #[test]
    fn test_short_duration_response() {
        let model = first_order_siso_model_with_d();
        let (y, _u, _x) = step_for_discrete_ss(&model, 0.05).unwrap();
        assert_eq!(y.ncols(), 0);
    }

}
