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

type ResponseInputState = (na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>);

/// Generates a unit-step response of a discrete state-space model.
pub fn step_for_discrete_ss(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> Result<ResponseInputState, SimulationError> {
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
) -> Result<ResponseInputState, SimulationError> {
    let discrete_model = DiscreteStateSpaceModel::from_continuous_zoh(model, sampling_dt)?;

    step_for_discrete_ss(&discrete_model, duration)
}

/// Generates an impulse response for a discrete model.
pub fn impulse_for_discrete_ss(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> Result<ResponseInputState, SimulationError> {
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
) -> Result<ResponseInputState, SimulationError> {
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

    fn integrator_siso_model() -> DiscreteStateSpaceModel {
        DiscreteStateSpaceModel::try_from_matrices(
            &na::dmatrix![1.0],
            &na::dmatrix![1.0],
            &na::dmatrix![1.0],
            &na::dmatrix![0.0],
            1.0,
        )
        .unwrap()
    }

    // Verifies simulation output includes direct feedthrough term D*u.
    #[test]
    fn test_simulate_includes_feedthrough_term() {
        let model = first_order_siso_model_with_d();
        let u = na::dmatrix![1.0, 1.0, 1.0];
        let x0 = na::dvector![0.0];

        let (y, _x) = simulate(&model, &u, &x0).unwrap();

        assert_eq!(y.ncols(), 3);
        assert!((y[0] - 2.0).abs() < 1e-12);
        assert!((y[1] - 3.0).abs() < 1e-12);
        assert!((y[2] - 3.5).abs() < 1e-12);
    }

    // Verifies step generation matches the number of model inputs for MIMO systems.
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

        let (y, u, x) = step_for_discrete_ss(&model, 1.0).unwrap();

        assert_eq!(u.nrows(), 2);
        assert_eq!(u.ncols(), 10);
        assert_eq!(y.shape(), (2, 10));
        assert_eq!(x.shape(), (2, 11));

        // U should be a 2x10 unit step for each input channel.
        for i in 0..u.nrows() {
            for k in 0..u.ncols() {
                assert!((u[(i, k)] - 1.0).abs() < 1e-12);
            }
        }

        // With A = I, B = I, x0 = 0 and u_k = 1:
        // x_k = [k, k]^T and y_k = x_k (C = I, D = 0).
        for k in 0..10 {
            let expected_state = k as f64;
            assert!((x[(0, k)] - expected_state).abs() < 1e-12);
            assert!((x[(1, k)] - expected_state).abs() < 1e-12);
            assert!((y[(0, k)] - expected_state).abs() < 1e-12);
            assert!((y[(1, k)] - expected_state).abs() < 1e-12);
        }

        // Last state sample is after 10 updates.
        assert!((x[(0, 10)] - 10.0).abs() < 1e-12);
        assert!((x[(1, 10)] - 10.0).abs() < 1e-12);
    }

    // Verifies zero-duration simulation returns empty I/O and initial-state-only trajectory.
    #[test]
    fn test_zero_duration_response() {
        let model = first_order_siso_model_with_d();
        let (y, u, x) = step_for_discrete_ss(&model, 0.0).unwrap();

        assert_eq!(y.shape(), (1, 0));
        assert_eq!(u.shape(), (1, 0));
        assert_eq!(x.shape(), (1, 1));
    }

    // Verifies durations shorter than one sampling period produce zero output samples.
    #[test]
    fn test_short_duration_response() {
        let model = first_order_siso_model_with_d();
        let (y, _u, _x) = step_for_discrete_ss(&model, 0.05).unwrap();
        assert_eq!(y.ncols(), 0);
    }

    // Verifies nominal step simulation values for a simple integrator model. x accumulates the input and y is equal to the beginning of x.
    #[test]
    fn test_step_response_nominal_values() {
        let model = integrator_siso_model();
        let (y, u, x) = step_for_discrete_ss(&model, 3.0).unwrap();

        assert_eq!(u, na::dmatrix![1.0, 1.0, 1.0]);
        assert_eq!(y, na::dmatrix![0.0, 1.0, 2.0]);
        assert_eq!(x, na::dmatrix![0.0, 1.0, 2.0, 3.0]);
    }

    // Verifies nominal impulse simulation values for a simple integrator model.
    #[test]
    fn test_impulse_response_nominal_values() {
        let model = integrator_siso_model();
        let (y, u, x) = impulse_for_discrete_ss(&model, 3.0).unwrap();

        assert_eq!(u, na::dmatrix![1.0, 0.0, 0.0]);
        assert_eq!(y, na::dmatrix![0.0, 1.0, 1.0]);
        assert_eq!(x, na::dmatrix![0.0, 1.0, 1.0, 1.0]);
    }

    // Verifies nominal ramp simulation values for a simple integrator model.
    #[test]
    fn test_ramp_response_nominal_values() {
        let model = integrator_siso_model();
        let (y, u, x) = ramp_for_discrete_ss(&model, 3.0).unwrap();

        assert_eq!(u, na::dmatrix![0.0, 1.0, 2.0]);
        assert_eq!(y, na::dmatrix![0.0, 0.0, 1.0]);
        assert_eq!(x, na::dmatrix![0.0, 0.0, 1.0, 3.0]);
    }

    // Verifies simulation fails when input row count does not match B.ncols().
    #[test]
    fn test_simulate_input_dimension_mismatch() {
        let model = first_order_siso_model_with_d();
        let bad_u = na::dmatrix![1.0, 1.0; 1.0, 1.0];
        let x0 = na::dvector![0.0];

        let result = simulate(&model, &bad_u, &x0);
        assert!(matches!(
            result,
            Err(SimulationError::DimensionMismatch {
                expected_u_rows: 1,
                actual_u_rows: 2,
                expected_x0_rows: 1,
                actual_x0_rows: 1
            })
        ));
    }

    // Verifies simulation fails when x0 length does not match A.nrows().
    #[test]
    fn test_simulate_state_dimension_mismatch() {
        let model = first_order_siso_model_with_d();
        let u = na::dmatrix![1.0, 1.0];
        let bad_x0 = na::dvector![0.0, 0.0];

        let result = simulate(&model, &u, &bad_x0);
        assert!(matches!(
            result,
            Err(SimulationError::DimensionMismatch {
                expected_u_rows: 1,
                actual_u_rows: 1,
                expected_x0_rows: 1,
                actual_x0_rows: 2
            })
        ));
    }

    // Verifies negative duration is rejected for step response.
    #[test]
    fn test_step_negative_duration_error() {
        let model = first_order_siso_model_with_d();
        let result = step_for_discrete_ss(&model, -1.0);
        assert!(matches!(
            result,
            Err(SimulationError::InvalidDuration(d)) if (d + 1.0).abs() < 1e-12
        ));
    }

    // Verifies negative duration is rejected for impulse response.
    #[test]
    fn test_impulse_negative_duration_error() {
        let model = first_order_siso_model_with_d();
        let result = impulse_for_discrete_ss(&model, -0.5);
        assert!(matches!(
            result,
            Err(SimulationError::InvalidDuration(d)) if (d + 0.5).abs() < 1e-12
        ));
    }

    // Verifies negative duration is rejected for ramp response.
    #[test]
    fn test_ramp_negative_duration_error() {
        let model = first_order_siso_model_with_d();
        let result = ramp_for_discrete_ss(&model, -0.25);
        assert!(matches!(
            result,
            Err(SimulationError::InvalidDuration(d)) if (d + 0.25).abs() < 1e-12
        ));
    }

    // Verifies invalid sampling periods are rejected in time vector helper.
    #[test]
    fn test_time_vector_invalid_sampling_dt() {
        let zero = time_vector(0.0, 10);
        let negative = time_vector(-0.1, 10);
        let nan = time_vector(f64::NAN, 10);
        let inf = time_vector(f64::INFINITY, 10);

        assert!(matches!(
            zero,
            Err(SimulationError::Model(ModelError::InvalidSamplingDt(dt))) if dt == 0.0
        ));
        assert!(matches!(
            negative,
            Err(SimulationError::Model(ModelError::InvalidSamplingDt(dt))) if (dt + 0.1).abs() < 1e-12
        ));
        assert!(matches!(
            nan,
            Err(SimulationError::Model(ModelError::InvalidSamplingDt(dt))) if dt.is_nan()
        ));
        assert!(matches!(
            inf,
            Err(SimulationError::Model(ModelError::InvalidSamplingDt(dt))) if dt.is_infinite()
        ));
    }
}
