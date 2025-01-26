use nalgebra as na;

use crate::model::{ContinuousStateSpaceModel, Discrete, DiscreteStateSpaceModel, StateSpaceModel};

/// Generates the step response of a discrete state-space model for a given duration.
///
/// # Arguments
///
/// * `model` - A reference to an object that implements the `StateSpaceModel` and `Discrete` traits.
/// * `duration` - The duration for which the step response is to be generated.
///
/// # Returns
///
/// A tuple containing:
/// * `response` - A matrix containing the output sequence of the step response.
/// * `input` - A matrix containing the input sequence (step input).
/// * `states` - A matrix containing the state sequence during the step response.
///
pub fn step_for_discrete_ss(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> (na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>) {
    // Initial state is zero for a step response
    let initial_state = na::DVector::<f64>::zeros(model.mat_a().nrows());

    // Generate step for given duration
    let n_samples = (duration / model.sampling_dt()).floor() as usize;
    let input = na::DMatrix::from_element(1, n_samples, 1.0f64);

    let (response, states) = simulate_ss_response(model, &input, &initial_state);

    (response, input, states)
}

/// Generates the step response of a continuous state-space model for a given duration by converting it to a discrete model using the forward Euler method.
///
/// # Arguments
///
/// * `model` - A reference to a `ContinuousStateSpaceModel` object.
/// * `sampling_dt` - The sampling time for the discrete model.
/// * `duration` - The duration for which the step response is to be generated.
///
/// # Returns
///
/// A tuple containing:
/// * `response` - A matrix containing the output sequence of the step response.
/// * `input` - A matrix containing the input sequence (step input).
/// * `states` - A matrix containing the state sequence during the step response.
///
pub fn step_for_continuous_ss(
    model: &ContinuousStateSpaceModel,
    sampling_dt: f64,
    duration: f64,
) -> (na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>) {
    let discrete_model =
        DiscreteStateSpaceModel::from_continuous_ss_forward_euler(model, sampling_dt);

    // Initial state is zero for a step response
    let initial_state = na::DVector::<f64>::zeros(model.mat_a().nrows());

    // Generate step for given duration
    let n_samples = (duration / discrete_model.sampling_dt()).floor() as usize;
    let input = na::DMatrix::from_element(1, n_samples, 1.0f64);

    let (response, states) = simulate_ss_response(&discrete_model, &input, &initial_state);

    (response, input, states)
}

fn simulate_ss_response(
    model: &(impl StateSpaceModel + Discrete),
    mat_u: &na::DMatrix<f64>,
    x0: &na::DVector<f64>,
) -> (na::DMatrix<f64>, na::DMatrix<f64>) {
    let sim_time = mat_u.ncols();
    let n_state = model.mat_a().nrows();
    let n_output = model.mat_c().nrows();
    let mut mat_x = na::DMatrix::<f64>::zeros(n_state, sim_time + 1);
    let mut mat_y = na::DMatrix::<f64>::zeros(n_output, sim_time);
    for i in 0..sim_time {
        if i == 0 {
            mat_x.column_mut(i).copy_from(&x0);
            mat_y.column_mut(i).copy_from(&(model.mat_c() * x0));
            mat_x
                .column_mut(i + 1)
                .copy_from(&(model.mat_a() * x0 + model.mat_b() * mat_u.column(i)));
        } else {
            mat_y
                .column_mut(i)
                .copy_from(&(model.mat_c() * mat_x.column(i)));

            let mat_x_slice = mat_x.column(i).into_owned();
            mat_x
                .column_mut(i + 1)
                .copy_from(&(model.mat_a() * mat_x_slice + model.mat_b() * mat_u.column(i)));
        }
    }

    (mat_y, mat_x)
}
