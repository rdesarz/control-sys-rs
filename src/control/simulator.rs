use nalgebra as na;

use crate::control::model::{StateSpaceModel, Discrete};

fn system_simulate(
    model: &(impl StateSpaceModel + Discrete),
    mat_u: &na::DMatrix<f64>,
    x0: &na::DVector<f64>,
) -> (na::DMatrix<f64>, na::DMatrix<f64>) {
    let sim_time = mat_u.ncols();
    let n_state = model.get_mat_a().nrows();
    let n_output = model.get_mat_c().nrows();
    let mut mat_x = na::DMatrix::<f64>::zeros(n_state, sim_time + 1);
    let mut mat_y = na::DMatrix::<f64>::zeros(n_output, sim_time);
    for i in 0..sim_time {
        if i == 0 {
            mat_x.column_mut(i).copy_from(&x0);
            mat_y.column_mut(i).copy_from(&(model.get_mat_c() * x0));
            mat_x
                .column_mut(i + 1)
                .copy_from(&(model.get_mat_a() * x0 + model.get_mat_b() * mat_u.column(i)));
        } else {
            mat_y
                .column_mut(i)
                .copy_from(&(model.get_mat_c() * mat_x.column(i)));

            let mat_x_slice = mat_x.column(i).into_owned();
            mat_x.column_mut(i + 1).copy_from(
                &(model.get_mat_a() * mat_x_slice + model.get_mat_b() * mat_u.column(i)),
            );
        }
    }

    (mat_y, mat_x)
}

pub fn compute_system_response(
    model: &(impl StateSpaceModel + Discrete),
    input: &na::DMatrix<f64>,
    initial_state: &na::DVector<f64>,
) -> Vec<f64> {
    // simulate the discrete-time system
    let (y_test, _x_test) = system_simulate(model, &input, &initial_state);

    let system_response: Vec<f64> = y_test.row(0).iter().map(|&val| val as f64).collect();

    system_response
}

pub fn step(
    model: &(impl StateSpaceModel + Discrete),
    duration: f64,
) -> (na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>) {
    // Initial state is zero for a step response
    let initial_state = na::DVector::<f64>::zeros(model.get_mat_a().nrows());

    // Generate step for given duration
    let n_samples = (duration / model.get_sampling_dt()).floor() as usize;
    let input = na::DMatrix::from_element(1, n_samples, 1.0f64);

    let (response, states) = system_simulate(model, &input, &initial_state);

    (response, input, states)
}
