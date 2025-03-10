extern crate nalgebra as na;

/// Generate a pulse trajectory
///
/// # Arguments
///
/// * `time_steps` - Number of time_steps on which the trajectory should be generated
///
/// # Returns
///
/// A matrix that contains the trajectory.
pub fn generate_pulse_trajectory(time_steps: usize) -> na::DMatrix<f64> {
    let mut traj = na::DMatrix::<f64>::zeros(time_steps, 1);
    traj.view_range_mut(0..time_steps / 3, 0..1)
        .copy_from(&na::DMatrix::from_element(time_steps / 3, 1, 1.0));
    traj.view_range_mut(2 * time_steps / 3..time_steps, 0..1)
        .copy_from(&na::DMatrix::from_element(time_steps / 3, 1, 1.0));

    traj
}
