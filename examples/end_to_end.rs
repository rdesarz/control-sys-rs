// End-to-end workflow:
// 1) Build a continuous model.
// 2) Discretize with ZOH.
// 3) Run a step simulation.
// 4) Check controllability.
use control_sys::{analysis, model, simulator};
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = na::dmatrix![0.0, 1.0; -2.0, -3.0];
    let b = na::dmatrix![0.0; 1.0];
    let c = na::dmatrix![1.0, 0.0];
    let d = na::dmatrix![0.0];

    let continuous = model::ContinuousStateSpaceModel::try_from_matrices(&a, &b, &c, &d)?;
    let discrete = model::DiscreteStateSpaceModel::from_continuous_zoh(&continuous, 0.05)?;

    let (y, _u, _x) = simulator::step_for_discrete_ss(&discrete, 5.0)?;
    let (is_controllable, _ctrb) = analysis::is_ss_controllable(&discrete);

    println!("step samples: {}", y.ncols());
    println!("controllable: {}", is_controllable);

    Ok(())
}
