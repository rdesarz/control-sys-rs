use control_sys::controller;
use control_sys::model;
use control_sys::model::StateSpaceModel;
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = na::dmatrix![0.0, 1.0; -2.0, -3.0];
    let b = na::dmatrix![0.0; 1.0];
    let c = na::dmatrix![1.0, 0.0];
    let d = na::dmatrix![0.0];

    let model = model::ContinuousStateSpaceModel::try_from_matrices(&a, &b, &c, &d)?;

    let q = na::dmatrix![10.0, 0.0; 0.0, 1.0];
    let r = na::dmatrix![1.0];

    let k = controller::continuous_lqr(model.mat_a(), model.mat_b(), &q, &r, 0.01, 5000, 1e-10)?;
    let acl = controller::closed_loop(model.mat_a(), model.mat_b(), &k)?;

    println!("LQR gain K = {}", k);
    println!("Closed-loop A matrix = {}", acl);

    Ok(())
}
