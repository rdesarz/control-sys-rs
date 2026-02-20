// Controllability example:
// Build a discrete state-space model and compute its controllability matrix/rank.
use control_sys::analysis;
use control_sys::model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ss_model = model::DiscreteStateSpaceModel::try_from_matrices(
        &nalgebra::dmatrix![1.0, -2.0;
                            2.0, 1.0],
        &nalgebra::dmatrix![1.0;
                            2.0],
        &nalgebra::dmatrix![1.0, 0.0],
        &nalgebra::dmatrix![0.0],
        0.05,
    )?;

    let (is_controllable, controllability_matrix) = analysis::is_ss_controllable(&ss_model);

    if is_controllable {
        println!("The system is controllable");
        println!("Its controllability matrix is: {}", controllability_matrix);
    }

    Ok(())
}
