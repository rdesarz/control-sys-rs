// LTI analysis-first example:
// Build a model and run a single consolidated analysis pass
// (poles, stability, controllability, observability).
use control_sys::{analysis, model};
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = model::DiscreteStateSpaceModel::try_from_matrices(
        &na::dmatrix![0.9, 0.1; 0.0, 0.8],
        &na::dmatrix![1.0; 0.5],
        &na::dmatrix![1.0, 0.0],
        &na::dmatrix![0.0],
        0.1,
    )?;

    let report = analysis::analyze_lti(&system, analysis::TimeDomain::Discrete, 1e-6)?;

    println!("stable: {}", report.is_stable);
    println!("spectral radius: {}", report.spectral_radius);
    println!(
        "controllable: {} (rank {}/{})",
        report.controllability.is_full_rank,
        report.controllability.rank,
        report.controllability.expected_rank
    );
    println!(
        "observable: {} (rank {}/{})",
        report.observability.is_full_rank,
        report.observability.rank,
        report.observability.expected_rank
    );

    Ok(())
}
