// Step-response plotting example:
// Build a DC motor model, simulate a unit-step input, and export a plot to img/step_response.png.
use control_sys::model::Pole;
use control_sys::simulator;

use plotters::prelude::*;

use std::fs;

extern crate nalgebra as na;

pub mod dc_motor {
    use control_sys::model::DiscreteStateSpaceModel;
    use std::default::Default;

    pub struct Parameters {
        b: f64,
        j: f64,
        k: f64,
        l: f64,
        r: f64,
    }

    impl Default for Parameters {
        fn default() -> Parameters {
            Parameters {
                b: 0.1,
                j: 0.01,
                k: 0.01,
                l: 0.5,
                r: 1.0,
            }
        }
    }

    pub fn build_model(
        params: Parameters,
        sampling_dt: f64,
    ) -> Result<DiscreteStateSpaceModel, control_sys::model::ModelError> {
        let mat_ac = na::dmatrix![
            -params.b / params.j, params.k / params.j;
            -params.k / params.l, -params.r / params.l;
        ];
        let mat_bc = na::dmatrix![0.0; 1.0 / params.l];
        let mat_cc = na::dmatrix![1.0, 0.0];

        DiscreteStateSpaceModel::from_continuous_matrix_zoh(
            &mat_ac,
            &mat_bc,
            &mat_cc,
            &na::dmatrix![0.0],
            sampling_dt,
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sampling_dt = 0.05;
    let params = dc_motor::Parameters::default();
    let model = dc_motor::build_model(params, sampling_dt)?;

    let _poles = model.poles();

    let (step_response, step, _) = simulator::step_for_discrete_ss(&model, 10.0)?;

    fs::create_dir("img").unwrap_or_else(|_| {
        println!("The folder img already exists, no need to create it.");
    });

    let root = BitMapBackend::new("img/step_response.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_y = step_response
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = step_response.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut chart = ChartBuilder::on(&root)
        .caption("System Output Y", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..step_response.ncols() as i32, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    let series_input: Vec<(i32, f64)> = step
        .row(0)
        .iter()
        .enumerate()
        .map(|(i, &val)| (i as i32, val))
        .collect();

    chart
        .draw_series(LineSeries::new(series_input, &Palette99::pick(0)))?
        .label("Input")
        .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], &Palette99::pick(0)));

    let series_y: Vec<(i32, f64)> = step_response
        .row(0)
        .iter()
        .enumerate()
        .map(|(i, &val)| (i as i32, val))
        .collect();

    chart.draw_series(LineSeries::new(series_y, &Palette99::pick(1)))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
