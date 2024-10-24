use control_sys_rs::control::model;
use control_sys_rs::control::model::Pole;
use control_sys_rs::control::simulator;

use plotters::prelude::*;

use std::borrow::Borrow;
use std::fs;
use std::rc::Rc;

extern crate nalgebra as na;

pub mod two_spring_damper_mass {
    extern crate nalgebra as na;

    use std::default::Default;

    use control_sys_rs::control::model::DiscreteStateSpaceModel;

    pub struct Parameters {
        m1: f64,
        m2: f64,
        k1: f64,
        k2: f64,
        d1: f64,
        d2: f64,
    }

    impl Default for Parameters {
        fn default() -> Parameters {
            Parameters {
                m1: 2.0,
                m2: 2.0,
                k1: 100.0,
                k2: 200.0,
                d1: 1.0,
                d2: 5.0,
            }
        }
    }

    pub fn build_model(params: Parameters, sampling_dt: f64) -> DiscreteStateSpaceModel {
        // Define the continuous-time system matrices
        let mat_ac = na::dmatrix![
            0.0, 1.0, 0.0, 0.0;
            -(params.k1 + params.k2) / params.m1, -(params.d1 + params.d2) / params.m1, params.k2 / params.m1, params.d2 / params.m1;
            0.0, 0.0, 0.0, 1.0;
            params.k2 / params.m2, params.d2 / params.m2, -params.k2 / params.m2, -params.d2 / params.m2
        ];
        let mat_bc = na::dmatrix![0.0; 0.0; 0.0; 1.0 / params.m2];
        let mat_cc = na::dmatrix![1.0, 0.0, 0.0, 0.0];
        let mat_dc = na::dmatrix![0.0];

        // Model discretization
        DiscreteStateSpaceModel::from_continuous_matrix_forward_euler(
            &mat_ac,
            &mat_bc,
            &mat_cc,
            &mat_dc,
            sampling_dt,
        )
    }
}

pub mod dc_motor {
    use control_sys_rs::control::model::DiscreteStateSpaceModel;
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

    pub fn build_model(params: Parameters, sampling_dt: f64) -> DiscreteStateSpaceModel {
        // Define the continuous-time system matrices
        let mat_ac = na::dmatrix![
            -params.b / params.j, params.k / params.j;
            -params.k / params.l, -params.r / params.l;
        ];
        let mat_bc = na::dmatrix![0.0; 1.0 / params.l];
        let mat_cc = na::dmatrix![1.0, 0.0];

        // Model discretization
        DiscreteStateSpaceModel::from_continuous_matrix_forward_euler(
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
    let model = Rc::new(dc_motor::build_model(params, sampling_dt));

    model.poles();

    let (step_response, step, _) = simulator::step_for_discrete_ss(
        <Rc<model::DiscreteStateSpaceModel> as Borrow<model::DiscreteStateSpaceModel>>::borrow(
            &model,
        ),
        10.0,
    );

    // Create a folder for results
    fs::create_dir("img").unwrap_or_else(|_| {
        println!("The folder img already exists, no need to create it.");
    });

    // Draw the step response
    {
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
            .build_cartesian_2d(0..step_response.len() as i32, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        // Plot input
        let series_input: Vec<(i32, f64)> = step
            .row(0)
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as i32, val as f64))
            .collect();

        chart
            .draw_series(LineSeries::new(series_input, &Palette99::pick(0)))?
            .label(format!("Output {}", 0))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], &Palette99::pick(0)));

        // Plot system response
        let series_y: Vec<(i32, f64)> = step_response
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as i32, val as f64))
            .collect();

        chart.draw_series(LineSeries::new(series_y, &Palette99::pick(1)))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(())
}
