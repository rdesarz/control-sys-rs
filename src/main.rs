use control_sys_rs::control::controller;
use control_sys_rs::control::model;
use control_sys_rs::control::model::DiscreteStateSpaceModel;
use control_sys_rs::control::simulator;
use control_sys_rs::control::trajectory;

use plotters::prelude::*;

use std::borrow::Borrow;
use std::rc::Rc;

extern crate nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sampling_dt = 0.05;
    let params = model::dc_motor::Parameters::default();
    let model = Rc::new(model::dc_motor::Model::new(params, sampling_dt));

    let (step_response, step, _) = simulator::step(
        <Rc<model::dc_motor::Model> as Borrow<model::dc_motor::Model>>::borrow(&model),
        10.0,
    );

    // Define parameters
    let f = 20usize;
    let v = 20usize;
    let time_steps = 300;

    // Define a pulse trajectory
    let trajectory = trajectory::generate_pulse_trajectory(time_steps);

    // Set the initial state
    let x0 = na::DVector::<f64>::zeros(model.get_mat_a().nrows());

    // Initialize the states
    let mut states = na::DMatrix::<f64>::zeros(x0.nrows(), 1);
    states.column_mut(0).copy_from(&x0);

    // We store the computed inputs
    let mut inputs = na::DMatrix::<f64>::zeros(0, 0);

    // We store the output vectors of the controlled state trajectory
    let mut outputs = na::DMatrix::<f64>::zeros(1, model.get_mat_c().nrows());
    outputs.row_mut(0).copy_from(&(model.get_mat_c() * x0));

    // Create the controller
    let pred_weight = 10f64;
    let q0 = 0.0000000011f64;
    let q_other = 0.0001f64;

    let mut mpc =
        controller::mpc::Controller::new(model, f, v, q0, q_other, pred_weight, &trajectory)?;

    for current_timestep in 0..time_steps - f {
        // Compute input
        let input = mpc.compute_control_input(
            current_timestep,
            &states.column(current_timestep).into_owned(),
        );

        // Compute the next state and output
        let (next_state, next_output) =
            mpc.propagate_dynamics(&input, &states.column(current_timestep).into_owned());

        // Append the lists
        states = na::stack![states, next_state];
        outputs = na::stack![outputs, next_output];

        if inputs.shape() == (0, 0) {
            inputs.resize_mut(1, input.nrows(), 0.);
            inputs.view_range_mut(0, ..).copy_from(&input);
        } else {
            inputs = na::stack![inputs, input];
        }
    }

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

    // Draw the control
    {
        let root = BitMapBackend::new("img/control.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_y = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_y = outputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut chart = ChartBuilder::on(&root)
            .caption("System Output Y", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..step_response.len() as i32, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        // Plot input
        let inputs_series: Vec<(i32, f64)> = inputs
            .row(0)
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as i32, val as f64))
            .collect();

        chart
            .draw_series(LineSeries::new(inputs_series, &Palette99::pick(0)))?
            .label(format!("Input"))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], &Palette99::pick(0)));

        // Plot system response
        let outputs_serie: Vec<(i32, f64)> = outputs
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as i32, val as f64))
            .collect();

        chart
            .draw_series(LineSeries::new(outputs_serie, &Palette99::pick(1)))?
            .label(format!("Output"))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 40, y)], &Palette99::pick(1)));

        let traj_series: Vec<(i32, f64)> = mpc
            .desired_ctrl_traj_total
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as i32, val as f64))
            .collect();

        chart
            .draw_series(LineSeries::new(traj_series, &Palette99::pick(2)))?
            .label(format!("Desired trajectory"))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 60, y)], &Palette99::pick(2)));

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(())
}
