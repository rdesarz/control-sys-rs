# control-sys.rs

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/rdesarz/control-sys-rs/rust.yml)

## A Control System Library

Control-sys.rs is a control system library written in Rust. It currently tools to define state-space model representing LTI systems and analyze them. The goal is to develop more and more features for designing and implementing control systems in Rust.

## Basic functions

### Defining a continuous state space model

A continuous state space model can be defined with `ContinuousStateSpaceModel`. As an example, a state-space model representing a DC motor can be defined this way. We are using `nalgebra` to represent the matrices: 

```rust
using control_sys::model;

// DC motor parameters
let b = 0.1;
let j = 0.01;
let k = 0.01;
let l = 0.5;
let r = 1.0;

let mat_ac = na::dmatrix![
        -b / j, k / j;
        -k / l, -r / l;
    ];
let mat_bc = na::dmatrix![0.0; 
                          1.0 / l];
let mat_cc = na::dmatrix![1.0, 0.0];

let model = model::ContinuousStateSpaceModel::new(&mat_ac, &mat_bc, &mat_cc, dmatrix![]);
```

### Building a discrete state-space model from a continuous one

A `DiscreteStateSpaceModel` can be built from a continuous one. You then need to specify the sampling step `ts`: 

```rust
using control_sys::model;

// DC motor parameters
let b = 0.1;
let j = 0.01;
let k = 0.01;
let l = 0.5;
let r = 1.0;

let mat_ac = na::dmatrix![
        -b / j, k / j;
        -k / l, -r / l;
    ];
let mat_bc = na::dmatrix![0.0; 
                          1.0 / l];
let mat_cc = na::dmatrix![1.0, 0.0];

let model = model::ContinuousStateSpaceModel::new(&mat_ac, &mat_bc, &mat_cc, dmatrix![]);

let discrete_model = 
model::DiscreteStateSpaceModel::from_continuous_matrix_forward_euler(&model, 0.05);
```

### Computing the step response of a system

You can compute the step response of a system. For a discrete system, the simulation steps are given by the sampling step of the discretization:

```rust
using control_sys::simulator;

// where model implements the traits `StateSpaceModel` and `Discrete`
let duration = 10.0; // [s]
let (response, input, states) = simulator::step_for_discrete_ss(
        &model,
        duration,
    );
```

You can also compute the step response for a continuous model. You will need to provide the sampling step and the model will be discretized before computing the step response:

```rust
using control_sys::simulator;

// where model is a continuous state space model
let sampling_dt = 0.05; // [s]
let duration = 10.0; // [s]
let (response, input, states) = simulator::step_for_continuous_ss(
        &model,
        sampling_dt,
        duration,
    );
```

### Controllability of a system

The controllability of a system can be evaluated using the `is_ss_controllable` method:

```rust
let ss_model = model::DiscreteStateSpaceModel::new(
        &nalgebra::dmatrix![1.0, -2.0; 
                            2.0, 1.0],
        &nalgebra::dmatrix![1.0;
                            2.0],
        &nalgebra::dmatrix![],
        &nalgebra::dmatrix![],
        0.05,
    );

let controllability_matrix = analysis::compute_controllability_matrix(ss_model.mat_a(), ss_model.mat_b())?;

if analysis::is_ss_controllable(&ss_model)
{
    println!("The system is controllable");
    println!("Its controllability matrix is: {}", controllability_matrix);
}
```


