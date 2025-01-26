#![warn(missing_docs)]
/*!
# control-sys.rs

**Control-sys.rs** is a control system library written in Rust. It implements tools to represent and analyze LTI systems using state-space model.

## Examples

### Continuous state space model

A continuous state space model can be defined with `ContinuousStateSpaceModel`. As an example, a state-space model representing a DC motor can be defined this way. We are using `nalgebra` to represent the matrices: 

```rust
use nalgebra as na;
use control_sys_rs::model;

// DC motor parameters
let b = 0.1f64;
let j = 0.01f64;
let k = 0.01f64;
let l = 0.5f64;
let r = 1.0f64;

let mat_ac = na::dmatrix![
        -b / j, k / j;
        -k / l, -r / l;
    ];
let mat_bc = na::dmatrix![0.0; 
                          1.0 / l];
let mat_cc = na::dmatrix![1.0, 0.0];

let cont_model = model::ContinuousStateSpaceModel::from_matrices(&mat_ac, &mat_bc, &mat_cc, &na::dmatrix![]);
```

### Continuous to discrete model conversion

A `DiscreteStateSpaceModel` can be built from a continuous one. You then need to specify the sampling step `ts`: 

```rust
use nalgebra as na;
use control_sys_rs::model;

// DC motor parameters
let b = 0.1f64;
let j = 0.01f64;
let k = 0.01f64;
let l = 0.5f64;
let r = 1.0f64;

let mat_ac = na::dmatrix![
        -b / j, k / j;
        -k / l, -r / l;
    ];
let mat_bc = na::dmatrix![0.0; 
                          1.0 / l];
let mat_cc = na::dmatrix![1.0, 0.0];

let cont_model = model::ContinuousStateSpaceModel::from_matrices(&mat_ac, &mat_bc, &mat_cc, &na::dmatrix![]);

let discrete_model = 
    model::DiscreteStateSpaceModel::from_continuous_ss_forward_euler(&cont_model, 0.05);
```

### Step response 

You can compute the step response of a system. For a discrete system, the simulation steps are given by the sampling step of the discretization:

```rust
use nalgebra as na;
use control_sys_rs::{model, simulator};

// DC motor parameters
let b = 0.1f64;
let j = 0.01f64;
let k = 0.01f64;
let l = 0.5f64;
let r = 1.0f64;

let mat_ac = na::dmatrix![
        -b / j, k / j;
        -k / l, -r / l;
    ];
let mat_bc = na::dmatrix![0.0; 
                          1.0 / l];
let mat_cc = na::dmatrix![1.0, 0.0];

let cont_model = model::ContinuousStateSpaceModel::from_matrices(&mat_ac, &mat_bc, &mat_cc, &na::dmatrix![]);

let discrete_model = 
    model::DiscreteStateSpaceModel::from_continuous_ss_forward_euler(&cont_model, 0.05);

// where model implements the traits `StateSpaceModel` and `Discrete`
let duration = 10.0; // [s]
let (response, input, states) = simulator::step_for_discrete_ss(
        &discrete_model,
        duration,
    );
```

You can also compute the step response for a continuous model. You will need to provide the sampling step and the model will be discretized before computing the step response:

```rust
use nalgebra as na;
use control_sys_rs::{model, simulator};

// DC motor parameters
let b = 0.1f64;
let j = 0.01f64;
let k = 0.01f64;
let l = 0.5f64;
let r = 1.0f64;

let mat_ac = na::dmatrix![
        -b / j, k / j;
        -k / l, -r / l;
    ];
let mat_bc = na::dmatrix![0.0; 
                          1.0 / l];
let mat_cc = na::dmatrix![1.0, 0.0];

let cont_model = model::ContinuousStateSpaceModel::from_matrices(&mat_ac, &mat_bc, &mat_cc, &na::dmatrix![]);

// where model is a continuous state space model
let sampling_dt = 0.05; // [s]
let duration = 10.0; // [s]
let (response, input, states) = simulator::step_for_continuous_ss(
        &cont_model,
        sampling_dt,
        duration,
    );
```

### Controllability 

The controllability of a system can be evaluated using the `is_ss_controllable` method:

```rust
use nalgebra as na;
use control_sys_rs::{model, analysis};

let ss_model = model::DiscreteStateSpaceModel::from_matrices(
        &nalgebra::dmatrix![1.0, -2.0; 
                            2.0, 1.0],
        &nalgebra::dmatrix![1.0;
                            2.0],
        &nalgebra::dmatrix![],
        &nalgebra::dmatrix![],
        0.05,
    );

let (is_controllable, controllability_matrix) = analysis::is_ss_controllable(&ss_model);

if is_controllable
{
    println!("The system is controllable");
    println!("Its controllability matrix is: {}", controllability_matrix);
}
```



 */

/// Methods to analyze control systems
pub mod analysis;

/// Structures to represent control systems
pub mod model;

/// Methods to simulate control systems in time-space
pub mod simulator;

/// Methods to generate trajectories
pub mod trajectory;
