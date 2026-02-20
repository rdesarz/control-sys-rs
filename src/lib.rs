#![warn(missing_docs)]
/*!
# control-sys

`control-sys` is a Rust control systems library focused on LTI state-space models.

## Quick Start

```rust
use nalgebra as na;
use control_sys::{analysis, model, simulator};
use control_sys::model::StateSpaceModel;

let a = na::dmatrix![0.0, 1.0; -2.0, -3.0];
let b = na::dmatrix![0.0; 1.0];
let c = na::dmatrix![1.0, 0.0];
let d = na::dmatrix![0.0];

let continuous = model::ContinuousStateSpaceModel::try_from_matrices(&a, &b, &c, &d).unwrap();
let discrete = model::DiscreteStateSpaceModel::from_continuous_zoh(&continuous, 0.05).unwrap();

let (y, u, x) = simulator::step_for_discrete_ss(&discrete, 2.0).unwrap();
assert_eq!(u.nrows(), discrete.mat_b().ncols());
assert_eq!(x.nrows(), discrete.mat_a().nrows());
assert_eq!(y.nrows(), discrete.mat_c().nrows());

let (is_controllable, _ctrb) = analysis::is_ss_controllable(&discrete);
assert!(is_controllable);
```

## Discretization Methods and Assumptions

`DiscreteStateSpaceModel` provides explicit conversion APIs:

- `from_continuous_zoh`: exact ZOH using augmented matrix exponential.
- `from_continuous_forward_euler`: explicit Euler.
- `from_continuous_backward_euler`: implicit Euler.
- `from_continuous_tustin`: bilinear transform.

Recommended default is ZOH for sampled-data systems driven by held inputs.
*/

/// Methods to analyze control systems.
pub mod analysis;

/// Methods and structs to create controllers.
pub mod controller;

/// Structures to represent control systems.
pub mod model;

/// Methods to simulate control systems in time-space.
pub mod simulator;

/// Methods to generate trajectories.
pub mod trajectory;
