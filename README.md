# control-sys, a control system library written in Rust

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/rdesarz/control-sys-rs/rust.yml)

**control-sys** is a Rust library to represent, analyze, simulate, and control LTI systems using state-space models.

## Highlights

- Checked model constructors with typed errors (`ModelError`)
- Continuous-to-discrete conversion with explicit methods:
  - ZOH
  - Tustin
- Time-domain simulation with full output equation `y = Cx + Du`
- Controllability and observability analysis
- Stability checks and rank diagnostics
- Controller helpers:
  - SISO pole placement
  - Discrete LQR
  - Continuous LQR approximation
  - Closed-loop assembly (`A-BK`)

## Example: End-to-End Workflow

```rust
use control_sys::{analysis, model, simulator};
use nalgebra as na;

let a = na::dmatrix![0.0, 1.0; -2.0, -3.0];
let b = na::dmatrix![0.0; 1.0];
let c = na::dmatrix![1.0, 0.0];
let d = na::dmatrix![0.0];

let cont = model::ContinuousStateSpaceModel::try_from_matrices(&a, &b, &c, &d).unwrap();
let disc = model::DiscreteStateSpaceModel::from_continuous_zoh(&cont, 0.05).unwrap();

let (y, u, _x) = simulator::step_for_discrete_ss(&disc, 5.0).unwrap();
assert_eq!(y.nrows(), 1);
assert_eq!(u.nrows(), 1);

let (is_ctrb, _ctrb_mat) = analysis::is_ss_controllable(&disc);
assert!(is_ctrb);
```

## Discretization Methods and Assumptions

`DiscreteStateSpaceModel` has explicit conversion methods:

- `from_continuous_zoh`: exact ZOH (recommended default)
- `from_continuous_tustin`: bilinear transform

Use ZOH by default for sampled-data systems with held inputs.

## Running checks

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
