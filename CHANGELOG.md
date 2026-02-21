# Changelog

## Branch: `11-refactor-model-analysis`

This changelog summarizes the work developed on branch `11-refactor-model-analysis`.

### Added
- Checked model constructors with typed errors:
  - `ContinuousStateSpaceModel::try_from_matrices(...)`
  - `DiscreteStateSpaceModel::try_from_matrices(...)`
  - `ModelError` (`MatrixANotSquare`, `DimensionMismatch`, `InvalidSamplingDt`, `SingularMatrix`)
- Consolidated LTI analysis API:
  - `analysis::analyze_lti(...)`
  - `TimeDomain` (`Continuous`, `Discrete`)
  - `LtiAnalysisReport` (poles, spectral radius, stability, controllability/observability diagnostics)
- Expanded simulator API and tests:
  - Generic `simulate(...)`
  - Step, impulse, and ramp response validation with nominal-value tests
  - Error-path tests for invalid dimensions/duration/sampling time
- New analysis-first example:
  - `examples/analysis_report.rs`

### Changed
- Discretization strategy simplified and refocused for control-library coherence:
  - Exposed discretization narrowed to **ZOH only**
- Simulation behavior uses full output equation consistently:
  - `y = Cx + Du`
- Documentation improvements:
  - Expanded Rustdoc in `model.rs` with equations and assumptions
  - Detailed comments on matrix exponential algorithm, including references
  - Added explanatory comments for all unit tests
  - Added top-of-file purpose comments in examples
- CI workflow hardening:
  - Reliable fontconfig install in GitHub Actions (`apt-get update` + `libfontconfig1-dev`)

### Removed
- Controller implementation from this branch scope (deferred to future work)
- LQR implementation/example behavior (placeholder kept for future controller work)
- `simulate_with_noise(...)` (kept simulator scope deterministic)
- Non-target discretization methods from public API:
  - Forward Euler
  - Backward Euler
  - Tustin

### Breaking Changes
- Legacy/unsafe model constructors were deprecated or removed in favor of checked constructors.
- Controller APIs introduced earlier in the branch were removed before final scope stabilization.
- Discretization API reduced to ZOH to keep the public surface minimal.

### Quality / Verification
- Unit test suite expanded substantially for analysis and simulation behavior.
- Edge/error cases now covered explicitly (dimension mismatch, invalid duration, invalid dt).
- Current branch state passes:
  - `cargo test` (unit tests + doctests)

### Notes
- Branch intent evolved toward a narrower milestone: robust LTI model validation, analysis, and simulation foundations.
- Controller synthesis and broader control-toolbox parity are intentionally postponed to later features.
