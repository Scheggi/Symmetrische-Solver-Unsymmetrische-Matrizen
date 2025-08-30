use serde::Deserialize;

/// Represents the configuration for a specific solver.
/// `#[serde(tag = "type")]` tells serde to use the "type" field
/// in the JSON to decide which enum variant to create.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum SolverConfig {
    LU,
    CG,
    CGNE,
    GMRES { m: usize},
    BiCGSTAB { tol: f64 },
}
/// Represents a single test run defined in the JSON config.
#[derive(Debug, Deserialize)]
pub(crate) struct TestRun {
    pub(crate) matrix_name: String,
    pub(crate) matrix_url: String,
    pub(crate) solver: SolverConfig,
}