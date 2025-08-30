use thiserror::Error;

/// Defines the specific errors that can occur in our benchmark application.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Failed to parse configuration from '{file_path}'")]
    ConfigParseError {
        file_path: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Failed to load matrix from source: '{source_url}'")]
    MatrixLoadError {
        source_url: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Solver '{solver_name}' failed for matrix '{matrix_name}': {reason}")]
    SolverFailed {
        solver_name: String,
        matrix_name: String,
        reason: String,
    },
}