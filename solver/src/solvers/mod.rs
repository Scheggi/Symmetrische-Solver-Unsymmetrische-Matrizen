// Declare the solver sub-modules
pub mod bicgstab;
pub mod cg;
pub mod gmres;
pub mod lu;
mod cgne;

// Re-export the public functions and structs for easy access from main.rs
pub use self::bicgstab::{bicgstab};
pub use self::cg::conjugate_gradient;
pub use self::gmres::gmres;
pub use self::lu::solve_lu;


// Symmetrische-Solver-Unsymmetrische-Matrizen
