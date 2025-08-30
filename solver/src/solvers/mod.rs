// Declare the solver sub-modules
pub mod bicgstab;
pub mod cg;
pub mod cgne;
pub mod gmres;
pub mod lu;

// Re-export the public functions and structs for easy access from main.rs
pub use self::bicgstab::{bicgstab};
pub use self::cg::conjugate_gradient;
pub use self::cgne::conjugate_gradient_normal_euqitation;
pub use self::gmres::gmres;
pub use self::lu::solve_lu;
