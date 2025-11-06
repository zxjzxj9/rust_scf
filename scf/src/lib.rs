//! SCF (Self-Consistent Field) Calculation Library
//!
//! This library provides implementations of various SCF algorithms for quantum chemistry
//! calculations, including restricted and unrestricted Hartree-Fock methods, as well as
//! geometry optimization algorithms.
//!
//! # Modules
//!
//! - `config`: Configuration management and command-line argument parsing
//! - `io`: Input/output operations, logging, and basis set loading
//! - `scf_impl`: SCF trait definition and implementations (SimpleSCF, SpinSCF)
//! - `optim_impl`: Geometry optimization algorithms (Conjugate Gradient, Steepest Descent)
//! - `force_validation`: Force calculation validation utilities

pub mod config;
pub mod force_validation;
pub mod io;
pub mod optim_impl;
pub mod scf_impl;

// Re-export commonly used items for convenience
pub use config::{Args, Config};
pub use optim_impl::{CGOptimizer, GeometryOptimizer, SteepestDescentOptimizer};
pub use scf_impl::{SimpleSCF, SpinSCF, SCF, DIIS};
