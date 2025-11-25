//! SCF (Self-Consistent Field) Calculation Library
//!
//! This library provides implementations of various SCF algorithms for quantum chemistry
//! calculations, including restricted and unrestricted Hartree-Fock methods, MP2
//! (Møller-Plesset perturbation theory), CCSD (Coupled Cluster Singles and Doubles),
//! CI (Configuration Interaction), and geometry optimization algorithms.
//!
//! # Modules
//!
//! - `config`: Configuration management and command-line argument parsing
//! - `io`: Input/output operations, logging, and basis set loading
//! - `scf_impl`: SCF trait definition and implementations (SimpleSCF, SpinSCF)
//! - `mp2_impl`: MP2 correlation energy calculations
//! - `ccsd_impl`: CCSD correlation energy calculations
//! - `ci_impl`: CI correlation and excited state calculations
//! - `optim_impl`: Geometry optimization algorithms (Conjugate Gradient, Steepest Descent)
//! - `force_validation`: Force calculation validation utilities

pub mod app;
pub mod ccsd_impl;
pub mod ci_impl;
pub mod config;
pub mod force_validation;
pub mod io;
pub mod mp2_impl;
pub mod optim_impl;
pub mod scf_impl;

// Re-export commonly used items for convenience
pub use ccsd_impl::CCSD;
pub use ci_impl::{CIMethod, CI};
pub use config::{Args, Config};
pub use mp2_impl::MP2;
pub use optim_impl::{CGOptimizer, GeometryOptimizer, SteepestDescentOptimizer};
pub use scf_impl::{SimpleSCF, SpinSCF, DIIS, SCF};

#[cfg(feature = "python")]
mod python;
