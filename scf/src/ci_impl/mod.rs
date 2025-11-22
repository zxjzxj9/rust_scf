//! Configuration Interaction (CI) module
//!
//! This module provides implementations of CI methods for computing electron correlation
//! and excited states. CI methods construct the electronic wave function as a linear
//! combination of Slater determinants.
//!
//! # Available Methods
//!
//! - **CIS** (Configuration Interaction Singles): For computing excited states
//! - **CISD** (Configuration Interaction Singles and Doubles): Ground state correlation
//!
//! # Usage
//!
//! ```rust,ignore
//! use scf::{SimpleSCF, CI};
//!
//! // After converged HF calculation
//! let mut ci = scf.create_ci(10, 1e-6); // 10 states, 1e-6 threshold
//!
//! // Run CISD for ground state
//! let cisd_energy = ci.calculate_cisd_energy();
//!
//! // Or run CIS for excited states
//! let cis_energies = ci.calculate_cis_energies(5); // 5 excited states
//! ```

mod ci;
mod tests;

pub use ci::CIMethod;
pub use ci::CI;
