//! CCSD (Coupled Cluster Singles and Doubles) implementation module
//!
//! This module provides the CCSD method for calculating electron correlation
//! energy beyond the Hartree-Fock approximation.
//!
//! # Overview
//!
//! CCSD is an ab initio method that uses an exponential ansatz for the wavefunction:
//!
//! |Ψ⟩ = exp(T₁ + T₂) |Φ₀⟩
//!
//! where T₁ and T₂ are the singles and doubles cluster operators.
//!
//! # Usage
//!
//! ```rust,ignore
//! use scf::{SimpleSCF, SCF, CCSD};
//!
//! // Perform HF calculation
//! let mut scf = SimpleSCF::new();
//! // ... setup and run SCF ...
//! let hf_energy = scf.calculate_total_energy();
//!
//! // Create CCSD calculator
//! let mut ccsd = scf.create_ccsd(50, 1e-7);
//!
//! // Solve CCSD equations
//! let correlation_energy = ccsd.solve();
//! let total_ccsd_energy = hf_energy + correlation_energy;
//! ```

mod ccsd;

pub use ccsd::CCSD;

#[cfg(test)]
mod tests;
