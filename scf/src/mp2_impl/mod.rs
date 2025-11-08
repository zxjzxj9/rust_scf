//! MP2 (Møller-Plesset perturbation theory, second order) implementation
//!
//! This module provides MP2 correlation energy calculations based on
//! converged Hartree-Fock results. MP2 adds electron correlation corrections
//! beyond the mean-field HF approximation.
//!
//! # Theory
//!
//! The MP2 correlation energy is given by:
//!
//! E_MP2 = Σ_{i<j,a<b} [(ia|jb) * (2*(ia|jb) - (ib|ja))] / (ε_i + ε_j - ε_a - ε_b)
//!
//! where:
//! - i, j are occupied molecular orbitals
//! - a, b are virtual (unoccupied) molecular orbitals
//! - (ia|jb) are two-electron integrals in MO basis
//! - ε are orbital energies
//!
//! # Usage
//!
//! MP2 calculations require a converged HF calculation first:
//!
//! ```ignore
//! // First perform HF calculation
//! let mut scf = SimpleSCF::new();
//! // ... setup and run SCF ...
//! scf.scf_cycle();
//!
//! // Then calculate MP2 energy
//! let mut mp2 = MP2::from_scf(&scf);
//! let correlation_energy = mp2.calculate_mp2_energy();
//! let total_mp2_energy = scf.calculate_total_energy() + correlation_energy;
//! ```

mod mp2;
#[cfg(test)]
mod tests;

pub use mp2::MP2;

