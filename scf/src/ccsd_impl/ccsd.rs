//! Core CCSD (Coupled Cluster Singles and Doubles) implementation
//!
//! This module implements the CCSD method for computing electron correlation energy.
//! CCSD is an iterative method that solves for T1 (singles) and T2 (doubles) amplitudes.

extern crate nalgebra as na;

use basis::basis::Basis;
use na::{DMatrix, DVector};
use periodic_table_on_an_enum::Element;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::info;

/// CCSD calculation structure
///
/// This struct holds all necessary data from a converged HF calculation
/// and provides methods to compute the CCSD correlation energy.
pub struct CCSD<B: Basis> {
    /// Number of basis functions
    pub num_basis: usize,

    /// Number of occupied orbitals
    pub num_occ: usize,

    /// Number of virtual orbitals
    pub num_virt: usize,

    /// Molecular orbital coefficients (from HF)
    pub mo_coeffs: DMatrix<f64>,

    /// Orbital energies (from HF)
    pub orbital_energies: DVector<f64>,

    /// Molecular orbital basis functions
    pub mo_basis: Vec<Arc<B>>,

    /// Elements (for electron counting)
    pub elems: Vec<Element>,

    /// T1 amplitudes (singles): t_i^a
    /// Dimensions: (num_occ, num_virt)
    pub t1: DMatrix<f64>,

    /// T2 amplitudes (doubles): t_ij^ab
    /// Stored as a vector in row-major order: [i*num_occ*num_virt*num_virt + j*num_virt*num_virt + a*num_virt + b]
    /// Dimensions: (num_occ, num_occ, num_virt, num_virt)
    pub t2: Vec<f64>,

    /// CCSD correlation energy (calculated)
    pub correlation_energy: Option<f64>,

    /// Maximum number of CCSD iterations
    pub max_iterations: usize,

    /// Convergence threshold for T amplitudes
    pub convergence_threshold: f64,

    /// Cache for frequently used MO integrals
    /// Using on-the-fly calculation to save memory
    integral_cache_enabled: bool,
}

impl<B: Basis + Send + Sync> CCSD<B> {
    /// Create a new CCSD calculator from converged HF data
    ///
    /// # Arguments
    ///
    /// * `mo_coeffs` - Molecular orbital coefficients from HF
    /// * `orbital_energies` - Orbital energies from HF
    /// * `mo_basis` - Basis functions
    /// * `elems` - Elements (for counting electrons)
    /// * `max_iterations` - Maximum number of CCSD iterations
    /// * `convergence_threshold` - Convergence threshold for amplitudes
    ///
    /// # Returns
    ///
    /// A new CCSD instance ready for correlation energy calculation
    pub fn new(
        mo_coeffs: DMatrix<f64>,
        orbital_energies: DVector<f64>,
        mo_basis: Vec<Arc<B>>,
        elems: Vec<Element>,
        max_iterations: usize,
        convergence_threshold: f64,
    ) -> Self {
        let num_basis = mo_basis.len();
        let total_electrons: usize = elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        let num_occ = total_electrons / 2; // Restricted HF
        let num_virt = num_basis - num_occ;

        info!("===========================================");
        info!("     CCSD Initialization");
        info!("===========================================");
        info!("Number of basis functions: {}", num_basis);
        info!("Number of occupied orbitals: {}", num_occ);
        info!("Number of virtual orbitals: {}", num_virt);
        info!("Total electrons: {}", total_electrons);
        info!("Max iterations: {}", max_iterations);
        info!("Convergence threshold: {:.2e}", convergence_threshold);

        // Initialize T1 amplitudes to zero
        let t1 = DMatrix::zeros(num_occ, num_virt);

        // Initialize T2 amplitudes to zero
        let t2_size = num_occ * num_occ * num_virt * num_virt;
        let t2 = vec![0.0; t2_size];

        info!("T1 amplitude matrix size: {} x {}", num_occ, num_virt);
        info!(
            "T2 amplitude tensor size: {} x {} x {} x {}",
            num_occ, num_occ, num_virt, num_virt
        );
        info!(
            "Total memory for amplitudes: {:.2} MB",
            (t1.len() * 8 + t2.len() * 8) as f64 / 1_048_576.0
        );
        info!("===========================================");

        CCSD {
            num_basis,
            num_occ,
            num_virt,
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            t1,
            t2,
            correlation_energy: None,
            max_iterations,
            convergence_threshold,
            integral_cache_enabled: false,
        }
    }

    /// Access T2 amplitude: t_ij^ab
    #[inline]
    fn get_t2(&self, i: usize, j: usize, a: usize, b: usize) -> f64 {
        let idx = i * (self.num_occ * self.num_virt * self.num_virt)
            + j * (self.num_virt * self.num_virt)
            + a * self.num_virt
            + b;
        self.t2[idx]
    }

    /// Set T2 amplitude: t_ij^ab
    #[inline]
    fn set_t2(&mut self, i: usize, j: usize, a: usize, b: usize, value: f64) {
        let idx = i * (self.num_occ * self.num_virt * self.num_virt)
            + j * (self.num_virt * self.num_virt)
            + a * self.num_virt
            + b;
        self.t2[idx] = value;
    }

    /// Calculate a single two-electron integral in the MO basis
    ///
    /// (pq|rs) = Σ_μνλσ C_μp C_νq C_λr C_σs (μν|λσ)
    ///
    /// # Arguments
    ///
    /// * `p, q, r, s` - MO indices
    ///
    /// # Returns
    ///
    /// The two-electron integral (pq|rs) in MO basis
    fn mo_two_electron_integral(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let mut integral = 0.0;

        for mu in 0..self.num_basis {
            let c_mu_p = self.mo_coeffs[(mu, p)];
            if c_mu_p.abs() < 1e-10 {
                continue;
            }

            for nu in 0..self.num_basis {
                let c_nu_q = self.mo_coeffs[(nu, q)];
                if c_nu_q.abs() < 1e-10 {
                    continue;
                }

                for lambda in 0..self.num_basis {
                    let c_lambda_r = self.mo_coeffs[(lambda, r)];
                    if c_lambda_r.abs() < 1e-10 {
                        continue;
                    }

                    for sigma in 0..self.num_basis {
                        let c_sigma_s = self.mo_coeffs[(sigma, s)];
                        if c_sigma_s.abs() < 1e-10 {
                            continue;
                        }

                        // Calculate AO integral (μν|λσ)
                        let ao_integral = B::JKabcd(
                            &self.mo_basis[mu],
                            &self.mo_basis[nu],
                            &self.mo_basis[lambda],
                            &self.mo_basis[sigma],
                        );

                        integral += c_mu_p * c_nu_q * c_lambda_r * c_sigma_s * ao_integral;
                    }
                }
            }
        }

        integral
    }

    /// Initialize T2 amplitudes from MP2 guess
    ///
    /// t_ij^ab = (ia|jb) / (ε_i + ε_j - ε_a - ε_b)
    fn initialize_t2_from_mp2(&mut self) {
        info!("Initializing T2 amplitudes with MP2 guess...");

        for i in 0..self.num_occ {
            for j in 0..self.num_occ {
                for a in 0..self.num_virt {
                    for b in 0..self.num_virt {
                        let ia_jb =
                            self.mo_two_electron_integral(i, self.num_occ + a, j, self.num_occ + b);

                        let denominator = self.orbital_energies[i] + self.orbital_energies[j]
                            - self.orbital_energies[self.num_occ + a]
                            - self.orbital_energies[self.num_occ + b];

                        if denominator.abs() > 1e-10 {
                            self.set_t2(i, j, a, b, ia_jb / denominator);
                        }
                    }
                }
            }

            if (i + 1) % 5 == 0 || i == self.num_occ - 1 {
                info!(
                    "  Initialized T2 for {}/{} occupied orbitals",
                    i + 1,
                    self.num_occ
                );
            }
        }

        info!("T2 initialization complete.");
    }

    /// Calculate CCSD correlation energy from current amplitudes
    ///
    /// E_CCSD = Σ_ia f_ia t_i^a + 1/4 Σ_ijab <ij||ab> t_ij^ab
    ///          + 1/2 Σ_ijab <ij||ab> t_i^a t_j^b
    ///
    /// where <ij||ab> = (ia|jb) - (ib|ja) are antisymmetrized integrals
    fn calculate_energy(&self) -> f64 {
        let mut energy = 0.0;

        // Singles contribution: Σ_ia f_ia t_i^a
        for i in 0..self.num_occ {
            for a in 0..self.num_virt {
                let f_ia = self.orbital_energies[self.num_occ + a];
                energy += f_ia * self.t1[(i, a)];
            }
        }

        // Doubles contribution
        for i in 0..self.num_occ {
            for j in 0..self.num_occ {
                for a in 0..self.num_virt {
                    for b in 0..self.num_virt {
                        let ia_jb =
                            self.mo_two_electron_integral(i, self.num_occ + a, j, self.num_occ + b);
                        let ib_ja =
                            self.mo_two_electron_integral(i, self.num_occ + b, j, self.num_occ + a);

                        let antisym = ia_jb - ib_ja;

                        // Pure doubles: 1/4 Σ <ij||ab> t_ij^ab
                        energy += 0.25 * antisym * self.get_t2(i, j, a, b);

                        // Singles × Singles: 1/2 Σ <ij||ab> t_i^a t_j^b
                        energy += 0.5 * antisym * self.t1[(i, a)] * self.t1[(j, b)];
                    }
                }
            }
        }

        energy
    }

    /// Update T1 amplitudes
    ///
    /// Simplified T1 equation for demonstration:
    /// t_i^a(new) = [f_ia + intermediates] / D_i^a
    ///
    /// where D_i^a = ε_i - ε_a
    fn update_t1(&mut self) -> DMatrix<f64> {
        let mut new_t1 = DMatrix::zeros(self.num_occ, self.num_virt);

        for i in 0..self.num_occ {
            for a in 0..self.num_virt {
                let mut residual = 0.0;

                // Fock matrix contribution
                let f_ia = self.orbital_energies[self.num_occ + a];
                residual += f_ia;

                // Add contributions from T2 (simplified)
                for j in 0..self.num_occ {
                    for b in 0..self.num_virt {
                        let ia_jb =
                            self.mo_two_electron_integral(i, self.num_occ + a, j, self.num_occ + b);
                        residual += ia_jb * self.t1[(j, b)];
                    }
                }

                // Energy denominator
                let denominator =
                    self.orbital_energies[i] - self.orbital_energies[self.num_occ + a];

                if denominator.abs() > 1e-10 {
                    new_t1[(i, a)] = residual / denominator;
                } else {
                    new_t1[(i, a)] = self.t1[(i, a)];
                }
            }
        }

        new_t1
    }

    /// Update T2 amplitudes
    ///
    /// Simplified T2 equation:
    /// t_ij^ab(new) = [<ij||ab> + intermediates] / D_ij^ab
    ///
    /// where D_ij^ab = ε_i + ε_j - ε_a - ε_b
    fn update_t2(&mut self) -> Vec<f64> {
        let t2_size = self.num_occ * self.num_occ * self.num_virt * self.num_virt;
        let mut new_t2 = vec![0.0; t2_size];

        // Parallel update of T2 amplitudes
        let chunks: Vec<_> = (0..self.num_occ)
            .flat_map(|i| (0..self.num_occ).map(move |j| (i, j)))
            .collect();

        let results: Vec<_> = chunks
            .par_iter()
            .map(|&(i, j)| {
                let mut local_results = Vec::new();

                for a in 0..self.num_virt {
                    for b in 0..self.num_virt {
                        // Get antisymmetrized integral
                        let ia_jb =
                            self.mo_two_electron_integral(i, self.num_occ + a, j, self.num_occ + b);
                        let ib_ja =
                            self.mo_two_electron_integral(i, self.num_occ + b, j, self.num_occ + a);

                        let mut residual = ia_jb - ib_ja;

                        // Add T1 contributions (simplified)
                        residual += ia_jb * (self.t1[(i, a)] + self.t1[(j, b)]);

                        // Energy denominator
                        let denominator = self.orbital_energies[i] + self.orbital_energies[j]
                            - self.orbital_energies[self.num_occ + a]
                            - self.orbital_energies[self.num_occ + b];

                        let new_amplitude = if denominator.abs() > 1e-10 {
                            residual / denominator
                        } else {
                            self.get_t2(i, j, a, b)
                        };

                        local_results.push((i, j, a, b, new_amplitude));
                    }
                }

                local_results
            })
            .collect();

        // Collect results
        for result_set in results {
            for (i, j, a, b, value) in result_set {
                let idx = i * (self.num_occ * self.num_virt * self.num_virt)
                    + j * (self.num_virt * self.num_virt)
                    + a * self.num_virt
                    + b;
                new_t2[idx] = value;
            }
        }

        new_t2
    }

    /// Calculate RMS change in amplitudes
    fn calculate_rms_change(&self, new_t1: &DMatrix<f64>, new_t2: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        let mut count = 0;

        // T1 contribution
        for i in 0..self.num_occ {
            for a in 0..self.num_virt {
                let diff = new_t1[(i, a)] - self.t1[(i, a)];
                sum_sq += diff * diff;
                count += 1;
            }
        }

        // T2 contribution
        for i in 0..new_t2.len() {
            let diff = new_t2[i] - self.t2[i];
            sum_sq += diff * diff;
            count += 1;
        }

        (sum_sq / count as f64).sqrt()
    }

    /// Solve CCSD equations iteratively
    ///
    /// This method iteratively updates T1 and T2 amplitudes until convergence.
    ///
    /// # Returns
    ///
    /// The CCSD correlation energy in atomic units
    pub fn solve(&mut self) -> f64 {
        info!("===========================================");
        info!("     Starting CCSD Iterations");
        info!("===========================================");

        // Initialize T2 with MP2 guess
        self.initialize_t2_from_mp2();

        // Initial energy
        let mut old_energy = self.calculate_energy();
        info!("Initial energy (MP2-like): {:.12} Eh", old_energy);
        info!("");

        info!(
            "{:>5} {:>18} {:>18} {:>15}",
            "Iter", "E_CCSD", "ΔE", "RMS(T)"
        );
        info!("{}", "-".repeat(60));

        for iteration in 0..self.max_iterations {
            // Update amplitudes
            let new_t1 = self.update_t1();
            let new_t2 = self.update_t2();

            // Calculate RMS change
            let rms = self.calculate_rms_change(&new_t1, &new_t2);

            // Update amplitudes
            self.t1 = new_t1;
            self.t2 = new_t2;

            // Calculate new energy
            let new_energy = self.calculate_energy();
            let delta_e = new_energy - old_energy;

            info!(
                "{:5} {:18.12} {:18.12} {:15.10}",
                iteration + 1,
                new_energy,
                delta_e,
                rms
            );

            // Check convergence
            if rms < self.convergence_threshold && delta_e.abs() < self.convergence_threshold {
                info!("");
                info!("===========================================");
                info!("       CCSD Converged!");
                info!("===========================================");
                info!("Final CCSD correlation energy: {:.12} Eh", new_energy);
                info!("Number of iterations: {}", iteration + 1);
                info!("Final RMS change: {:.10}", rms);
                info!("Final energy change: {:.12} Eh", delta_e);
                info!("===========================================");

                self.correlation_energy = Some(new_energy);
                return new_energy;
            }

            old_energy = new_energy;
        }

        info!("");
        info!("===========================================");
        info!("       CCSD NOT Converged");
        info!("===========================================");
        info!("Maximum iterations ({}) reached", self.max_iterations);
        info!("Final CCSD correlation energy: {:.12} Eh", old_energy);
        info!("Consider increasing max_iterations or adjusting convergence_threshold");
        info!("===========================================");

        self.correlation_energy = Some(old_energy);
        old_energy
    }

    /// Get the correlation energy
    pub fn get_correlation_energy(&self) -> Option<f64> {
        self.correlation_energy
    }

    /// Print a summary of the CCSD calculation
    pub fn print_summary(&self, hf_energy: f64) {
        info!("===========================================");
        info!("        CCSD Results Summary");
        info!("===========================================");
        info!("Hartree-Fock energy:       {:.12} Eh", hf_energy);
        if let Some(corr_e) = self.correlation_energy {
            info!("CCSD correlation energy:   {:.12} Eh", corr_e);
            info!("Total CCSD energy:         {:.12} Eh", hf_energy + corr_e);
            info!("===========================================");
        } else {
            info!("CCSD correlation energy not yet calculated.");
            info!("===========================================");
        }
    }

    /// Get T1 diagnostics (measure of multireference character)
    ///
    /// T1 diagnostic = ||T1|| / √(2 * N_occ)
    ///
    /// Values > 0.02 for closed-shell systems may indicate significant multireference character
    pub fn t1_diagnostic(&self) -> f64 {
        let t1_norm: f64 = self.t1.iter().map(|x| x * x).sum::<f64>().sqrt();
        t1_norm / (2.0 * self.num_occ as f64).sqrt()
    }
}
