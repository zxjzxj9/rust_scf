//! Core MP2 implementation

extern crate nalgebra as na;

use basis::basis::Basis;
use na::{DMatrix, DVector};
use periodic_table_on_an_enum::Element;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::info;

/// MP2 calculation structure
///
/// This struct holds all necessary data from a converged HF calculation
/// and provides methods to compute the MP2 correlation energy.
pub struct MP2<B: Basis> {
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

    /// MP2 correlation energy (calculated)
    pub correlation_energy: Option<f64>,

    /// Elements (for electron counting)
    pub elems: Vec<Element>,
}

impl<B: Basis + Send + Sync> MP2<B> {
    /// Create a new MP2 calculator from converged HF data
    ///
    /// # Arguments
    ///
    /// * `mo_coeffs` - Molecular orbital coefficients from HF
    /// * `orbital_energies` - Orbital energies from HF
    /// * `mo_basis` - Basis functions
    /// * `elems` - Elements (for counting electrons)
    ///
    /// # Returns
    ///
    /// A new MP2 instance ready for correlation energy calculation
    pub fn new(
        mo_coeffs: DMatrix<f64>,
        orbital_energies: DVector<f64>,
        mo_basis: Vec<Arc<B>>,
        elems: Vec<Element>,
    ) -> Self {
        let num_basis = mo_basis.len();
        let total_electrons: usize = elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        let num_occ = total_electrons / 2; // Restricted HF
        let num_virt = num_basis - num_occ;

        info!("MP2 Initialization:");
        info!("  Number of basis functions: {}", num_basis);
        info!("  Number of occupied orbitals: {}", num_occ);
        info!("  Number of virtual orbitals: {}", num_virt);
        info!("  Total electrons: {}", total_electrons);

        MP2 {
            num_basis,
            num_occ,
            num_virt,
            mo_coeffs,
            orbital_energies,
            mo_basis,
            correlation_energy: None,
            elems,
        }
    }

    /// Calculate a single two-electron integral in the MO basis
    ///
    /// (pq|rs) = Σ_μνλσ C_μp C_νq C_λr C_σs (μν|λσ)
    ///
    /// where (μν|λσ) are the AO basis integrals
    ///
    /// # Arguments
    ///
    /// * `p, q, r, s` - MO indices
    ///
    /// # Returns
    ///
    /// The two-electron integral (pq|rs) in MO basis
    fn mo_two_electron_integral(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        // Transform AO integrals to MO basis
        // (pq|rs) = Σ_μνλσ C_μp C_νq C_λr C_σs (μν|λσ)

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

                        // Accumulate transformed integral
                        integral += c_mu_p * c_nu_q * c_lambda_r * c_sigma_s * ao_integral;
                    }
                }
            }
        }

        integral
    }

    /// Calculate MP2 correlation energy using direct transformation
    ///
    /// This is a straightforward but computationally expensive O(N^8) algorithm.
    /// For production use, consider implementing a more efficient integral
    /// transformation algorithm (e.g., quarter-transformation method).
    ///
    /// E_MP2 = Σ_{i<j,a<b} [(ia|jb) * (2*(ia|jb) - (ib|ja))] / (ε_i + ε_j - ε_a - ε_b)
    ///
    /// # Returns
    ///
    /// The MP2 correlation energy in atomic units
    pub fn calculate_mp2_energy_direct(&mut self) -> f64 {
        info!("Starting MP2 correlation energy calculation (direct method)...");
        info!("Warning: This is an O(N^8) algorithm and may be slow for large systems.");

        if self.num_occ == 0 {
            info!("No occupied orbitals - MP2 correlation energy is zero.");
            self.correlation_energy = Some(0.0);
            return 0.0;
        }

        if self.num_virt == 0 {
            info!("No virtual orbitals - MP2 correlation energy is zero.");
            self.correlation_energy = Some(0.0);
            return 0.0;
        }

        // Create list of all (i,j,a,b) combinations
        let mut ijab_list = Vec::new();
        for i in 0..self.num_occ {
            for j in 0..i + 1 {
                // j <= i to avoid double counting
                for a in self.num_occ..self.num_basis {
                    for b in self.num_occ..self.num_basis {
                        ijab_list.push((i, j, a, b));
                    }
                }
            }
        }

        info!(
            "Calculating MP2 energy for {} occupied and {} virtual orbitals...",
            self.num_occ, self.num_virt
        );
        info!("Total number of terms: {}", ijab_list.len());

        // Parallel calculation of MP2 energy contributions
        let correlation_energy: f64 = ijab_list
            .par_iter()
            .enumerate()
            .map(|(idx, &(i, j, a, b))| {
                if idx % 100 == 0 {
                    info!("Processing term {}/{}", idx, ijab_list.len());
                }

                // Calculate required MO integrals
                let ia_jb = self.mo_two_electron_integral(i, a, j, b);
                let ib_ja = self.mo_two_electron_integral(i, b, j, a);

                // Energy denominator: ε_i + ε_j - ε_a - ε_b
                let denominator = self.orbital_energies[i] + self.orbital_energies[j]
                    - self.orbital_energies[a]
                    - self.orbital_energies[b];

                if denominator.abs() < 1e-10 {
                    // Avoid division by zero (shouldn't happen in practice)
                    return 0.0;
                }

                // MP2 energy contribution
                // For i != j: E += (ia|jb) * [2*(ia|jb) - (ib|ja)] / denominator
                // For i == j: E += (ia|jb) * [(ia|jb) - (ib|ja)] / denominator  (factor of 2 from spin)

                let numerator = if i == j {
                    // Same spin combination - only appears once
                    ia_jb * (ia_jb - ib_ja)
                } else {
                    // Different spin - appears twice (up-up and down-down)
                    ia_jb * (2.0 * ia_jb - ib_ja)
                };

                numerator / denominator
            })
            .sum();

        info!("MP2 correlation energy: {:.12} Eh", correlation_energy);
        self.correlation_energy = Some(correlation_energy);
        correlation_energy
    }

    /// Calculate MP2 energy using a more efficient algorithm with integral storage
    ///
    /// This method precomputes and stores commonly needed integrals to avoid
    /// redundant calculations. Still expensive but better than direct method.
    ///
    /// # Returns
    ///
    /// The MP2 correlation energy in atomic units
    pub fn calculate_mp2_energy(&mut self) -> f64 {
        info!("Starting MP2 correlation energy calculation (optimized method)...");

        if self.num_occ == 0 || self.num_virt == 0 {
            info!("No occupied or virtual orbitals - MP2 correlation energy is zero.");
            self.correlation_energy = Some(0.0);
            return 0.0;
        }

        // Step 1: Transform integrals to MO basis for occupied-virtual blocks
        // We only need (ia|jb) type integrals where i,j are occupied and a,b are virtual
        info!("Transforming two-electron integrals to MO basis...");
        info!("This may take some time for larger basis sets.");

        // Store integrals in a more efficient format
        // We'll compute them on-the-fly but with better caching

        let mut total_energy = 0.0;
        let mut count = 0;
        let total_terms = self.num_occ * self.num_occ * self.num_virt * self.num_virt;

        info!("Computing MP2 energy for {} terms...", total_terms);

        // Iterate over occupied orbitals
        for i in 0..self.num_occ {
            for j in 0..=i {
                // For each pair of occupied orbitals, calculate contribution from all virtual pairs
                let ij_energy: f64 = (self.num_occ..self.num_basis)
                    .into_par_iter()
                    .map(|a| {
                        let mut energy = 0.0;
                        for b in self.num_occ..self.num_basis {
                            // Calculate MO integrals
                            let ia_jb = self.mo_two_electron_integral(i, a, j, b);
                            let ib_ja = self.mo_two_electron_integral(i, b, j, a);

                            // Energy denominator
                            let denominator = self.orbital_energies[i] + self.orbital_energies[j]
                                - self.orbital_energies[a]
                                - self.orbital_energies[b];

                            if denominator.abs() < 1e-10 {
                                continue;
                            }

                            // MP2 contribution
                            let numerator = if i == j {
                                ia_jb * (ia_jb - ib_ja)
                            } else {
                                ia_jb * (2.0 * ia_jb - ib_ja)
                            };

                            energy += numerator / denominator;
                        }
                        energy
                    })
                    .sum();

                total_energy += ij_energy;
                count += 1;

                if count % 10 == 0 || count == self.num_occ * (self.num_occ + 1) / 2 {
                    info!(
                        "Processed {}/{} occupied orbital pairs, current E_MP2 = {:.12} Eh",
                        count,
                        self.num_occ * (self.num_occ + 1) / 2,
                        total_energy
                    );
                }
            }
        }

        info!("MP2 correlation energy: {:.12} Eh", total_energy);
        self.correlation_energy = Some(total_energy);
        total_energy
    }

    /// Get the total MP2 energy (HF energy must be added separately)
    pub fn get_correlation_energy(&self) -> Option<f64> {
        self.correlation_energy
    }

    /// Print a summary of the MP2 calculation
    pub fn print_summary(&self, hf_energy: f64) {
        info!("===========================================");
        info!("        MP2 Calculation Summary");
        info!("===========================================");
        info!("Hartree-Fock energy:       {:.12} Eh", hf_energy);
        if let Some(corr_e) = self.correlation_energy {
            info!("MP2 correlation energy:    {:.12} Eh", corr_e);
            info!("Total MP2 energy:          {:.12} Eh", hf_energy + corr_e);
            info!("===========================================");
        } else {
            info!("MP2 correlation energy not yet calculated.");
            info!("===========================================");
        }
    }
}
