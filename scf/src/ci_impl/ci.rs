//! Core CI (Configuration Interaction) implementation
//!
//! This module implements both CIS (singles) and CISD (singles + doubles) methods
//! for computing electron correlation and excited states.

extern crate nalgebra as na;

use basis::basis::Basis;
use na::{DMatrix, DVector};
use periodic_table_on_an_enum::Element;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::info;

/// CI method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIMethod {
    /// Configuration Interaction Singles (excited states)
    CIS,
    /// Configuration Interaction Singles and Doubles (ground + excited states)
    CISD,
}

/// CI calculation structure
///
/// This struct holds all necessary data from a converged HF calculation
/// and provides methods to compute CI energies and excited states.
pub struct CI<B: Basis> {
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

    /// CI correlation energy (calculated)
    pub correlation_energy: Option<f64>,

    /// Excited state energies (for CIS)
    pub excitation_energies: Vec<f64>,

    /// CI coefficients for the ground state
    pub ci_coeffs: Option<DVector<f64>>,

    /// Maximum number of states to compute
    pub max_states: usize,

    /// Convergence threshold for eigenvalue solver
    pub convergence_threshold: f64,

    /// Reference (HF) energy
    pub hf_energy: f64,
}

impl<B: Basis + Send + Sync> CI<B> {
    /// Create a new CI calculator from converged HF data
    ///
    /// # Arguments
    ///
    /// * `mo_coeffs` - Molecular orbital coefficients from HF
    /// * `orbital_energies` - Orbital energies from HF
    /// * `mo_basis` - Basis functions
    /// * `elems` - Elements (for counting electrons)
    /// * `hf_energy` - Reference Hartree-Fock energy
    /// * `max_states` - Maximum number of states to compute
    /// * `convergence_threshold` - Convergence threshold for diagonalization
    ///
    /// # Returns
    ///
    /// A new CI instance ready for correlation/excited state calculations
    pub fn new(
        mo_coeffs: DMatrix<f64>,
        orbital_energies: DVector<f64>,
        mo_basis: Vec<Arc<B>>,
        elems: Vec<Element>,
        hf_energy: f64,
        max_states: usize,
        convergence_threshold: f64,
    ) -> Self {
        let num_basis = mo_basis.len();
        let total_electrons: usize = elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        let num_occ = total_electrons / 2; // Restricted HF
        let num_virt = num_basis - num_occ;

        info!("===========================================");
        info!("        CI Initialization");
        info!("===========================================");
        info!("Number of basis functions: {}", num_basis);
        info!("Number of occupied orbitals: {}", num_occ);
        info!("Number of virtual orbitals: {}", num_virt);
        info!("Total electrons: {}", total_electrons);
        info!("HF reference energy: {:.10} Eh", hf_energy);
        info!("Max states to compute: {}", max_states);
        info!("Convergence threshold: {:.2e}", convergence_threshold);
        info!("===========================================");

        CI {
            num_basis,
            num_occ,
            num_virt,
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            correlation_energy: None,
            excitation_energies: Vec::new(),
            ci_coeffs: None,
            max_states,
            convergence_threshold,
            hf_energy,
        }
    }

    /// Calculate two-electron integral in MO basis: (pq|rs)
    ///
    /// Transforms from AO to MO basis using:
    /// (pq|rs) = Σ_{μνλσ} C_{μp} C_{νq} C_{λr} C_{σs} (μν|λσ)
    fn mo_two_electron_integral(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let num_basis = self.num_basis;
        let mo_coeffs = &self.mo_coeffs;

        let mut integral = 0.0;

        // Transform from AO to MO basis
        for mu in 0..num_basis {
            let c_mu_p = mo_coeffs[(mu, p)];
            if c_mu_p.abs() < 1e-10 {
                continue;
            }

            for nu in 0..num_basis {
                let c_nu_q = mo_coeffs[(nu, q)];
                if c_nu_q.abs() < 1e-10 {
                    continue;
                }

                for lambda in 0..num_basis {
                    let c_lambda_r = mo_coeffs[(lambda, r)];
                    if c_lambda_r.abs() < 1e-10 {
                        continue;
                    }

                    for sigma in 0..num_basis {
                        let c_sigma_s = mo_coeffs[(sigma, s)];
                        if c_sigma_s.abs() < 1e-10 {
                            continue;
                        }

                        // Calculate AO integral (μν|λσ)
                        let ao_integral = self.ao_two_electron_integral(mu, nu, lambda, sigma);

                        // Accumulate transformed integral
                        integral += c_mu_p * c_nu_q * c_lambda_r * c_sigma_s * ao_integral;
                    }
                }
            }
        }

        integral
    }

    /// Calculate two-electron integral in AO basis: (μν|λσ)
    fn ao_two_electron_integral(&self, mu: usize, nu: usize, lambda: usize, sigma: usize) -> f64 {
        let basis_mu = &self.mo_basis[mu];
        let basis_nu = &self.mo_basis[nu];
        let basis_lambda = &self.mo_basis[lambda];
        let basis_sigma = &self.mo_basis[sigma];

        B::JKabcd(basis_mu, basis_nu, basis_lambda, basis_sigma)
    }

    /// Calculate CIS (Configuration Interaction Singles) energies
    ///
    /// CIS constructs excited states from single excitations i → a.
    /// The Hamiltonian matrix elements are:
    /// H_{ia,jb} = δ_ij δ_ab (ε_a - ε_i) + (ia|jb) - (ib|ja)
    ///
    /// # Arguments
    ///
    /// * `num_states` - Number of excited states to compute
    ///
    /// # Returns
    ///
    /// Vector of excitation energies (not total energies)
    pub fn calculate_cis_energies(&mut self, num_states: usize) -> Vec<f64> {
        info!("");
        info!("===========================================");
        info!("    Starting CIS Calculation");
        info!("===========================================");
        info!("Computing {} excited states", num_states);

        if self.num_occ == 0 || self.num_virt == 0 {
            info!("No occupied or virtual orbitals - cannot perform CIS.");
            return Vec::new();
        }

        let n_singles = self.num_occ * self.num_virt;
        info!("Number of single excitations: {}", n_singles);
        info!("CI matrix size: {} x {}", n_singles, n_singles);

        if n_singles == 0 {
            return Vec::new();
        }

        // Build CIS Hamiltonian matrix
        info!("Building CIS Hamiltonian matrix...");
        let h_cis = self.build_cis_hamiltonian();

        // Diagonalize to get excitation energies
        info!("Diagonalizing CIS Hamiltonian...");
        let eigen = h_cis.symmetric_eigen();

        // Extract excitation energies (sorted automatically)
        let excitation_energies: Vec<f64> = eigen
            .eigenvalues
            .iter()
            .take(num_states.min(n_singles))
            .copied()
            .collect();

        // Store results
        self.excitation_energies = excitation_energies.clone();

        info!("");
        info!("===========================================");
        info!("       CIS Results Summary");
        info!("===========================================");
        info!("HF Ground State Energy:  {:.10} Eh", self.hf_energy);
        info!("");
        info!("Excited States:");
        for (i, &exc_energy) in excitation_energies.iter().enumerate() {
            let total_energy = self.hf_energy + exc_energy;
            info!(
                "  State {}: Excitation = {:.6} Eh ({:.2} eV), Total = {:.10} Eh",
                i + 1,
                exc_energy,
                exc_energy * 27.2114,
                total_energy
            );
        }
        info!("===========================================");

        excitation_energies
    }

    /// Build the CIS Hamiltonian matrix
    ///
    /// Matrix element: H_{ia,jb} = δ_ij δ_ab (ε_a - ε_i) + (ia|jb) - (ib|ja)
    fn build_cis_hamiltonian(&self) -> DMatrix<f64> {
        let n_singles = self.num_occ * self.num_virt;
        let mut h_matrix = DMatrix::zeros(n_singles, n_singles);

        // Create index mapping: (i, a) -> index
        let mut singles_list = Vec::new();
        for i in 0..self.num_occ {
            for a_rel in 0..self.num_virt {
                let a = self.num_occ + a_rel;
                singles_list.push((i, a));
            }
        }

        info!("Computing {} CIS matrix elements...", n_singles * n_singles);

        // Build Hamiltonian matrix (parallelized over rows)
        let rows: Vec<Vec<f64>> = (0..n_singles)
            .into_par_iter()
            .map(|idx_ia| {
                let (i, a) = singles_list[idx_ia];
                let mut row = vec![0.0; n_singles];

                for idx_jb in 0..n_singles {
                    let (j, b) = singles_list[idx_jb];

                    let mut h_element = 0.0;

                    // Diagonal part: (ε_a - ε_i) when i=j and a=b
                    if i == j && a == b {
                        h_element += self.orbital_energies[a] - self.orbital_energies[i];
                    }

                    // Two-electron contributions (Coulomb - Exchange)
                    // (ia|jb) - (ib|ja)
                    if i == j || a == b {
                        let coulomb = self.mo_two_electron_integral(i, a, j, b);
                        let exchange = self.mo_two_electron_integral(i, b, j, a);
                        h_element += coulomb - exchange;
                    }

                    row[idx_jb] = h_element;
                }

                row
            })
            .collect();

        // Fill the matrix from computed rows
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                h_matrix[(i, j)] = val;
            }
        }

        info!("CIS Hamiltonian construction complete.");
        h_matrix
    }

    /// Calculate CISD (Configuration Interaction Singles and Doubles) energy
    ///
    /// CISD includes the HF reference, all single excitations i → a,
    /// and all double excitations i,j → a,b.
    ///
    /// The CI wave function is:
    /// |Ψ⟩ = c_0 |HF⟩ + Σ_ia c_i^a |Φ_i^a⟩ + Σ_{i<j,a<b} c_ij^ab |Φ_ij^ab⟩
    ///
    /// # Returns
    ///
    /// CISD correlation energy (difference from HF energy)
    pub fn calculate_cisd_energy(&mut self) -> f64 {
        info!("");
        info!("===========================================");
        info!("      Starting CISD Calculation");
        info!("===========================================");

        if self.num_occ == 0 || self.num_virt == 0 {
            info!("No occupied or virtual orbitals - CISD correlation energy is zero.");
            self.correlation_energy = Some(0.0);
            return 0.0;
        }

        // Calculate configuration space size
        let n_singles = self.num_occ * self.num_virt;
        let n_doubles =
            (self.num_occ * (self.num_occ - 1) / 2) * (self.num_virt * (self.num_virt - 1) / 2);
        let n_config = 1 + n_singles + n_doubles; // Reference + singles + doubles

        info!("Configuration space:");
        info!("  Reference configurations: 1");
        info!("  Single excitations: {}", n_singles);
        info!("  Double excitations: {}", n_doubles);
        info!("  Total configurations: {}", n_config);
        info!("CI matrix size: {} x {}", n_config, n_config);

        if n_config > 10000 {
            info!(
                "Warning: Large CI matrix ({} x {}). This may take significant time and memory.",
                n_config, n_config
            );
        }

        // Build CISD Hamiltonian matrix
        info!("Building CISD Hamiltonian matrix...");
        let h_cisd = self.build_cisd_hamiltonian();

        // Diagonalize to get ground state energy
        info!("Diagonalizing CISD Hamiltonian...");
        let eigen = h_cisd.symmetric_eigen();

        // Lowest eigenvalue is the CISD ground state energy
        let cisd_energy = eigen.eigenvalues[0];
        let correlation_energy = cisd_energy - self.hf_energy;

        // Store the ground state CI coefficients
        self.ci_coeffs = Some(eigen.eigenvectors.column(0).into_owned());
        self.correlation_energy = Some(correlation_energy);

        info!("");
        info!("===========================================");
        info!("       CISD Results Summary");
        info!("===========================================");
        info!("Hartree-Fock energy:      {:.10} Eh", self.hf_energy);
        info!("CISD correlation energy:  {:.10} Eh", correlation_energy);
        info!("Total CISD energy:        {:.10} Eh", cisd_energy);
        info!("===========================================");

        correlation_energy
    }

    /// Build the CISD Hamiltonian matrix
    ///
    /// Constructs the full CI matrix including:
    /// - Reference determinant (HF)
    /// - Single excitations
    /// - Double excitations
    fn build_cisd_hamiltonian(&self) -> DMatrix<f64> {
        let n_singles = self.num_occ * self.num_virt;
        let n_doubles =
            (self.num_occ * (self.num_occ - 1) / 2) * (self.num_virt * (self.num_virt - 1) / 2);
        let n_config = 1 + n_singles + n_doubles;

        let mut h_matrix = DMatrix::zeros(n_config, n_config);

        // Create configuration list
        // Index 0: Reference (HF)
        // Index 1 to n_singles: Singles i → a
        // Index n_singles+1 to end: Doubles i,j → a,b (i<j, a<b)

        let mut singles_list = Vec::new();
        for i in 0..self.num_occ {
            for a_rel in 0..self.num_virt {
                let a = self.num_occ + a_rel;
                singles_list.push((i, a));
            }
        }

        let mut doubles_list = Vec::new();
        for i in 0..self.num_occ {
            for j in (i + 1)..self.num_occ {
                for a_rel in 0..self.num_virt {
                    let a = self.num_occ + a_rel;
                    for b_rel in (a_rel + 1)..self.num_virt {
                        let b = self.num_occ + b_rel;
                        doubles_list.push((i, j, a, b));
                    }
                }
            }
        }

        info!("Computing CISD Hamiltonian matrix elements...");

        // H(0,0): Reference energy (HF energy)
        h_matrix[(0, 0)] = self.hf_energy;

        // H(0, singles) and H(singles, 0): Zero by Brillouin's theorem for canonical HF
        // (These are zero for canonical orbitals)

        // H(singles, singles): Similar to CIS matrix
        info!("Computing singles-singles block...");
        for (idx_i, &(i, a)) in singles_list.iter().enumerate() {
            let row = 1 + idx_i;

            for (idx_j, &(j, b)) in singles_list.iter().enumerate() {
                let col = 1 + idx_j;

                let mut h_element = 0.0;

                // Diagonal part
                if i == j && a == b {
                    h_element +=
                        self.hf_energy + self.orbital_energies[a] - self.orbital_energies[i];
                }

                // Two-electron contributions
                if i == j || a == b {
                    let coulomb = self.mo_two_electron_integral(i, a, j, b);
                    let exchange = self.mo_two_electron_integral(i, b, j, a);
                    h_element += coulomb - exchange;
                }

                h_matrix[(row, col)] = h_element;
            }
        }

        // H(singles, doubles): Coupling between singles and doubles
        info!("Computing singles-doubles block...");
        for (idx_single, &(i, a)) in singles_list.iter().enumerate() {
            let row = 1 + idx_single;

            for (idx_double, &(j, k, b, c)) in doubles_list.iter().enumerate() {
                let col = 1 + n_singles + idx_double;

                let h_element = self.hamiltonian_single_double(i, a, j, k, b, c);
                h_matrix[(row, col)] = h_element;
                h_matrix[(col, row)] = h_element; // Hermitian
            }
        }

        // H(doubles, doubles): Double excitation block
        info!("Computing doubles-doubles block...");
        let double_rows: Vec<(usize, Vec<f64>)> = (0..doubles_list.len())
            .into_par_iter()
            .map(|idx_i| {
                let (i, j, a, b) = doubles_list[idx_i];
                let mut row = vec![0.0; doubles_list.len()];

                for (idx_k, &(k, l, c, d)) in doubles_list.iter().enumerate() {
                    row[idx_k] = self.hamiltonian_double_double(i, j, a, b, k, l, c, d);
                }

                (idx_i, row)
            })
            .collect();

        for (idx_i, row) in double_rows {
            let row_idx = 1 + n_singles + idx_i;
            for (idx_j, &val) in row.iter().enumerate() {
                let col_idx = 1 + n_singles + idx_j;
                h_matrix[(row_idx, col_idx)] = val;
            }
        }

        info!("CISD Hamiltonian construction complete.");
        h_matrix
    }

    /// Hamiltonian matrix element between single and double excitations
    /// ⟨Φ_i^a | H | Φ_jk^bc⟩
    fn hamiltonian_single_double(
        &self,
        i: usize,
        a: usize,
        j: usize,
        k: usize,
        b: usize,
        c: usize,
    ) -> f64 {
        // Slater-Condon rules: singles and doubles differ by 3 or 4 spin-orbitals
        // These matrix elements are typically zero or involve specific integral patterns

        let mut h_element = 0.0;

        // Case 1: i = j, one virtual orbital matches
        if i == j && a == b {
            // ⟨Φ_i^a | H | Φ_ik^ac⟩ = (ia|kc)
            h_element += self.mo_two_electron_integral(i, a, k, c);
        } else if i == j && a == c {
            // ⟨Φ_i^a | H | Φ_ik^bc⟩ = -(ia|kb)
            h_element -= self.mo_two_electron_integral(i, a, k, b);
        }

        // Case 2: i = k
        if i == k && a == b {
            h_element -= self.mo_two_electron_integral(i, a, j, c);
        } else if i == k && a == c {
            h_element += self.mo_two_electron_integral(i, a, j, b);
        }

        h_element
    }

    /// Hamiltonian matrix element between two double excitations
    /// ⟨Φ_ij^ab | H | Φ_kl^cd⟩
    fn hamiltonian_double_double(
        &self,
        i: usize,
        j: usize,
        a: usize,
        b: usize,
        k: usize,
        l: usize,
        c: usize,
        d: usize,
    ) -> f64 {
        // Slater-Condon rules for doubles
        let mut h_element = 0.0;

        // Same determinant: i=k, j=l, a=c, b=d
        if i == k && j == l && a == c && b == d {
            // Diagonal element: HF energy + orbital energy differences + two-electron terms
            h_element += self.hf_energy;
            h_element += self.orbital_energies[a] + self.orbital_energies[b];
            h_element -= self.orbital_energies[i] + self.orbital_energies[j];

            // Add two-electron integrals for double excitation
            h_element += self.mo_two_electron_integral(a, b, a, b);
            h_element -= self.mo_two_electron_integral(a, b, b, a);
        }
        // Differ by one occupied and one virtual orbital
        else if (i == k && j == l && a == c)
            || (i == k && j == l && b == d)
            || (i == k && a == c && b == d)
            || (j == l && a == c && b == d)
        {
            // Two spin-orbitals the same, two different
            // These contribute specific two-electron integrals

            if i == k && j == l && a == c {
                // Only b ≠ d
                h_element += self.mo_two_electron_integral(a, b, a, d);
                h_element -= self.mo_two_electron_integral(a, b, d, a);
            } else if i == k && j == l && b == d {
                // Only a ≠ c
                h_element += self.mo_two_electron_integral(a, b, c, b);
                h_element -= self.mo_two_electron_integral(a, b, b, c);
            } else if i == k && a == c && b == d {
                // Only j ≠ l
                h_element += self.mo_two_electron_integral(i, j, i, l);
                h_element -= self.mo_two_electron_integral(i, j, l, i);
            } else if j == l && a == c && b == d {
                // Only i ≠ k
                h_element += self.mo_two_electron_integral(i, j, k, j);
                h_element -= self.mo_two_electron_integral(i, j, j, k);
            }
        }
        // Differ by two virtual orbitals (same occupied)
        else if i == k && j == l {
            // ⟨Φ_ij^ab | H | Φ_ij^cd⟩ = (ab|cd)
            h_element += self.mo_two_electron_integral(a, b, c, d);
            h_element -= self.mo_two_electron_integral(a, b, d, c);
        }
        // Differ by two occupied orbitals (same virtual)
        else if a == c && b == d {
            // ⟨Φ_ij^ab | H | Φ_kl^ab⟩ = (ij|kl)
            h_element += self.mo_two_electron_integral(i, j, k, l);
            h_element -= self.mo_two_electron_integral(i, j, l, k);
        }

        h_element
    }

    /// Get the CISD correlation energy
    pub fn get_correlation_energy(&self) -> Option<f64> {
        self.correlation_energy
    }

    /// Get the CIS excitation energies
    pub fn get_excitation_energies(&self) -> &[f64] {
        &self.excitation_energies
    }

    /// Print a formatted summary of CI results
    pub fn print_summary(&self, method: CIMethod) {
        info!("");
        info!("===========================================");
        info!("         CI Summary");
        info!("===========================================");
        info!("Method: {:?}", method);
        info!("HF Reference Energy: {:.10} Eh", self.hf_energy);

        match method {
            CIMethod::CIS => {
                if !self.excitation_energies.is_empty() {
                    info!("");
                    info!("Excited States:");
                    for (i, &exc_energy) in self.excitation_energies.iter().enumerate() {
                        let total_energy = self.hf_energy + exc_energy;
                        info!(
                            "  State {}: {:.6} Eh ({:.2} eV), Total: {:.10} Eh",
                            i + 1,
                            exc_energy,
                            exc_energy * 27.2114,
                            total_energy
                        );
                    }
                }
            }
            CIMethod::CISD => {
                if let Some(corr_energy) = self.correlation_energy {
                    info!("CISD Correlation Energy: {:.10} Eh", corr_energy);
                    info!("Total CISD Energy: {:.10} Eh", self.hf_energy + corr_energy);
                }
            }
        }

        info!("===========================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_initialization() {
        // This is a placeholder test
        // Real tests are in tests.rs
        assert!(true);
    }
}
