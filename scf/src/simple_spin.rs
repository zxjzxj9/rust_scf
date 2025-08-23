// simple_spin.rs - SCF implementation with spin polarization support

extern crate nalgebra as na;

use crate::scf::{SCF, DIIS};
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, Vector3};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::info;

#[derive(Clone)]
pub struct SpinSCF<B: AOBasis> {
    pub(crate) num_atoms: usize,
    pub(crate) num_basis: usize,
    pub(crate) ao_basis: Vec<Arc<Mutex<B>>>,
    mo_basis: Vec<Arc<B::BasisType>>,
    pub(crate) coords: Vec<Vector3<f64>>,
    pub(crate) elems: Vec<Element>,
    // Separate coefficients for alpha and beta
    pub(crate) coeffs_alpha: DMatrix<f64>,
    pub(crate) coeffs_beta: DMatrix<f64>,
    pub(crate) integral_matrix: DMatrix<f64>,
    pub density_mixing: f64,
    // Separate density matrices for alpha and beta
    density_matrix_alpha: DMatrix<f64>,
    density_matrix_beta: DMatrix<f64>,
    // Separate Fock matrices for alpha and beta
    pub(crate) fock_matrix_alpha: DMatrix<f64>,
    pub(crate) fock_matrix_beta: DMatrix<f64>,
    pub(crate) overlap_matrix: DMatrix<f64>,
    // Core Hamiltonian (kinetic + nuclear attraction)
    pub h_core: DMatrix<f64>,
    // Separate energy levels for alpha and beta
    pub e_level_alpha: DVector<f64>,
    pub e_level_beta: DVector<f64>,
    pub max_cycle: usize,
    // Spin multiplicity (2S+1)
    pub multiplicity: usize,
    // Molecular charge
    pub charge: i32,
}

/// Helper function to align eigenvectors
fn align_eigenvectors(mut eigvecs: DMatrix<f64>) -> DMatrix<f64> {
    for j in 0..eigvecs.ncols() {
        let col = eigvecs.column(j);
        let (_, &max_val) = col
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        if max_val < 0.0 {
            for i in 0..eigvecs.nrows() {
                eigvecs[(i, j)] = -eigvecs[(i, j)];
            }
        }
    }
    eigvecs
}

impl<B: AOBasis + Clone> SpinSCF<B> {
    pub fn new() -> SpinSCF<B> {
        SpinSCF {
            num_atoms: 0,
            num_basis: 0,
            ao_basis: Vec::new(),
            mo_basis: Vec::new(),
            coords: Vec::new(),
            elems: Vec::new(),
            coeffs_alpha: DMatrix::zeros(0, 0),
            coeffs_beta: DMatrix::zeros(0, 0),
            density_matrix_alpha: DMatrix::zeros(0, 0),
            density_matrix_beta: DMatrix::zeros(0, 0),
            integral_matrix: DMatrix::zeros(0, 0),
            density_mixing: 0.2,
            fock_matrix_alpha: DMatrix::zeros(0, 0),
            fock_matrix_beta: DMatrix::zeros(0, 0),
            overlap_matrix: DMatrix::zeros(0, 0),
            h_core: DMatrix::zeros(0, 0),
            e_level_alpha: DVector::zeros(0),
            e_level_beta: DVector::zeros(0),
            max_cycle: 1000,
            multiplicity: 1, // Default to singlet (no unpaired electrons)
            charge: 0, // Default to neutral molecule
        }
    }

    pub fn set_charge(&mut self, charge: i32) {
        self.charge = charge;
        info!("Molecular charge set to {}", self.charge);
    }

    pub fn set_multiplicity(&mut self, multiplicity: usize) {
        if multiplicity < 1 {
            panic!("Multiplicity must be at least 1");
        }
        self.multiplicity = multiplicity;
        info!("Spin multiplicity set to {}", self.multiplicity);
    }

    /// Get reference to alpha density matrix for validation
    pub fn get_density_matrix_alpha(&self) -> &DMatrix<f64> {
        &self.density_matrix_alpha
    }

    /// Get reference to beta density matrix for validation
    pub fn get_density_matrix_beta(&self) -> &DMatrix<f64> {
        &self.density_matrix_beta
    }

    /// Get reference to alpha molecular orbital coefficients
    pub fn get_coeffs_alpha(&self) -> &DMatrix<f64> {
        &self.coeffs_alpha
    }

    /// Get reference to beta molecular orbital coefficients
    pub fn get_coeffs_beta(&self) -> &DMatrix<f64> {
        &self.coeffs_beta
    }

    /// Get reference to alpha energy levels
    pub fn get_e_level_alpha(&self) -> &DVector<f64> {
        &self.e_level_alpha
    }

    /// Get reference to beta energy levels
    pub fn get_e_level_beta(&self) -> &DVector<f64> {
        &self.e_level_beta
    }

    /// Validates that the multiplicity is consistent with the number of electrons
    pub fn validate_multiplicity(&self) -> Result<(), String> {
        if self.elems.is_empty() {
            return Ok(()); // Cannot validate without atoms
        }

        let nuclear_charge: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        
        // Account for molecular charge
        let total_electrons = (nuclear_charge as i32 - self.charge) as usize;

        let unpaired_electrons = self.multiplicity - 1;
        
        // Check if we have enough electrons for the requested unpaired electrons
        if total_electrons < unpaired_electrons {
            return Err(format!(
                "Invalid multiplicity {}: only {} electrons available, but {} unpaired electrons requested",
                self.multiplicity, total_electrons, unpaired_electrons
            ));
        }
        
        // Check if the remaining electrons can be paired
        let remaining_electrons = total_electrons - unpaired_electrons;
        if remaining_electrons % 2 != 0 {
            return Err(format!(
                "Invalid multiplicity {}: after {} unpaired electrons, {} electrons remain which cannot be paired",
                self.multiplicity, unpaired_electrons, remaining_electrons
            ));
        }

        // Additional checks for common physical constraints
        if unpaired_electrons > total_electrons {
            return Err(format!(
                "Invalid multiplicity {}: cannot have more unpaired electrons ({}) than total electrons ({})",
                self.multiplicity, unpaired_electrons, total_electrons
            ));
        }

        Ok(())
    }
}

impl<B: AOBasis + Clone> SCF for SpinSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        self.elems = elems.clone();
        self.num_atoms = elems.len();
        info!("\n#####################################################");
        info!("------------- Initializing Basis Set -------------");
        info!("#####################################################");

        self.num_basis = 0;
        for elem in elems {
            let b: &Self::BasisType = *basis.get(elem.get_symbol()).unwrap();
            info!(
                "  Element: {}, Basis Size: {}",
                elem.get_symbol(),
                b.basis_size()
            );

            let b_arc = Arc::new(Mutex::new((*b).clone()));
            self.ao_basis.push(b_arc.clone());
        }

        info!("  Number of Atoms: {}", self.num_atoms);
        info!("-----------------------------------------------------\n");
    }

    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        info!("#####################################################");
        info!("--------------- Initializing Geometry ---------------");
        info!("#####################################################");
        assert!(coords.len() == elems.len());
        let size = coords.len();
        for i in 0..size {
            info!(
                "  Element: {}, Coordinates: {:?}",
                elems[i].get_symbol(),
                coords[i]
            );
            self.ao_basis[i].lock().unwrap().set_center(coords[i]);
            info!(
                "    Center set to: {:?}",
                self.ao_basis[i].lock().unwrap().get_center()
            );
        }
        self.coords = coords.clone();

        self.mo_basis.clear();
        self.num_basis = 0;
        for ao in &self.ao_basis {
            let ao_locked = ao.lock().unwrap();
            for tb in ao_locked.get_basis() {
                self.mo_basis.push(tb.clone());
            }
            self.num_basis += ao_locked.basis_size();
        }
        info!(
            "  Rebuilt MO basis with {} basis functions.",
            self.num_basis
        );
        info!("-----------------------------------------------------\n");
    }

    fn init_density_matrix(&mut self) {
        info!("#####################################################");
        info!("------------ Initializing Density Matrix ------------");
        info!("#####################################################");
        info!("  Building Overlap and Fock Matrices...");

        // Initialize matrices with zeros
        self.fock_matrix_alpha = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
        self.fock_matrix_beta = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
        self.overlap_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
        self.h_core = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);

        // Build one-electron matrices
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                // Overlap matrix is the same for both spins
                self.overlap_matrix[(i, j)] =
                    B::BasisType::Sab(&self.mo_basis[i], &self.mo_basis[j]);

                // Build core Hamiltonian (kinetic + nuclear attraction)
                let t_ij = B::BasisType::Tab(&self.mo_basis[i], &self.mo_basis[j]);
                let mut v_ij = 0.0;
                for k in 0..self.num_atoms {
                    v_ij += B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[k],
                        self.elems[k].get_atomic_number() as u32,
                    );
                }
                let h_core_ij = t_ij + v_ij;
                
                // Store core Hamiltonian
                self.h_core[(i, j)] = h_core_ij;
                
                // Add symmetry breaking to help convergence for both open and closed shell
                if self.multiplicity > 1 {
                    // Add larger perturbation for open-shell systems
                    let perturbation = 0.1 * (((i + j + 1) as f64).sin());
                    self.fock_matrix_alpha[(i, j)] = h_core_ij + perturbation;
                    self.fock_matrix_beta[(i, j)] = h_core_ij - perturbation;
                } else {
                    // For closed-shell systems, start with identical matrices
                    self.fock_matrix_alpha[(i, j)] = h_core_ij;
                    self.fock_matrix_beta[(i, j)] = h_core_ij;
                }
            }
        }

        info!("  Diagonalizing Fock matrices to get initial coefficients...");

        // Calculate number of alpha and beta electrons based on multiplicity
        let nuclear_charge: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        
        // Account for molecular charge
        let total_electrons = (nuclear_charge as i32 - self.charge) as usize;

        // Validate multiplicity before proceeding
        if let Err(err_msg) = self.validate_multiplicity() {
            panic!("Multiplicity validation failed: {}", err_msg);
        }

        let unpaired_electrons = self.multiplicity - 1;

        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        info!(
            "  Total electrons: {}, Alpha: {}, Beta: {}, Basis size: {}",
            total_electrons, n_alpha, n_beta, self.num_basis
        );
        
        if self.multiplicity > 1 {
            info!("  Symmetry breaking applied for open-shell system (multiplicity = {})", self.multiplicity);
        } else {
            info!("  Closed-shell system detected (multiplicity = 1)");
        }

        // Diagonalize initial Fock matrices with numerical stability checks
        info!("  Checking overlap matrix conditioning...");
        let overlap_det = self.overlap_matrix.determinant();
        let overlap_condition = self.calculate_condition_number(&self.overlap_matrix);
        
        info!("    Overlap determinant: {:.2e}", overlap_det);
        info!("    Overlap condition number: {:.2e}", overlap_condition);
        
        if overlap_det.abs() < 1e-12 {
            panic!("Overlap matrix is nearly singular (det = {:.2e}). Check basis set linear dependence.", overlap_det);
        }
        
        if overlap_condition > 1e12 {
            info!("    ⚠️  Warning: Poor overlap matrix conditioning ({:.2e})", overlap_condition);
        }

        let l = match self.overlap_matrix.clone().cholesky() {
            Some(chol) => chol,
            None => {
                panic!("Cholesky decomposition failed. Overlap matrix is not positive definite.");
            }
        };
        let l_inv = l.inverse();

        // For alpha electrons
        let f_alpha_prime =
            l_inv.clone() * self.fock_matrix_alpha.clone() * l_inv.clone().transpose();
        let eig_alpha = f_alpha_prime
            .clone()
            .try_symmetric_eigen(1e-6, 1000)
            .unwrap();

        // Sort alpha eigenvalues and eigenvectors
        let eigenvalues_alpha = eig_alpha.eigenvalues.clone();
        let eigenvectors_alpha = eig_alpha.eigenvectors.clone();
        let mut indices_alpha: Vec<usize> = (0..eigenvalues_alpha.len()).collect();
        indices_alpha.sort_by(|&a, &b| {
            eigenvalues_alpha[a]
                .partial_cmp(&eigenvalues_alpha[b])
                .unwrap()
        });

        let sorted_eigenvalues_alpha = DVector::from_fn(eigenvalues_alpha.len(), |i, _| {
            eigenvalues_alpha[indices_alpha[i]]
        });
        let sorted_eigenvectors_alpha = eigenvectors_alpha.select_columns(&indices_alpha);
        let mut eigvecs_alpha = l_inv.clone().transpose() * sorted_eigenvectors_alpha;
        
        // Normalize the molecular orbitals to ensure orthonormality
        for j in 0..eigvecs_alpha.ncols() {
            let mut col = eigvecs_alpha.column_mut(j);
            let norm_squared = (&col).transpose() * &self.overlap_matrix * &col;
            let norm = norm_squared[(0, 0)].sqrt();
            if norm > 1e-12 {
                col /= norm;
            }
        }
        
        self.coeffs_alpha = eigvecs_alpha;
        self.e_level_alpha = sorted_eigenvalues_alpha;

        // For beta electrons - always use UHF (independent alpha and beta orbitals)
        let f_beta_prime =
            l_inv.clone() * self.fock_matrix_beta.clone() * l_inv.clone().transpose();
        let eig_beta = f_beta_prime
            .clone()
            .try_symmetric_eigen(1e-6, 1000)
            .unwrap();

        // Sort beta eigenvalues and eigenvectors
        let eigenvalues_beta = eig_beta.eigenvalues.clone();
        let eigenvectors_beta = eig_beta.eigenvectors.clone();
        let mut indices_beta: Vec<usize> = (0..eigenvalues_beta.len()).collect();
        indices_beta.sort_by(|&a, &b| {
            eigenvalues_beta[a]
                .partial_cmp(&eigenvalues_beta[b])
                .unwrap()
        });

        let sorted_eigenvalues_beta = DVector::from_fn(eigenvalues_beta.len(), |i, _| {
            eigenvalues_beta[indices_beta[i]]
        });
        let sorted_eigenvectors_beta = eigenvectors_beta.select_columns(&indices_beta);
        let mut eigvecs_beta = l_inv.clone().transpose() * sorted_eigenvectors_beta;
        
        // Normalize the molecular orbitals to ensure orthonormality
        for j in 0..eigvecs_beta.ncols() {
            let mut col = eigvecs_beta.column_mut(j);
            let norm_squared = (&col).transpose() * &self.overlap_matrix * &col;
            let norm = norm_squared[(0, 0)].sqrt();
            if norm > 1e-12 {
                col /= norm;
            }
        }
        
        self.coeffs_beta = eigvecs_beta;
        self.e_level_beta = sorted_eigenvalues_beta;

        info!("  Initial Energy Levels:");
        info!("    Alpha electrons:");
        for i in 0..self.e_level_alpha.len() {
            info!("      Level {}: {:.8} au", i + 1, self.e_level_alpha[i]);
        }
        info!("    Beta electrons:");
        for i in 0..self.e_level_beta.len() {
            info!("      Level {}: {:.8} au", i + 1, self.e_level_beta[i]);
        }

        self.update_density_matrix();
        info!("  Initial Density Matrices built.");
        info!("-----------------------------------------------------\n");
    }

    fn update_density_matrix(&mut self) {
        info!("  Updating Density Matrices...");

        // Calculate number of alpha and beta electrons based on multiplicity
        let nuclear_charge: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        
        // Account for molecular charge
        let total_electrons = (nuclear_charge as i32 - self.charge) as usize;

        let unpaired_electrons = self.multiplicity - 1;
        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        // Update alpha density matrix - standard UHF approach
        // Fill lowest n_alpha orbitals (Aufbau principle with UHF)
        if n_alpha > 0 {
            if n_alpha > self.coeffs_alpha.ncols() {
                panic!("Not enough alpha orbitals ({}) for {} alpha electrons", 
                       self.coeffs_alpha.ncols(), n_alpha);
            }
            let occupied_coeffs_alpha = self.coeffs_alpha.columns(0, n_alpha);
            if self.density_matrix_alpha.shape() == (0, 0) {
                self.density_matrix_alpha = &occupied_coeffs_alpha * occupied_coeffs_alpha.transpose();
            } else {
                let new_density_alpha = &occupied_coeffs_alpha * occupied_coeffs_alpha.transpose();
                self.density_matrix_alpha = self.density_mixing * new_density_alpha
                    + (1.0 - self.density_mixing) * self.density_matrix_alpha.clone();
            }
        } else {
            // No alpha electrons - zero density matrix
            self.density_matrix_alpha = DMatrix::zeros(self.num_basis, self.num_basis);
        }

        // Update beta density matrix
        if n_beta > 0 {
            if n_beta > self.coeffs_beta.ncols() {
                panic!("Not enough beta orbitals ({}) for {} beta electrons", 
                       self.coeffs_beta.ncols(), n_beta);
            }
            let occupied_coeffs_beta = self.coeffs_beta.columns(0, n_beta);
            if self.density_matrix_beta.shape() == (0, 0) {
                self.density_matrix_beta = &occupied_coeffs_beta * occupied_coeffs_beta.transpose();
            } else {
                let new_density_beta = &occupied_coeffs_beta * occupied_coeffs_beta.transpose();
                self.density_matrix_beta = self.density_mixing * new_density_beta
                    + (1.0 - self.density_mixing) * self.density_matrix_beta.clone();
            }
        } else {
            // No beta electrons - zero density matrix
            self.density_matrix_beta = DMatrix::zeros(self.num_basis, self.num_basis);
        }

        info!(
            "  Density Matrices updated with mixing factor {:.2}.",
            self.density_mixing
        );
    }

    fn init_fock_matrix(&mut self) {
        info!("#####################################################");
        info!("------------- Initializing Fock Matrix -------------");
        info!("#####################################################");
        info!("  Building Integral Matrix (Two-electron integrals)...");

        self.integral_matrix = DMatrix::from_element(
            self.num_basis * self.num_basis,
            self.num_basis * self.num_basis,
            0.0,
        );

        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                for k in 0..self.num_basis {
                    for l in 0..self.num_basis {
                        let integral_ijkl = B::BasisType::JKabcd(
                            &self.mo_basis[i],
                            &self.mo_basis[j],
                            &self.mo_basis[k],
                            &self.mo_basis[l],
                        );
                        let row = i * self.num_basis + j;
                        let col = k * self.num_basis + l;
                        self.integral_matrix[(row, col)] = integral_ijkl;
                    }
                }
            }
        }

        info!("  Integral Matrix (Two-electron integrals) built.");
        info!("-----------------------------------------------------\n");
    }

    fn scf_cycle(&mut self) {
        info!("#####################################################");
        info!("--------------- Performing SCF Cycle ---------------");
        info!("#####################################################");

        let mut previous_e_level_alpha = DVector::zeros(self.num_basis);
        let mut previous_e_level_beta = DVector::zeros(self.num_basis);
        let mut previous_density_alpha = DMatrix::zeros(self.num_basis, self.num_basis);
        let mut previous_density_beta = DMatrix::zeros(self.num_basis, self.num_basis);
        let mut previous_total_energy = 0.0;
        let mut diis_alpha = DIIS::new(12); // Increased further for difficult systems
        let mut diis_beta = DIIS::new(12);
        let mut diis_start_cycle = 3; // Start DIIS after a few cycles
        let mut diis_alpha_enabled = true;
        let mut diis_beta_enabled = true;
        let mut diis_reset_counter = 0;
        let mut adaptive_mixing_factor = self.density_mixing;
        let mut level_shift = 0.0;
        let mut consecutive_increases = 0;
        let mut consecutive_decreases = 0;
        const DENSITY_CONVERGENCE_THRESHOLD: f64 = 1e-6;
        const ENERGY_CONVERGENCE_THRESHOLD: f64 = 1e-8;
        const DIIS_RESET_THRESHOLD: f64 = 5.0; // Reset DIIS if energy change is too large
        let mut cycle = 0;

        for _ in 0..self.max_cycle {
            cycle += 1;
            info!(
                "\n------------------ SCF Cycle: {} ------------------",
                cycle
            );

            // Skip flattening - we'll build matrices directly

            // Step 2: Build G matrices
            info!("  Step 2: Building G Matrices from Density Matrices and Integrals...");

            // Build Coulomb and Exchange matrices directly
            let mut j_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
            let mut k_alpha = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
            let mut k_beta = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);

            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    for k in 0..self.num_basis {
                        for l in 0..self.num_basis {
                            let coulomb = B::BasisType::JKabcd(
                                &self.mo_basis[i],
                                &self.mo_basis[j],
                                &self.mo_basis[k],
                                &self.mo_basis[l],
                            );
                            let exchange = B::BasisType::JKabcd(
                                &self.mo_basis[i],
                                &self.mo_basis[k],
                                &self.mo_basis[j],
                                &self.mo_basis[l],
                            );

                            let p_total_kl = self.density_matrix_alpha[(k, l)] + self.density_matrix_beta[(k, l)];
                            let p_alpha_kl = self.density_matrix_alpha[(k, l)];
                            let p_beta_kl = self.density_matrix_beta[(k, l)];

                            // Coulomb contribution (from all electrons)
                            j_matrix[(i, j)] += coulomb * p_total_kl;

                            // Exchange contributions (same-spin only)
                            k_alpha[(i, j)] += exchange * p_alpha_kl;
                            k_beta[(i, j)] += exchange * p_beta_kl;
                        }
                    }
                }
            }

            // Step 3: Build Fock matrices
            info!("  Step 3: Building Fock Matrices...");

            // Core hamiltonian (one-electron terms)
            let mut h_core = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    h_core[(i, j)] = B::BasisType::Tab(&self.mo_basis[i], &self.mo_basis[j]);
                    for k in 0..self.num_atoms {
                        h_core[(i, j)] += B::BasisType::Vab(
                            &self.mo_basis[i],
                            &self.mo_basis[j],
                            self.coords[k],
                            self.elems[k].get_atomic_number() as u32,
                        );
                    }
                }
            }

            // Build Fock matrices: H_core + J - K
            let mut fock_alpha = h_core.clone() + j_matrix.clone() - k_alpha;
            let mut fock_beta = h_core + j_matrix - k_beta;

            // Apply level shifting for numerical stability if needed
            if level_shift > 0.0 {
                info!("  Applying level shifting: {:.3} au", level_shift);
                
                // Add level shift to diagonal elements (virtual orbitals will be shifted up)
                for i in 0..self.num_basis {
                    fock_alpha[(i, i)] += level_shift;
                    fock_beta[(i, i)] += level_shift;
                }
            }

            // Step 4: Apply enhanced DIIS acceleration
            if cycle >= diis_start_cycle {
                info!("  Step 4: Applying enhanced DIIS acceleration...");

                // Check if DIIS should be reset due to poor performance
                if cycle > 1 {
                    let current_total_energy = self.calculate_total_energy();
                    let energy_change = (current_total_energy - previous_total_energy).abs();
                    
                    if energy_change > DIIS_RESET_THRESHOLD {
                        diis_reset_counter += 1;
                        info!("    DIIS reset triggered (energy change: {:.2e})", energy_change);
                        
                        if diis_reset_counter >= 3 {
                            info!("    Resetting DIIS after {} poor cycles", diis_reset_counter);
                            diis_alpha = DIIS::new(12);
                            diis_beta = DIIS::new(12);
                            diis_reset_counter = 0;
                            diis_start_cycle = cycle + 2; // Delay restart
                        }
                    } else {
                        diis_reset_counter = 0; // Reset counter on good convergence
                    }
                }

                // Apply DIIS for alpha if enabled
                if diis_alpha_enabled {
                    diis_alpha.update(
                        fock_alpha.clone(),
                        &self.density_matrix_alpha,
                        &self.overlap_matrix,
                    );
                    if let Some(diis_fock_alpha) = diis_alpha.extrapolate() {
                        fock_alpha = diis_fock_alpha;
                        info!("    DIIS extrapolation applied to alpha");
                    } else {
                        info!("    DIIS extrapolation failed for alpha, using regular Fock");
                    }
                }

                // Apply DIIS for beta if enabled  
                if diis_beta_enabled {
                    diis_beta.update(
                        fock_beta.clone(),
                        &self.density_matrix_beta,
                        &self.overlap_matrix,
                    );
                    if let Some(diis_fock_beta) = diis_beta.extrapolate() {
                        fock_beta = diis_fock_beta;
                        info!("    DIIS extrapolation applied to beta");
                    } else {
                        info!("    DIIS extrapolation failed for beta, using regular Fock");
                    }
                }
            } else {
                info!("  Step 4: DIIS delayed until cycle {}", diis_start_cycle);
            }

            // Step 5: Diagonalize Fock matrices with stability checks
            info!("  Step 5: Diagonalizing Fock matrices...");
            
            // Calculate electron counts for sanity checks
            let nuclear_charge: usize = self.elems.iter().map(|e| e.get_atomic_number() as usize).sum();
            let total_electrons = (nuclear_charge as i32 - self.charge) as usize;
            let unpaired_electrons = self.multiplicity - 1;
            let n_alpha = (total_electrons + unpaired_electrons) / 2;
            let n_beta = (total_electrons - unpaired_electrons) / 2;
            
            // Check for NaN or infinite values in Fock matrices
            if !fock_alpha.iter().all(|x| x.is_finite()) {
                panic!("Alpha Fock matrix contains NaN or infinite values at cycle {}", cycle);
            }
            if !fock_beta.iter().all(|x| x.is_finite()) {
                panic!("Beta Fock matrix contains NaN or infinite values at cycle {}", cycle);
            }

            let l = match self.overlap_matrix.clone().cholesky() {
                Some(chol) => chol,
                None => {
                    panic!("Cholesky decomposition failed at cycle {}. SCF may have diverged.", cycle);
                }
            };
            let l_inv = l.inverse();

            // Diagonalize alpha Fock matrix
            let f_alpha_prime = l_inv.clone() * &fock_alpha * l_inv.transpose();
            let eig_alpha = f_alpha_prime.try_symmetric_eigen(1e-6, 1000).unwrap();

            let eigenvalues_alpha = eig_alpha.eigenvalues.clone();
            let eigenvectors_alpha = eig_alpha.eigenvectors.clone();
            let mut indices_alpha: Vec<usize> = (0..eigenvalues_alpha.len()).collect();
            indices_alpha.sort_by(|&a, &b| {
                eigenvalues_alpha[a]
                    .partial_cmp(&eigenvalues_alpha[b])
                    .unwrap()
            });

            let sorted_eigenvalues_alpha = DVector::from_fn(eigenvalues_alpha.len(), |i, _| {
                eigenvalues_alpha[indices_alpha[i]]
            });
            let sorted_eigenvectors_alpha = eigenvectors_alpha.select_columns(&indices_alpha);
            let mut eigvecs_alpha = l_inv.clone().transpose() * sorted_eigenvectors_alpha;
            
            // Normalize the molecular orbitals to ensure orthonormality
            for j in 0..eigvecs_alpha.ncols() {
                let mut col = eigvecs_alpha.column_mut(j);
                let norm_squared = (&col).transpose() * &self.overlap_matrix * &col;
                let norm = norm_squared[(0, 0)].sqrt();
                if norm > 1e-12 {
                    col /= norm;
                }
            }
            
            self.coeffs_alpha = eigvecs_alpha;
            let current_e_level_alpha = sorted_eigenvalues_alpha;
            
            // Sanity checks for alpha orbital energies
            if !current_e_level_alpha.iter().all(|x| x.is_finite()) {
                panic!("Alpha orbital energies contain NaN or infinite values at cycle {}", cycle);
            }
            
            let alpha_homo_energy = if n_alpha > 0 { current_e_level_alpha[n_alpha-1] } else { 0.0 };
            let alpha_lumo_energy = if n_alpha < current_e_level_alpha.len() { current_e_level_alpha[n_alpha] } else { 0.0 };
            
            if cycle > 5 && alpha_homo_energy > 10.0 {
                info!("    ⚠️  Warning: Alpha HOMO energy suspiciously high: {:.3} au", alpha_homo_energy);
            }

            // Diagonalize beta Fock matrix - always use independent UHF orbitals
            let f_beta_prime = l_inv.clone() * &fock_beta * l_inv.transpose();
            let eig_beta = f_beta_prime.try_symmetric_eigen(1e-6, 1000).unwrap();

            let eigenvalues_beta = eig_beta.eigenvalues.clone();
            let eigenvectors_beta = eig_beta.eigenvectors.clone();
            let mut indices_beta: Vec<usize> = (0..eigenvalues_beta.len()).collect();
            indices_beta.sort_by(|&a, &b| {
                eigenvalues_beta[a]
                    .partial_cmp(&eigenvalues_beta[b])
                    .unwrap()
            });

            let sorted_eigenvalues_beta = DVector::from_fn(eigenvalues_beta.len(), |i, _| {
                eigenvalues_beta[indices_beta[i]]
            });
            let sorted_eigenvectors_beta = eigenvectors_beta.select_columns(&indices_beta);
            let mut eigvecs_beta = l_inv.transpose() * sorted_eigenvectors_beta;
            
            // Normalize the molecular orbitals to ensure orthonormality
            for j in 0..eigvecs_beta.ncols() {
                let mut col = eigvecs_beta.column_mut(j);
                let norm_squared = (&col).transpose() * &self.overlap_matrix * &col;
                let norm = norm_squared[(0, 0)].sqrt();
                if norm > 1e-12 {
                    col /= norm;
                }
            }
            
            self.coeffs_beta = eigvecs_beta;
            let current_e_level_beta = sorted_eigenvalues_beta;
            
            // Sanity checks for beta orbital energies
            if !current_e_level_beta.iter().all(|x| x.is_finite()) {
                panic!("Beta orbital energies contain NaN or infinite values at cycle {}", cycle);
            }
            
            let beta_homo_energy = if n_beta > 0 { current_e_level_beta[n_beta-1] } else { 0.0 };
            let beta_lumo_energy = if n_beta < current_e_level_beta.len() { current_e_level_beta[n_beta] } else { 0.0 };
            
            if cycle > 5 && beta_homo_energy > 10.0 {
                info!("    ⚠️  Warning: Beta HOMO energy suspiciously high: {:.3} au", beta_homo_energy);
            }

            // Update energy levels
            info!("  Step 6: Energy Levels obtained:");
            info!("    Alpha electrons:");
            for i in 0..current_e_level_alpha.len() {
                info!("      Level {}: {:.8} au", i + 1, current_e_level_alpha[i]);
            }
            info!("    Beta electrons:");
            for i in 0..current_e_level_beta.len() {
                info!("      Level {}: {:.8} au", i + 1, current_e_level_beta[i]);
            }

            // Update density matrices with adaptive mixing
            self.update_density_matrix_with_mixing(adaptive_mixing_factor);

            // Check convergence using multiple criteria
            if cycle > 1 {
                info!("  Step 7: Checking for Convergence...");
                
                // Calculate current total energy
                let current_total_energy = self.calculate_total_energy();
                
                // Density matrix RMSD convergence
                let density_change_alpha = (&self.density_matrix_alpha - &previous_density_alpha)
                    .iter().map(|x| x * x).sum::<f64>().sqrt() / (self.num_basis as f64);
                let density_change_beta = (&self.density_matrix_beta - &previous_density_beta)
                    .iter().map(|x| x * x).sum::<f64>().sqrt() / (self.num_basis as f64);
                let max_density_change = density_change_alpha.max(density_change_beta);
                
                // Total energy change
                let energy_change = (current_total_energy - previous_total_energy).abs();
                
                // Legacy energy level change for comparison
                let energy_level_change_alpha = (current_e_level_alpha.clone() - previous_e_level_alpha.clone()).norm();
                let energy_level_change_beta = (current_e_level_beta.clone() - previous_e_level_beta.clone()).norm();

                info!("    Density RMSD (Alpha): {:.2e}", density_change_alpha);
                info!("    Density RMSD (Beta):  {:.2e}", density_change_beta);
                info!("    Max Density RMSD:     {:.2e}", max_density_change);
                info!("    Total Energy Change:   {:.2e} au", energy_change);
                info!("    Total Energy:          {:.8} au", current_total_energy);
                                    info!("    Legacy Energy Levels:  {:.2e} au", energy_level_change_alpha + energy_level_change_beta);

                // Adaptive parameter adjustment based on convergence behavior
                if energy_change > previous_total_energy.abs() * 0.1 {
                    // Energy is increasing or oscillating - be more conservative
                    consecutive_increases += 1;
                    consecutive_decreases = 0;
                    
                    if consecutive_increases > 2 {
                        adaptive_mixing_factor = (adaptive_mixing_factor * 0.7).max(0.1);
                        level_shift += 0.5; // Add level shifting to stabilize
                        info!("    🔧 Adaptive: Reduced mixing to {:.3}, level shift: {:.3}", 
                              adaptive_mixing_factor, level_shift);
                    }
                } else {
                    // Energy is decreasing - can be more aggressive
                    consecutive_decreases += 1;
                    consecutive_increases = 0;
                    
                    if consecutive_decreases > 3 && adaptive_mixing_factor < 0.8 {
                        adaptive_mixing_factor = (adaptive_mixing_factor * 1.2).min(0.8);
                        level_shift = (level_shift - 0.1).max(0.0); // Reduce level shift
                        info!("    🚀 Adaptive: Increased mixing to {:.3}, level shift: {:.3}", 
                              adaptive_mixing_factor, level_shift);
                    }
                }

                // Multi-criteria convergence
                let density_converged = max_density_change < DENSITY_CONVERGENCE_THRESHOLD;
                let energy_converged = energy_change < ENERGY_CONVERGENCE_THRESHOLD;
                
                if density_converged && energy_converged {
                    info!("  ✅ SCF CONVERGED at cycle {} ✅", cycle);
                    info!("    Density RMSD: {:.2e} < {:.2e}", max_density_change, DENSITY_CONVERGENCE_THRESHOLD);
                    info!("    Energy Change: {:.2e} < {:.2e}", energy_change, ENERGY_CONVERGENCE_THRESHOLD);
                    info!("-------------------- SCF Converged ---------------------\n");
                    break;
                } else {
                    info!("    SCF not yet converged:");
                    if !density_converged {
                        info!("      Density RMSD: {:.2e} >= {:.2e}", max_density_change, DENSITY_CONVERGENCE_THRESHOLD);
                    }
                    if !energy_converged {
                        info!("      Energy Change: {:.2e} >= {:.2e}", energy_change, ENERGY_CONVERGENCE_THRESHOLD);
                    }
                }
                
                // Store current values for next iteration
                previous_total_energy = current_total_energy;
            } else {
                info!("  Convergence check not performed for the first cycle.");
                // Initialize for first iteration
                previous_total_energy = self.calculate_total_energy();
            }

            // Store current energy levels and density matrices for the next cycle
            previous_e_level_alpha = current_e_level_alpha.clone();
            previous_e_level_beta = current_e_level_beta.clone();
            previous_density_alpha = self.density_matrix_alpha.clone();
            previous_density_beta = self.density_matrix_beta.clone();
            self.e_level_alpha = current_e_level_alpha.clone();
            self.e_level_beta = current_e_level_beta.clone();

            // Update Fock matrices for next cycle
            self.fock_matrix_alpha = fock_alpha;
            self.fock_matrix_beta = fock_beta;
        }

        if cycle == self.max_cycle {
            info!("\n------------------- SCF Not Converged -------------------");
            info!("  SCF did not converge within {} cycles.", self.max_cycle);
            info!("  Please increase MAX_CYCLE or check system setup.");
            info!("-----------------------------------------------------\n");
        } else {
            // Calculate total energy
            let total_energy = self.calculate_total_energy();
            info!("  Total energy: {:.10} au", total_energy);
            info!("-----------------------------------------------------\n");
        }
    }

    fn calculate_total_energy(&self) -> f64 {
        // Use efficient energy formula similar to SimpleSCF
        // E = Tr(P_alpha * H_core) + Tr(P_beta * H_core) + 
        //     0.5 * [Tr(P_alpha * G_alpha) + Tr(P_beta * G_beta)]
        // where G = F - H_core

        // One-electron energy: Tr(P_total * H_core)
        let mut one_electron_energy = 0.0;
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                let p_total_ij = self.density_matrix_alpha[(i, j)] + self.density_matrix_beta[(i, j)];
                one_electron_energy += self.h_core[(i, j)] * p_total_ij;
            }
        }

        // Two-electron energy: 0.5 * [Tr(P_alpha * G_alpha) + Tr(P_beta * G_beta)]
        let mut two_electron_energy = 0.0;
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                let g_alpha_ij = self.fock_matrix_alpha[(i, j)] - self.h_core[(i, j)];
                let g_beta_ij = self.fock_matrix_beta[(i, j)] - self.h_core[(i, j)];
                
                two_electron_energy += 0.5 * self.density_matrix_alpha[(i, j)] * g_alpha_ij;
                two_electron_energy += 0.5 * self.density_matrix_beta[(i, j)] * g_beta_ij;
            }
        }

        // Nuclear repulsion energy
        let mut nuclear_repulsion = 0.0;
        for i in 0..self.num_atoms {
            for j in (i + 1)..self.num_atoms {
                let z_i = self.elems[i].get_atomic_number() as f64;
                let z_j = self.elems[j].get_atomic_number() as f64;
                let r_ij = (self.coords[i] - self.coords[j]).norm();
                if r_ij > 1e-10 {
                    nuclear_repulsion += z_i * z_j / r_ij;
                }
            }
        }

        let total_energy = one_electron_energy + two_electron_energy + nuclear_repulsion;
        if total_energy.is_finite() {
            total_energy
        } else {
            0.0
        }
    }

    fn calculate_forces(&self) -> Vec<Vector3<f64>> {
        info!("#####################################################");
        info!("----------- Calculating Complete Forces -------------");
        info!("#####################################################");

        let mut forces = vec![Vector3::zeros(); self.num_atoms];

        // Calculate n_alpha and n_beta for W matrix construction
        let nuclear_charge: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        
        // Account for molecular charge
        let total_electrons = (nuclear_charge as i32 - self.charge) as usize;
        let unpaired_electrons = self.multiplicity - 1;
        
        // Validate multiplicity for force calculation
        if let Err(err_msg) = self.validate_multiplicity() {
            panic!("Multiplicity validation failed during force calculation: {}", err_msg);
        }
        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        // Construct Energy-Weighted Density Matrices W_alpha and W_beta
        let c_occ_alpha = self.coeffs_alpha.columns(0, n_alpha);
        let e_occ_alpha_vec = self.e_level_alpha.rows(0, n_alpha);
        let e_occ_alpha_diag = DMatrix::from_diagonal(&e_occ_alpha_vec);
        let w_matrix_alpha = &c_occ_alpha * e_occ_alpha_diag * c_occ_alpha.transpose();

        let c_occ_beta = self.coeffs_beta.columns(0, n_beta);
        let e_occ_beta_vec = self.e_level_beta.rows(0, n_beta);
        let e_occ_beta_diag = DMatrix::from_diagonal(&e_occ_beta_vec);
        let w_matrix_beta = &c_occ_beta * e_occ_beta_diag * c_occ_beta.transpose();

        // Step 1: Calculate nuclear-nuclear repulsion forces
        info!("  Step 1: Nuclear-nuclear repulsion forces...");
        for i in 0..self.num_atoms {
            for j in 0..self.num_atoms {
                if i == j {
                    continue;
                }

                let z_i = self.elems[i].get_atomic_number() as f64;
                let z_j = self.elems[j].get_atomic_number() as f64;
                let r_ij = self.coords[i] - self.coords[j];
                let r_ij_norm = r_ij.norm();

                // Nuclear-nuclear repulsion force
                forces[i] += z_i * z_j * r_ij / (r_ij_norm * r_ij_norm * r_ij_norm);
            }
        }

        // Step 2: Calculate electron-nuclear attraction forces
        info!("  Step 2: Electron-nuclear attraction forces...");
        // For spin-polarized calculation, use total density (alpha + beta)
        for atom_idx in 0..self.num_atoms {
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    // Get combined density matrix elements
                    let p_ij = self.density_matrix_alpha[(i, j)] + self.density_matrix_beta[(i, j)];

                    // Calculate derivative of nuclear attraction integrals
                    let dv_dr = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );

                    // Add contribution to force
                    forces[atom_idx] -= p_ij * dv_dr;
                }
            }
        }

        // Step 3: Calculate forces from two-electron integrals
        info!("  Step 3: Two-electron integral derivatives...");
        for atom_idx in 0..self.num_atoms {
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    for k in 0..self.num_basis {
                        for l in 0..self.num_basis {
                            let p_alpha_ij = self.density_matrix_alpha[(i, j)];
                            let p_beta_ij = self.density_matrix_beta[(i, j)];
                            let p_alpha_kl = self.density_matrix_alpha[(k, l)];
                            let p_beta_kl = self.density_matrix_beta[(k, l)];

                            // Get derivatives of two-electron integrals
                            let coulomb_deriv = B::BasisType::dJKabcd_dR(
                                &self.mo_basis[i],
                                &self.mo_basis[j],
                                &self.mo_basis[k],
                                &self.mo_basis[l],
                                self.coords[atom_idx],
                            );

                            let exchange_deriv = B::BasisType::dJKabcd_dR(
                                &self.mo_basis[i],
                                &self.mo_basis[k],
                                &self.mo_basis[j],
                                &self.mo_basis[l],
                                self.coords[atom_idx],
                            );

                            // Coulomb contribution (both spins interact)
                            forces[atom_idx] -= ((p_alpha_ij + p_beta_ij) * (p_alpha_kl + p_beta_kl)) * coulomb_deriv;

                            // Exchange contribution (same-spin only)
                            forces[atom_idx] += 0.5 * (p_alpha_ij * p_alpha_kl + p_beta_ij * p_beta_kl) * exchange_deriv;
                        }
                    }
                }
            }
        }

        // Step 4: Calculate Pulay forces (derivatives w.r.t. basis function centers)
        info!("  Step 4: Pulay forces (basis function derivatives)...");
        for atom_idx in 0..self.num_atoms {
            // Core Hamiltonian Pulay forces
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    let p_total_ij = self.density_matrix_alpha[(i, j)] + self.density_matrix_beta[(i, j)];

                    // Overlap matrix derivatives (weighted by Energy Weighted Density matrix elements)
                    let ds_dr = B::BasisType::dSab_dR(&self.mo_basis[i], &self.mo_basis[j], atom_idx);
                    let w_total_ij = w_matrix_alpha[(i,j)] + w_matrix_beta[(i,j)];
                    forces[atom_idx] -= w_total_ij * ds_dr;

                    // Kinetic energy derivatives
                    let dt_dr = B::BasisType::dTab_dR(&self.mo_basis[i], &self.mo_basis[j], atom_idx);
                    forces[atom_idx] -= p_total_ij * dt_dr;

                    // Nuclear attraction Pulay forces
                    for k in 0..self.num_atoms {
                        let dv_dr_basis = B::BasisType::dVab_dRbasis(
                            &self.mo_basis[i],
                            &self.mo_basis[j],
                            self.coords[k],
                            self.elems[k].get_atomic_number() as u32,
                            atom_idx,
                        );
                        forces[atom_idx] -= p_total_ij * dv_dr_basis;
                    }
                }
            }
        }

        info!("  Complete forces calculated on atoms:");
        for (i, force) in forces.iter().enumerate() {
            info!(
                "    Atom {}: [{:.6}, {:.6}, {:.6}] au",
                i + 1,
                force.x,
                force.y,
                force.z
            );
        }
        info!("-----------------------------------------------------\n");

        forces
    }

}

impl<B: AOBasis + Clone> SpinSCF<B> {
    /// Update density matrices with custom mixing factor (for adaptive SCF)
    pub fn update_density_matrix_with_mixing(&mut self, mixing_factor: f64) {
        info!("  Updating Density Matrices with adaptive mixing factor {:.3}...", mixing_factor);

        // Calculate number of alpha and beta electrons based on multiplicity
        let nuclear_charge: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        
        // Account for molecular charge
        let total_electrons = (nuclear_charge as i32 - self.charge) as usize;

        let unpaired_electrons = self.multiplicity - 1;
        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        // Update alpha density matrix with adaptive mixing
        if n_alpha > 0 {
            if n_alpha > self.coeffs_alpha.ncols() {
                panic!("Not enough alpha orbitals ({}) for {} alpha electrons", 
                       self.coeffs_alpha.ncols(), n_alpha);
            }
            let occupied_coeffs_alpha = self.coeffs_alpha.columns(0, n_alpha);
            if self.density_matrix_alpha.shape() == (0, 0) {
                self.density_matrix_alpha = &occupied_coeffs_alpha * occupied_coeffs_alpha.transpose();
            } else {
                let new_density_alpha = &occupied_coeffs_alpha * occupied_coeffs_alpha.transpose();
                self.density_matrix_alpha = mixing_factor * new_density_alpha
                    + (1.0 - mixing_factor) * self.density_matrix_alpha.clone();
            }
        } else {
            self.density_matrix_alpha = DMatrix::zeros(self.num_basis, self.num_basis);
        }

        // Update beta density matrix with adaptive mixing
        if n_beta > 0 {
            if n_beta > self.coeffs_beta.ncols() {
                panic!("Not enough beta orbitals ({}) for {} beta electrons", 
                       self.coeffs_beta.ncols(), n_beta);
            }
            let occupied_coeffs_beta = self.coeffs_beta.columns(0, n_beta);
            if self.density_matrix_beta.shape() == (0, 0) {
                self.density_matrix_beta = &occupied_coeffs_beta * occupied_coeffs_beta.transpose();
            } else {
                let new_density_beta = &occupied_coeffs_beta * occupied_coeffs_beta.transpose();
                self.density_matrix_beta = mixing_factor * new_density_beta
                    + (1.0 - mixing_factor) * self.density_matrix_beta.clone();
            }
        } else {
            self.density_matrix_beta = DMatrix::zeros(self.num_basis, self.num_basis);
        }

        info!("  Density Matrices updated with adaptive mixing factor {:.3}.", mixing_factor);
    }

    /// Calculate condition number of a matrix (ratio of largest to smallest eigenvalue)
    pub fn calculate_condition_number(&self, matrix: &DMatrix<f64>) -> f64 {
        match matrix.clone().try_symmetric_eigen(1e-10, 1000) {
            Some(eigen) => {
                let eigenvalues = eigen.eigenvalues;
                let max_eigenvalue = eigenvalues.max();
                let min_eigenvalue = eigenvalues.min();
                
                if min_eigenvalue.abs() < 1e-15 {
                    1e15 // Return very large condition number for near-singular matrix
                } else {
                    (max_eigenvalue / min_eigenvalue).abs()
                }
            }
            None => {
                1e15 // Return very large condition number if eigenvalue decomposition fails
            }
        }
    }
}