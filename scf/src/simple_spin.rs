// simple_spin.rs - SCF implementation with spin polarization support

extern crate nalgebra as na;

use crate::scf::{DIIS, SCF};
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, Norm, SymmetricEigen, Vector3};
use nalgebra::{Const, Dyn};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, span, Level};

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
    // Separate energy levels for alpha and beta
    pub e_level_alpha: DVector<f64>,
    pub e_level_beta: DVector<f64>,
    pub max_cycle: usize,
    // Spin multiplicity (2S+1)
    pub multiplicity: usize,
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
            e_level_alpha: DVector::zeros(0),
            e_level_beta: DVector::zeros(0),
            max_cycle: 1000,
            multiplicity: 1, // Default to singlet (no unpaired electrons)
        }
    }

    pub fn set_multiplicity(&mut self, multiplicity: usize) {
        self.multiplicity = multiplicity;
        info!("Spin multiplicity set to {}", self.multiplicity);
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
            let b = *basis.get(elem.get_symbol()).unwrap();
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

        // Build one-electron matrices
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                // Overlap matrix is the same for both spins
                self.overlap_matrix[(i, j)] =
                    B::BasisType::Sab(&self.mo_basis[i], &self.mo_basis[j]);

                // Initial Fock matrices are the same for both spins (before SCF)
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
                self.fock_matrix_alpha[(i, j)] = t_ij + v_ij;
                self.fock_matrix_beta[(i, j)] = t_ij + v_ij;
            }
        }

        info!("  Diagonalizing Fock matrices to get initial coefficients...");

        // Calculate number of alpha and beta electrons based on multiplicity
        let total_electrons: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();

        let unpaired_electrons = self.multiplicity - 1;
        assert!(
            total_electrons >= unpaired_electrons,
            "Invalid multiplicity: not enough electrons"
        );
        assert_eq!((total_electrons - unpaired_electrons) % 2, 0, 
                   "Invalid electron count for given multiplicity");

        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        info!(
            "  Total electrons: {}, Alpha: {}, Beta: {}",
            total_electrons, n_alpha, n_beta
        );

        // Diagonalize initial Fock matrices (same for alpha and beta at this point)
        let l = self.overlap_matrix.clone().cholesky().unwrap();
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
        let eigvecs_alpha = l_inv.clone().transpose() * sorted_eigenvectors_alpha;
        self.coeffs_alpha = eigvecs_alpha;
        self.e_level_alpha = sorted_eigenvalues_alpha;

        // For beta electrons
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
        let eigvecs_beta = l_inv.clone().transpose() * sorted_eigenvectors_beta;
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
        let total_electrons: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();

        let unpaired_electrons = self.multiplicity - 1;
        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        // Update alpha density matrix
        let occupied_coeffs_alpha = self.coeffs_alpha.columns(0, n_alpha);
        if self.density_matrix_alpha.shape() == (0, 0) {
            self.density_matrix_alpha = &occupied_coeffs_alpha * occupied_coeffs_alpha.transpose();
        } else {
            let new_density_alpha = &occupied_coeffs_alpha * occupied_coeffs_alpha.transpose();
            self.density_matrix_alpha = self.density_mixing * new_density_alpha
                + (1.0 - self.density_mixing) * self.density_matrix_alpha.clone();
        }

        // Update beta density matrix
        let occupied_coeffs_beta = self.coeffs_beta.columns(0, n_beta);
        if self.density_matrix_beta.shape() == (0, 0) {
            self.density_matrix_beta = &occupied_coeffs_beta * occupied_coeffs_beta.transpose();
        } else {
            let new_density_beta = &occupied_coeffs_beta * occupied_coeffs_beta.transpose();
            self.density_matrix_beta = self.density_mixing * new_density_beta
                + (1.0 - self.density_mixing) * self.density_matrix_beta.clone();
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
                        let integral_ikjl = B::BasisType::JKabcd(
                            &self.mo_basis[i],
                            &self.mo_basis[k],
                            &self.mo_basis[j],
                            &self.mo_basis[l],
                        );
                        let row = i * self.num_basis + j;
                        let col = k * self.num_basis + l;
                        self.integral_matrix[(row, col)] = integral_ijkl - 0.5 * integral_ikjl;
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
        let mut diis_alpha = DIIS::new(5);
        let mut diis_beta = DIIS::new(5);
        const CONVERGENCE_THRESHOLD: f64 = 1e-6;
        let mut cycle = 0;

        for _ in 0..self.max_cycle {
            cycle += 1;
            info!(
                "\n------------------ SCF Cycle: {} ------------------",
                cycle
            );

            // Step 1: Flatten density matrices
            info!("  Step 1: Flattening Density Matrices...");
            let density_alpha_flattened = self
                .density_matrix_alpha
                .clone()
                .reshape_generic(Dyn(self.num_basis * self.num_basis), Dyn(1));

            let density_beta_flattened = self
                .density_matrix_beta
                .clone()
                .reshape_generic(Dyn(self.num_basis * self.num_basis), Dyn(1));

            let total_density_flattened =
                density_alpha_flattened.clone() + density_beta_flattened.clone();

            // Step 2: Build G matrices
            info!("  Step 2: Building G Matrices from Density Matrices and Integrals...");

            // Coulomb terms (same for both spins)
            let j_matrix_flattened = &self.integral_matrix * &total_density_flattened;
            let j_matrix =
                j_matrix_flattened.reshape_generic(Dyn(self.num_basis), Dyn(self.num_basis));

            // Exchange terms (different for alpha and beta)
            let k_alpha_flattened = &self.integral_matrix * &density_alpha_flattened;
            let k_alpha =
                k_alpha_flattened.reshape_generic(Dyn(self.num_basis), Dyn(self.num_basis));

            let k_beta_flattened = &self.integral_matrix * &density_beta_flattened;
            let k_beta = k_beta_flattened.reshape_generic(Dyn(self.num_basis), Dyn(self.num_basis));

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

            // Step 4: Apply DIIS (if not first cycle)
            if cycle > 1 {
                info!("  Step 4: Applying DIIS acceleration...");

                diis_alpha.update(
                    fock_alpha.clone(),
                    &self.density_matrix_alpha,
                    &self.overlap_matrix,
                );
                if let Some(diis_fock_alpha) = diis_alpha.extrapolate() {
                    fock_alpha = diis_fock_alpha;
                }

                diis_beta.update(
                    fock_beta.clone(),
                    &self.density_matrix_beta,
                    &self.overlap_matrix,
                );
                if let Some(diis_fock_beta) = diis_beta.extrapolate() {
                    fock_beta = diis_fock_beta;
                }

                info!("  DIIS extrapolation applied.");
            }

            // Step 5: Diagonalize Fock matrices
            info!("  Step 5: Diagonalizing Fock matrices...");
            let l = self.overlap_matrix.clone().cholesky().unwrap();
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
            let eigvecs_alpha = l_inv.clone().transpose() * sorted_eigenvectors_alpha;
            self.coeffs_alpha = eigvecs_alpha;
            let current_e_level_alpha = sorted_eigenvalues_alpha;

            // Diagonalize beta Fock matrix
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
            let eigvecs_beta = l_inv.transpose() * sorted_eigenvectors_beta;
            self.coeffs_beta = eigvecs_beta;
            let current_e_level_beta = sorted_eigenvalues_beta;

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

            // Update density matrices
            self.update_density_matrix();

            // Check convergence
            if cycle > 1 {
                info!("  Step 7: Checking for Convergence...");
                let energy_change_alpha =
                    (current_e_level_alpha.clone() - previous_e_level_alpha.clone()).norm();
                let energy_change_beta =
                    (current_e_level_beta.clone() - previous_e_level_beta.clone()).norm();
                let total_energy_change = energy_change_alpha + energy_change_beta;

                info!("    Alpha energy change: {:.8} au", energy_change_alpha);
                info!("    Beta energy change: {:.8} au", energy_change_beta);
                info!("    Total energy change: {:.8} au", total_energy_change);

                if total_energy_change < CONVERGENCE_THRESHOLD {
                    info!("  SCF converged early at cycle {}.", cycle);
                    info!("-------------------- SCF Converged ---------------------\n");
                    break;
                } else {
                    info!("    SCF not yet converged.");
                }
            } else {
                info!("  Convergence check not performed for the first cycle.");
            }

            // Store current energy levels for the next cycle
            previous_e_level_alpha = current_e_level_alpha.clone();
            previous_e_level_beta = current_e_level_beta.clone();
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
        // Calculate number of alpha and beta electrons
        let total_electrons: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();

        let unpaired_electrons = self.multiplicity - 1;
        let n_alpha = (total_electrons + unpaired_electrons) / 2;
        let n_beta = (total_electrons - unpaired_electrons) / 2;

        // Calculate one-electron energy contribution
        let mut one_electron_energy = 0.0;

        // Core hamiltonian
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

        // One-electron contribution
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                one_electron_energy += h_core[(i, j)]
                    * (self.density_matrix_alpha[(i, j)] + self.density_matrix_beta[(i, j)]);
            }
        }

        // Two-electron contribution
        let mut two_electron_energy = 0.0;
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

                        let p_alpha_ij = self.density_matrix_alpha[(i, j)];
                        let p_beta_ij = self.density_matrix_beta[(i, j)];
                        let p_alpha_kl = self.density_matrix_alpha[(k, l)];
                        let p_beta_kl = self.density_matrix_beta[(k, l)];

                        // Coulomb contribution
                        two_electron_energy +=
                            0.5 * coulomb * ((p_alpha_ij + p_beta_ij) * (p_alpha_kl + p_beta_kl));

                        // Exchange contribution
                        two_electron_energy -=
                            0.5 * exchange * (p_alpha_ij * p_alpha_kl + p_beta_ij * p_beta_kl);
                    }
                }
            }
        }

        // Nuclear repulsion energy
        let mut nuclear_repulsion = 0.0;
        for i in 0..self.num_atoms {
            for j in (i + 1)..self.num_atoms {
                let z_i = self.elems[i].get_atomic_number() as f64;
                let z_j = self.elems[j].get_atomic_number() as f64;
                let r_ij = (self.coords[i] - self.coords[j]).norm();
                nuclear_repulsion += z_i * z_j / r_ij;
            }
        }

        one_electron_energy + two_electron_energy + nuclear_repulsion
    }

    fn calculate_forces(&self) -> Vec<Vector3<f64>> {
        info!("#####################################################");
        info!("----------- Calculating Hellman-Feynman Forces -------");
        info!("#####################################################");

        let mut forces = vec![Vector3::zeros(); self.num_atoms];

        // Step 1: Calculate nuclear-nuclear repulsion forces
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
        // This is a simplified approach - for accurate calculations,
        // derivatives of two-electron integrals are needed

        info!("  Forces calculated on atoms:");
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