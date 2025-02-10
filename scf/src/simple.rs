// a simple implementation of the scf trait

extern crate nalgebra as na;

use crate::scf::SCF;
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, SymmetricEigen, Vector3, Norm};
use nalgebra::{Const, Dyn};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct SimpleSCF<B: AOBasis> {
    pub(crate) num_atoms: usize,
    pub(crate) num_basis: usize,
    pub(crate) ao_basis: Vec<Arc<Mutex<B>>>,
    mo_basis: Vec<Arc<B::BasisType>>,
    pub(crate) coords: Vec<Vector3<f64>>,
    pub(crate) elems: Vec<Element>,
    pub(crate) coeffs: DMatrix<f64>,
    pub(crate) integral_matrix: DMatrix<f64>,
    density_mixing: f64,
    density_matrix: DMatrix<f64>,
    pub(crate) fock_matrix: DMatrix<f64>,
    pub(crate) overlap_matrix: DMatrix<f64>,
    pub e_level: DVector<f64>,
    max_cycle: usize,
}

/// Given a matrix where each column is an eigenvector,
/// this function aligns each eigenvector so that the entry with the largest
/// absolute value is positive.
fn align_eigenvectors(mut eigvecs: DMatrix<f64>) -> DMatrix<f64> {
    for j in 0..eigvecs.ncols() {
        // Extract column j as a slice
        let col = eigvecs.column(j);
        // Find the index and value of the entry with maximum absolute value.
        let (max_idx, &max_val) = col
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        // If that maximum element is negative, flip the whole column.
        if max_val < 0.0 {
            for i in 0..eigvecs.nrows() {
                eigvecs[(i, j)] = -eigvecs[(i, j)];
            }
        }
    }
    eigvecs
}

impl<B: AOBasis + Clone> SimpleSCF<B> {
    pub fn new() -> SimpleSCF<B> {
        SimpleSCF {
            num_atoms: 0,
            num_basis: 0,
            ao_basis: Vec::new(),
            mo_basis: Vec::new(),
            coords: Vec::new(),
            elems: Vec::new(),
            coeffs: DMatrix::zeros(0, 0),
            density_matrix: DMatrix::zeros(0, 0),
            integral_matrix: DMatrix::zeros(0, 0),
            density_mixing: 0.2,
            fock_matrix: DMatrix::zeros(0, 0),
            overlap_matrix: DMatrix::zeros(0, 0),
            e_level: DVector::zeros(0),
            max_cycle: 1000,
        }
    }
}

impl<B: AOBasis + Clone> SCF for SimpleSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        self.elems = elems.clone();
        self.num_atoms = elems.len();
        println!("\n#####################################################");
        println!("------------- Initializing Basis Set -------------");
        println!("#####################################################");

        self.num_basis = 0;
        for elem in elems {
            let b = *basis.get(elem.get_symbol()).unwrap();
            println!(
                "  Element: {}, Basis Size: {}",
                elem.get_symbol(),
                b.basis_size()
            );

            let b_arc = Arc::new(Mutex::new((*b).clone()));
            self.ao_basis.push(b_arc.clone());
        }

        println!("  Number of Atoms: {}", self.num_atoms);
        println!("-----------------------------------------------------\n");
    }

    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        println!("#####################################################");
        println!("--------------- Initializing Geometry ---------------");
        println!("#####################################################");
        assert!(coords.len() == elems.len());
        let size = coords.len();
        for i in 0..size {
            println!(
                "  Element: {}, Coordinates: {:?}",
                elems[i].get_symbol(),
                coords[i]
            );
            self.ao_basis[i].lock().unwrap().set_center(coords[i]);
            println!(
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
        println!("  Rebuilt MO basis with {} basis functions.", self.num_basis);
        println!("-----------------------------------------------------\n");
    }



    fn init_density_matrix(&mut self) {
        println!("#####################################################");
        println!("------------ Initializing Density Matrix ------------");
        println!("#####################################################");
        println!("  Building Overlap and Fock Matrices...");
        self.fock_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
        self.overlap_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);

        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                self.overlap_matrix[(i, j)] =
                    B::BasisType::Sab(&self.mo_basis[i], &self.mo_basis[j]);
                self.fock_matrix[(i, j)] = B::BasisType::Tab(&self.mo_basis[i], &self.mo_basis[j]);
                for k in 0..self.num_atoms {
                    self.fock_matrix[(i, j)] += B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[k],
                        self.elems[k].get_atomic_number() as u32,
                    );
                }
            }
        }

        println!("  Diagonalizing Fock matrix to get initial coefficients...");
        let l = self.overlap_matrix.clone().cholesky().unwrap();
        let l_inv = l.inverse();
        let f_prime = l_inv.clone() * self.fock_matrix.clone_owned() * l_inv.clone().transpose();
        let eig = f_prime.clone().try_symmetric_eigen(1e-6, 1000).unwrap();

        // Sort eigenvalues and eigenvectors
        let eigenvalues = eig.eigenvalues.clone();
        let eigenvectors = eig.eigenvectors.clone();
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
        let sorted_eigenvalues =
            DVector::from_fn(eigenvalues.len(), |i, _| eigenvalues[indices[i]]);
        let sorted_eigenvectors = eigenvectors.select_columns(&indices);
        let eigvecs = l_inv.clone().transpose() * sorted_eigenvectors;
        // Corrected line: Remove l_inv multiplication here
        self.coeffs = eigvecs;
        self.e_level = sorted_eigenvalues;

        println!("  Initial Energy Levels:");
        for i in 0..self.e_level.len() {
            println!("    Level {}: {:.8} au", i + 1, self.e_level[i]);
        }

        self.update_density_matrix();
        println!("  Initial Density Matrix built.");
        println!("-----------------------------------------------------\n");
    }

    fn update_density_matrix(&mut self) {
        println!("  Updating Density Matrix...");
        let total_electrons: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        // Ensure even number of electrons for closed-shell
        assert!(total_electrons % 2 == 0, "Total number of electrons must be even");
        let n_occ = total_electrons / 2;

        let occupied_coeffs = self.coeffs.columns(0, n_occ);
        if self.density_matrix.shape() == (0, 0) {
            self.density_matrix = 2.0 * &occupied_coeffs * occupied_coeffs.transpose();
        } else {
            let new_density = 2.0 * &occupied_coeffs * occupied_coeffs.transpose();
            self.density_matrix = self.density_mixing * new_density
                + (1.0 - self.density_mixing) * self.density_matrix.clone();
        }
        println!("  Density Matrix updated with mixing factor {:.2}.", self.density_mixing);
    }

    fn init_fock_matrix(&mut self) {
        println!("#####################################################");
        println!("------------- Initializing Fock Matrix -------------");
        println!("#####################################################");
        println!("  Building Integral Matrix (Two-electron integrals)...");

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
        println!("  Integral Matrix (Two-electron integrals) built.");
        println!("-----------------------------------------------------\n");
    }

    fn scf_cycle(&mut self) {
        println!("#####################################################");
        println!("--------------- Performing SCF Cycle ---------------");
        println!("#####################################################");

        let mut previous_e_level = DVector::zeros(self.num_basis); // Initialize previous energy level
        const CONVERGENCE_THRESHOLD: f64 = 1e-6; // Define convergence threshold
        let mut cycle = 0;

        for _ in 0..self.max_cycle {
            cycle += 1;
            println!("\n------------------ SCF Cycle: {} ------------------", cycle);

            println!("  Step 1: Flattening Density Matrix...");
            let density_flattened =
                self.density_matrix.clone()
                    .reshape_generic(Dyn(self.num_basis * self.num_basis), Dyn(1));
            println!("  Density Matrix flattened.");

            println!("  Step 2: Building G Matrix from Density Matrix and Integrals...");
            let g_matrix_flattened = &self.integral_matrix * &density_flattened;
            let g_matrix =
                g_matrix_flattened.reshape_generic(Dyn(self.num_basis), Dyn(self.num_basis));
            println!("  G Matrix built.");

            println!("  Step 3: Building Hamiltonian (Fock + G) Matrix...");
            let hamiltonian = self.fock_matrix.clone() + g_matrix;
            println!("  Hamiltonian Matrix built.");

            println!("  Step 4: Diagonalizing Hamiltonian Matrix...");
            let l = self.overlap_matrix.clone().cholesky().unwrap();
            let l_inv = l.inverse();
            let f_prime = l_inv.clone() * &hamiltonian * l_inv.transpose();
            let eig = f_prime.try_symmetric_eigen(1e-6, 1000).unwrap();
            println!("  Hamiltonian Matrix diagonalized.");

            // Sort eigenvalues and eigenvectors
            let eigenvalues = eig.eigenvalues.clone();
            let eigenvectors = eig.eigenvectors.clone();
            let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
            indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
            let sorted_eigenvalues =
                DVector::from_fn(eigenvalues.len(), |i, _| eigenvalues[indices[i]]);
            let sorted_eigenvectors = eigenvectors.select_columns(&indices);

            let eigvecs = l_inv.transpose() * sorted_eigenvectors;
            // Corrected line: Remove l_inv multiplication here
            self.coeffs = eigvecs;
            let current_e_level = sorted_eigenvalues;

            println!("  Step 5: Energy Levels obtained:");
            for i in 0..current_e_level.len() {
                println!("    Level {}: {:.8} au", i + 1, current_e_level[i]);
            }

            self.update_density_matrix();

            if cycle > 1 { // Start convergence check from the second cycle
                println!("  Step 6: Checking for Convergence...");
                let energy_change = (current_e_level.clone() - previous_e_level.clone()).norm();
                println!("    Energy change: {:.8} au", energy_change);
                if energy_change < CONVERGENCE_THRESHOLD {
                    println!("  SCF converged early at cycle {}.", cycle);
                    println!("-------------------- SCF Converged ---------------------\n");
                    break; // Exit the loop if converged
                } else {
                    println!("    SCF not yet converged.");
                }
            } else {
                println!("  Convergence check not performed for the first cycle.");
            }
            previous_e_level = current_e_level.clone();
        }
        if cycle == self.max_cycle {
            println!("\n------------------- SCF Not Converged -------------------");
            println!("  SCF did not converge within {} cycles.", self.max_cycle);
            println!("  Please increase MAX_CYCLE or check system setup.");
            println!("-----------------------------------------------------\n");
        } else {
            println!("-----------------------------------------------------\n");
        }
    }
}