// a simple implementation of the scf trait

extern crate nalgebra as na;

use crate::scf::SCF;
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, Vector3};
use periodic_table_on_an_enum::Element;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::info;

#[derive(Clone)]
pub struct SimpleSCF<B: AOBasis> {
    pub num_atoms: usize,
    pub num_basis: usize,
    pub ao_basis: Vec<Arc<Mutex<B>>>,
    mo_basis: Vec<Arc<B::BasisType>>,
    pub coords: Vec<Vector3<f64>>,
    pub elems: Vec<Element>,
    pub coeffs: DMatrix<f64>,
    pub density_mixing: f64,
    pub density_matrix: DMatrix<f64>,
    pub fock_matrix: DMatrix<f64>,
    pub h_core: DMatrix<f64>,
    pub overlap_matrix: DMatrix<f64>,
    pub e_level: DVector<f64>,
    pub max_cycle: usize,
}

/// Given a matrix where each column is an eigenvector,
/// this function aligns each eigenvector so that the entry with the largest
/// absolute value is positive.
pub fn align_eigenvectors(mut eigvecs: DMatrix<f64>) -> DMatrix<f64> {
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

impl<B: AOBasis + Clone + Send> SimpleSCF<B>
where
    B::BasisType: Send + Sync,
{
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
            density_mixing: 0.2,
            fock_matrix: DMatrix::zeros(0, 0),
            h_core: DMatrix::zeros(0, 0),
            overlap_matrix: DMatrix::zeros(0, 0),
            e_level: DVector::zeros(0),
            max_cycle: 100,
        }
    }

    pub fn get_density_matrix(&self) -> DMatrix<f64> {
        self.density_matrix.clone()
    }

    pub fn set_initial_density_matrix(&mut self, density_matrix: DMatrix<f64>) {
        self.density_matrix = density_matrix;
    }

    pub fn update_fock_matrix(&mut self) {
        let mut g_matrix = DMatrix::zeros(self.num_basis, self.num_basis);
        let p = &self.density_matrix;
        
        // Create a vector of (i, j) pairs for parallel iteration
        let ij_pairs: Vec<(usize, usize)> = (0..self.num_basis)
            .flat_map(|i| (0..self.num_basis).map(move |j| (i, j)))
            .collect();
        
        // Parallel computation of G matrix elements
        let g_values: Vec<f64> = ij_pairs
            .par_iter()
            .map(|&(i, j)| {
                let mut g_ij = 0.0;
                for k in 0..self.num_basis {
                    for l in 0..self.num_basis {
                        let coulomb = B::BasisType::JKabcd(&self.mo_basis[i], &self.mo_basis[j], &self.mo_basis[k], &self.mo_basis[l]);
                        let exchange = B::BasisType::JKabcd(&self.mo_basis[i], &self.mo_basis[k], &self.mo_basis[j], &self.mo_basis[l]);
                        g_ij += p[(k, l)] * (coulomb - 0.5 * exchange);
                    }
                }
                g_ij
            })
            .collect();
        
        // Assign computed values back to matrix
        for (idx, &(i, j)) in ij_pairs.iter().enumerate() {
            g_matrix[(i, j)] = g_values[idx];
        }
        
        self.fock_matrix = self.h_core.clone() + g_matrix;
    }
}

impl<B: AOBasis + Clone + Send> SCF for SimpleSCF<B>
where
    B::BasisType: Send + Sync,
{
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &B>) {
        self.elems = elems.clone();
        self.num_atoms = elems.len();
        self.ao_basis.clear();
        for elem in elems {
            let b = *basis.get(elem.get_symbol()).unwrap();
            let b_arc = Arc::new(Mutex::new((*b).clone()));
            self.ao_basis.push(b_arc.clone());
        }
    }

    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        self.coords = coords.clone();
        for i in 0..self.num_atoms {
            self.ao_basis[i].lock().unwrap().set_center(coords[i]);
        }

        self.mo_basis.clear();
        self.num_basis = 0;
        for ao in &self.ao_basis {
            let ao_locked = ao.lock().unwrap();
            self.mo_basis.extend(ao_locked.get_basis());
            self.num_basis += ao_locked.basis_size();
        }

        self.density_matrix = DMatrix::zeros(self.num_basis, self.num_basis);
        self.fock_matrix = DMatrix::zeros(self.num_basis, self.num_basis);
        self.h_core = DMatrix::zeros(self.num_basis, self.num_basis);
        self.coeffs = DMatrix::zeros(self.num_basis, self.num_basis);
        self.e_level = DVector::zeros(self.num_basis);
        self.overlap_matrix = DMatrix::zeros(self.num_basis, self.num_basis);
    }

    fn init_density_matrix(&mut self) {
        self.overlap_matrix = DMatrix::zeros(self.num_basis, self.num_basis);
        self.h_core = DMatrix::zeros(self.num_basis, self.num_basis);

        // Create a vector of (i, j) pairs for parallel iteration
        let ij_pairs: Vec<(usize, usize)> = (0..self.num_basis)
            .flat_map(|i| (0..self.num_basis).map(move |j| (i, j)))
            .collect();

        // Parallel computation of overlap matrix elements
        let overlap_values: Vec<f64> = ij_pairs
            .par_iter()
            .map(|&(i, j)| B::BasisType::Sab(&self.mo_basis[i], &self.mo_basis[j]))
            .collect();

        // Parallel computation of kinetic energy matrix elements
        let kinetic_values: Vec<f64> = ij_pairs
            .par_iter()
            .map(|&(i, j)| B::BasisType::Tab(&self.mo_basis[i], &self.mo_basis[j]))
            .collect();

        // Parallel computation of nuclear attraction matrix elements
        let nuclear_values: Vec<f64> = ij_pairs
            .par_iter()
            .map(|&(i, j)| {
                let mut nuclear_sum = 0.0;
                for k in 0..self.num_atoms {
                    nuclear_sum += B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[k],
                        self.elems[k].get_atomic_number() as u32,
                    );
                }
                nuclear_sum
            })
            .collect();

        // Assign computed values back to matrices
        for (idx, &(i, j)) in ij_pairs.iter().enumerate() {
            self.overlap_matrix[(i, j)] = overlap_values[idx];
            self.h_core[(i, j)] = kinetic_values[idx] + nuclear_values[idx];
        }
        self.fock_matrix = self.h_core.clone();

        let l = self.overlap_matrix.clone().cholesky().unwrap();
        let l_inv = l.inverse();
        let f_prime = l_inv.clone() * self.fock_matrix.clone() * l_inv.transpose();
        let eig = f_prime.symmetric_eigen();

        let mut indices: Vec<usize> = (0..eig.eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
        let sorted_eigenvalues = DVector::from_fn(eig.eigenvalues.len(), |i, _| eig.eigenvalues[indices[i]]);
        let sorted_eigenvectors = eig.eigenvectors.select_columns(&indices);
        
        let eigvecs = l_inv.transpose() * sorted_eigenvectors;
        
        self.coeffs = align_eigenvectors(eigvecs);
        self.e_level = sorted_eigenvalues;

        self.update_density_matrix();
    }

    fn update_density_matrix(&mut self) {
        let total_electrons: usize = self.elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        let n_occ = total_electrons / 2;

        let occupied_coeffs = self.coeffs.columns(0, n_occ);
        let new_density = 2.0 * &occupied_coeffs * occupied_coeffs.transpose();
        
        if self.density_matrix.iter().all(|&x| x == 0.0) {
            self.density_matrix = new_density;
        } else {
            self.density_matrix = self.density_mixing * new_density + (1.0 - self.density_mixing) * self.density_matrix.clone();
        }
    }

    fn init_fock_matrix(&mut self) {
        // Obsolete
    }

    fn scf_cycle(&mut self) {
        let mut old_energy = 0.0;
        for cycle in 0..self.max_cycle {
            info!("Starting SCF cycle {}", cycle);
            self.update_fock_matrix();
            info!("Finished updating Fock matrix for cycle {}", cycle);

            let l = self.overlap_matrix.clone().cholesky().unwrap();
            let l_inv = l.inverse();
            let f_prime = l_inv.clone() * self.fock_matrix.clone() * l_inv.transpose();
            let eig = f_prime.symmetric_eigen();

            let mut indices: Vec<usize> = (0..eig.eigenvalues.len()).collect();
            indices.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
            let sorted_eigenvalues = DVector::from_fn(eig.eigenvalues.len(), |i, _| eig.eigenvalues[indices[i]]);
            let sorted_eigenvectors = eig.eigenvectors.select_columns(&indices);
            
            let eigvecs = l_inv.transpose() * sorted_eigenvectors;

            self.coeffs = align_eigenvectors(eigvecs);
            self.e_level = sorted_eigenvalues;

            self.update_density_matrix();

            let total_energy = self.calculate_total_energy();
            let energy_change = total_energy - old_energy;

            info!("Cycle {}: E = {:.12} au, dE = {:.12} au", cycle, total_energy, energy_change);

            if energy_change.abs() < 1e-5 {
                info!("SCF converged in {} cycles.", cycle + 1);
                break;
            }
            old_energy = total_energy;
        }
    }

    fn calculate_total_energy(&self) -> f64 {
        // Parallel computation of one-electron energy
        let ij_pairs: Vec<(usize, usize)> = (0..self.num_basis)
            .flat_map(|i| (0..self.num_basis).map(move |j| (i, j)))
            .collect();
            
        let one_electron_energy: f64 = ij_pairs
            .par_iter()
            .map(|&(i, j)| {
                self.density_matrix[(i, j)] * (self.h_core[(i, j)] + self.fock_matrix[(i, j)])
            })
            .sum::<f64>() * 0.5;

        // Parallel computation of nuclear repulsion
        let atom_pairs: Vec<(usize, usize)> = (0..self.num_atoms)
            .flat_map(|i| ((i + 1)..self.num_atoms).map(move |j| (i, j)))
            .collect();
            
        let nuclear_repulsion: f64 = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let z_i = self.elems[i].get_atomic_number() as f64;
                let z_j = self.elems[j].get_atomic_number() as f64;
                let r_ij = (self.coords[i] - self.coords[j]).norm();
                if r_ij > 1e-10 {
                    z_i * z_j / r_ij
                } else {
                    0.0
                }
            })
            .sum();

        one_electron_energy + nuclear_repulsion
    }

    fn calculate_forces(&self) -> Vec<Vector3<f64>> {
        info!("-----------------------------------------------------");
        info!("  Computing forces using Hellmann-Feynman theorem");
        info!("-----------------------------------------------------");
        
        let mut forces = vec![Vector3::zeros(); self.num_atoms];
        
        // Calculate Energy Weighted Density Matrix for Pulay forces
        // W = C_occ * ε_occ * C_occ^T
        let total_electrons: usize = self.elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        let n_occ = total_electrons / 2;
        
        let c_occ = self.coeffs.columns(0, n_occ);
        let e_occ_vec = self.e_level.rows(0, n_occ);
        let e_occ_diag = DMatrix::from_diagonal(&e_occ_vec);
        let w_matrix = &c_occ * e_occ_diag * c_occ.transpose();

        // Step 1: Calculate nuclear-nuclear repulsion forces
        info!("  Step 1: Nuclear-nuclear repulsion forces...");
        let nuclear_force_contributions: Vec<Vector3<f64>> = (0..self.num_atoms)
            .into_par_iter()
            .map(|i| {
                let mut force_i = Vector3::zeros();
                for j in 0..self.num_atoms {
                    if i == j {
                        continue;
                    }

                    let z_i = self.elems[i].get_atomic_number() as f64;
                    let z_j = self.elems[j].get_atomic_number() as f64;
                    let r_ij = self.coords[i] - self.coords[j];
                    let r_ij_norm = r_ij.norm();

                    // Nuclear-nuclear repulsion force
                    force_i += z_i * z_j * r_ij / (r_ij_norm * r_ij_norm * r_ij_norm);
                }
                force_i
            })
            .collect();
        
        for (i, force_contrib) in nuclear_force_contributions.into_iter().enumerate() {
            forces[i] += force_contrib;
        }

        // Step 2: Calculate electron-nuclear attraction forces
        info!("  Step 2: Electron-nuclear attraction forces...");
        let electron_nuclear_forces: Vec<Vector3<f64>> = (0..self.num_atoms)
            .into_par_iter()
            .map(|atom_idx| {
                let mut force_atom = Vector3::zeros();
                for i in 0..self.num_basis {
                    for j in 0..self.num_basis {
                        let p_ij = self.density_matrix[(i, j)];

                        // Calculate derivative of nuclear attraction integrals
                        let dv_dr = B::BasisType::dVab_dR(
                            &self.mo_basis[i],
                            &self.mo_basis[j],
                            self.coords[atom_idx],
                            self.elems[atom_idx].get_atomic_number() as u32,
                        );

                        // Add contribution to force
                        force_atom -= p_ij * dv_dr;
                    }
                }
                force_atom
            })
            .collect();
        
        for (atom_idx, force_contrib) in electron_nuclear_forces.into_iter().enumerate() {
            forces[atom_idx] += force_contrib;
        }

        // Step 3: Calculate forces from two-electron integrals
        info!("  Step 3: Two-electron integral derivatives...");
        let two_electron_forces: Vec<Vector3<f64>> = (0..self.num_atoms)
            .into_par_iter()
            .map(|atom_idx| {
                let mut force_atom = Vector3::zeros();
                for i in 0..self.num_basis {
                    for j in 0..self.num_basis {
                        for k in 0..self.num_basis {
                            for l in 0..self.num_basis {
                                let p_ij = self.density_matrix[(i, j)];
                                let p_kl = self.density_matrix[(k, l)];

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

                                // Coulomb contribution
                                force_atom -= p_ij * p_kl * coulomb_deriv;

                                // Exchange contribution
                                force_atom += 0.5 * p_ij * p_kl * exchange_deriv;
                            }
                        }
                    }
                }
                force_atom
            })
            .collect();
        
        for (atom_idx, force_contrib) in two_electron_forces.into_iter().enumerate() {
            forces[atom_idx] += force_contrib;
        }

        // Step 4: Calculate Pulay forces (derivatives w.r.t. basis function centers)
        info!("  Step 4: Pulay forces (basis function derivatives)...");
        let pulay_forces: Vec<Vector3<f64>> = (0..self.num_atoms)
            .into_par_iter()
            .map(|atom_idx| {
                let mut force_atom = Vector3::zeros();
                // Core Hamiltonian Pulay forces
                for i in 0..self.num_basis {
                    for j in 0..self.num_basis {
                        let p_ij = self.density_matrix[(i, j)];

                        // Overlap matrix derivatives (weighted by Energy Weighted Density matrix elements)
                        let ds_dr = B::BasisType::dSab_dR(&self.mo_basis[i], &self.mo_basis[j], atom_idx);
                        let w_ij = w_matrix[(i, j)];
                        force_atom -= w_ij * ds_dr;

                        // Kinetic energy derivatives
                        let dt_dr = B::BasisType::dTab_dR(&self.mo_basis[i], &self.mo_basis[j], atom_idx);
                        force_atom -= p_ij * dt_dr;

                        // Nuclear attraction Pulay forces
                        for k in 0..self.num_atoms {
                            let dv_dr_basis = B::BasisType::dVab_dRbasis(
                                &self.mo_basis[i],
                                &self.mo_basis[j],
                                self.coords[k],
                                self.elems[k].get_atomic_number() as u32,
                                atom_idx,
                            );
                            force_atom -= p_ij * dv_dr_basis;
                        }
                    }
                }
                force_atom
            })
            .collect();
        
        for (atom_idx, force_contrib) in pulay_forces.into_iter().enumerate() {
            forces[atom_idx] += force_contrib;
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

