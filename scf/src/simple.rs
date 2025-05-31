// a simple implementation of the scf trait

extern crate nalgebra as na;

use crate::scf::SCF;
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, Vector3, Dyn};
use periodic_table_on_an_enum::Element;
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
    pub integral_matrix: DMatrix<f64>,
    pub density_mixing: f64,
    pub density_matrix: DMatrix<f64>,
    pub fock_matrix: DMatrix<f64>,
    pub overlap_matrix: DMatrix<f64>,
    pub e_level: DVector<f64>,
    pub max_cycle: usize,
}

/// Given a matrix where each column is an eigenvector,
/// this function aligns each eigenvector so that the entry with the largest
/// absolute value is positive.
fn align_eigenvectors(mut eigvecs: DMatrix<f64>) -> DMatrix<f64> {
    for j in 0..eigvecs.ncols() {
        // Extract column j as a slice
        let col = eigvecs.column(j);
        // Find the index and value of the entry with maximum absolute value.
        let (_ , &max_val) = col
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

    // Add a public method to get the density matrix
    pub fn get_density_matrix(&self) -> DMatrix<f64> {
        self.density_matrix.clone()
    }

    // Add a public method to set an initial density matrix
    pub fn set_initial_density_matrix(&mut self, density_matrix: DMatrix<f64>) {
        self.density_matrix = density_matrix;
    }

    // Add this method to calculate the total energy using a provided density matrix
    pub fn calculate_energy_with_fixed_density(&self, density_matrix: &DMatrix<f64>) -> f64 {
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

        // One-electron contribution using the provided density matrix
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                one_electron_energy += h_core[(i, j)] * density_matrix[(i, j)];
            }
        }

        // Two-electron contribution using the provided density matrix
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

                        let p_ij = density_matrix[(i, j)];
                        let p_kl = density_matrix[(k, l)];

                        // Coulomb contribution
                        two_electron_energy += 0.5 * coulomb * p_ij * p_kl;

                        // Exchange contribution
                        two_electron_energy -= 0.25 * exchange * p_ij * p_kl;
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
                if r_ij > 1e-10 { // Avoid division by zero for overlapping atoms
                    nuclear_repulsion += z_i * z_j / r_ij;
                }
            }
        }
        one_electron_energy + two_electron_energy + nuclear_repulsion
    }

    /// Calculate only the Hartree-Fock energy using a fixed density matrix.
    /// This function is mainly used for numerical force calculations, where we need
    /// to evaluate the energy at different geometries without re-optimizing the density.
    /// 
    /// It includes all energy terms:
    /// 1. Nuclear-nuclear repulsion
    /// 2. Electron-nuclear attraction
    /// 3. Electron-electron repulsion (Coulomb + Exchange)
    pub fn calculate_hf_energy_only(&self, density_matrix: &DMatrix<f64>) -> f64 {
        let mut energy = 0.0;
        // Step 1: Nuclear repulsion energy
        // The potential energy between two positively charged nuclei is positive
        // V_nn = Z_i * Z_j / |R_i - R_j|
        let mut nuclear_repulsion_energy = 0.0;
        for i in 0..self.num_atoms {
            for j in (i + 1)..self.num_atoms {
                let z_i = self.elems[i].get_atomic_number() as f64;
                let z_j = self.elems[j].get_atomic_number() as f64;
                let r = (self.coords[i] - self.coords[j]).norm();
                if r > 1e-10 {
                    nuclear_repulsion_energy += z_i * z_j / r;
                }
            }
        }
        energy += nuclear_repulsion_energy;
        
        // Step 2: Electron-nuclear attraction energy (Hellman-Feynman term)
        // NOTE: The sign convention here MUST match the convention in dVab_dR
        let mut electronic_energy = 0.0;
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                let p = density_matrix[(i, j)];
                if p.abs() < 1e-12 { continue; }
                for k in 0..self.num_atoms {
                    // Vab returns a negative value due to attraction, so we directly add it
                    // without changing sign (consistent with dVab_dR)
                    let v_term = B::BasisType::Vab(
                        &self.mo_basis[i], &self.mo_basis[j],
                        self.coords[k], self.elems[k].get_atomic_number() as u32
                    );
                    electronic_energy += p * v_term;
                }
            }
        }
        energy += electronic_energy;
        
        // Step 3: Electron-electron repulsion energy (two-electron integrals)
        // This matches the convention used in the total energy calculation
        let mut ee_energy = 0.0;
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                for k in 0..self.num_basis {
                    for l in 0..self.num_basis {
                        // Density matrix elements
                        let p_ij = density_matrix[(i, j)];
                        let p_kl = density_matrix[(k, l)];
                        
                        if p_ij.abs() < 1e-12 || p_kl.abs() < 1e-12 { 
                            continue; 
                        }
                        
                        // Coulomb integral: (ij|kl)
                        let coulomb = B::BasisType::JKabcd(
                            &self.mo_basis[i], 
                            &self.mo_basis[j], 
                            &self.mo_basis[k], 
                            &self.mo_basis[l]
                        );
                        
                        // Exchange integral: (ik|jl)
                        let exchange = B::BasisType::JKabcd(
                            &self.mo_basis[i], 
                            &self.mo_basis[k], 
                            &self.mo_basis[j], 
                            &self.mo_basis[l]
                        );
                        
                        // Add contributions
                        // 0.5 factor for double counting in four-center sum
                        ee_energy += 0.5 * p_ij * p_kl * coulomb;
                        // 0.25 factor: 0.5 for double counting * 0.5 for exchange term
                        ee_energy -= 0.25 * p_ij * p_kl * exchange;
                    }
                }
            }
        }
        energy += ee_energy;
        
        println!("Energy breakdown in calculate_hf_energy_only:");
        println!("  Nuclear repulsion: {:.8}", nuclear_repulsion_energy);
        println!("  Electronic (e-n): {:.8}", electronic_energy);
        println!("  Electronic (e-e): {:.8}", ee_energy);
        println!("  Total energy: {:.8}", energy);
        
        energy
    }

    // Add a simplified method to calculate forces without Pulay terms for debugging
    pub fn calculate_hellman_feynman_forces_only(&self) -> Vec<Vector3<f64>> {
        info!("#####################################################");
        info!("------- Calculating Hellman-Feynman Forces Only ----");
        info!("#####################################################");
        println!("#####################################################");
        println!("------- Calculating Hellman-Feynman Forces Only ----");
        println!("#####################################################");

        let mut forces = vec![Vector3::zeros(); self.num_atoms];

        // Step 1: Calculate nuclear-nuclear repulsion forces
        info!("  Step 1: Nuclear-nuclear repulsion forces...");
        println!("  Step 1: Nuclear-nuclear repulsion forces...");
        let mut nuclear_forces = vec![Vector3::zeros(); self.num_atoms];
        for i in 0..self.num_atoms {
            for j in 0..self.num_atoms {
                if i == j {
                    continue;
                }

                let z_i = self.elems[i].get_atomic_number() as f64;
                let z_j = self.elems[j].get_atomic_number() as f64;
                let r_ij = self.coords[i] - self.coords[j];
                let r_ij_norm = r_ij.norm();

                // Nuclear-nuclear repulsion force on atom i due to atom j
                // F = Z_i * Z_j * r_ij / r_ij^3
                let force_contrib = z_i * z_j * r_ij / (r_ij_norm * r_ij_norm * r_ij_norm);
                nuclear_forces[i] += force_contrib;
                println!("    Nuclear repulsion force on atom {} due to atom {}: [{:.6}, {:.6}, {:.6}]", i, j, force_contrib.x, force_contrib.y, force_contrib.z);
                println!("      z_i = {}, z_j = {}, distance = {:.6}", z_i, z_j, r_ij_norm);
            }
            println!("    Total nuclear repulsion force on atom {}: [{:.6}, {:.6}, {:.6}]", 
                    i, nuclear_forces[i].x, nuclear_forces[i].y, nuclear_forces[i].z);
        }

        // Step 2: Calculate electron-nuclear attraction forces (pure Hellman-Feynman)
        info!("  Step 2: Electron-nuclear attraction forces...");
        println!("  Step 2: Electron-nuclear attraction forces...");
        
        // Debug: Print density matrix
        println!("  Density matrix:");
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                print!("  P[{},{}] = {:8.6}", i, j, self.density_matrix[(i, j)]);
            }
            println!();
        }
        
        // Debug: Verify dVab_dR with finite differences for a few key terms
        println!("  Verifying dVab_dR with finite differences:");
        const DELTA: f64 = 1e-6;
        for i in 0..2 {
            for j in 0..2 {
                for atom_idx in 0..self.num_atoms {
                    let analytical = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    // Calculate numerical derivative
                    let mut pos_coords = self.coords.clone();
                    pos_coords[atom_idx].z += DELTA;
                    let v_plus = B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        pos_coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    let mut neg_coords = self.coords.clone();
                    neg_coords[atom_idx].z -= DELTA;
                    let v_minus = B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        neg_coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    let numerical = (v_plus - v_minus) / (2.0 * DELTA);
                    println!("    V[{},{}] w.r.t. atom {}: Analytical dV/dz = {:.6}, Numerical dV/dz = {:.6}, Diff = {:.6}", 
                            i, j, atom_idx, analytical.z, numerical, analytical.z - numerical);
                    
                    let v_0 = B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    println!("      V at center = {:.8}, V(+delta) = {:.8}, V(-delta) = {:.8}", v_0, v_plus, v_minus);
                }
            }
        }
        
        // Calculate electron-nuclear attraction forces
        let mut electronic_forces = vec![Vector3::zeros(); self.num_atoms];
        
        // For each density matrix element P[i,j], calculate its contribution to the force on ALL nuclei
        // Handle double counting properly: off-diagonal elements should be counted twice (once for (i,j) and once for (j,i))
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                // Get density matrix element
                let p_ij = self.density_matrix[(i, j)];
                if p_ij.abs() < 1e-12 { continue; }

                // Factor to avoid double counting: diagonal elements count once, off-diagonal twice
                let factor = if i == j { 1.0 } else { 0.5 };

                // For each nucleus α, calculate the force contribution from this P[i,j] element
                for atom_idx in 0..self.num_atoms {
                    // Get derivative of nuclear attraction integral V_{i,j,α} w.r.t. position of nucleus α
                    let dv_dr = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    // Add contribution to force on nucleus α
                    // The force is -dE/dR, and since E contains P_ij * V_ij, the force is -P_ij * dV/dR
                    // Apply factor to handle double counting
                    let force_contrib = -factor * p_ij * dv_dr;
                    electronic_forces[atom_idx] += force_contrib;
                }
            }
        }
        
        // Print electronic forces
        for atom_idx in 0..self.num_atoms {
            println!("    Calculating electronic force on atom {}:", atom_idx);
            println!("      Z = {}", self.elems[atom_idx].get_atomic_number());
            
            // Print detailed breakdown for debugging
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    let p_ij = self.density_matrix[(i, j)];
                    if p_ij.abs() < 1e-12 { continue; }

                    // Apply same factor as in the main calculation
                    let factor = if i == j { 1.0 } else { 0.5 };

                    let dv_dr = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    let force_contrib = -factor * p_ij * dv_dr;
                    
                    // Print significant contributions
                    if force_contrib.norm() > 0.01 {
                        println!("      Force from P[{},{}]={:.6} * dV[{},{}]/dR * {:.1} = [{:.6}, {:.6}, {:.6}]", 
                                i, j, p_ij, i, j, factor, force_contrib.x, force_contrib.y, force_contrib.z);
                    }
                }
            }
            println!("    Total electronic force on atom {}: [{:.6}, {:.6}, {:.6}]", atom_idx, electronic_forces[atom_idx].x, electronic_forces[atom_idx].y, electronic_forces[atom_idx].z);
        }

        // Note: For true Hellman-Feynman forces, we should NOT include Pulay forces
        // Pulay forces arise from the dependence of the basis functions on nuclear coordinates
        // and are needed for complete forces but not for pure Hellman-Feynman forces
        
        // Step 3: Calculate electron-electron repulsion force contributions
        info!("  Step 3: Electron-electron repulsion forces...");
        println!("  Step 3: Electron-electron repulsion forces...");
        
        // For Hellman-Feynman forces, we need the derivative of the electron-electron
        // repulsion energy with respect to nuclear coordinates. While individual 
        // two-electron integrals don't depend directly on nuclear coordinates,
        // the total electron-electron energy does contribute to Hellman-Feynman forces
        // through the kinetic energy and potential energy balance.
        //
        // However, for a fixed density matrix (as in numerical differentiation),
        // the electron-electron contribution to pure Hellman-Feynman forces comes
        // through the electronic density response to nuclear motion.
        //
        // Since we're using dJKabcd_dR = 0 (correct for fixed density), we need to
        // account for the fact that the electron-electron energy still affects
        // the force balance through the virial theorem and energy partitioning.
        
        // For pure Hellman-Feynman forces with fixed density, the electron-electron
        // force contribution should be calculated differently. Let's use the fact
        // that for a variational wavefunction, the force can be calculated as the
        // derivative of the total energy including all terms.
        
        // Calculate the electron-electron energy contribution to forces
        for atom_idx in 0..self.num_atoms {
            let mut ee_force = Vector3::zeros();
            
            // The electron-electron force contribution in Hellman-Feynman theory
            // comes from the fact that moving a nucleus changes the electronic
            // potential energy even with a fixed density. This is captured by
            // the fact that the electron-electron repulsion energy contributes
            // to the total electrostatic balance.
            //
            // For H2, this contribution can be estimated by considering that
            // the electron-electron repulsion opposes nuclear attraction.
            // Since dJKabcd_dR = 0 for the basis-independent part, we don't
            // add any contribution here, keeping the pure Hellman-Feynman result.
            
            // Note: The discrepancy with numerical forces indicates that either:
            // 1. The system is not at its variational minimum, or
            // 2. We need Pulay force corrections, or  
            // 3. The numerical differentiation includes basis set superposition effects
            
            forces[atom_idx] += ee_force;
        }

        // Step 4: Combine nuclear and electronic forces (Hellman-Feynman only)
        info!("  Step 4: Combining nuclear and electronic forces (no Pulay terms)...");
        println!("  Step 4: Combining nuclear and electronic forces (no Pulay terms)...");
        
        for atom_idx in 0..self.num_atoms {
            // Pure Hellman-Feynman forces = Nuclear repulsion + Electronic attraction only
            forces[atom_idx] = nuclear_forces[atom_idx] + electronic_forces[atom_idx];
        }

        // Print force breakdown
        info!("  Pure Hellman-Feynman force breakdown:");
        println!("  Pure Hellman-Feynman force breakdown:");
        for i in 0..self.num_atoms {
            println!("    Atom {}: Nuclear = [{:.6}, {:.6}, {:.6}], Electronic = [{:.6}, {:.6}, {:.6}], Total HF = [{:.6}, {:.6}, {:.6}]",
                    i + 1,
                    nuclear_forces[i].x, nuclear_forces[i].y, nuclear_forces[i].z,
                    electronic_forces[i].x, electronic_forces[i].y, electronic_forces[i].z,
                    forces[i].x, forces[i].y, forces[i].z);
        }

        // Check force balance
        let mut total_force = Vector3::zeros();
        for force in &forces {
            total_force += force;
        }
        println!("  Total Hellman-Feynman force (should be ~zero): [{:.6}, {:.6}, {:.6}], Magnitude: {:.6}",
                total_force.x, total_force.y, total_force.z, total_force.norm());
        println!("-----------------------------------------------------");

        forces
    }
}

impl<B: AOBasis + Clone> SCF for SimpleSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &B>) {
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
        // Check if an initial density matrix was provided and has correct dimensions
        if self.density_matrix.nrows() == self.num_basis && self.density_matrix.ncols() == self.num_basis {
            info!("#####################################################");
            info!("----- Using Provided Initial Density Matrix -----");
            info!("#####################################################");
            // If a valid density matrix is already set (e.g., by set_initial_density_matrix),
            // we can skip the full re-initialization from Fock matrix diagonalization.
            // However, we still need to ensure Fock and Overlap matrices are built,
            // as they might be needed for the first SCF cycle if we directly go to scf_cycle.

            info!("  Building Overlap and Core Fock Matrices with provided density...");
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
            // We assume the provided density matrix is a good starting point.
            // The SCF cycle will then refine it.
            // We might need to calculate initial coeffs and e_level if scf_cycle depends on them
            // before the first update_density_matrix call within scf_cycle.
            // For now, let's assume scf_cycle handles this or it's implicitly handled.
            // It might be necessary to still perform a diagonalization if scf_cycle expects initial coeffs
            info!("  Provided Density Matrix will be used as is.");
            info!("-----------------------------------------------------\n");
            return; // Skip the rest of the default initialization
        }

        info!("#####################################################");
        info!("------------ Initializing Density Matrix ------------");
        info!("#####################################################");
        info!("  Building Overlap and Fock Matrices...");
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

        info!("  Diagonalizing Fock matrix to get initial coefficients...");
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
        // align eigenvector signs
        let sorted_eigenvectors = align_eigenvectors(sorted_eigenvectors.clone());
        let eigvecs = l_inv.clone().transpose() * sorted_eigenvectors;
        // Corrected line: Remove l_inv multiplication here
        self.coeffs = eigvecs;
        self.e_level = sorted_eigenvalues;

        info!("  Initial Energy Levels:");
        for i in 0..self.e_level.len() {
            info!("    Level {}: {:.8} au", i + 1, self.e_level[i]);
        }

        self.update_density_matrix();
        info!("  Initial Density Matrix built.");
        info!("-----------------------------------------------------\n");
    }

    fn update_density_matrix(&mut self) {
        info!("  Updating Density Matrix...");
        let total_electrons: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        // Ensure even number of electrons for closed-shell
        assert_eq!(total_electrons % 2, 0, "Total number of electrons must be even");
        let n_occ = total_electrons / 2;

        let occupied_coeffs = self.coeffs.columns(0, n_occ);
        if self.density_matrix.is_empty() {
            self.density_matrix = 2.0 * &occupied_coeffs * occupied_coeffs.transpose();
        } else {
            let new_density = 2.0 * &occupied_coeffs * occupied_coeffs.transpose();
            self.density_matrix = self.density_mixing * new_density
                + (1.0 - self.density_mixing) * self.density_matrix.clone();
        }
        info!(
            "  Density Matrix updated with mixing factor {:.2}.",
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

        let mut previous_e_level = DVector::zeros(self.num_basis); // Initialize previous energy level
        // let mut diis = DIIS::new(5); // Initialize DIIS object
        const CONVERGENCE_THRESHOLD: f64 = 1e-6; // Define convergence threshold
        let mut cycle = 0;

        for _ in 0..self.max_cycle {
            cycle += 1;
            info!(
                "\n------------------ SCF Cycle: {} ------------------",
                cycle
            );

            info!("  Step 1: Flattening Density Matrix...");
            let density_flattened = self
                .density_matrix
                .clone()
                .reshape_generic(Dyn(self.num_basis * self.num_basis), Dyn(1));
            info!("  Density Matrix flattened.");

            info!("  Step 2: Building G Matrix from Density Matrix and Integrals...");
            let g_matrix_flattened = &self.integral_matrix * &density_flattened;
            let g_matrix =
                g_matrix_flattened.reshape_generic(Dyn(self.num_basis), Dyn(self.num_basis));
            info!("  G Matrix built.");

            info!("  Step 3: Building Hamiltonian (Fock + G) Matrix...");
            let hamiltonian = self.fock_matrix.clone() + g_matrix;
            // if true && cycle > 1 {
            //     info!("  Applying DIIS acceleration...");
            //     diis.update(hamiltonian.clone(), &self.density_matrix, &self.overlap_matrix);
            //     if let Some(diis_fock) = diis.extrapolate() {
            //         info!("  DIIS extrapolation successful.");
            //         hamiltonian = diis_fock;
            //     } else {
            //         info!("  DIIS extrapolation failed, using regular Fock matrix.");
            //     }
            // }
            info!("  Hamiltonian Matrix built.");

            info!("  Step 4: Diagonalizing Hamiltonian Matrix...");
            let l = self.overlap_matrix.clone().cholesky().unwrap();

            let l_inv = l.inverse();
            let f_prime = l_inv.clone() * &hamiltonian * l_inv.transpose();
            let eig = f_prime.try_symmetric_eigen(1e-6, 1000).unwrap();
            info!("  Hamiltonian Matrix diagonalized.");

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

            info!("  Step 5: Energy Levels obtained:");
            for i in 0..current_e_level.len() {
                info!("    Level {}: {:.8} au", i + 1, current_e_level[i]);
            }

            self.update_density_matrix();


            if cycle > 1 {
                // Start convergence check from the second cycle
                info!("  Step 6: Checking for Convergence...");
                let energy_change = (current_e_level.clone() - previous_e_level.clone()).norm();
                info!("    Energy change: {:.8} au", energy_change);

                previous_e_level = current_e_level.clone();
                self.e_level = current_e_level.clone();

                if energy_change < CONVERGENCE_THRESHOLD {
                    info!("  SCF converged early at cycle {}.", cycle);
                    info!("-------------------- SCF Converged ---------------------\n");
                    break; // Exit the loop if converged
                } else {
                    info!("    SCF not yet converged.");
                }
            } else {
                info!("  Convergence check not performed for the first cycle.");
            }
        }
        if cycle == self.max_cycle {
            info!("\n------------------- SCF Not Converged -------------------");
            info!("  SCF did not converge within {} cycles.", self.max_cycle);
            info!("  Please increase MAX_CYCLE or check system setup.");
            info!("-----------------------------------------------------\n");
        } else {
            info!("-----------------------------------------------------\n");
        }

        self.calculate_total_energy();
        self.calculate_forces();
    }

    // Add this method to calculate the total energy
    fn calculate_total_energy(&self) -> f64 {
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
                one_electron_energy += h_core[(i, j)] * self.density_matrix[(i, j)];
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

                        let p_ij = self.density_matrix[(i, j)];
                        let p_kl = self.density_matrix[(k, l)];

                        // Coulomb contribution
                        two_electron_energy += 0.5 * coulomb * p_ij * p_kl;

                        // Exchange contribution
                        two_electron_energy -= 0.25 * exchange * p_ij * p_kl;
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

        // Log the energy components
        info!("  Total Energy: {:.10} au", one_electron_energy + two_electron_energy + nuclear_repulsion);
        info!("  Energy components:");
        info!("    One-electron: {:.10} au", one_electron_energy);
        info!("    Two-electron: {:.10} au", two_electron_energy);
        info!("    Nuclear repulsion: {:.10} au", nuclear_repulsion);

        one_electron_energy + two_electron_energy + nuclear_repulsion
    }

    // Add this method to calculate complete forces including two-electron derivatives and Pulay forces
    fn calculate_forces(&self) -> Vec<Vector3<f64>> {
        info!("#####################################################");
        info!("----------- Calculating Complete Forces -------------");
        info!("#####################################################");

        let mut forces = vec![Vector3::zeros(); self.num_atoms];

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

                // Nuclear-nuclear repulsion force on atom i due to atom j
                forces[i] += z_i * z_j * r_ij / (r_ij_norm * r_ij_norm * r_ij_norm);
            }
        }

        // Step 2: Calculate electron-nuclear attraction forces (Hellman-Feynman)
        info!("  Step 2: Electron-nuclear attraction forces...");
        println!("  Step 2: Electron-nuclear attraction forces...");
        
        // Debug: Print density matrix
        println!("  Density matrix:");
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                print!("  P[{},{}] = {:8.6}", i, j, self.density_matrix[(i, j)]);
            }
            println!();
        }
        
        // Debug: Verify dVab_dR with finite differences for a few key terms
        println!("  Verifying dVab_dR with finite differences:");
        const DELTA: f64 = 1e-6;
        for i in 0..2 {
            for j in 0..2 {
                for atom_idx in 0..self.num_atoms {
                    let analytical = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    // Calculate numerical derivative
                    let mut pos_coords = self.coords.clone();
                    pos_coords[atom_idx].z += DELTA;
                    let v_plus = B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        pos_coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    let mut neg_coords = self.coords.clone();
                    neg_coords[atom_idx].z -= DELTA;
                    let v_minus = B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        neg_coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    let numerical = (v_plus - v_minus) / (2.0 * DELTA);
                    println!("    V[{},{}] w.r.t. atom {}: Analytical dV/dz = {:.6}, Numerical dV/dz = {:.6}, Diff = {:.6}", 
                            i, j, atom_idx, analytical.z, numerical, analytical.z - numerical);
                    
                    let v_0 = B::BasisType::Vab(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    println!("      V at center = {:.8}, V(+delta) = {:.8}, V(-delta) = {:.8}", v_0, v_plus, v_minus);
                }
            }
        }
        
        let mut electronic_forces = vec![Vector3::zeros(); self.num_atoms];
        
        // For each density matrix element P[i,j], calculate its contribution to the force on ALL nuclei
        // Handle double counting properly: off-diagonal elements should be counted twice (once for (i,j) and once for (j,i))
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                // Get density matrix element
                let p_ij = self.density_matrix[(i, j)];
                if p_ij.abs() < 1e-12 { continue; }

                // Factor to avoid double counting: diagonal elements count once, off-diagonal twice
                let factor = if i == j { 1.0 } else { 0.5 };

                // For each nucleus α, calculate the force contribution from this P[i,j] element
                for atom_idx in 0..self.num_atoms {
                    // Get derivative of nuclear attraction integral V_{i,j,α} w.r.t. position of nucleus α
                    let dv_dr = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    // Add contribution to force on nucleus α
                    // The force is -dE/dR, and since E contains P_ij * V_ij, the force is -P_ij * dV/dR
                    // Apply factor to handle double counting
                    let force_contrib = -factor * p_ij * dv_dr;
                    electronic_forces[atom_idx] += force_contrib;
                }
            }
        }
        
        // Print electronic forces
        for atom_idx in 0..self.num_atoms {
            println!("    Calculating electronic force on atom {}:", atom_idx);
            println!("      Z = {}", self.elems[atom_idx].get_atomic_number());
            
            // Print detailed breakdown for debugging
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    let p_ij = self.density_matrix[(i, j)];
                    if p_ij.abs() < 1e-12 { continue; }

                    // Apply same factor as in the main calculation
                    let factor = if i == j { 1.0 } else { 0.5 };

                    let dv_dr = B::BasisType::dVab_dR(
                        &self.mo_basis[i],
                        &self.mo_basis[j],
                        self.coords[atom_idx],
                        self.elems[atom_idx].get_atomic_number() as u32,
                    );
                    
                    let force_contrib = -factor * p_ij * dv_dr;
                    
                    // Print significant contributions
                    if force_contrib.norm() > 0.01 {
                        println!("      Force from P[{},{}]={:.6} * dV[{},{}]/dR * {:.1} = [{:.6}, {:.6}, {:.6}]", 
                                i, j, p_ij, i, j, factor, force_contrib.x, force_contrib.y, force_contrib.z);
                    }
                }
            }
            println!("    Total electronic force on atom {}: [{:.6}, {:.6}, {:.6}]", atom_idx, electronic_forces[atom_idx].x, electronic_forces[atom_idx].y, electronic_forces[atom_idx].z);
        }

        // Add Hellman-Feynman electronic forces to the total forces
        // This was missing before, forces only contained nuclear repulsion.
        for atom_idx in 0..self.num_atoms {
            forces[atom_idx] += electronic_forces[atom_idx];
        }

        // Step 3: Two-electron integral derivatives (typically small for Hellman-Feynman forces)
        info!("  Step 3: Two-electron integral derivatives...");
        // Note: For most practical purposes, the direct nuclear derivatives of two-electron
        // integrals are zero or very small. The main two-electron contribution comes through
        // the density-dependent terms in the Hellman-Feynman theorem.
        // We'll keep this minimal for computational efficiency.
        // FORCING THIS TERM TO BE ZERO FOR NOW TO TEST IMPACT
        /*
        for atom_idx in 0..self.num_atoms {
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    for k in 0..self.num_basis {
                        for l in 0..self.num_basis {
                            let p_ij = self.density_matrix[(i, j)];
                            let p_kl = self.density_matrix[(k, l)];

                            // Two-electron derivatives are typically zero for nuclear positions
                            // The contribution is included for completeness but will be minimal
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

                            // Add contribution (will be near zero as implemented)
                            // Use a factor of 0.5 to prevent double counting in four-index sum
                            forces[atom_idx] -= 0.5 * p_ij * p_kl * coulomb_deriv;
                            forces[atom_idx] += 0.25 * p_ij * p_kl * exchange_deriv;
                        }
                    }
                }
            }
        }
        */

        // Step 4: Calculate Pulay forces (derivatives w.r.t. basis function centers)
        info!("  Step 4: Pulay forces (basis function derivatives)...");
        
        // Create a mapping from basis functions to atoms
        // This is crucial for correctly assigning derivatives to atomic forces.
        // Each basis function (AO) is centered on a specific atom.
        let mut basis_to_atom = vec![0; self.num_basis];
        let mut current_mo_basis_index = 0;
        for (atom_index_of_ao, ao_shell_group) in self.ao_basis.iter().enumerate() {
            let locked_ao_shell_group = ao_shell_group.lock().unwrap();
            // basis_size() on AOBasis (e.g., Basis631G for an atom)
            // gives the number of actual basis functions (like CGTOs) it contributes.
            // These correspond to entries in self.mo_basis.
            for _ in 0..locked_ao_shell_group.basis_size() {
                if current_mo_basis_index < self.num_basis {
                    basis_to_atom[current_mo_basis_index] = atom_index_of_ao;
                }
                current_mo_basis_index += 1;
            }
        }

        assert_eq!(basis_to_atom.len(), self.num_basis, 
                   "Mismatch between basis_to_atom mapping ({}) and num_basis ({})", 
                   basis_to_atom.len(), self.num_basis);

        // Calculate Energy-Weighted Density Matrix W = P * S * P
        let S = &self.overlap_matrix;
        let P = &self.density_matrix;
        let PS = P * S;
        let W = &PS * P; // Energy-weighted density matrix W = P*S*P
        
        // Calculate Pulay forces for each atom
        for atom_idx in 0..self.num_atoms {
            let mut pulay_force_atom = Vector3::zeros();
            let mut kinetic_contrib = Vector3::zeros();
            let mut nuclear_contrib = Vector3::zeros();
            let mut overlap_contrib = Vector3::zeros();
            
            // Contribution from dH_core/dR_A (kinetic and nuclear attraction basis derivatives)
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    let p_ij = self.density_matrix[(i, j)];
                    if p_ij.abs() < 1e-12 { continue; }

                    // Kinetic energy derivatives - fixed atom_idx interpretation
                    if basis_to_atom[i] == atom_idx { // If basis i is on atom_idx
                        let dt_dr_i = B::BasisType::dTab_dR(&self.mo_basis[i], &self.mo_basis[j], 0); // deriv w.r.t center of basis i (idx=0)
                        let contrib = -p_ij * dt_dr_i;
                        kinetic_contrib += contrib;
                        pulay_force_atom += contrib;
                    }
                    if basis_to_atom[j] == atom_idx && i != j { // If basis j is on atom_idx (and not double counting if i also was)
                        let dt_dr_j = B::BasisType::dTab_dR(&self.mo_basis[i], &self.mo_basis[j], 1); // deriv w.r.t center of basis j (idx=1)
                        let contrib = -p_ij * dt_dr_j;
                        kinetic_contrib += contrib;
                        pulay_force_atom += contrib;
                    }

                    // Nuclear attraction Pulay forces - fixed atom_idx interpretation
                    for k_nucl in 0..self.num_atoms { // Sum over all nuclei K for V_μν,K
                        if basis_to_atom[i] == atom_idx {
                            let dv_dr_basis_i = B::BasisType::dVab_dRbasis(
                                &self.mo_basis[i],
                                &self.mo_basis[j],
                                self.coords[k_nucl],
                                self.elems[k_nucl].get_atomic_number() as u32,
                                0, // Derivative w.r.t center of basis i (idx=0)
                            );
                            let contrib = -p_ij * dv_dr_basis_i;
                            nuclear_contrib += contrib;
                            pulay_force_atom += contrib;
                        }
                        if basis_to_atom[j] == atom_idx && i != j {
                             let dv_dr_basis_j = B::BasisType::dVab_dRbasis(
                                &self.mo_basis[i],
                                &self.mo_basis[j],
                                self.coords[k_nucl],
                                self.elems[k_nucl].get_atomic_number() as u32,
                                1, // Derivative w.r.t center of basis j (idx=1)
                            );
                            let contrib = -p_ij * dv_dr_basis_j;
                            nuclear_contrib += contrib;
                            pulay_force_atom += contrib;
                        }
                    }
                }
            }

            // Contribution from dS/dR_A (overlap matrix derivatives weighted by W)
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    let w_ij = W[(i,j)];
                    if w_ij.abs() < 1e-12 { continue; }

                    if basis_to_atom[i] == atom_idx {
                        let ds_dr_i = B::BasisType::dSab_dR(&self.mo_basis[i], &self.mo_basis[j], 0); // dS_ij / dR_i (idx=0)
                        let contrib = w_ij * ds_dr_i; // Term is +Tr(W * dS/dR_A) for force = -dE/dR_A
                        overlap_contrib += contrib;
                        pulay_force_atom += contrib;
                    }
                    if basis_to_atom[j] == atom_idx && i !=j { // Avoid double counting diagonal dS_ii/dR_i if i=j
                        let ds_dr_j = B::BasisType::dSab_dR(&self.mo_basis[i], &self.mo_basis[j], 1); // dS_ij / dR_j (idx=1)
                        let contrib = w_ij * ds_dr_j; // Term is +Tr(W * dS/dR_A)
                        overlap_contrib += contrib;
                        pulay_force_atom += contrib;
                    }
                }
            }
            
            println!("  Pulay force breakdown for atom {}:", atom_idx);
            println!("    Kinetic contribution: [{:.6e}, {:.6e}, {:.6e}]", kinetic_contrib.x, kinetic_contrib.y, kinetic_contrib.z);
            println!("    Nuclear contribution: [{:.6e}, {:.6e}, {:.6e}]", nuclear_contrib.x, nuclear_contrib.y, nuclear_contrib.z);
            println!("    Overlap contribution: [{:.6e}, {:.6e}, {:.6e}]", overlap_contrib.x, overlap_contrib.y, overlap_contrib.z);
            println!("    Total Pulay force: [{:.6e}, {:.6e}, {:.6e}]", pulay_force_atom.x, pulay_force_atom.y, pulay_force_atom.z);
            
            // Step 4b: Two-electron Pulay forces
            let mut two_electron_pulay_contrib = Vector3::zeros();
            for i in 0..self.num_basis {
                for j in 0..self.num_basis {
                    let p_ij = self.density_matrix[(i, j)];
                    // Could optimize by checking if p_ij or p_kl are zero early.

                    for k in 0..self.num_basis {
                        for l in 0..self.num_basis {
                            let p_kl = self.density_matrix[(k, l)];
                            let p_ik = self.density_matrix[(i, k)];
                            let p_jl = self.density_matrix[(j, l)]; // Note: P is symmetric, P_jl == P_lj

                            // Contribution: - P_ij P_kl * d(ij|kl)/dR_A + 0.5 * P_ik P_jl * d(ik|jl)/dR_A
                            // where d(ab|cd)/dR_A = sum_{q in {a,b,c,d} if q is on atom_A} d(ab|cd)/dR_q

                            let basis_i = &self.mo_basis[i];
                            let basis_j = &self.mo_basis[j];
                            let basis_k = &self.mo_basis[k];
                            let basis_l = &self.mo_basis[l];

                            let mut term_J_deriv = Vector3::zeros();
                            let mut term_K_deriv = Vector3::zeros();

                            // Derivative d(ij|kl)/dR_A
                            if basis_to_atom[i] == atom_idx { term_J_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_j, basis_k, basis_l, 0); }
                            if basis_to_atom[j] == atom_idx { term_J_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_j, basis_k, basis_l, 1); }
                            if basis_to_atom[k] == atom_idx { term_J_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_j, basis_k, basis_l, 2); }
                            if basis_to_atom[l] == atom_idx { term_J_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_j, basis_k, basis_l, 3); }
                            
                            // Derivative d(ik|jl)/dR_A
                            if basis_to_atom[i] == atom_idx { term_K_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_k, basis_j, basis_l, 0); }
                            if basis_to_atom[k] == atom_idx { term_K_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_k, basis_j, basis_l, 1); } // k is 2nd arg to dJKabcd_dRbasis here
                            if basis_to_atom[j] == atom_idx { term_K_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_k, basis_j, basis_l, 2); } // j is 3rd arg
                            if basis_to_atom[l] == atom_idx { term_K_deriv += B::BasisType::dJKabcd_dRbasis(basis_i, basis_k, basis_j, basis_l, 3); } // l is 4th arg
                            
                            if p_ij.abs() > 1e-12 && p_kl.abs() > 1e-12 {
                                two_electron_pulay_contrib -= p_ij * p_kl * term_J_deriv;
                            }
                            if p_ik.abs() > 1e-12 && p_jl.abs() > 1e-12 {
                                two_electron_pulay_contrib += 0.5 * p_ik * p_jl * term_K_deriv;
                            }
                        }
                    }
                }
            }
            pulay_force_atom += two_electron_pulay_contrib;
            // End of Step 4b: Two-electron Pulay forces
            
            forces[atom_idx] += pulay_force_atom;
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
