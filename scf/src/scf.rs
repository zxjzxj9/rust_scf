extern crate nalgebra as na;
use basis::basis::AOBasis;
use na::Vector3;
use nalgebra::{DMatrix, DVector};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;


pub trait SCF {
    type BasisType: AOBasis;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>);
    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>);
    fn init_density_matrix(&mut self);
    fn update_density_matrix(&mut self);
    fn init_fock_matrix(&mut self);
    fn scf_cycle(&mut self);
    fn calculate_total_energy(&self) -> f64; // Add this method to calculate Hellman-Feynman forces
    fn calculate_forces(&self) -> Vec<Vector3<f64>>;
}

pub struct DIIS {
    error_matrices: Vec<DMatrix<f64>>,
    fock_matrices: Vec<DMatrix<f64>>,
    max_subspace_size: usize,
}

impl DIIS {
    pub fn new(max_subspace_size: usize) -> Self {
        DIIS {
            error_matrices: Vec::new(),
            fock_matrices: Vec::new(),
            max_subspace_size,
        }
    }

    // Calculate DIIS error matrix: FDS - SDF (transformed commutator)
    pub fn calculate_error_matrix(
        &self,
        fock: &DMatrix<f64>,
        density: &DMatrix<f64>,
        overlap: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        // Calculate FDS
        let fds = fock * density * overlap;
        // Calculate SDF
        let sdf = overlap * density * fock;
        // Error matrix is the commutator: FDS - SDF
        fds - sdf
    }

    pub fn update(
        &mut self,
        fock_matrix: DMatrix<f64>,
        density_matrix: &DMatrix<f64>,
        overlap_matrix: &DMatrix<f64>,
    ) {
        // Calculate the error matrix
        let error = self.calculate_error_matrix(&fock_matrix, density_matrix, overlap_matrix);

        if self.error_matrices.len() >= self.max_subspace_size {
            self.error_matrices.remove(0);
            self.fock_matrices.remove(0);
        }

        self.error_matrices.push(error);
        self.fock_matrices.push(fock_matrix);
    }

    pub fn extrapolate(&self) -> Option<DMatrix<f64>> {
        let n = self.error_matrices.len();
        if n == 0 {
            return None;
        }

        // Build the B matrix for DIIS equations
        let mut b = DMatrix::zeros(n + 1, n + 1);
        for i in 0..n {
            for j in 0..n {
                // Flatten matrices to vectors for dot product
                let e_i_flat: Vec<f64> = self.error_matrices[i].iter().cloned().collect();
                let e_j_flat: Vec<f64> = self.error_matrices[j].iter().cloned().collect();

                // Calculate dot product
                b[(i, j)] = e_i_flat.iter().zip(&e_j_flat).map(|(a, b)| a * b).sum();
            }
            // Set constraint rows/columns
            b[(i, n)] = -1.0;
            b[(n, i)] = -1.0;
        }
        b[(n, n)] = 0.0;

        // Right-hand side vector
        let mut rhs = DVector::zeros(n + 1);
        rhs[n] = -1.0;

        // Solve the DIIS equations
        let lu_result = b.lu();
        let coeffs = match lu_result.solve(&rhs) {
            Some(x) => x,
            None => return None,
        };

        // Extrapolate the Fock matrix
        let mut fock_extrapolated =
            DMatrix::zeros(self.fock_matrices[0].nrows(), self.fock_matrices[0].ncols());

        for i in 0..n {
            fock_extrapolated += &self.fock_matrices[i] * coeffs[i];
        }

        Some(fock_extrapolated)
    }
}
