extern crate nalgebra as na;
use basis::basis::AOBasis;
use na::Vector3;
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};

pub trait SCF {
    type BasisType: AOBasis;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>);
    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>);
    fn init_density_matrix(&mut self);
    fn update_density_matrix(&mut self);
    fn init_fock_matrix(&mut self);
    fn scf_cycle(&mut self);
}

pub(crate) struct DIIS {
    error_vectors: Vec<DVector<f64>>,
    fock_matrices: Vec<DMatrix<f64>>,
    max_subspace_size: usize,
}

impl DIIS {
    pub fn new(max_subspace_size: usize) -> Self {
        DIIS {
            error_vectors: Vec::new(),
            fock_matrices: Vec::new(),
            max_subspace_size,
        }
    }

    pub(crate) fn update(&mut self, error_vector: DVector<f64>, fock_matrix: DMatrix<f64>) {
        if self.error_vectors.len() >= self.max_subspace_size {
            self.error_vectors.remove(0);
            self.fock_matrices.remove(0);
        }
        self.error_vectors.push(error_vector);
        self.fock_matrices.push(fock_matrix);
    }

    pub(crate) fn extrapolate(&self) -> DMatrix<f64> {
        let n = self.error_vectors.len();
        if n == 0 {
            panic!("DIIS: No stored error vectors available for extrapolation.");
        }

        // Build the extended DIIS matrix of size (n+1) x (n+1)
        let mut B = DMatrix::zeros(n + 1, n + 1);
        for i in 0..n {
            for j in 0..n {
                // Dot product of error vectors
                B[(i, j)] = self.error_vectors[i].dot(&self.error_vectors[j]);
            }
            // Set the off-diagonals for the constraint
            B[(i, n)] = -1.0;
            B[(n, i)] = -1.0;
        }
        B[(n, n)] = 0.0;

        // Build the right-hand side vector with the constraint
        let mut rhs = DVector::zeros(n + 1);
        rhs[n] = -1.0;

        // Solve the linear system B * x = rhs for the coefficients
        let x = B.lu().solve(&rhs)
            .expect("DIIS extrapolation failed: could not solve the DIIS equations.");

        // The desired DIIS coefficients are the first n entries
        let coeffs = x.rows(0, n);

        // Extrapolate the Fock matrix as a weighted sum of stored Fock matrices
        let mut fock_extrapolated = DMatrix::zeros(
            self.fock_matrices[0].nrows(),
            self.fock_matrices[0].ncols()
        );
        for i in 0..n {
            fock_extrapolated += &self.fock_matrices[i] * coeffs[i];
        }

        fock_extrapolated
    }
}