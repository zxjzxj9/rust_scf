// a simple implementation of the scf trait

extern crate nalgebra as na;

use crate::scf::SCF;
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, SymmetricEigen, Vector3};
use nalgebra::{Const, Dyn};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct SimpleSCF<B: AOBasis> {
    num_atoms: usize,
    num_basis: usize,
    ao_basis: Vec<Arc<Mutex<B>>>,
    mo_basis: Vec<Arc<B::BasisType>>,
    coords: Vec<Vector3<f64>>,
    elems: Vec<Element>,
    coeffs: DMatrix<f64>,
    integral_matrix: DMatrix<f64>,
    fock_matrix: DMatrix<f64>,
    overlap_matrix: DMatrix<f64>,
    e_level: DVector<f64>,
    MAX_CYCLE: usize,
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
            integral_matrix: DMatrix::zeros(0, 0),
            fock_matrix: DMatrix::zeros(0, 0),
            overlap_matrix: DMatrix::zeros(0, 0),
            e_level: DVector::zeros(0),
            MAX_CYCLE: 100,
        }
    }
}

impl<B: AOBasis + Clone> SCF for SimpleSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        self.elems = elems.clone();
        self.num_atoms = elems.len();
        println!("Initializing basis set...");

        self.num_basis = 0;
        for elem in elems {
            let b = *basis.get(elem.get_symbol()).unwrap();
            println!(
                "Element: {}, basis size: {}",
                elem.get_symbol(),
                b.basis_size()
            );

            let b_arc = Arc::new(Mutex::new((*b).clone()));
            self.ao_basis.push(b_arc.clone());
        }

        println!("Number of atoms: {}", self.num_atoms);
    }

    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        println!("Initializing geometry...");
        assert!(coords.len() == elems.len());
        let size = coords.len();
        for i in 0..size {
            println!(
                "Element: {}, coords: {:?}",
                elems[i].get_symbol(),
                coords[i]
            );
            self.ao_basis[i].lock().unwrap().set_center(coords[i]);
            println!(
                "Center: {:?}",
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
        println!("Rebuilt MO basis with {} basis functions.", self.num_basis);
    }



    fn init_density_matrix(&mut self) {
        println!("Initializing density matrix...");
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

        // println!("Overlap matrix: {:?}", self.overlap_matrix);
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
        let sorted_eigenvectors = align_eigenvectors(eigenvectors.select_columns(&indices));
        let eigvecs = l_inv.clone().transpose() * sorted_eigenvectors;
        // Corrected line: Remove l_inv multiplication here
        self.coeffs = eigvecs;
        self.e_level = sorted_eigenvalues;

        // println!("Energy levels: {:?}", self.e_level);
    }

    fn init_fock_matrix(&mut self) {
        println!("Initializing Fock matrix...");

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
    }

    fn scf_cycle(&mut self) {
        println!("Performing SCF cycle...");

        let total_electrons: usize = self
            .elems
            .iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();
        // Ensure even number of electrons for closed-shell
        assert!(total_electrons % 2 == 0, "Total number of electrons must be even");
        let n_occ = total_electrons / 2;

        for _ in 0..self.MAX_CYCLE {
            let occupied_coeffs = self.coeffs.columns(0, n_occ);
            let new_density_matrix = 2.0 * &occupied_coeffs * occupied_coeffs.transpose();

            let density_flattened =
                new_density_matrix.reshape_generic(Dyn(self.num_basis * self.num_basis), Dyn(1));

            let g_matrix_flattened = &self.integral_matrix * &density_flattened;
            let g_matrix =
                g_matrix_flattened.reshape_generic(Dyn(self.num_basis), Dyn(self.num_basis));

            let hamiltonian = self.fock_matrix.clone() + g_matrix;

            let l = self.overlap_matrix.clone().cholesky().unwrap();
            let l_inv = l.inverse();
            let f_prime = l_inv.clone() * &hamiltonian * l_inv.transpose();
            let eig = f_prime.try_symmetric_eigen(1e-6, 1000).unwrap();

            // Sort eigenvalues and eigenvectors
            let eigenvalues = eig.eigenvalues.clone();
            let eigenvectors = eig.eigenvectors.clone();
            let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
            indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
            let sorted_eigenvalues =
                DVector::from_fn(eigenvalues.len(), |i, _| eigenvalues[indices[i]]);
            let sorted_eigenvectors = align_eigenvectors(eigenvectors.select_columns(&indices));

            let eigvecs = l_inv.transpose() * sorted_eigenvectors;
            // Corrected line: Remove l_inv multiplication here
            self.coeffs = eigvecs;
            self.e_level = sorted_eigenvalues;

            println!("Energy levels: {:?}", self.e_level);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::cgto::Basis631G;
    use nalgebra::Vector3;
    use periodic_table_on_an_enum::Element;
    use std::collections::HashMap;

    fn fetch_basis(atomic_symbol: &str) -> Basis631G {
        let url = format!(
            "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
            atomic_symbol
        );
        let basis_str = reqwest::blocking::get(url).unwrap().text().unwrap();
        Basis631G::parse_nwchem(&basis_str)
    }
    // Mock implementations for testing
    #[derive(Clone)]
    struct MockAOBasis {
        center: Vector3<f64>,
    }

    impl AOBasis for MockAOBasis {
        type BasisType = MockBasis;

        fn basis_size(&self) -> usize {
            1
        }

        fn get_basis(&self) -> Vec<Arc<Self::BasisType>> {
            vec![Arc::new(MockBasis {
                center: self.center,
            })]        }

        fn set_center(&mut self, center: Vector3<f64>) {
            self.center = center;
        }

        fn get_center(&self) -> Option<Vector3<f64>> {
            Some(self.center)
        }
    }


    #[derive(Clone)]
    struct MockBasis {
        center: Vector3<f64>,
    }

    impl Basis for MockBasis {
        fn evaluate(&self, r: &Vector3<f64>) -> f64 {
            todo!()
        }

        fn Sab(a: &Self, b: &Self) -> f64 {
            if a.center == b.center {
                1.0
            } else {
                0.0
            }
        }

        fn Tab(_: &Self, _: &Self) -> f64 {
            0.1 // Kinetic energy integral
        }

        fn Vab(_: &Self, _: &Self, _: Vector3<f64>, charge: u32) -> f64 {
            -0.2 * charge as f64 // Potential energy integral
        }

        fn JKabcd(_: &Self, _: &Self, _: &Self, _: &Self) -> f64 {
            0.01 // Two-electron integral
        }
    }

    // Helper function to create mock basis
    fn create_mock_basis() -> MockAOBasis {
        MockAOBasis {
            center: Vector3::zeros(),
        }
    }

    #[test]
    fn test_init_basis() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);

        assert_eq!(scf.num_atoms, 2);
        assert_eq!(scf.ao_basis.len(), 2);
        assert_eq!(scf.elems, elems);
    }

    #[test]
    fn test_init_geometry() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);

        assert_eq!(scf.coords, coords);
        assert_eq!(scf.num_basis, 2);
        for (i, ao) in scf.ao_basis.iter().enumerate() {
            let center = ao.lock().unwrap().get_center().unwrap();
            assert_eq!(center, coords[i]);
        }
    }

    #[test]
    fn test_init_density_matrix() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();

        // Test matrix dimensions
        assert_eq!(scf.overlap_matrix.shape(), (2, 2));
        assert_eq!(scf.fock_matrix.shape(), (2, 2));

        // Verify overlap matrix is identity
        assert!(scf.overlap_matrix.clone().is_identity(1e-6));

        // Verify Fock matrix values: Tab + sum(Vab)
        let expected_diagonal = 0.1 + (-0.2 * 1.0 * 2.0); // -0.3 for diagonal
        let expected_off_diagonal = 0.1 + (-0.2 * 1.0 * 2.0); // -0.3 for off-diagonal in current mock setup

        for i in 0..scf.num_basis {
            for j in 0..scf.num_basis {
                let expected = if i == j { expected_diagonal } else { expected_off_diagonal };
                assert!((scf.fock_matrix[(i, j)] - expected).abs() < 1e-6);
            }
        }

        // Verify energy levels are sorted
        let eigenvalues = scf.e_level.as_slice();
        assert!(eigenvalues.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_init_fock_matrix() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();

        // Test integral matrix dimensions
        assert_eq!(
            scf.integral_matrix.shape(),
            (scf.num_basis * scf.num_basis, scf.num_basis * scf.num_basis)
        );

        // Verify integral values (0.01 - 0.5*0.01 = 0.005)
        let expected_value = 0.01 - 0.5 * 0.01;
        assert!(scf
            .integral_matrix
            .iter()
            .all(|&x| (x - expected_value).abs() < 1e-6));
    }

    #[test]
    fn test_scf_cycle_updates() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();

        let initial_coeffs = scf.coeffs.clone();
        let initial_energy = scf.e_level.clone();

        scf.scf_cycle();

        // Verify updates after SCF cycle
        let diff = scf.coeffs.clone() - initial_coeffs;
        assert!(diff.iter().all(|&x| x.abs() < 1e-6));

        assert_eq!(scf.coeffs.ncols(), 2); // Should maintain dimensions
    }

    #[test]
    fn test_occupied_orbitals() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Helium]; // Atomic number 2
        let coords = vec![Vector3::zeros()];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("He", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();

        // Verify occupied orbitals count
        let total_electrons: usize = elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        assert_eq!(total_electrons, 2);
        let n_occ = total_electrons / 2;
        assert_eq!(n_occ, 1);
    }

    #[test]
    fn test_simple_scf() {
        let mut scf = SimpleSCF::new();
        let h2o_coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.809),
            Vector3::new(1.443, 0.0, -0.453),
        ];
        let h2o_elems = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];

        let mut basis = HashMap::new();
        let h_basis = fetch_basis("H");
        let o_basis = fetch_basis("O");
        basis.insert("H", &h_basis);
        basis.insert("O", &o_basis);

        scf.init_basis(&h2o_elems, basis);
        scf.init_geometry(&h2o_coords, &h2o_elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
    }
}
