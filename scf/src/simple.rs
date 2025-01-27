// a simple implementation of the scf trait

extern crate nalgebra as na;

use crate::scf::SCF;
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, Vector3, SymmetricEigen};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use nalgebra::{Const, Dyn};

pub struct SimpleSCF<B: AOBasis> {
    num_atoms: usize,
    num_basis: usize,
    ao_basis: Vec<Arc<Mutex<B>>>,
    mo_basis: Vec<Arc<B::BasisType>>,
    coords: Vec<Vector3<f64>>,
    elems: Vec<Element>,
    // use nalgebra for the density matrix, fock matrix, etc.
    coeffs: DMatrix<f64>,
    integral_matrix:  DMatrix<f64>, // <ij|1/r12|kl>
    fock_matrix: DMatrix<f64>,
    overlap_matrix: DMatrix<f64>,
    e_level: DVector<f64>,
    MAX_CYCLE: usize,
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

// implement the scf trait for the simple scf struct
impl<B: AOBasis + Clone> SCF for SimpleSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        self.elems = elems.clone();
        self.num_atoms = elems.len();
        println!("Initializing basis set...");

        self.num_basis = 0;
        for elem in elems {
            let b = *basis.get(elem.get_symbol()).unwrap();
            // b.get_basis().iter().for_each(|tb| {
            //     // println!("tb: {:?}", tb);
            // });
            println!(
                "Element: {}, basis size: {}",
                elem.get_symbol(),
                b.basis_size()
            );

            let b_arc = Arc::new(Mutex::new((*b).clone()));

            // Push to ao_basis
            self.ao_basis.push(b_arc.clone());
        }

        println!("Number of atoms: {}", self.num_atoms);
        // println!("Number of basis functions: {}", self.mo_basis.len());
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
            println!("Center: {:?}", self.ao_basis[i].lock().unwrap().get_center());
        }
        self.coords = coords.clone();

        // Rebuild the molecular orbital basis with updated coordinates
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
        // core hamiltonian initialization
        // self.density_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
        self.fock_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);
        self.overlap_matrix = DMatrix::from_element(self.num_basis, self.num_basis, 0.0);

        // solve Hc = ScE to initialize density matrix
        // let eig =
        //     self.overlap_matrix.clone().try_symmetric_eigen(1e-6, 1000).unwrap();
        // let eigvecs = eig.eigenvectors;
        // let eigvals = eig.eigenvalues;

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

        // print overlap matrix for debugging
        // println!("Overlap Matrix: {:?}", self.overlap_matrix);

        let l = self.overlap_matrix.clone().cholesky().unwrap();
        let l_inv = l.inverse();
        let f_prime = l_inv.clone() * self.fock_matrix.clone_owned() * l_inv.clone().transpose();
        let eig = f_prime.clone().try_symmetric_eigen(1e-6, 1000).unwrap();
        let eigvecs = l_inv.clone().transpose() * eig.eigenvectors;
        let eigvals = eig.eigenvalues;
        self.coeffs = l_inv * eigvecs;
        self.e_level = eigvals;

        // print!("Coeffs shape: {:?}", self.coeffs.shape());

        // print energy levels
        println!("Energy levels: {:?}", self.e_level);
    }

    fn init_fock_matrix(&mut self) {
        println!("Initializing Fock matrix...");

        // suppose all the shell is occupied
        self.integral_matrix= DMatrix::from_element(
            self.num_basis * self.num_basis,
            self.num_basis * self.num_basis, 0.0);

        // Precompute two-electron integrals and store them in a hashmap or tensor
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                for k in 0..self.num_basis {
                    for l in 0..self.num_basis {
                        let integral_ijkl = B::BasisType::JKabcd(
                            &self.mo_basis[i], &self.mo_basis[j],
                            &self.mo_basis[k], &self.mo_basis[l]);
                        let integral_ikjl = B::BasisType::JKabcd(
                            &self.mo_basis[i], &self.mo_basis[k],
                            &self.mo_basis[j], &self.mo_basis[l]);
                        let row = i * self.num_basis + j;
                        let col = k * self.num_basis + l;
                        // self.integral_matrix[(row, col)] = integral;
                        self.integral_matrix[(row, col)] = integral_ijkl - 0.5 * integral_ikjl;
                    }
                }
            }
        }

    }

    fn scf_cycle(&mut self) {
        println!("Performing SCF cycle...");

        // 1. calculate all the electron numbers and number of occupied orbitals
        let total_electrons: usize = self.elems.iter()
            .map(|e| e.get_atomic_number() as usize)
            .sum();

        // 2. calculate the number of occupied orbitals
        let n_occ = total_electrons / 2;

        for _ in 0..self.MAX_CYCLE {
            // 0. truncate the coeffs matrix to the number of occupied orbitals
            self.coeffs = <DMatrix<f64>>::from(self.coeffs.columns(0, n_occ));

            // 1. close shell density matrix
            let new_density_matrix = 2.0 * &self.coeffs * self.coeffs.transpose();

            // 2. flatten density matrix as the column vector (num_basis² x 1)
            let density_flattened = new_density_matrix.reshape_generic(
                Dyn(self.num_basis * self.num_basis),
                Dyn(1)
            );

            // 3. calculate g matrix (for electron repulsion)
            let g_matrix_flattened = &self.integral_matrix * &density_flattened;

            // 4. reshape it back to matrix form
            let g_matrix = g_matrix_flattened.reshape_generic(
                Dyn(self.num_basis),
                Dyn(self.num_basis)
            );

            // 5. construct fock matrix
            let hamiltonian = self.fock_matrix.clone() + g_matrix;

            // 6. orthogonalize the fock matrix
            let l = self.overlap_matrix.clone().cholesky().unwrap();
            let l_inv = l.inverse();
            let f_prime = l_inv.clone() * &hamiltonian * l_inv.transpose();
            let eig = f_prime.try_symmetric_eigen(1e-6, 1000).unwrap();
            let eigvecs = l_inv.transpose() * eig.eigenvectors;
            self.coeffs = l_inv * eigvecs;
            self.e_level = eig.eigenvalues;

            println!("Energy levels: {:?}", self.e_level);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use periodic_table_on_an_enum::Element;
    use std::collections::HashMap;
    use basis::cgto::Basis631G;

    fn fetch_basis(atomic_symbol: &str) -> Basis631G {
        let url = format!(
            "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
            atomic_symbol);
        let basis_str = reqwest::blocking::get(url).unwrap().text().unwrap();
        println!("Basis set for {}: {}", atomic_symbol, basis_str);
        Basis631G::parse_nwchem(&basis_str)
    }

    #[test]
    fn test_simple_scf() {
        let mut scf = SimpleSCF::new(); // h2o coordinates in Bohr
        let h2o_coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.809),
            Vector3::new(1.443, 0.0, -0.453),
        ];
        let h2o_elems = vec![
            Element::Oxygen,
            Element::Hydrogen,
            Element::Hydrogen,
        ];

        let mut basis = HashMap::new();

        // download basis first
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