// a simple implementation of the scf trait

extern crate nalgebra as na;

use crate::scf::SCF;
use basis::basis::{AOBasis, Basis};
use na::{DMatrix, DVector, Vector3};
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
    // use nalgebra for the density matrix, fock matrix, etc.
    coeffs: DMatrix<f64>,
    // density_matrix: DMatrix<f64>,
    integral_matrix:  DMatrix<f64>, // <ij|1/r12|kl>
    fock_matrix: DMatrix<f64>,
    overlap_matrix: DMatrix<f64>,
    e_level: DVector<f64>,
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
            coeffs: DMatrix::from_element(0, 0, 0.0),
            integral_matrix: DMatrix::from_element(0, 0, 0.0),
            fock_matrix: DMatrix::from_element(0, 0, 0.0),
            overlap_matrix: DMatrix::from_element(0, 0, 0.0),
            e_level: DVector::from_element(0, 0.0),
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
            println!(
                "Element: {}, basis size: {}",
                elem.get_symbol(),
                b.basis_size()
            );

            let b_arc = Arc::new(Mutex::new((*b).clone()));
            // Push to ao_basis
            self.ao_basis.push(b_arc.clone());

            for tb in b_arc.lock().unwrap().get_basis() {
                // If tb is Arc<...> already:
                self.mo_basis.push(tb.clone());
            }

            self.num_basis += b.basis_size();
        }

        println!("Number of atoms: {}", self.num_atoms);
        println!("Number of basis functions: {}", self.mo_basis.len());
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
        }
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

        let l = self.overlap_matrix.clone().cholesky().unwrap();
        let l_inv = l.inverse();
        let f_prime = l_inv.clone() * self.fock_matrix.clone_owned() * l_inv.clone().transpose();
        let eig = f_prime.clone().try_symmetric_eigen(1e-6, 1000).unwrap();
        let eigvecs = l_inv.clone().transpose() * eig.eigenvectors;
        let eigvals = eig.eigenvalues;
        self.coeffs = l_inv * eigvecs;
        self.e_level = eigvals;

        // print energy levels
        println!("Energy levels: {:?}", self.e_level);
    }

    fn init_fock_matrix(&mut self) {
        println!("Initializing Fock matrix...");

        // Precompute two-electron integrals and store them in a hashmap or tensor
        for i in 0..self.num_basis {
            for j in 0..self.num_basis {
                for k in 0..self.num_basis {
                    for l in 0..self.num_basis {
                        let integral = B::BasisType::JKabcd(
                            &self.mo_basis[i], &self.mo_basis[j],
                            &self.mo_basis[k], &self.mo_basis[l]);
                        let row = i * self.num_basis + j;
                        let col = k * self.num_basis + l;
                        self.integral_matrix[(row, col)] = integral;
                    }
                }
            }
        }

    }

    fn scf_cycle(&mut self) {
        // println!("Performing SCF cycle...");
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
        let url =
            "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}"
                .format(atomic_symbol);
        let basis_str = reqwest::blocking::get(url).unwrap().text().unwrap();
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
        basis.insert("O", &fetch_basis("O"));
        basis.insert("H", &fetch_basis("H"));



        scf.init_basis(&h2o_elems, basis);
        scf.init_geometry(&h2o_coords, &h2o_elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
    }
}