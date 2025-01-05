// a simple implementation of the scf trait

extern crate nalgebra as na;

use std::collections::HashMap;
use na::{Vector3, DVector, DMatrix};
use periodic_table_on_an_enum::Element;
use crate::scf::SCF;
use basis::basis::AOBasis;

pub struct SimpleSCF<B: AOBasis> {
    num_atoms: usize,
    num_basis: usize,
    mo_basis: Vec<B>,
    coords: Vec<Vector3<f64>>,
    // use nalgebra for the density matrix, fock matrix, etc.
    coeffs: DVector<f64>,
    density_matrix: DMatrix<f64>,
}

// implement the scf trait for the simple scf struct
impl <B: AOBasis + Clone> SCF for SimpleSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        self.num_atoms = elems.len();
        println!("Initializing basis set...");

        self.num_basis = 0;
        for elem in elems {
            let b = *basis.get(elem.get_symbol()).unwrap();
            println!("Element: {}, basis size: {}", elem.get_symbol(), b.basis_size());
            self.mo_basis.push(b.clone());
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
            println!("Element: {}, coords: {:?}", elems[i].get_symbol(), coords[i]);
            self.mo_basis[i].set_center(coords[i]);
        }
    }

    fn init_density_matrix(&mut self) {
        println!("Initializing density matrix...");
    }

    fn init_fock_matrix(&mut self) {
        println!("Initializing Fock matrix...");
    }

    fn scf_cycle(&mut self) {
        println!("Performing SCF cycle...");
    }
}