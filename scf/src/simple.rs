// a simple implementation of the scf trait

extern crate nalgebra as na;

use std::collections::HashMap;
use na::Vector3;
use periodic_table_on_an_enum::Element;
use crate::scf::SCF;
use basis::basis::AOBasis;

pub struct SimpleSCF<B: AOBasis> {
    num_atoms: u32,
    mo_basis: Vec<B>,
    coords: Vec<Vector3<f64>>,
}

// implement the scf trait for the simple scf struct
impl <B: AOBasis> SCF for SimpleSCF<B> {
    type BasisType = B;

    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        self.num_atoms = elems.len() as u32;
        println!("Initializing basis set...");

        for elem in elems {
            println!("Element: {}", elem);
        }

        println!("Number of atoms: {}", self.num_atoms);
        println!("Number of basis functions: {}", self.mo_basis.len());
    }

    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        println!("Initializing geometry...");
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