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

    fn init_basis(&self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>) {
        println!("Initializing basis set...");
    }

    fn init_geometry(&self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        println!("Initializing geometry...");
    }

    fn init_density_matrix(&self) {
        println!("Initializing density matrix...");
    }

    fn init_fock_matrix(&self) {
        println!("Initializing Fock matrix...");
    }

    fn scf_cycle(&self) {
        println!("Performing SCF cycle...");
    }
}