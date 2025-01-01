// a simple implementation of the scf trait

extern crate nalgebra as na;

use std::collections::HashMap;
use na::Vector3;
use periodic_table_on_an_enum::Element;
use crate::scf::SCF;
use basis::basis::Basis;

struct SimpleSCF {
    num_atoms: u32,
    coords: Vec<Vector3<f64>>,
}

impl SCF for SimpleSCF {
    // fn init_basis(&self, basis: HashMap<&str, &dyn Basis>) {
    //     println!("Initializing basis...");
    // }
    
    fn init_geometry(&self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>) {
        println!("Initializing geometry...");
    }

    fn init_density_matrix(&self) {
        println!("Initializing density matrix...");
    }

    fn init_fock_matrix(&self) {
        println!("Initializing Fock matrix...");
    }
}