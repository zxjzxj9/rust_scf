extern crate nalgebra as na;

use std::collections::HashMap;
use basis::basis::AOBasis;
use na::Vector3;
use periodic_table_on_an_enum::Element;

pub trait SCF {
    type BasisType: AOBasis;
    
    fn init_basis(&mut self, elems: &Vec<Element>, basis: HashMap<&str, &Self::BasisType>);
    fn init_geometry(&mut self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>);
    fn init_density_matrix(&mut self);
    fn init_fock_matrix(&mut self);
    fn scf_cycle(&mut self);
}