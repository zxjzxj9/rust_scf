extern crate nalgebra as na;

use std::collections::HashMap;
use basis::basis::Basis;
use na::Vector3;
use periodic_table_on_an_enum::Element;

pub trait SCF {
    type BasisType: Basis;
    
    fn init_basis(&self, basis: HashMap<&str, &Self::BasisType>);
    
    fn init_geometry(&self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>);
    fn init_density_matrix(&self);
    fn init_fock_matrix(&self);
}