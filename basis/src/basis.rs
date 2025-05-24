/*  Abstraction of the basis trait

   Author: Victor Zhang
   Date: 2024/12/31
*/

// define the basis trait

#![allow(non_snake_case, unused)]

use nalgebra::Vector3;
use std::sync::Arc;

pub trait Basis {
    fn evaluate(&self, r: &Vector3<f64>) -> f64;

    fn Sab(a: &Self, b: &Self) -> f64;
    fn Tab(a: &Self, b: &Self) -> f64;

    fn Vab(a: &Self, b: &Self, R: Vector3<f64>, Z: u32) -> f64;

    // Add this method to calculate derivative of nuclear attraction integral
    fn dVab_dR(a: &Self, b: &Self, R: Vector3<f64>, Z: u32) -> Vector3<f64>;

    fn JKabcd(a: &Self, b: &Self, c: &Self, d: &Self) -> f64;
    
    // Add derivatives for two-electron integrals w.r.t. nuclear positions
    fn dJKabcd_dR(a: &Self, b: &Self, c: &Self, d: &Self, R: Vector3<f64>) -> Vector3<f64>;
    
    // Add derivatives for overlap and kinetic integrals w.r.t. basis function centers (Pulay forces)
    fn dSab_dR(a: &Self, b: &Self, atom_idx: usize) -> Vector3<f64>;
    fn dTab_dR(a: &Self, b: &Self, atom_idx: usize) -> Vector3<f64>;
    fn dVab_dRbasis(a: &Self, b: &Self, R: Vector3<f64>, Z: u32, atom_idx: usize) -> Vector3<f64>;
    fn dJKabcd_dRbasis(a: &Self, b: &Self, c: &Self, d: &Self, atom_idx: usize) -> Vector3<f64>;
}

// define a trait for atomic orbital basis sets
pub trait AOBasis {
    type BasisType: Basis;
    fn set_center(&mut self, center: Vector3<f64>);
    fn get_center(&self) -> Option<Vector3<f64>>;
    fn basis_size(&self) -> usize;
    fn get_basis(&self) -> Vec<Arc<Self::BasisType>>;
}

pub enum BasisFormat {
    NWChem,
    Json,
}

//
// pub enum BasisType {
//     // STO,
// }
