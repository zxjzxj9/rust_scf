/*  Abstraction of the basis trait

    Author: Victor Zhang
    Date: 2024/12/31
 */

// define the basis trait

#![allow(non_snake_case, unused)]
use nalgebra::Vector3;

pub trait Basis {
    fn evaluate(&self, r: &Vector3<f64>) -> f64;
    fn Sab(a: &Self, b: &Self) -> f64;
    fn Tab(a: &Self, b: &Self) -> f64;

    fn Vab(a: &Self, b: &Self, R: Vector3<f64>) -> f64;
    fn JKabcd(a: &Self, b: &Self, c: &Self, d: &Self) -> f64;

}

pub enum BasisFormat {
    NWChem,
    Json,
}