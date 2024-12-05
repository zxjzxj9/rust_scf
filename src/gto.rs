// import necessary libraries
extern crate nalgebra as na;

use std::f64;
use na::Vector3;

// define a struct for Gaussian-type orbital:
// reference: https://joshuagoings.com/assets/integrals.pdf
// \phi_{\alpha}(\mathbf{r}) = N_{\alpha} x^{l} y^{m} z^{n} e^{-\alpha r^{2}}
#[derive(Debug)]
pub struct GTO {
    pub alpha: f64,
    pub l_xyz: Vector3<i32>,
    pub center: Vector3<f64>,
    pub norm: f64,
}


// implement new function for GTO
impl GTO {
    pub fn new(alpha: f64, l_xyz: Vector3<i32>, center: Vector3<f64>) -> Self {
        let norm = Self::compute_norm(alpha, l_xyz);
        Self { alpha, l_xyz, center, norm }
    }

    // compute the normalization constant for GTO
    fn compute_norm(alpha: f64, l_xyz: Vector3<i32>) -> f64 {
        let l = l_xyz.x + l_xyz.y + l_xyz.z;
        let norm = (2.0 * alpha / f64::consts::PI).powf(0.75) * (2.0_f64).powf(-l as f64)
            * Self::factorial(l) / Self::factorial(2 * l);
        norm
    }

    // compute the factorial of a number
    fn factorial(n: i32) -> f64 {
        let mut fact = 1.0;
        for i in 1..=n {
            fact *= i as f64;
        }
        fact
    }
}

trait Basis {
    fn evaluate(&self, r: &Vector3<f64>) -> f64;
    fn overlap(&self, r1: &Vector3<f64>, r2: &Vector3<f64>) -> f64;

    fn kinetic(&self, r1: &Vector3<f64>, r2: &Vector3<f64>) -> f64;
    fn potential(&self, r1: &Vector3<f64>, r2: &Vector3<f64>, R: &Vector3<f64>) -> f64;
    fn two_electron(&self, r1: &Vector3<f64>, r2: &Vector3<f64>) -> f64;
}

// parameter see this https://gitlab.com/Molcas/OpenMolcas/-/blob/master/basis_library/6-31G