extern crate nalgebra as na;

use crate::basis;
// use basis::basis::Basis;
use na::Vector3;
use std::f64::consts::PI;
use basis::helper::{simpson_integration, simpson_integration_3d};

#[derive(Debug)]
pub struct GTO {
    pub gto1d: [GTO1d; 3],
    pub norm: f64,
}

#[derive(Debug)]
pub struct GTO1d {
    pub alpha: f64,
    pub l: i32,
    pub center: f64,
    pub norm: f64,
}

fn factorial(n: i32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

#[allow(non_snake_case)]
impl GTO1d {
    pub fn new(alpha: f64, l: i32, center: f64) -> Self {
        let norm = GTO1d::compute_norm(alpha, l);
        Self {
            alpha,
            l,
            center,
            norm,
        }
    }

    fn compute_norm(alpha: f64, l: i32) -> f64 {
        let pi = std::f64::consts::PI;

        // (2l)!
        let double_l_factorial = factorial(2 * l);

        // l!
        let l_factorial = factorial(l);

        // Compute components of the normalization factor:
        // N^2 = (2^(3l) * l! * alpha^l * sqrt(2 alpha / pi)) / (2l)!
        let numerator = (2.0_f64).powi(3 * l) * l_factorial as f64 * alpha.powi(l);
        let denominator = double_l_factorial;
        let factor = (2.0 * alpha / pi).sqrt();

        let n_squared = numerator * factor / denominator;
        n_squared.sqrt()
    }

    pub(crate) fn evaluate(&self, x: f64) -> f64 {
        let x = x - self.center;
        self.norm * x.powi(self.l) * (-self.alpha * x.powi(2)).exp()
    }

    pub fn Eab(i: i32, j: i32, t: i32, Qx: f64, a: f64, b: f64) -> f64 {
        let p = a + b;
        let q = a * b / p;

        if t < 0 || t > i + j {
            0.0
        } else if i == 0 && j == 0 && t == 0 {
            let r = (-q * Qx.powi(2)).exp();
            // println!("r: {}, Qx: {}, q: {}", r, Qx, q);
            r
        } else if j == 0 {
            // how to recursively call Eab
            let r = GTO1d::Eab(i - 1, j, t - 1, Qx, a, b) / (2.0 * p)
                - GTO1d::Eab(i - 1, j, t, Qx, a, b) * q * Qx / a
                + GTO1d::Eab(i - 1, j, t + 1, Qx, a, b) * ((t + 1) as f64);
            // println!("r: {}", r);
            r
        } else {
            let r = GTO1d::Eab(i, j - 1, t - 1, Qx, a, b) / (2.0 * p)
                + GTO1d::Eab(i, j - 1, t, Qx, a, b) * q * Qx / b
                + GTO1d::Eab(i, j - 1, t + 1, Qx, a, b) * ((t + 1) as f64);
            // println!("r: {}", r);
            r
        }
    }

    // overlap integral
    pub(crate) fn Sab(a: &GTO1d, b: &GTO1d) -> f64 {
        let p = a.alpha + b.alpha;
        // let q = a.alpha * b.alpha / p;
        let Qx = a.center - b.center;
        // let P = (a.alpha * a.c + b.alpha * b.c) / p;

        // Base case (i=0, j=0)
        GTO1d::Eab(a.l, b.l, 0, Qx, a.alpha, b.alpha) * (PI / p).sqrt() * a.norm * b.norm
    }

    // kinetic integral
    pub (crate) fn Tab(a: &GTO1d, b: &GTO1d) -> f64 {
        todo!("Implement Tab")
    }

    // potential integral
    pub(crate) fn Vab(a: &GTO1d, b: &GTO1d, R: f64) -> f64 {
        todo!("Implement Vab")
    }

    // Coulomb integral and Exchange integral
    pub(crate) fn JKab(a: &GTO1d, b: &GTO1d, c: &GTO1d, d: &GTO1d) -> f64 {
        todo!("Implement Iab")
    }
}



#[allow(non_snake_case)]
impl GTO {
    pub(crate) fn new(alpha: f64, l_xyz: Vector3<i32>, center: Vector3<f64>) -> Self {
        let gto1d = [
            GTO1d::new(alpha, l_xyz.x, center.x),
            GTO1d::new(alpha, l_xyz.y, center.y),
            GTO1d::new(alpha, l_xyz.z, center.z),
        ];

        let norm = gto1d[0].norm * gto1d[1].norm * gto1d[2].norm;
        Self { gto1d, norm }
    }

    pub(crate) fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        self.gto1d[0].evaluate(r.x) * self.gto1d[1].evaluate(r.y) * self.gto1d[2].evaluate(r.z)
    }

    pub(crate) fn Sab(a: &GTO, b: &GTO) -> f64 {
        GTO1d::Sab(&a.gto1d[0], &b.gto1d[0])
            * GTO1d::Sab(&a.gto1d[1], &b.gto1d[1])
            * GTO1d::Sab(&a.gto1d[2], &b.gto1d[2])
    }
}




//
// impl Basis for GTO {
//     fn evaluate(&self, r: &Vector3<f64>) -> f64 {
//         todo!()
//     }
//
//     fn overlap(&self, other: &Self) -> f64 {
//         todo!()
//     }
//
//     fn kinetic(&self, other: &Self) -> f64 {
//         todo!()
//     }
//
//     fn potential(&self, other: &Self, R: &Vector3<f64>) -> f64 {
//         todo!()
//     }
//
//     fn two_electron(&self, other: &Self) -> f64 {
//         todo!()
//     }
// }
