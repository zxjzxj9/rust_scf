extern crate nalgebra as na;

use crate::basis;
// use basis::basis::Basis;
use basis::helper::{simpson_integration, simpson_integration_3d};
use na::Vector3;
use nalgebra::{ArrayStorage, Const, Matrix};
use std::f64::consts::PI;

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
        let numerator = 2.0_f64.powi(3 * l) * l_factorial * alpha.powi(l);
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

        if t < 0 || t > i + j || i < 0 || j < 0 {
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

    // for test only
    pub fn derivative(&self, x: f64) -> f64 {
        let x = x - self.center;
        if self.l == 0 {
            // For l = 0, the derivative is just the Gaussian part
            -2.0 * self.alpha * x * self.norm * (-self.alpha * x.powi(2)).exp()
        } else {
            // For l > 0, use the derivative formula
            let term1 = self.l as f64 * x.powi((self.l - 1) as i32);
            let term2 = -2.0 * self.alpha * x.powi(self.l);
            self.norm * (term1 + term2) * (-self.alpha * x.powi(2)).exp()
        }
    }

    // for test only
    pub fn laplacian(&self, x: f64) -> f64 {
        let x = x - self.center;
        if self.l == 0 {
            // For l = 0, the Laplacian simplifies to Gaussian part
            self.norm
                * (-2.0 * self.alpha + 4.0 * self.alpha.powi(2) * x.powi(2))
                * (-self.alpha * x.powi(2)).exp()
        } else if self.l == 1 {
            // For l = 1
            self.norm
                * (2.0 * x * self.alpha)
                * (2.0 * self.alpha * x.powi(2) - 3.0)
                * (-self.alpha * x.powi(2)).exp()
        } else {
            // General case for l > 1
            let term1 = self.l as f64 * (self.l as f64 - 1.0) * x.powi((self.l - 2) as i32);
            let term2 = -2.0 * self.alpha * (2.0 * self.l as f64 + 1.0) * x.powi(self.l as i32);
            let term3 = 4.0 * self.alpha.powi(2) * x.powi((self.l + 2) as i32);
            self.norm * (term1 + term2 + term3) * (-self.alpha * x.powi(2)).exp()
        }
    }

    // kinetic integral
    pub(crate) fn Tab(a: &GTO1d, b: &GTO1d) -> f64 {
        let p = a.alpha + b.alpha; // Combined Gaussian exponent
        let Qx = a.center - b.center; // Center separation

        // Normalization factors
        let norm = a.norm * b.norm * (std::f64::consts::PI / p).sqrt();

        // Terms in the Laplacian
        let term1 =
            b.l as f64 * (b.l as f64 - 1.0) * GTO1d::Eab(a.l, b.l - 2, 0, Qx, a.alpha, b.alpha);
        let term2 = -2.0
            * b.alpha
            * (2.0 * b.l as f64 + 1.0)
            * GTO1d::Eab(a.l, b.l, 0, Qx, a.alpha, b.alpha);
        let term3 = 4.0 * b.alpha.powi(2) * GTO1d::Eab(a.l, b.l + 2, 0, Qx, a.alpha, b.alpha);

        // Combine terms
        -0.5 * norm * (term1 + term2 + term3)
    }

    // // potential integral
    // pub(crate) fn Vab(a: &GTO1d, b: &GTO1d, R: f64) -> f64 {
    //     todo!("Implement Vab")
    // }
    //
    // // Coulomb integral and Exchange integral
    // pub(crate) fn JKab(a: &GTO1d, b: &GTO1d, c: &GTO1d, d: &GTO1d) -> f64 {
    //     todo!("Implement Iab")
    // }
}

#[derive(Debug)]
pub struct GTO {
    pub alpha: f64,
    pub l_xyz: Vector3<i32>,
    pub center: Vector3<f64>,
    pub norm: f64,
    pub gto1d: [GTO1d; 3],
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
        Self {
            alpha,
            l_xyz,
            center,
            norm,
            gto1d,
        }
    }

    pub(crate) fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        self.gto1d[0].evaluate(r.x) * self.gto1d[1].evaluate(r.y) * self.gto1d[2].evaluate(r.z)
    }

    pub fn laplacian(&self, r: &Vector3<f64>) -> f64 {
        let laplacian_x = self.gto1d[0].laplacian(r.x);
        let laplacian_y = self.gto1d[1].laplacian(r.y);
        let laplacian_z = self.gto1d[2].laplacian(r.z);

        laplacian_x * self.gto1d[1].evaluate(r.y) * self.gto1d[2].evaluate(r.z)
            + self.gto1d[0].evaluate(r.x) * laplacian_y * self.gto1d[2].evaluate(r.z)
            + self.gto1d[0].evaluate(r.x) * self.gto1d[1].evaluate(r.y) * laplacian_z
    }

    pub(crate) fn Sab(a: &GTO, b: &GTO) -> f64 {
        GTO1d::Sab(&a.gto1d[0], &b.gto1d[0])
            * GTO1d::Sab(&a.gto1d[1], &b.gto1d[1])
            * GTO1d::Sab(&a.gto1d[2], &b.gto1d[2])
    }

    pub(crate) fn Tab(a: &GTO, b: &GTO) -> f64 {
        GTO1d::Tab(&a.gto1d[0], &b.gto1d[0])
            * GTO1d::Sab(&a.gto1d[1], &b.gto1d[1])
            * GTO1d::Sab(&a.gto1d[2], &b.gto1d[2])
            + GTO1d::Tab(&a.gto1d[1], &b.gto1d[1])
                * GTO1d::Sab(&a.gto1d[0], &b.gto1d[0])
                * GTO1d::Sab(&a.gto1d[2], &b.gto1d[2])
            + GTO1d::Tab(&a.gto1d[2], &b.gto1d[2])
                * GTO1d::Sab(&a.gto1d[0], &b.gto1d[0])
                * GTO1d::Sab(&a.gto1d[1], &b.gto1d[1])
    }

    pub(crate) fn merge(a: &GTO, b: &GTO) -> GTO {
        // merge two GTOs into one GTO
        let center = (a.center * a.alpha + b.center * b.alpha) / (a.alpha + b.alpha);
        let l_xyz = a.l_xyz + b.l_xyz;
        let alpha = a.alpha + b.alpha;
        GTO::new(alpha, l_xyz, center)
    }

    pub(crate) fn Vab(a: &GTO, b: &GTO, R: Vector3<f64>) -> f64 {
        todo!("Implement Vab")
    }

    pub(crate) fn JKabcd(a: &GTO, b: &GTO, c: &GTO, d: &GTO) -> f64 {
        todo!("Implement Exab")
    }
}

// https://chemistry.montana.edu/callis/courses/chmy564/460water.pdf

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
