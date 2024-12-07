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

    fn evaluate(&self, x: f64) -> f64 {
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

    fn Sab(a: &GTO1d, b: &GTO1d) -> f64 {
        let p = a.alpha + b.alpha;
        // let q = a.alpha * b.alpha / p;
        let Qx = a.center - b.center;
        // let P = (a.alpha * a.c + b.alpha * b.c) / p;

        // Base case (i=0, j=0)
        GTO1d::Eab(a.l, b.l, 0, Qx, a.alpha, b.alpha) * (PI / p).sqrt() * a.norm * b.norm
    }
}



#[allow(non_snake_case)]
impl GTO {
    fn new(alpha: f64, l_xyz: Vector3<i32>, center: Vector3<f64>) -> Self {
        let gto1d = [
            GTO1d::new(alpha, l_xyz.x, center.x),
            GTO1d::new(alpha, l_xyz.y, center.y),
            GTO1d::new(alpha, l_xyz.z, center.z),
        ];

        let norm = gto1d[0].norm * gto1d[1].norm * gto1d[2].norm;
        Self { gto1d, norm }
    }

    fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        self.gto1d[0].evaluate(r.x) * self.gto1d[1].evaluate(r.y) * self.gto1d[2].evaluate(r.z)
    }

    fn Sab(a: &GTO, b: &GTO) -> f64 {
        GTO1d::Sab(&a.gto1d[0], &b.gto1d[0])
            * GTO1d::Sab(&a.gto1d[1], &b.gto1d[1])
            * GTO1d::Sab(&a.gto1d[2], &b.gto1d[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gto1d_normalization() {
        let gto = GTO1d::new(1.0, 2, 1.0);

        // Integrand for normalization check: (N * x^l * e^{-alpha x^2})^2
        // = N^2 * x^{2l} * e^{-2 alpha x^2}
        let integrand = |x: f64| {
            // let x_pow = x.powi(l);
            // (gto.norm * x_pow * (-gto.alpha * x.powi(2)).exp()).powi(2)
            gto.evaluate(x).powi(2)
        };

        // Integrate from -10 to 10
        let integral = simpson_integration(integrand, -10.0, 10.0, 10_000);
        // println!("wfn integral: {}", integral);
        // Check if integral is close to 1
        let diff = (integral - 1.0).abs();
        assert!(diff < 1e-5, "Integral is not close to 1: got {}", integral);
    }

    #[test]
    fn test_gto1d_overlap() {
        let gto1 = GTO1d::new(1.2, 1, 1.0);
        let gto2 = GTO1d::new(0.8, 1, 3.0);
        let integrand = |x: f64| gto1.evaluate(x) * gto2.evaluate(x);

        let integral = simpson_integration(integrand, -10.0, 10.0, 10_000);
        // println!("integral: {}", integral);
        let overlap = GTO1d::Sab(&gto1, &gto2);
        // println!("overlap: {}", overlap);
        assert!(
            (integral - overlap).abs() < 1e-5,
            "Overlap is not close to integral: got {}",
            overlap
        );
    }

    #[test]
    fn test_gto_normalization() {
        let gto = GTO::new(1.0, Vector3::new(1, 1, 1), Vector3::new(0.0, 0.0, 0.0));

        // Integrand for normalization check: (N * x^l * e^{-alpha x^2})^2
        // = N^2 * x^{2l} * e^{-2 alpha x^2}
        let integrand = |x, y, z| gto.evaluate(&Vector3::new(x, y, z)).powi(2);

        let lower = Vector3::new(-10.0, -10.0, -10.0);
        let upper = Vector3::new(10.0, 10.0, 10.0);
        // Integrate from -10 to 10
        let integral = simpson_integration_3d(integrand, lower, upper, 100, 100, 100);
        // println!("wfn integral: {}", integral);
        // Check if integral is close to 1
        let diff = (integral - 1.0).abs();
        assert!(diff < 1e-5, "Integral is not close to 1: got {}", integral);
    }
}

// impl GTO {
//     pub fn new(alpha: f64, l_xyz: Vector3<i32>, center: Vector3<f64>) -> Self {
//         todo!()
//     }
//
//     fn compute_norm(alpha: f64, l_xyz: Vector3<i32>) -> f64 {
//         let (l, m, n) = (l_xyz.x, l_xyz.y, l_xyz.z);
//
//         // Convert to f64 once
//         let lf = l as f64;
//         let mf = m as f64;
//         let nf = n as f64;
//
//         // The normalization factor for a Cartesian GTO is:
//         // N = sqrt( (2^(l+m+n) * (2α)^(l+m+n+3/2) ) / (π^(3/2) (2l)!(2m)!(2n)!) )
//         let numerator = 2f64.powf(lf + mf + nf) * (2.0 * alpha).powf(lf + mf + nf + 1.5);
//         let denominator = PI.powf(1.5)
//             * factorial((2 * l) as i32)
//             * factorial((2 * m) as i32)
//             * factorial((2 * n) as i32);
//
//         (numerator / denominator).sqrt()
//     }
// }
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
