extern crate nalgebra as na;

use crate::basis::helper::boys_function;
use itertools::iproduct;
use na::Vector3;
use rayon::prelude::*;
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
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
        // let norm = GTO1d::compute_norm(alpha, l);
        let norm = 1.0;
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

#[derive(Debug, Serialize, Deserialize)]
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

    /// Computes the Coulomb auxiliary Hermite integrals.
    ///
    /// # Arguments
    ///
    /// * `t, u, v` - Orders of the Coulomb Hermite derivative in x, y, z directions, respectively
    /// * `n` - Order of the Boys function
    /// * `p` - A parameter (usually related to the exponents in the GTO integrals)
    /// * `PCx, PCy, PCz` - Cartesian distance components between the Gaussian composite center P and nuclear center C
    /// * `RPC` - Distance between P and C
    ///
    /// This function implements the recursion defined in Helgaker, Jørgensen, and Taylor for Coulomb integrals.
    pub fn hermite_coulomb(
        t: i32, u: i32, v: i32, n: i32, p: f64,
        PCx: f64, PCy: f64, PCz: f64, RPC: f64,
    ) -> f64 {
        let T = p * RPC * RPC;
        let mut val = 0.0;

        if t == 0 && u == 0 && v == 0 {
            val += (-2.0 * p).powi(n as i32) * boys_function(n, T);
        } else if t == 0 && u == 0 {
            if v > 1 {
                val += (v as f64 - 1.0)
                    * GTO::hermite_coulomb(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC);
            }
            val += PCz * GTO::hermite_coulomb(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC);
        } else if t == 0 {
            if u > 1 {
                val += (u as f64 - 1.0)
                    * GTO::hermite_coulomb(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC);
            }
            val += PCy * GTO::hermite_coulomb(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC);
        } else {
            if t > 1 {
                val += (t as f64 - 1.0)
                    * GTO::hermite_coulomb(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC);
            }
            val += PCx * GTO::hermite_coulomb(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC);
        }

        val
    }

    pub(crate) fn Vab(a: &GTO, b: &GTO, R: Vector3<f64>) -> f64 {
        let c = GTO::merge(a, b);
        // println!("a: {:?}, b: {:?}, c: {:?}", a.l_xyz, b.l_xyz, c.l_xyz);
        let mut val = 0.0;
        let dr = c.center - R;

        let val = iproduct!(0..=c.l_xyz.x, 0..=c.l_xyz.y, 0..=c.l_xyz.z)
            .par_bridge()
            .map(|(i, j, k)| {
                let eab_x = GTO1d::Eab(a.l_xyz.x, b.l_xyz.x, i,
                    a.center.x - b.center.x, a.alpha, b.alpha);

                let eab_y = GTO1d::Eab(a.l_xyz.y, b.l_xyz.y, j,
                    a.center.y - b.center.y, a.alpha, b.alpha);

                let eab_z = GTO1d::Eab(a.l_xyz.z, b.l_xyz.z, k,
                    a.center.z - b.center.z, a.alpha, b.alpha);

                let hermite_val =
                    GTO::hermite_coulomb(i, j, k, 0, c.alpha, dr.x, dr.y, dr.z, dr.norm());

                eab_x * eab_y * eab_z * hermite_val
            }).sum::<f64>();

        a.norm * b.norm * val * 2.0 * PI / c.alpha
    }

    pub(crate) fn JKabcd(a: &GTO, b: &GTO, c: &GTO, d: &GTO) -> f64 {
        let e = GTO::merge(a, b);
        let f = GTO::merge(c, d);
        let dr = e.center - f.center;
        let alpha = e.alpha * f.alpha / (e.alpha + f.alpha);

        let  val = iproduct!(0..=e.l_xyz.x, 0..=e.l_xyz.y, 0..=e.l_xyz.z,
                0..=f.l_xyz.x, 0..=f.l_xyz.y, 0..=f.l_xyz.z)
            .par_bridge()
            .map(|(i, j, k, l, m, n)| {
                let eab_x = GTO1d::Eab(a.l_xyz.x, b.l_xyz.x, i,
                    a.center.x - b.center.x, a.alpha, b.alpha);

                let eab_y = GTO1d::Eab(a.l_xyz.y, b.l_xyz.y, j,
                    a.center.y - b.center.y, a.alpha, b.alpha);

                let eab_z = GTO1d::Eab(a.l_xyz.z, b.l_xyz.z, k,
                    a.center.z - b.center.z, a.alpha, b.alpha);

                let ecd_x = GTO1d::Eab(c.l_xyz.x, d.l_xyz.x, l,
                    c.center.x - d.center.x, c.alpha, d.alpha);

                let ecd_y = GTO1d::Eab(c.l_xyz.y, d.l_xyz.y, m,
                    c.center.y - d.center.y, c.alpha, d.alpha);

                let ecd_z = GTO1d::Eab(c.l_xyz.z, d.l_xyz.z, n,
                    c.center.z - d.center.z, c.alpha, d.alpha);

                let hermite_val =
                    GTO::hermite_coulomb(i + l, j + m, k + n, 0,
                                         alpha, dr.x, dr.y, dr.z, dr.norm());

                // if l + m + n is odd, then sgn is -1, otherwise sgn is 1
                let sgn = if (l + m + n) % 2 == 0 { 1.0 } else { -1.0 };
                eab_x * eab_y * eab_z * ecd_x * ecd_y * ecd_z * sgn * hermite_val
            }).sum::<f64>();

        a.norm * b.norm * c.norm * d.norm * val
            * 2.0 * PI.powf(2.5) / (e.alpha * f.alpha * (e.alpha + f.alpha).sqrt())
    }
}

// https://chemistry.montana.edu/callis/courses/chmy564/460water.pdf