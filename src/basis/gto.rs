extern crate nalgebra as na;

use std::f64::consts::PI;
use na::Vector3;

#[derive(Debug)]
pub struct GTO {
    pub alpha: f64,
    pub l_xyz: Vector3<i32>,
    pub center: Vector3<f64>,
    pub norm: f64,
}

impl GTO {
    pub fn new(alpha: f64, l_xyz: Vector3<i32>, center: Vector3<f64>) -> Self {
        let norm = Self::compute_norm(alpha, l_xyz);
        Self { alpha, l_xyz, center, norm }
    }

    fn compute_norm(alpha: f64, l_xyz: Vector3<i32>) -> f64 {
        let (l, m, n) = (l_xyz.x, l_xyz.y, l_xyz.z);

        // Convert to f64 once
        let lf = l as f64;
        let mf = m as f64;
        let nf = n as f64;

        // The normalization factor for a Cartesian GTO is:
        // N = sqrt( (2^(l+m+n) * (2α)^(l+m+n+3/2) ) / (π^(3/2) (2l)!(2m)!(2n)!) )
        let numerator = 2f64.powf(lf + mf + nf) * (2.0 * alpha).powf(lf + mf + nf + 1.5);
        let denominator = PI.powf(1.5)
            * factorial((2 * l) as i32)
            * factorial((2 * m) as i32)
            * factorial((2 * n) as i32);

        (numerator / denominator).sqrt()
    }
}

// A simple factorial function (sufficient for small integers)
fn factorial(n: i32) -> f64 {
    (1..=n).map(|x| x as f64).product()
}

