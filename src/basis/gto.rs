extern crate nalgebra as na;

use std::f64::consts::PI;
use na::Vector3;
use crate::basis;
use basis::basis::Basis;

#[derive(Debug)]
pub struct GTO {
    pub gto1d: Vector3<GTO1d>,
    pub norm: f64,
}

#[derive(Debug)]
pub struct GTO1d {
    pub alpha: f64,
    pub l: i32,
    pub c: f64,
    pub norm: f64,
}

impl GTO1d {
    pub fn new(alpha: f64, l: i32, c: f64) -> Self {
        let norm = GTO1d::compute_norm(alpha, l);
        Self { alpha, l, c, norm }
    }

    fn compute_norm(alpha: f64, l: i32) -> f64 {
        let pi = std::f64::consts::PI;

        // Factorials
        fn factorial(n: usize) -> f64 {
            (1..=n).fold(1.0, |acc, x| acc * x as f64)
        }

        let l_usize = l as usize;

        // (2l)!
        let double_l_factorial = factorial(2 * l_usize);

        // l!
        let l_factorial = factorial(l_usize);

        // Compute components of the normalization factor:
        // N^2 = (2^(3l) * l! * alpha^l * sqrt(2 alpha / pi)) / (2l)!
        let numerator = (2.0_f64).powi(3 * l) * l_factorial as f64 * alpha.powi(l);
        let denominator = double_l_factorial;
        let factor = (2.0 * alpha / pi).sqrt();

        let n_squared = numerator * factor / denominator;
        n_squared.sqrt()
    }
}

// Simpson's rule integration
fn simpson_integration<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gto_normalization() {
        let alpha = 1.0;
        let l = 0;
        let gto = GTO1d::new(alpha, l, 1.0);

        // Integrand for normalization check: (N * x^l * e^{-alpha x^2})^2
        // = N^2 * x^{2l} * e^{-2 alpha x^2}
        let integrand = |x: f64| {
            let x_pow = x.powi(2 * l);
            (gto.norm * x_pow * (-gto.alpha * x * x).exp()).powi(2)
        };

        // Integrate from -10 to 10
        let integral = simpson_integration(integrand, -10.0, 10.0, 10_000);

        // Check if integral is close to 1
        let diff = (integral - 1.0).abs();
        assert!(diff < 1e-5, "Integral is not close to 1: got {}", integral);
    }
}

impl GTO {
    pub fn new(alpha: f64, l_xyz: Vector3<i32>, center: Vector3<f64>) -> Self {
        todo!()
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

impl Basis for GTO {
    fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        todo!()
    }

    fn overlap(&self, other: &Self) -> f64 {
        todo!()
    }

    fn kinetic(&self, other: &Self) -> f64 {
        todo!()
    }

    fn potential(&self, other: &Self, R: &Vector3<f64>) -> f64 {
        todo!()
    }

    fn two_electron(&self, other: &Self) -> f64 {
        todo!()
    }
}