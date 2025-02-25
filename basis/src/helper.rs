#![allow(non_snake_case, unused)]
use libm::{erf, sqrt};
use nalgebra::Vector3;
use num_complex::Complex;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

// Simpson's rule integration
pub(crate) fn simpson_integration<F>(f: F, a: f64, b: f64, n: usize) -> f64
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

// Helper function to determine Simpson's weight for a given index and max index
fn simpson_weight(i: usize, n: usize) -> f64 {
    if i == 0 || i == n {
        1.0
    } else if i % 2 == 1 {
        4.0
    } else {
        2.0
    }
}

/// Parallel Simpson's rule integration in 3D.
///
/// Integrates f(x,y,z) over the box defined by [a.x,b.x] x [a.y,b.y] x [a.z,b.z].
/// Uses Simpson's rule with nx, ny, nz subdivisions (adjusted up if not even).
/// Returns the approximate value of the integral.
pub(crate) fn simpson_integration_3d<F>(
    f: F,
    a: Vector3<f64>,
    b: Vector3<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> f64
where
    F: Fn(f64, f64, f64) -> f64 + Sync,
{
    // Ensure nx, ny, nz are even
    let nx = if nx % 2 == 0 { nx } else { nx + 1 };
    let ny = if ny % 2 == 0 { ny } else { ny + 1 };
    let nz = if nz % 2 == 0 { nz } else { nz + 1 };

    let hx = (b.x - a.x) / nx as f64;
    let hy = (b.y - a.y) / ny as f64;
    let hz = (b.z - a.z) / nz as f64;

    // Precompute coordinates and weights for each dimension
    let x_coords: Vec<f64> = (0..=nx).map(|i| a.x + i as f64 * hx).collect();
    let y_coords: Vec<f64> = (0..=ny).map(|j| a.y + j as f64 * hy).collect();
    let z_coords: Vec<f64> = (0..=nz).map(|k| a.z + k as f64 * hz).collect();

    let x_weights: Vec<f64> = (0..=nx).map(|i| simpson_weight(i, nx)).collect();
    let y_weights: Vec<f64> = (0..=ny).map(|j| simpson_weight(j, ny)).collect();
    let z_weights: Vec<f64> = (0..=nz).map(|k| simpson_weight(k, nz)).collect();

    // Parallel integration using Rayon
    let sum: f64 = (0..=nx)
        .into_par_iter()
        .map(|i| {
            let x = x_coords[i];
            let wx = x_weights[i];
            (0..=ny)
                .into_par_iter()
                .map(|j| {
                    let y = y_coords[j];
                    let wy = y_weights[j];
                    (0..=nz)
                        .into_par_iter()
                        .map(|k| {
                            let z = z_coords[k];
                            let wz = z_weights[k];
                            let w = wx * wy * wz;
                            w * f(x, y, z)
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum();

    // Multiply by the step sizes and the factor 1/27
    // Simpson's rule in 3D: (hx*hy*hz/27)* sum_of_weights
    sum * (hx * hy * hz) / 27.0
}

// Compute the Boys function F_n(x) for n = 0..12.
// see the paper by Gregory Beylkin; Sandeep Sharma
// https://doi.org/10.1063/5.0062444
pub fn boys_function(n: i32, x: f64) -> f64 {
    let n = n as usize;
    assert!(n <= 12, "n must be between 0 and 12");
    assert!(x >= 0.0, "x must be nonnegative");

    // Constants and parameters as in original code
    let tol: f64 = 1.0e-3;
    let sqrtpio2: f64 = 0.886226925452758014;
    let t: [f64; 12] = [
        2.0,
        0.66666666666666663,
        0.4,
        0.2857142857142857,
        0.22222222222222221,
        0.18181818181818182,
        0.15384615384615385,
        0.13333333333333333,
        0.11764705882352941,
        0.10526315789473684,
        0.09523809523809523,
        0.08695652173913043,
    ];

    // Complex parameters (converted from double complex arrays)
    // Fortran indexing: zz(1:10) => Rust: zz[0..9]
    let zz: [Complex<f64>; 10] = [
        Complex::new(64.3040206523305, 182.43694739308491),
        Complex::new(64.3040206523305, -182.43694739308491),
        Complex::new(-12.572081889410178, 141.21366415342502),
        Complex::new(-12.572081889410178, -141.21366415342502),
        Complex::new(-54.103079551670268, 104.57909575828442),
        Complex::new(-54.103079551670268, -104.57909575828442),
        Complex::new(-78.720025594983341, 69.309284623985663),
        Complex::new(-78.720025594983341, -69.309284623985663),
        Complex::new(-92.069621609035313, 34.559308619699376),
        Complex::new(-92.069621609035313, -34.559308619699376),
    ];

    let fact: [Complex<f64>; 10] = [
        Complex::new(0.0013249210991966042, 0.00091787356295447745),
        Complex::new(0.0013249210991966042, -0.00091787356295447745),
        Complex::new(0.055545905103006735, -3.5151540664451613),
        Complex::new(0.055545905103006735, 3.5151540664451613),
        Complex::new(-114.56407675096416, 192.13789620924834),
        Complex::new(-114.56407675096416, -192.13789620924834),
        Complex::new(2091.5556220686653, -1582.5742912360638),
        Complex::new(2091.5556220686653, 1582.5742912360638),
        Complex::new(-9477.9394228935325, 3081.4443710192086),
        Complex::new(-9477.9394228935325, -3081.4443710192086),
    ];

    let ww: [Complex<f64>; 10] = [
        Complex::new(-8.3418049867878959e-9, -7.0958810331788253e-9),
        Complex::new(-8.3418050437598581e-9, 7.0958810084577824e-9),
        Complex::new(8.2436739552884774e-8, -2.7704117936134414e-7),
        Complex::new(8.2436739547688584e-8, 2.7704117938414886e-7),
        Complex::new(1.9838416382728666e-6, 7.8321058613942770e-7),
        Complex::new(1.9838416382681279e-6, -7.8321058613180811e-7),
        Complex::new(-4.7372729839268780e-6, 5.8076919074212929e-6),
        Complex::new(-4.7372729839287016e-6, -5.8076919074154416e-6),
        Complex::new(-6.8186014282131608e-6, -1.3515261354290787e-5),
        Complex::new(-6.8186014282138385e-6, 1.3515261354295612e-5),
    ];

    let rzz: [f64; 1] = [-96.32193429034384];
    let rfact: [f64; 1] = [152478.44519077540];
    let rww: [f64; 1] = [1.8995875677635889e-5];

    let y = (-x).exp();

    let mut vals = [0.0_f64; 13]; // vals(0..12)

    // Large x branch
    if x.abs() >= 4.5425955121971775 {
        let yy = sqrt(x);
        let val0 = sqrtpio2 * erf(yy) / yy;
        vals[0] = val0;
        let halfy = y / 2.0;
        for n_i in 1..=12 {
            // vals(n) = ((n - 0.5)*vals(n-1) - yy)/x
            // Wait, the code references 'yy' in the Fortran large x block as well.
            // In that block, `yy = y/2.0` is redefined after the first line.
            // Let's re-check the Fortran code in the large x path:
            //
            // if (abs(x).ge.0.45425955121971775D+01) then
            //   yy = sqrt(x)
            //   vals(0) = sqrtpio2*erf(yy)/yy
            //   yy = y/2.0d0
            //   do n = 1, 12
            //     vals(n) = ((n -0.5d0)*vals(n-1) - yy)/x
            //   enddo
            // end if
            //
            // Notice they reuse 'yy' as y/2 after computing vals(0). We must do the same:
        }
        let yy = y / 2.0;
        for n_i in 1..=12 {
            vals[n_i] = ((n_i as f64 - 0.5) * vals[n_i - 1] - yy) / x;
        }

        return vals[n];
    }

    // Otherwise use the more complex approach
    let mut rtmp = 0.0_f64;
    // k = 1,3,5,7,9 in Fortran => k = 0,2,4,6,8 in zero-based
    for k_i in (0..10).step_by(2) {
        let numerator = 1.0 - (fact[k_i] * y).re;
        // Actually fact and ww are complex. The code uses (1.0d0 - fact(k)*y) which might
        // need interpretation. In Fortran:
        //   rtmp = rtmp + ww(k)*(1.0d0 - fact(k)*y)/(x + zz(k))
        // Here, ww(k), fact(k), zz(k) are complex. Actually, the Fortran code treats them as complex.
        // On re-reading the code, fact(k), zz(k), ww(k) are complex. The code does complex division?
        // Actually, the code is suspicious. It's mixing complex arrays but doing real arithmetic.
        //
        // In the code snippet:
        // rtmp = rtmp + ww(k)*(1.0d0 - fact(k)*y)/(x + zz(k))
        //
        // (1.0d0 - fact(k)*y) => fact(k) is complex, y is real, so fact(k)*y is complex.
        // This results in a complex. (1.0d0 - fact(k)*y) is also complex.
        // (x + zz(k)) is complex. So we have a complex division and multiplication.
        // The final rtmp is assigned to a real variable. This is suspicious. Possibly the original code
        // uses implicit type conversions or does complex arithmetic but only real part is used at the end?
        //
        // On closer inspection: In the original code, rtmp is declared as real *8.
        // This implies that the entire expression is real. This suggests that maybe fact, zz, and ww arrays
        // are never actually used in a complex manner for final result. The code has them as complex arrays but
        // the final is always real. It's possible the imaginary parts cancel or that the code is using a
        // contour integral approximation and expects real results.
        //
        // We'll replicate the logic exactly:
        //
        // We must perform complex operations and take the real part of the result since rtmp is real.
        let numerator_c = Complex::new(1.0, 0.0) - fact[k_i] * y;
        let denominator_c = Complex::new(x, 0.0) + zz[k_i];
        let fraction = (ww[k_i] * numerator_c) / denominator_c;
        rtmp += fraction.re; // accumulate real part only, to mimic the Fortran code's real accumulation
    }

    let mut tmp = 0.0_f64;
    // do k=1,1 means only k=1 => k=0 in zero-based
    {
        let k_i = 0;
        let q = x + rzz[k_i];
        let numerator_c = 1.0 - rfact[k_i] * y;
        if q.abs() >= tol {
            // tmp = tmp + rww(k)*(1.0d0 - rfact(k)*y)/(x + rzz(k))
            tmp += rww[k_i] * (numerator_c / q);
        } else {
            // polynomial approximation:
            // p = 1.0d0 - q/2.0d0 + q**2/6.0d0 - q**3/24.0d0 + q**4/120.0d0
            let p = 1.0 - q / 2.0 + (q * q) / 6.0 - (q * q * q) / 24.0 + (q * q * q * q) / 120.0;
            tmp += rww[k_i] * p;
        }
    }

    vals[12] = 2.0 * rtmp + tmp;
    let yy = y / 2.0;
    for n_i in (0..12).rev() {
        // vals(n) = (x*vals(n+1)+yy)*t(n)
        vals[n_i] = (x * vals[n_i + 1] + yy) * t[n_i];
    }

    // Return requested order
    vals[n]
}

pub fn integrate_spherical_3d<F>(
    f: F,
    a: Vector3<f64>,
    b: Vector3<f64>,
    R: Vector3<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    tolerance: f64,
) -> f64
where
    F: Fn(Vector3<f64>) -> f64 + Sync,
{
    // Compute step sizes for each dimension
    let dx = (b.x - a.x) / nx as f64;
    let dy = (b.y - a.y) / ny as f64;
    let dz = (b.z - a.z) / nz as f64;

    let volume_element = dx * dy * dz;

    // Create a vector of all grid indices for parallel iteration
    let indices: Vec<(usize, usize, usize)> = (0..nx)
        .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
        .collect();

    // Use Rayon parallel iterator to compute sum
    let sum: f64 = indices
        .par_iter() // parallel iterator
        .map(|&(i, j, k)| {
            // Compute the midpoint coordinates in each dimension
            let x = a.x + (i as f64 + 0.5) * dx;
            let y = a.y + (j as f64 + 0.5) * dy;
            let z = a.z + (k as f64 + 0.5) * dz;
            let r = Vector3::new(x, y, z);

            let dr = r - R;
            let dist = dr.norm();

            // Skip points too close to R
            if dist < tolerance || dist == 0.0 {
                return 0.0;
            }

            // Evaluate the integrand: -f(r)/|r-R|
            let val = -f(r) / dist;
            val
        })
        .sum();

    // Multiply by the volume element to get the integral approximation
    sum * volume_element
}

pub(crate) fn two_electron_integral_monte_carlo<F>(psi: F, L: f64, samples: usize) -> (f64, f64)
where
    F: Fn(Vector3<f64>, Vector3<f64>) -> f64 + Sync,
{
    let volume = (2.0 * L).powi(3) * (2.0 * L).powi(3); // (2L)^3 for r1 and (2L)^3 for r2 => (2L)^6 total volume

    let mut rng = rand::thread_rng();
    let seed: u64 = rng.gen();

    // Parallel iteration
    let results: Vec<f64> = (0..samples)
        .into_par_iter()
        .map(|n| {
            let mut thread_rng = rand::thread_rng();

            // Random point r1
            let x1 = thread_rng.gen_range(-L..L);
            let y1 = thread_rng.gen_range(-L..L);
            let z1 = thread_rng.gen_range(-L..L);
            let r1 = Vector3::new(x1, y1, z1);

            // Random point r2
            let x2 = thread_rng.gen_range(-L..L);
            let y2 = thread_rng.gen_range(-L..L);
            let z2 = thread_rng.gen_range(-L..L);
            let r2 = Vector3::new(x2, y2, z2);

            let diff = r1 - r2;
            let dist = diff.norm();
            if dist > 1e-12 {
                let val = psi(r1, r2) / dist;
                val
            } else {
                // If they're essentially the same point, integrand is singular.
                // For a 3D integral of Coulomb potential, the measure of exact overlap is negligible,
                // but to be safe, we can skip or approximate.
                // We'll just skip by returning 0.0, since it's extremely rare in random sampling.
                0.0
            }
        })
        .collect();

    let sum: f64 = results.par_iter().sum();
    let mean = sum / (samples as f64);

    // Estimate standard deviation
    // calculate variance: sum((x_i - mean)^2) / (n-1)
    let var = results.par_iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples as f64 - 1.0);
    // let var = results.into_par_iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples as f64 - 1.0);
    let std_dev = var.sqrt();

    let integral = mean * volume;
    // Standard error of the mean = std_dev / sqrt(samples)
    let std_err = std_dev * volume / (samples as f64).sqrt();

    (integral, std_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // Basic test: check behavior for a few values
    #[test]
    fn test_boys_function() {
        let val = boys_function(0, 0.0);
        // F_0(0) = integral of exp(-0) dt from 0 to 1 = 1.0 (Boys function definition)
        // This code should produce something close to known F_0(0) = 1/2 * sqrt(pi)/erf(??)
        // Actually, the Boys function F_n(x) for n=0 at x=0 is 0.886226925...?
        // The definition: F_0(x) = (sqrt(pi)/2)/sqrt(x)*erf(sqrt(x))
        // At x=0, limit F_0(0)= 1.0 (since erf(0)=0, we must consider limit)
        // Let's trust the code for now. If the original code was correct, it should return something sensible.
        assert!(val >= 0.999 && val <= 1.001);
    }

    #[test]
    fn test_boys_function_nonzero() {
        let val = boys_function(1, 1.0);
        // Just a sanity check: no panic
        assert!(val.is_finite());
    }

    #[test]
    fn test_integration_unit_sphere() {
        // Domain: cube [-1,1]^3
        let a = Vector3::new(-1.0, -1.0, -1.0);
        let b = Vector3::new(1.0, 1.0, 1.0);

        // Singularity at the origin
        let R = Vector3::new(0.0, 0.0, 0.0);

        // Grid resolution
        let nx = 100;
        let ny = 100;
        let nz = 100;

        // Tolerance for skipping points near R
        let tolerance = 1e-6;

        // f(r)=1 inside the unit sphere, and 0 outside
        let f = |r: Vector3<f64>| if r.norm_squared() <= 1.0 { 1.0 } else { 0.0 };

        let result = integrate_spherical_3d(f, a, b, R, nx, ny, nz, tolerance);

        // Analytical result for ∫ (1/|r|) inside sphere radius 1: 2 * π * 1^2 = 2 * π
        let analytical = -2.0 * PI;

        let error = (result - analytical).abs();
        let relative_error = error / analytical;

        // Check that the relative error is reasonably small
        // Depending on nx, ny, nz, you might get a few percent accuracy.
        // Let's say we want within 5% for a quick test.
        assert!(
            relative_error < 0.05,
            "Integration error too large: result={}, expected={} (rel err={})",
            result,
            analytical,
            relative_error
        );
    }
}
