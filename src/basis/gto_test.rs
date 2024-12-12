use crate::basis::gto::{GTO1d, GTO};
use crate::basis::helper::{simpson_integration, simpson_integration_3d};
use nalgebra::Vector3;
use std::f64::consts::PI;

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

    #[test]
    fn test_gto_overlap() {
        let gto1 = GTO::new(1.2, Vector3::new(1, 1, 1), Vector3::new(0.0, 0.0, 0.0));
        let gto2 = GTO::new(0.8, Vector3::new(1, 1, 1), Vector3::new(3.0, 3.0, 3.0));
        let integrand =
            |x, y, z| gto1.evaluate(&Vector3::new(x, y, z)) * gto2.evaluate(&Vector3::new(x, y, z));

        let lower = Vector3::new(-10.0, -10.0, -10.0);
        let upper = Vector3::new(10.0, 10.0, 10.0);
        let integral = simpson_integration_3d(integrand, lower, upper, 100, 100, 100);
        // println!("integral: {}", integral);
        let overlap = GTO::Sab(&gto1, &gto2);
        // println!("overlap: {}", overlap);
        assert!(
            (integral - overlap).abs() < 1e-5,
            "Overlap is not close to integral: got {}",
            overlap
        );
    }

    #[test]
    fn test_gto1d_laplacian() {
        // Gaussian with alpha=0.8, l=0, center=3.0
        let gto = GTO1d::new(0.8, 1, 1.0);

        let x = 3.0;

        // Numerical Laplacian (finite differences)
        let h = 1e-5;
        let numerical_laplacian =
            (gto.evaluate(x + h) - 2.0 * gto.evaluate(x) + gto.evaluate(x - h)) / h.powi(2);

        // Analytical Laplacian
        let analytical_laplacian = gto.laplacian(x);

        // Check if the two are close
        assert!(
            (numerical_laplacian - analytical_laplacian).abs() < 1e-5,
            "Laplacian mismatch: numerical = {}, analytical = {}",
            numerical_laplacian,
            analytical_laplacian
        );
    }

    fn gto_kinetic_integral(alpha1: f64, alpha2: f64, ax: f64, bx: f64) -> f64 {
        // Pre-calculate common terms to improve readability and efficiency
        let alpha_sum = alpha1 + alpha2;
        let alpha_ratio = alpha1 / alpha_sum;

        // Calculate the exponential term components
        let exp_term = alpha1
            * (ax * ax * alpha_ratio - ax * ax - 2.0 * ax * bx * alpha_ratio
                + 2.0 * ax * bx
                + bx * bx * alpha_ratio
                - bx * bx);

        // Calculate the polynomial term
        let poly_term = -4.0 * ax * ax * alpha1 * alpha2 + 8.0 * ax * bx * alpha1 * alpha2
            - 4.0 * bx * bx * alpha1 * alpha2
            + 2.0 * alpha1
            + 2.0 * alpha2;

        // Calculate denominator term
        let denom = (2.0 * alpha1 * alpha1 + 4.0 * alpha1 * alpha2 + 2.0 * alpha2 * alpha2)
            * alpha_sum.sqrt();

        // Combine all terms
        let result = PI.sqrt() * alpha1 * alpha2 * poly_term * exp_term.exp() / denom;

        result
    }

    #[test]
    fn test_gto1d_kinetic_T00() {
        // Gaussian with alpha=1.2, l=0, center=1.0
        let gto1 = GTO1d::new(1.2, 0, 0.0);
        // Gaussian with alpha=0.8, l=0, center=3.0
        let gto2 = GTO1d::new(0.8, 0, 1.0);

        let p = gto1.alpha + gto2.alpha;
        let q = gto1.alpha * gto2.alpha / p;
        let Qx = gto1.center - gto2.center;

        let integrand = |x: f64| {
            let f1 = gto1.evaluate(x); // g1(x)
            let df2 = gto2.laplacian(x); // g2'(x)
            -0.5 * (f1 * df2)
        };

        let integral = simpson_integration(integrand, -1000.0, 1000.0, 10_000);
        let integral_analytical1 = gto1.norm
            * gto2.norm
            * gto_kinetic_integral(gto1.alpha, gto2.alpha, gto1.center, gto2.center);
        let integral_analytical2 = GTO1d::Tab(&gto1, &gto2);

        assert!(
            (integral - integral_analytical1).abs() < 1e-5,
            "Kinetic energy integral is not close: got {}, expected {}",
            integral,
            integral_analytical1
        );

        assert!(
            (integral_analytical1 - integral_analytical2).abs() < 1e-5,
            "Kinetic energy integral is not close: got {}, expected {}",
            integral_analytical2,
            integral_analytical1
        )
    }

    fn test_gto1d_kinetic(alpha1: f64, l1: i32, center1: f64, alpha2: f64, l2: i32, center2: f64) {
        let gto1 = GTO1d::new(alpha1, l1, center1);
        let gto2 = GTO1d::new(alpha2, l2, center2);

        // Integrand for kinetic energy: product of derivatives and basis functions
        let integrand = |x: f64| {
            let f1 = gto1.evaluate(x); // g1(x)
            let df2 = gto2.laplacian(x); // g2'(x)
            -0.5 * (f1 * df2)
        };

        // Integrate numerically using Simpson's rule
        let integral = simpson_integration(integrand, -10.0, 10.0, 10_000);

        // Compute the kinetic energy integral using the analytical method
        let kinetic = GTO1d::Tab(&gto1, &gto2);

        // Assert that the numerical and analytical results are close
        assert!(
            (integral - kinetic).abs() < 1e-5,
            "Kinetic energy integral is not close: got {}, expected {}, \
             params: alpha1={}, l1={}, center1={}, alpha2={}, l2={}, center2={}",
            kinetic,
            integral,
            alpha1,
            l1,
            center1,
            alpha2,
            l2,
            center2
        );
    }

    #[test]
    fn test_gto1d_kinetic_with_params() {
        test_gto1d_kinetic(1.2, 0, 0.0, 0.8, 0, 1.0);
        test_gto1d_kinetic(1.2, 0, 0.0, 0.8, 1, 1.0);
        test_gto1d_kinetic(1.2, 1, 0.0, 0.8, 0, 1.0);
        test_gto1d_kinetic(1.2, 1, 0.0, 0.8, 1, 1.0);
        test_gto1d_kinetic(1.2, 2, 0.0, 0.8, 0, 1.0);
        test_gto1d_kinetic(1.2, 0, 0.0, 0.8, 2, 1.0);
        test_gto1d_kinetic(1.2, 2, 0.0, 0.8, 1, 1.0);
        test_gto1d_kinetic(1.2, 1, 0.0, 0.8, 2, 1.0);
        test_gto1d_kinetic(1.2, 2, 0.0, 0.8, 2, 1.0);
    }

    fn test_gto_kinetic(alpha1: f64, l1: i32, center1: Vector3<f64>, alpha2: f64, l2: i32, center2: Vector3<f64>) {
        let gto1 = GTO::new(alpha1, Vector3::new(l1, l1, l1), center1);
        let gto2 = GTO::new(alpha2, Vector3::new(l2, l2, l2), center2);

        // Integrand for kinetic energy: product of derivatives and basis functions
        let integrand = |x, y, z| {
            let f1 = gto1.evaluate(&Vector3::new(x, y, z)); // g1(x)
            let df2 = gto2.laplacian(&Vector3::new(x, y, z)); // g2'(x)
            -0.5 * (f1 * df2)
        };

        let lower = Vector3::new(-10.0, -10.0, -10.0);
        let upper = Vector3::new(10.0, 10.0, 10.0);
        // Integrate numerically using Simpson's rule
        let integral = simpson_integration_3d(integrand, lower, upper, 100, 100, 100);

        // Compute the kinetic energy integral using the analytical method
        let kinetic = GTO::Tab(&gto1, &gto2);

        // Assert that the numerical and analytical results are close
        assert!(
            (integral - kinetic).abs() < 1e-5,
            "Kinetic energy integral is not close: got {}, expected {}, \
             params: alpha1={}, l1={}, center1={:?}, alpha2={}, l2={}, center2={:?}",
            kinetic,
            integral,
            alpha1,
            l1,
            center1,
            alpha2,
            l2,
            center2
        );
    }

    #[test]
    fn test_gto_kinetic_with_params() {
        test_gto_kinetic(1.0, 1, Vector3::new(0.0, 0.0, 0.0), 1.0, 1, Vector3::new(1.0, 1.0, 1.0));
        test_gto_kinetic(0.8, 2, Vector3::new(-1.0, 0.5, 2.0), 1.2, 1, Vector3::new(0.0, 0.0, 0.0));
        test_gto_kinetic(2.0, 0, Vector3::new(0.0, 0.0, 0.0), 0.5, 2, Vector3::new(1.0, -1.0, 1.0));
        // test_gto_kinetic(1.2, 0, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 0, Vector3::new(0.0, 0.0, 0.0));
        // test_gto_kinetic(1.2, 0, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 1, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 1, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 0, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 1, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 1, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 2, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 0, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 0, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 2, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 2, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 1, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 1, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 2, Vector3::new(1.0, 1.0, 1.0));
        // test_gto_kinetic(1.2, 2, Vector3::new(0.0, 0.0, 0.0),
        //                  0.8, 2, Vector3::new(1.0, 1.0, 1.0));
    }
}
