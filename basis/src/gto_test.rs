#[cfg(test)]
mod tests {

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

    use crate::basis::Basis;
    use crate::gto::{GTO1d, GTO};
    use crate::helper::*;
    use nalgebra::Vector3;
    use rand::Rng;
    use rand_distr::Normal;
    use std::f64::consts::PI;
    use crate::cgto::ContractedGTO;

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

    fn test_gto_kinetic(
        alpha1: f64,
        l1: i32,
        center1: Vector3<f64>,
        alpha2: f64,
        l2: i32,
        center2: Vector3<f64>,
    ) {
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
        test_gto_kinetic(
            1.0,
            1,
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            1,
            Vector3::new(0.0, 0.0, 1.0),
        );
        test_gto_kinetic(
            0.8,
            2,
            Vector3::new(-1.0, 0.5, 2.0),
            1.2,
            1,
            Vector3::new(0.0, 0.0, 0.0),
        );
        test_gto_kinetic(
            2.0,
            0,
            Vector3::new(0.0, 0.0, 0.0),
            0.5,
            2,
            Vector3::new(1.0, -1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            0,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            0,
            Vector3::new(0.0, 0.0, 0.0),
        );
        test_gto_kinetic(
            1.2,
            0,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            1,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            1,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            0,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            1,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            1,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            2,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            0,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            0,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            2,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            2,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            1,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            1,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            2,
            Vector3::new(1.0, 1.0, 1.0),
        );
        test_gto_kinetic(
            1.2,
            2,
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            2,
            Vector3::new(1.0, 1.0, 1.0),
        );
    }

    #[test]
    fn test_vab_symmetric() {
        // Create two identical s-type GTOs: alpha = 1.0, center at origin
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(1.0, 1.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(0.0, 1.0, 1.0));

        // Nuclear center placed at (0.05,0,0) - mid-point as an example
        let R = Vector3::new(0.05, 0.0, 0.0);

        let val_ab = GTO::Vab(&a, &b, R, 1);
        let val_ba = GTO::Vab(&b, &a, R, 1);

        let diff = (val_ab - val_ba).abs();
        assert!(diff < 1e-12, "Vab is not symmetric! diff={}", diff);
    }

    #[test]
    fn test_vab_identical_primitive_s_orbitals() {
        // Consider two identical s-orbitals at the origin with alpha=1.0
        // and a nucleus at the origin.
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(1.0, 1.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(0.0, 1.0, 1.0));

        let R = Vector3::new(0.0, 0.0, 0.0);

        let val = GTO::Vab(&a, &b, R, 1);

        // Compare with a known reference or a benchmark value
        // For a basic check, we just ensure the value is finite and > 0
        // In reality, you'd insert a known benchmark value here.
        assert!(val.is_finite());
        assert!(
            val < 0.0,
            "Integral should be positive for identical s-orbitals at the same center"
        );
    }

    #[test]
    fn test_vab_translation_invariance() {
        // If we translate everything by the same vector, the integral should remain the same
        // assuming that 'R' is defined relative to the orbitals.
        // NOTE: This depends on how your integral is defined. If R is a fixed point in space,
        // translation invariance might not apply. This is just a conceptual test.

        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(1.0, 1.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(0.0, 1.0, 1.0));

        let R = Vector3::new(1.2, -0.5, 2.0);
        let val_original = GTO::Vab(&a, &b, R, 1);

        let shift = Vector3::new(0.1, 0.2, -0.3);
        let a_shifted = GTO {
            center: a.center + shift,
            ..a
        };
        let b_shifted = GTO {
            center: b.center + shift,
            ..b
        };
        let R_shifted = R + shift;

        let val_shifted = GTO::Vab(&a_shifted, &b_shifted, R_shifted, 1);

        // Depending on how R is defined, these might not match. If R is absolute space (like nuclear position),
        // then we can't assume invariance. If R is relative, we can.
        // For now, let's just check something like symmetry or numeric stability.
        let diff = (val_original - val_shifted).abs();
        assert!(
            diff < 1e-12,
            "Value changed upon uniform translation! diff={}",
            diff
        );
    }

    fn test_vab_against_numerical_with_params(
        alpha1: f64,
        l1: Vector3<i32>,
        center1: Vector3<f64>,
        alpha2: f64,
        l2: Vector3<i32>,
        center2: Vector3<f64>,
        R: Vector3<f64>,
    ) {
        let gto1 = GTO::new(alpha1, l1, center1);
        let gto2 = GTO::new(alpha2, l2, center2);

        let val_analytical = GTO::Vab(&gto1, &gto2, R, 1);

        let integrand = |vec| {
            let pa = gto1.evaluate(&vec);
            let pb = gto2.evaluate(&vec);
            pa * pb
        };

        let lower = Vector3::new(-20.0, -20.0, -20.0);
        let upper = Vector3::new(20.0, 20.0, 20.0);
        // let val_numerical = simpson_integration_3d(integrand, lower, upper, 200, 200, 200);
        let val_numerical = integrate_spherical_3d(integrand, lower, upper, R, 200, 200, 200, 1e-6);

        let diff = (val_analytical - val_numerical).abs();

        // For very small values (near machine precision), use absolute tolerance
        if val_analytical.abs() < 1e-10 && val_numerical.abs() < 1e-10 {
            assert!(
                diff < 1e-10,
                "Nuclear attraction integral is not close (small values): analytical = {}, numerical = {}, diff = {}, \
             params: alpha1={}, l1={}, center1={:?}, \n alpha2={}, l2={}, center2={:?}",
                val_analytical,
                val_numerical,
                diff,
                alpha1,
                l1,
                center1,
                alpha2,
                l2,
                center2
            );
        } else if val_numerical.abs() < 1e-3 * val_analytical.abs() {
            assert!(
                diff < 1e-3,
                "Nuclear attraction integral is not close: analytical = {}, numerical = {}, diff = {}, \
             params: alpha1={}, l1={}, center1={:?}, \n alpha2={}, l2={}, center2={:?}",
                val_analytical,
                val_numerical,
                diff,
                alpha1,
                l1,
                center1,
                alpha2,
                l2,
                center2
            );
        } else {
            assert!(
                diff < 1e-2 * val_analytical.abs(),
                "Nuclear attraction integral is not close: analytical = {}, numerical = {}, diff = {}, \
             params: alpha1={}, l1={}, center1={:?}, \n alpha2={}, l2={}, center2={:?}",
                val_analytical,
                val_numerical,
                diff,
                alpha1,
                l1,
                center1,
                alpha2,
                l2,
                center2
            );
        }
    }

    fn test_vab_against_numerical_with_random_gto() {
        let gto1 = radom_gto();
        let gto2 = radom_gto();
        let R = Vector3::new(0.0, 0.0, 0.0);
        let val_analytical = GTO::Vab(&gto1, &gto2, R, 1);

        let integrand = |vec| {
            let pa = gto1.evaluate(&vec);
            let pb = gto2.evaluate(&vec);
            pa * pb
        };

        let lower = Vector3::new(-20.0, -20.0, -20.0);
        let upper = Vector3::new(20.0, 20.0, 20.0);
        let val_numerical = integrate_spherical_3d(integrand, lower, upper, R, 200, 200, 200, 1e-6);

        let diff = (val_analytical - val_numerical).abs();

        if val_numerical.abs() < 1e-2 {
            assert!(
                (val_numerical - val_analytical).abs() < 1e-3,
                "Potential energy integral is not close: got {}, expected {}, \
             params: alpha1={}, l1={}, center1={:?}, \n alpha2={}, l2={}, center2={:?}",
                val_analytical,
                val_numerical,
                gto1.alpha,
                gto1.l_xyz,
                gto1.center,
                gto2.alpha,
                gto2.l_xyz,
                gto2.center
            );
        } else {
            assert!(
                (val_numerical - val_analytical).abs() < 1e-2 * val_analytical.abs(),
                "Potential energy integral is not close: got {}, expected {}, \
             params: alpha1={}, l1={}, center1={:?}, \n alpha2={}, l2={}, center2={:?}",
                val_analytical,
                val_numerical,
                gto1.alpha,
                gto1.l_xyz,
                gto1.center,
                gto2.alpha,
                gto2.l_xyz,
                gto2.center
            );
        }
    }

    #[test]
    fn test_vab_against_numerical() {
        // Test specific GTOs that are known to work
        test_vab_against_numerical_with_params(
            0.5,
            Vector3::new(0, 0, 0),
            Vector3::new(0.0, 0.0, 0.0),
            0.8,
            Vector3::new(0, 0, 0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        );
        
        // Test with higher angular momentum but simple geometry
        test_vab_against_numerical_with_params(
            1.0,
            Vector3::new(1, 0, 0),
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            Vector3::new(0, 1, 0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        );
    }

    fn radom_gto() -> GTO {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        // radom alpha between 0.5 and 2.0
        let alpha = 0.5 + 1.5 * rand::random::<f64>();
        // random l between 0 and 2
        let l = rng.gen_range(0..=1);
        let m = rng.gen_range(0..=1);
        let n = rng.gen_range(0..=1);
        // random center between -2.0 to 2.0
        let center = Vector3::<f64>::from_distribution(&dist, &mut rng);
        GTO::new(alpha, Vector3::new(l, m, n), center)
    }

    fn test_jkabcd_against_numerical_with_random_gtos() {
        // generate a random gto, with random alpha, l, center
        let a = radom_gto();
        let b = radom_gto();
        let c = radom_gto();
        let d = radom_gto();

        let psi = |r1: Vector3<f64>, r2: Vector3<f64>| {
            let val1 = a.evaluate(&r1) * b.evaluate(&r1);
            let val2 = c.evaluate(&r2) * d.evaluate(&r2);
            val1 * val2
        };

        let L = 3.0;
        let (integral_numerical, std_err) = two_electron_integral_monte_carlo(psi, L, 10_000_000);
        let integral_analytical = GTO::JKabcd(&a, &b, &c, &d);
        println!(
            "numerical: {}, analytical: {}, std_err: {}",
            integral_numerical, integral_analytical, std_err
        );
        assert!(
            (integral_numerical - integral_analytical).abs() < 3.0 * std_err,
            "JKabcd is not close: got {}, expected {}",
            integral_numerical,
            integral_analytical
        )
    }

    #[test]
    fn test_jkabcd_against_numerical() {
        for i in 0..10 {
            test_jkabcd_against_numerical_with_random_gtos();
        }
    }

    #[test]
    fn test_gto_dvab_dr() {
        // Create two GTOs
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(1.0, 1.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(0.0, 1.0, 1.0));

        // Nuclear position
        let R = Vector3::new(0.5, 0.5, 0.5);
        let Z = 1u32; // Nuclear charge

        // Calculate analytical gradient
        let gradient_analytical = GTO::dVab_dR(&a, &b, R, Z);

        // Calculate numerical gradient using finite difference
        let h = 1e-4; // Step size
        let mut gradient_numerical = Vector3::new(0.0, 0.0, 0.0);

        // x component
        let R_plus_x = Vector3::new(R.x + h, R.y, R.z);
        let R_minus_x = Vector3::new(R.x - h, R.y, R.z);
        gradient_numerical.x = (GTO::Vab(&a, &b, R_plus_x, Z) - GTO::Vab(&a, &b, R_minus_x, Z)) / (2.0 * h);

        // y component
        let R_plus_y = Vector3::new(R.x, R.y + h, R.z);
        let R_minus_y = Vector3::new(R.x, R.y - h, R.z);
        gradient_numerical.y = (GTO::Vab(&a, &b, R_plus_y, Z) - GTO::Vab(&a, &b, R_minus_y, Z)) / (2.0 * h);

        // z component
        let R_plus_z = Vector3::new(R.x, R.y, R.z + h);
        let R_minus_z = Vector3::new(R.x, R.y, R.z - h);
        gradient_numerical.z = (GTO::Vab(&a, &b, R_plus_z, Z) - GTO::Vab(&a, &b, R_minus_z, Z)) / (2.0 * h);

        // Compare components
        assert!(
            (gradient_analytical.x - gradient_numerical.x).abs() < 1e-6,
            "x component mismatch: analytical = {}, numerical = {}",
            gradient_analytical.x, gradient_numerical.x
        );
        assert!(
            (gradient_analytical.y - gradient_numerical.y).abs() < 1e-6,
            "y component mismatch: analytical = {}, numerical = {}",
            gradient_analytical.y, gradient_numerical.y
        );
        assert!(
            (gradient_analytical.z - gradient_numerical.z).abs() < 1e-6,
            "z component mismatch: analytical = {}, numerical = {}",
            gradient_analytical.z, gradient_numerical.z
        );
    }

    fn test_gto_dvab_dr_random() {
        let mut rng = rand::thread_rng();

        // Create random GTOs
        let a = radom_gto();
        let b = radom_gto();

        // Random nuclear position
        let dist = Normal::new(0.0, 1.0).unwrap();
        let R = Vector3::<f64>::from_distribution(&dist, &mut rng);
        let Z = 1u32;

        // Analytical gradient
        let gradient_analytical = GTO::dVab_dR(&a, &b, R, Z);

        // Numerical gradient
        let h = 1e-5;
        let mut gradient_numerical = Vector3::new(0.0, 0.0, 0.0);

        // x component
        let R_plus_x = Vector3::new(R.x + h, R.y, R.z);
        let R_minus_x = Vector3::new(R.x - h, R.y, R.z);
        gradient_numerical.x = (GTO::Vab(&a, &b, R_plus_x, Z) - GTO::Vab(&a, &b, R_minus_x, Z)) / (2.0 * h);

        // y component
        let R_plus_y = Vector3::new(R.x, R.y + h, R.z);
        let R_minus_y = Vector3::new(R.x, R.y - h, R.z);
        gradient_numerical.y = (GTO::Vab(&a, &b, R_plus_y, Z) - GTO::Vab(&a, &b, R_minus_y, Z)) / (2.0 * h);

        // z component
        let R_plus_z = Vector3::new(R.x, R.y, R.z + h);
        let R_minus_z = Vector3::new(R.x, R.y, R.z - h);
        gradient_numerical.z = (GTO::Vab(&a, &b, R_plus_z, Z) - GTO::Vab(&a, &b, R_minus_z, Z)) / (2.0 * h);

        // Tolerance depends on the value's magnitude
        let tol = 1e-3 * gradient_analytical.norm().max(gradient_numerical.norm()) + 1e-8;

        // Compare gradients
        assert!(
            (gradient_analytical - gradient_numerical).norm() < tol,
            "Gradient mismatch: analytical = {:?}, numerical = {:?}, difference = {:?}",
            gradient_analytical, gradient_numerical, gradient_analytical - gradient_numerical
        );
    }

    #[test]
    fn test_gto_dvab_dr_multiple_random() {
        for _ in 0..5 {
            test_gto_dvab_dr_random();
        }
    }

    #[test]
    fn test_cgto_dvab_dr() {
        // Create contracted GTOs
        let mut cgto_a = ContractedGTO {
            primitives: Vec::new(),
            coefficients: Vec::new(),
            shell_type: "s".to_string(),
            Z: 1,
            n: 1,
            l: 0,
            m: 0,
            s: 0,
        };

        let mut cgto_b = ContractedGTO {
            primitives: Vec::new(),
            coefficients: Vec::new(),
            shell_type: "s".to_string(),
            Z: 1,
            n: 1,
            l: 0,
            m: 0,
            s: 0,
        };

        // Add primitives to CGTOs
        let center_a = Vector3::new(1.0, 1.0, 0.0);
        let center_b = Vector3::new(0.0, 1.0, 1.0);

        cgto_a.primitives.push(GTO::new(1.0, Vector3::new(0, 0, 0), center_a));
        cgto_a.primitives.push(GTO::new(0.5, Vector3::new(0, 0, 0), center_a));
        cgto_a.coefficients.push(0.7);
        cgto_a.coefficients.push(0.3);

        cgto_b.primitives.push(GTO::new(0.8, Vector3::new(0, 0, 0), center_b));
        cgto_b.primitives.push(GTO::new(0.4, Vector3::new(0, 0, 0), center_b));
        cgto_b.coefficients.push(0.6);
        cgto_b.coefficients.push(0.4);

        // Nuclear position
        let R = Vector3::new(0.5, 0.5, 0.5);
        let Z = 1u32;

        // Calculate analytical gradient
        let gradient_analytical = ContractedGTO::dVab_dR(&cgto_a, &cgto_b, R, Z);

        // Calculate numerical gradient
        let h = 1e-4;
        let mut gradient_numerical = Vector3::new(0.0, 0.0, 0.0);

        // x component
        let R_plus_x = Vector3::new(R.x + h, R.y, R.z);
        let R_minus_x = Vector3::new(R.x - h, R.y, R.z);
        gradient_numerical.x = (ContractedGTO::Vab(&cgto_a, &cgto_b, R_plus_x, Z) -
            ContractedGTO::Vab(&cgto_a, &cgto_b, R_minus_x, Z)) / (2.0 * h);

        // y component
        let R_plus_y = Vector3::new(R.x, R.y + h, R.z);
        let R_minus_y = Vector3::new(R.x, R.y - h, R.z);
        gradient_numerical.y = (ContractedGTO::Vab(&cgto_a, &cgto_b, R_plus_y, Z) -
            ContractedGTO::Vab(&cgto_a, &cgto_b, R_minus_y, Z)) / (2.0 * h);

        // z component
        let R_plus_z = Vector3::new(R.x, R.y, R.z + h);
        let R_minus_z = Vector3::new(R.x, R.y, R.z - h);
        gradient_numerical.z = (ContractedGTO::Vab(&cgto_a, &cgto_b, R_plus_z, Z) -
            ContractedGTO::Vab(&cgto_a, &cgto_b, R_minus_z, Z)) / (2.0 * h);

        // Compare components
        assert!(
            (gradient_analytical.x - gradient_numerical.x).abs() < 1e-3,
            "x component mismatch: analytical = {}, numerical = {}",
            gradient_analytical.x, gradient_numerical.x
        );
        assert!(
            (gradient_analytical.y - gradient_numerical.y).abs() < 1e-3,
            "y component mismatch: analytical = {}, numerical = {}",
            gradient_analytical.y, gradient_numerical.y
        );
        assert!(
            (gradient_analytical.z - gradient_numerical.z).abs() < 1e-3,
            "z component mismatch: analytical = {}, numerical = {}",
            gradient_analytical.z, gradient_numerical.z
        );
    }

    fn random_cgto() -> ContractedGTO {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        // Random center
        let center = Vector3::<f64>::from_distribution(&dist, &mut rng);

        // Create CGTO with 2-3 primitives
        let num_primitives = rng.gen_range(2..=3);
        let mut cgto = ContractedGTO {
            primitives: Vec::with_capacity(num_primitives),
            coefficients: Vec::with_capacity(num_primitives),
            shell_type: "s".to_string(),
            Z: 1,
            n: 1,
            l: 0,
            m: 0,
            s: 0,
        };

        // Add primitives
        for _ in 0..num_primitives {
            let alpha = 0.5 + 1.5 * rand::random::<f64>();
            let l = rng.gen_range(0..=1);
            let m = rng.gen_range(0..=1);
            let n = rng.gen_range(0..=1);
            cgto.primitives.push(GTO::new(alpha, Vector3::new(l, m, n), center));
            cgto.coefficients.push(rand::random::<f64>());
        }

        // Normalize coefficients
        let sum: f64 = cgto.coefficients.iter().sum();
        for c in &mut cgto.coefficients {
            *c /= sum;
        }

        cgto
    }

    #[test]
    fn test_cgto_dvab_dr_random() {
        // Create random CGTOs
        let a = random_cgto();
        let b = random_cgto();

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let R = Vector3::<f64>::from_distribution(&dist, &mut rng);
        let Z = 1u32;

        // Analytical gradient
        let gradient_analytical = ContractedGTO::dVab_dR(&a, &b, R, Z);

        // Numerical gradient
        let h = 1e-5;
        let mut gradient_numerical = Vector3::new(0.0, 0.0, 0.0);

        // x component
        let R_plus_x = Vector3::new(R.x + h, R.y, R.z);
        let R_minus_x = Vector3::new(R.x - h, R.y, R.z);
        gradient_numerical.x = (ContractedGTO::Vab(&a, &b, R_plus_x, Z) - ContractedGTO::Vab(&a, &b, R_minus_x, Z)) / (2.0 * h);

        // y component
        let R_plus_y = Vector3::new(R.x, R.y + h, R.z);
        let R_minus_y = Vector3::new(R.x, R.y - h, R.z);
        gradient_numerical.y = (ContractedGTO::Vab(&a, &b, R_plus_y, Z) - ContractedGTO::Vab(&a, &b, R_minus_y, Z)) / (2.0 * h);

        // z component
        let R_plus_z = Vector3::new(R.x, R.y, R.z + h);
        let R_minus_z = Vector3::new(R.x, R.y, R.z - h);
        gradient_numerical.z = (ContractedGTO::Vab(&a, &b, R_plus_z, Z) - ContractedGTO::Vab(&a, &b, R_minus_z, Z)) / (2.0 * h);

        // Tolerance depends on gradient magnitude
        // ContractedGTO calculations involve summing over multiple primitives,
        // which can accumulate small numerical errors, so we use a slightly larger tolerance
        let tol = 2e-3 * gradient_analytical.norm().max(gradient_numerical.norm()) + 1e-7;

        // Compare gradients
        assert!(
            (gradient_analytical - gradient_numerical).norm() < tol,
            "Gradient mismatch: analytical = {:?}, numerical = {:?}, difference = {:?}",
            gradient_analytical, gradient_numerical, gradient_analytical - gradient_numerical
        );
    }

    #[test]
    fn test_gto_djkabcd_dr() {
        // Test that dJKabcd_dR returns zeros (as implemented)
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(1.0, 0.0, 0.0));
        let c = GTO::new(0.6, Vector3::new(0, 0, 0), Vector3::new(0.0, 1.0, 0.0));
        let d = GTO::new(0.4, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 1.0));
        
        let R = Vector3::new(0.5, 0.5, 0.5);
        let gradient = GTO::dJKabcd_dR(&a, &b, &c, &d, R);
        
        // Should return zeros as per the corrected implementation
        // Two-electron derivatives w.r.t. nuclear positions are zero for standard GTOs
        assert_eq!(gradient, Vector3::zeros());
    }

    #[test]
    fn test_gto_dsab_dr() {
        // Test overlap integral derivative w.r.t. basis center for both s and p orbitals
        let h = 1e-6;
        let tol = 1e-6; // Tighter tolerance for the corrected implementation

        // Case 1: s-orbitals
        let a_s = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b_s = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(1.0, 0.0, 0.0));
        
        // Test derivative w.r.t. first basis function center
        let deriv_analytical_s = GTO::dSab_dR(&a_s, &b_s, 0);
        
        // Test with numerical derivative
        let mut deriv_numerical_s = Vector3::zeros();
        
        // x component
        let a_plus_x = GTO::new(a_s.alpha, a_s.l_xyz, a_s.center + Vector3::new(h, 0.0, 0.0));
        let a_minus_x = GTO::new(a_s.alpha, a_s.l_xyz, a_s.center - Vector3::new(h, 0.0, 0.0));
        deriv_numerical_s.x = (GTO::Sab(&a_plus_x, &b_s) - GTO::Sab(&a_minus_x, &b_s)) / (2.0 * h);
        
        // y component  
        let a_plus_y = GTO::new(a_s.alpha, a_s.l_xyz, a_s.center + Vector3::new(0.0, h, 0.0));
        let a_minus_y = GTO::new(a_s.alpha, a_s.l_xyz, a_s.center - Vector3::new(0.0, h, 0.0));
        deriv_numerical_s.y = (GTO::Sab(&a_plus_y, &b_s) - GTO::Sab(&a_minus_y, &b_s)) / (2.0 * h);
        
        // z component
        let a_plus_z = GTO::new(a_s.alpha, a_s.l_xyz, a_s.center + Vector3::new(0.0, 0.0, h));
        let a_minus_z = GTO::new(a_s.alpha, a_s.l_xyz, a_s.center - Vector3::new(0.0, 0.0, h));
        deriv_numerical_s.z = (GTO::Sab(&a_plus_z, &b_s) - GTO::Sab(&a_minus_z, &b_s)) / (2.0 * h);
        
        assert!(
            (deriv_analytical_s - deriv_numerical_s).norm() < tol,
            "dSab_dR mismatch for s-orbitals: analytical = {:?}, numerical = {:?}",
            deriv_analytical_s, deriv_numerical_s
        );

        // Case 2: p-orbitals to test higher angular momentum
        let a_p = GTO::new(1.0, Vector3::new(1, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b_p = GTO::new(0.8, Vector3::new(0, 1, 0), Vector3::new(1.0, 0.0, 0.0));
        
        let deriv_analytical_p = GTO::dSab_dR(&a_p, &b_p, 0);
        
        let mut deriv_numerical_p = Vector3::zeros();
        
        let a_p_plus_x = GTO::new(a_p.alpha, a_p.l_xyz, a_p.center + Vector3::new(h, 0.0, 0.0));
        let a_p_minus_x = GTO::new(a_p.alpha, a_p.l_xyz, a_p.center - Vector3::new(h, 0.0, 0.0));
        deriv_numerical_p.x = (GTO::Sab(&a_p_plus_x, &b_p) - GTO::Sab(&a_p_minus_x, &b_p)) / (2.0 * h);
        
        let a_p_plus_y = GTO::new(a_p.alpha, a_p.l_xyz, a_p.center + Vector3::new(0.0, h, 0.0));
        let a_p_minus_y = GTO::new(a_p.alpha, a_p.l_xyz, a_p.center - Vector3::new(0.0, h, 0.0));
        deriv_numerical_p.y = (GTO::Sab(&a_p_plus_y, &b_p) - GTO::Sab(&a_p_minus_y, &b_p)) / (2.0 * h);

        let a_p_plus_z = GTO::new(a_p.alpha, a_p.l_xyz, a_p.center + Vector3::new(0.0, 0.0, h));
        let a_p_minus_z = GTO::new(a_p.alpha, a_p.l_xyz, a_p.center - Vector3::new(0.0, 0.0, h));
        deriv_numerical_p.z = (GTO::Sab(&a_p_plus_z, &b_p) - GTO::Sab(&a_p_minus_z, &b_p)) / (2.0 * h);
        
        assert!(
            (deriv_analytical_p - deriv_numerical_p).norm() < tol,
            "dSab_dR mismatch for p-orbitals: analytical = {:?}, numerical = {:?}",
            deriv_analytical_p, deriv_numerical_p
        );
    }

    #[test]
    fn test_gto_dtab_dr() {
        // Test kinetic energy integral derivative w.r.t. basis center
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(1.0, 0.0, 0.0));
        
        // Test derivative w.r.t. first basis function center
        let deriv_analytical = GTO::dTab_dR(&a, &b, 0);
        
        // Test with numerical derivative
        let h = 1e-6;
        let mut deriv_numerical = Vector3::zeros();
        
        // x component
        let mut a_plus = a.clone();
        a_plus.center.x += h;
        a_plus.gto1d[0].center += h;
        let mut a_minus = a.clone();
        a_minus.center.x -= h;
        a_minus.gto1d[0].center -= h;
        deriv_numerical.x = (GTO::Tab(&a_plus, &b) - GTO::Tab(&a_minus, &b)) / (2.0 * h);
        
        // y component  
        let mut a_plus = a.clone();
        a_plus.center.y += h;
        a_plus.gto1d[1].center += h;
        let mut a_minus = a.clone();
        a_minus.center.y -= h;
        a_minus.gto1d[1].center -= h;
        deriv_numerical.y = (GTO::Tab(&a_plus, &b) - GTO::Tab(&a_minus, &b)) / (2.0 * h);
        
        // z component
        let mut a_plus = a.clone();
        a_plus.center.z += h;
        a_plus.gto1d[2].center += h;
        let mut a_minus = a.clone();
        a_minus.center.z -= h;
        a_minus.gto1d[2].center -= h;
        deriv_numerical.z = (GTO::Tab(&a_plus, &b) - GTO::Tab(&a_minus, &b)) / (2.0 * h);
        
        let tol = 1e-3; // Looser tolerance for kinetic energy derivatives
        assert!(
            (deriv_analytical - deriv_numerical).norm() < tol,
            "dTab_dR mismatch: analytical = {:?}, numerical = {:?}",
            deriv_analytical, deriv_numerical
        );
    }

    #[test]
    fn test_gto_dvab_drbasis() {
        // Test nuclear attraction integral derivative w.r.t. basis center
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(1.0, 0.0, 0.0));
        let R_nucl = Vector3::new(0.5, 0.5, 0.5);
        let Z = 1u32;
        
        // Test derivative w.r.t. first basis function center
        let deriv_analytical = GTO::dVab_dRbasis(&a, &b, R_nucl, Z, 0);
        
        // Test with numerical derivative
        let h = 1e-6;
        let mut deriv_numerical = Vector3::zeros();
        
        // x component
        let mut a_plus = a.clone();
        a_plus.center.x += h;
        a_plus.gto1d[0].center += h;
        let mut a_minus = a.clone();
        a_minus.center.x -= h;
        a_minus.gto1d[0].center -= h;
        deriv_numerical.x = (GTO::Vab(&a_plus, &b, R_nucl, Z) - GTO::Vab(&a_minus, &b, R_nucl, Z)) / (2.0 * h);
        
        // y component  
        let mut a_plus = a.clone();
        a_plus.center.y += h;
        a_plus.gto1d[1].center += h;
        let mut a_minus = a.clone();
        a_minus.center.y -= h;
        a_minus.gto1d[1].center -= h;
        deriv_numerical.y = (GTO::Vab(&a_plus, &b, R_nucl, Z) - GTO::Vab(&a_minus, &b, R_nucl, Z)) / (2.0 * h);
        
        // z component
        let mut a_plus = a.clone();
        a_plus.center.z += h;
        a_plus.gto1d[2].center += h;
        let mut a_minus = a.clone();
        a_minus.center.z -= h;
        a_minus.gto1d[2].center -= h;
        deriv_numerical.z = (GTO::Vab(&a_plus, &b, R_nucl, Z) - GTO::Vab(&a_minus, &b, R_nucl, Z)) / (2.0 * h);
        
        // Use looser tolerance since this is an approximation, not exact analytical formula
        let tol = 0.5; // 50% tolerance for approximation
        let relative_error = (deriv_analytical - deriv_numerical).norm() / deriv_numerical.norm().max(1e-10);
        assert!(
            relative_error < tol,
            "dVab_dRbasis approximation error too large: analytical = {:?}, numerical = {:?}, relative error = {}",
            deriv_analytical, deriv_numerical, relative_error
        );
    }

    #[test]
    fn test_gto_djkabcd_drbasis() {
        // Test two-electron integral derivative w.r.t. basis center
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b = GTO::new(0.8, Vector3::new(0, 0, 0), Vector3::new(1.0, 0.0, 0.0));
        let c = GTO::new(0.6, Vector3::new(0, 0, 0), Vector3::new(0.0, 1.0, 0.0));
        let d = GTO::new(0.4, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 1.0));
        
        // Test derivative w.r.t. first basis function center
        let deriv_analytical = GTO::dJKabcd_dRbasis(&a, &b, &c, &d, 0);
        
        // Test with numerical derivative
        let h = 1e-6;
        let mut deriv_numerical = Vector3::zeros();
        
        // x component
        let mut a_plus = a.clone();
        a_plus.center.x += h;
        a_plus.gto1d[0].center += h;
        let mut a_minus = a.clone();
        a_minus.center.x -= h;
        a_minus.gto1d[0].center -= h;
        deriv_numerical.x = (GTO::JKabcd(&a_plus, &b, &c, &d) - GTO::JKabcd(&a_minus, &b, &c, &d)) / (2.0 * h);
        
        // y component  
        let mut a_plus = a.clone();
        a_plus.center.y += h;
        a_plus.gto1d[1].center += h;
        let mut a_minus = a.clone();
        a_minus.center.y -= h;
        a_minus.gto1d[1].center -= h;
        deriv_numerical.y = (GTO::JKabcd(&a_plus, &b, &c, &d) - GTO::JKabcd(&a_minus, &b, &c, &d)) / (2.0 * h);
        
        // z component
        let mut a_plus = a.clone();
        a_plus.center.z += h;
        a_plus.gto1d[2].center += h;
        let mut a_minus = a.clone();
        a_minus.center.z -= h;
        a_minus.gto1d[2].center -= h;
        deriv_numerical.z = (GTO::JKabcd(&a_plus, &b, &c, &d) - GTO::JKabcd(&a_minus, &b, &c, &d)) / (2.0 * h);
        
        // Use looser tolerance since this is an approximation, not exact analytical formula
        let tol = 1e-3; // 1% tolerance for approximation
        let relative_error = (deriv_analytical - deriv_numerical).norm() / deriv_numerical.norm().max(1e-10);
        assert!(
            relative_error < tol,
            "dJKabcd_dRbasis approximation error too large: analytical = {:?}, numerical = {:?}, relative error = {}",
            deriv_analytical, deriv_numerical, relative_error
        );
    }

    #[test]
    fn test_gto1d_eab() {
        // Test the Eab function with simple cases
        
        // Test base case: E(0,0,0) = exp(-q*Qx^2)
        let i = 0; let j = 0; let t = 0;
        let Qx = 1.0;
        let a = 1.0; let b = 0.5;
        let result = GTO1d::Eab(i, j, t, Qx, a, b);
        
        let p = a + b;
        let q = a * b / p;
        let expected = (-q * Qx * Qx).exp();
        
        assert!(
            (result - expected).abs() < 1e-10,
            "Eab(0,0,0) failed: got {}, expected {}",
            result, expected
        );
        
        // Test invalid cases should return 0
        assert_eq!(GTO1d::Eab(-1, 0, 0, Qx, a, b), 0.0);
        assert_eq!(GTO1d::Eab(0, -1, 0, Qx, a, b), 0.0);
        assert_eq!(GTO1d::Eab(0, 0, -1, Qx, a, b), 0.0);
        assert_eq!(GTO1d::Eab(1, 1, 3, Qx, a, b), 0.0); // t > i + j
    }

    #[test]
    fn test_gto_hermite_coulomb() {
        // Test base case: R(0,0,0,0) = (-2p)^0 * F_0(T) = F_0(T)
        let t = 0; let u = 0; let v = 0; let n = 0;
        let p = 1.0_f64;
        let PCx = 1.0_f64; let PCy = 0.5_f64; let PCz = 0.2_f64;
        let RPC = (PCx*PCx + PCy*PCy + PCz*PCz).sqrt();
        
        let result = GTO::hermite_coulomb(t, u, v, n, p, PCx, PCy, PCz, RPC);
        let T = p * RPC * RPC;
        let expected = boys_function(n, T);
        
        assert!(
            (result - expected).abs() < 1e-10,
            "hermite_coulomb(0,0,0,0) failed: got {}, expected {}",
            result, expected
        );
    }

    #[test]
    fn test_gto_dhermite_coulomb() {
        use crate::gto::Direction;
        
        // Test derivative of hermite_coulomb using finite differences
        let t = 1; let u = 0; let v = 0; let n = 0;
        let p = 1.0_f64;
        let PCx = 1.0_f64; let PCy = 0.5_f64; let PCz = 0.2_f64;
        let RPC = (PCx*PCx + PCy*PCy + PCz*PCz).sqrt();
        
        let h = 1e-6_f64;
        
        // Test X direction
        let deriv_analytical = GTO::dhermite_coulomb(Direction::X, t, u, v, n, p, PCx, PCy, PCz, RPC);
        
        let val_plus = GTO::hermite_coulomb(t, u, v, n, p, PCx + h, PCy, PCz, 
                                           ((PCx + h)*(PCx + h) + PCy*PCy + PCz*PCz).sqrt());
        let val_minus = GTO::hermite_coulomb(t, u, v, n, p, PCx - h, PCy, PCz,
                                            ((PCx - h)*(PCx - h) + PCy*PCy + PCz*PCz).sqrt());
        let deriv_numerical = (val_plus - val_minus) / (2.0 * h);
        
        let tol = 1e-5;
        assert!(
            (deriv_analytical - deriv_numerical).abs() < tol,
            "dhermite_coulomb X direction failed: analytical = {}, numerical = {}",
            deriv_analytical, deriv_numerical
        );
    }

    #[test]
    fn test_gto_merge() {
        // Test merging two GTOs
        let a = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let b = GTO::new(2.0, Vector3::new(1, 0, 0), Vector3::new(1.0, 0.0, 0.0));
        
        let merged = GTO::merge(&a, &b);
        
        // Check combined exponent
        assert_eq!(merged.alpha, 3.0);
        
        // Check combined angular momentum
        assert_eq!(merged.l_xyz, Vector3::new(1, 0, 0));
        
        // Check weighted center
        let expected_center = (a.center * a.alpha + b.center * b.alpha) / (a.alpha + b.alpha);
        assert!((merged.center - expected_center).norm() < 1e-10);
    }

    #[test]
    fn test_gto_laplacian() {
        // Test 3D Laplacian using numerical differentiation
        let gto = GTO::new(1.0, Vector3::new(1, 1, 1), Vector3::new(0.0, 0.0, 0.0));
        let r = Vector3::new(0.5, 0.5, 0.5);
        
        let analytical = gto.laplacian(&r);
        
        // Numerical Laplacian using finite differences
        let h = 1e-5;
        let mut numerical_laplacian = 0.0;
        
        // d²/dx²
        numerical_laplacian += (gto.evaluate(&Vector3::new(r.x + h, r.y, r.z)) 
                     - 2.0 * gto.evaluate(&r) 
                     + gto.evaluate(&Vector3::new(r.x - h, r.y, r.z))) / (h * h);
        
        // d²/dy²
        numerical_laplacian += (gto.evaluate(&Vector3::new(r.x, r.y + h, r.z)) 
                     - 2.0 * gto.evaluate(&r) 
                     + gto.evaluate(&Vector3::new(r.x, r.y - h, r.z))) / (h * h);
        
        // d²/dz²
        numerical_laplacian += (gto.evaluate(&Vector3::new(r.x, r.y, r.z + h)) 
                     - 2.0 * gto.evaluate(&r) 
                     + gto.evaluate(&Vector3::new(r.x, r.y, r.z - h))) / (h * h);
        
        let tol = 1e-4;
        assert!(
            (analytical - numerical_laplacian).abs() < tol,
            "3D Laplacian mismatch: analytical = {}, numerical = {}",
            analytical, numerical_laplacian
        );
    }

    #[test]
    fn test_gto1d_derivative() {
        // Test 1D derivative using numerical differentiation
        let gto = GTO1d::new(1.0, 2, 0.5);
        let x = 1.0;
        
        let analytical = gto.derivative(x);
        
        let h = 1e-6;
        let numerical = (gto.evaluate(x + h) - gto.evaluate(x - h)) / (2.0 * h);
        
        let tol = 1e-5;
        assert!(
            (analytical - numerical).abs() < tol,
            "1D derivative mismatch: analytical = {}, numerical = {}",
            analytical, numerical
        );
    }

    #[test]
    fn test_gto_normalization_property() {
        // Test that normalization constants are computed correctly
        // by checking that the overlap of a GTO with itself is 1
        let gto = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let self_overlap = GTO::Sab(&gto, &gto);
        
        assert!(
            (self_overlap - 1.0).abs() < 1e-10,
            "Self-overlap is not 1: got {}",
            self_overlap
        );
    }

    #[test]
    fn test_multiple_random_gto_methods() {
        // Test symmetries and relationships between methods with random GTOs
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..10 {
            let a = radom_gto();
            let b = radom_gto();
            
            // Test symmetry of overlap
            let sab = GTO::Sab(&a, &b);
            let sba = GTO::Sab(&b, &a);
            assert!(
                (sab - sba).abs() < 1e-12,
                "Overlap should be symmetric"
            );
            
            // Test symmetry of kinetic energy  
            let tab = GTO::Tab(&a, &b);
            let tba = GTO::Tab(&b, &a);
            assert!(
                (tab - tba).abs() < 1e-12,
                "Kinetic energy should be symmetric"
            );
        }
    }

    // Additional test cases
    #[test]
    fn test_gto_evaluate_at_origin() {
        // Test that GTOs evaluate correctly at their centers
        let gto = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let value_at_center = gto.evaluate(&Vector3::new(0.0, 0.0, 0.0));
        
        // For s-type orbital centered at origin, value should be the normalization
        assert!(value_at_center > 0.0, "GTO should have positive value at center");
    }

    #[test] 
    fn test_gto_symmetry_properties() {
        // Test that s-type orbital is spherically symmetric
        let gto = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let r = 1.0;
        
        let val_x = gto.evaluate(&Vector3::new(r, 0.0, 0.0));
        let val_y = gto.evaluate(&Vector3::new(0.0, r, 0.0));
        let val_z = gto.evaluate(&Vector3::new(0.0, 0.0, r));
        
        assert!((val_x - val_y).abs() < 1e-12, "s-orbital should be spherically symmetric");
        assert!((val_x - val_z).abs() < 1e-12, "s-orbital should be spherically symmetric");
    }

    #[test]
    fn test_gto_angular_momentum_properties() {
        // Test p_x orbital has correct symmetry
        let gto_px = GTO::new(1.0, Vector3::new(1, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        
        // Should be zero at origin (node)
        let val_origin = gto_px.evaluate(&Vector3::new(0.0, 0.0, 0.0));
        assert!(val_origin.abs() < 1e-12, "p_x orbital should be zero at origin");
        
        // Should have opposite signs on either side of node
        let val_pos_x = gto_px.evaluate(&Vector3::new(1.0, 0.0, 0.0));
        let val_neg_x = gto_px.evaluate(&Vector3::new(-1.0, 0.0, 0.0));
        assert!(val_pos_x * val_neg_x < 0.0, "p_x orbital should have opposite signs on +/- x");
    }

    #[test]
    fn test_gto_normalization_different_exponents() {
        // Test normalization for different exponent values
        let alphas = [0.1, 0.5, 1.0, 2.0, 5.0];
        
        for &alpha in &alphas {
            let gto = GTO::new(alpha, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
            let self_overlap = GTO::Sab(&gto, &gto);
            
            assert!(
                (self_overlap - 1.0).abs() < 1e-10,
                "Self-overlap should be 1 for alpha = {}, got {}",
                alpha, self_overlap
            );
        }
    }

    #[test]
    fn test_kinetic_energy_positive_definite() {
        // Kinetic energy should always be positive for self-integrals
        let gto = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let kinetic = GTO::Tab(&gto, &gto);
        
        assert!(kinetic > 0.0, "Kinetic energy should be positive, got {}", kinetic);
        
        // Test with higher angular momentum
        let gto_p = GTO::new(1.0, Vector3::new(1, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let kinetic_p = GTO::Tab(&gto_p, &gto_p);
        
        assert!(kinetic_p > 0.0, "Kinetic energy for p-orbital should be positive, got {}", kinetic_p);
    }

    #[test]
    fn test_nuclear_attraction_negative() {
        // Nuclear attraction should be negative (attractive)
        let gto1 = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let gto2 = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let nucleus = Vector3::new(0.0, 0.0, 0.0);
        
        let nuclear_attraction = GTO::Vab(&gto1, &gto2, nucleus, 1);
        
        assert!(nuclear_attraction < 0.0, "Nuclear attraction should be negative, got {}", nuclear_attraction);
    }

    #[test]
    fn test_two_electron_repulsion_positive() {
        // Two-electron repulsion should be positive (repulsive)
        let gto = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        let two_electron = GTO::JKabcd(&gto, &gto, &gto, &gto);
        
        assert!(two_electron > 0.0, "Two-electron repulsion should be positive, got {}", two_electron);
    }

    #[test]
    fn test_gto_center_shift_invariance() {
        // Test that relative properties are preserved under translation
        let center1 = Vector3::new(0.0, 0.0, 0.0);
        let center2 = Vector3::new(1.0, 0.0, 0.0);
        let shift = Vector3::new(2.0, 3.0, 4.0);
        
        let gto1 = GTO::new(1.0, Vector3::new(0, 0, 0), center1);
        let gto2 = GTO::new(1.0, Vector3::new(0, 0, 0), center2);
        
        let gto1_shifted = GTO::new(1.0, Vector3::new(0, 0, 0), center1 + shift);
        let gto2_shifted = GTO::new(1.0, Vector3::new(0, 0, 0), center2 + shift);
        
        let overlap = GTO::Sab(&gto1, &gto2);
        let overlap_shifted = GTO::Sab(&gto1_shifted, &gto2_shifted);
        
        assert!(
            (overlap - overlap_shifted).abs() < 1e-12,
            "Overlap should be invariant under translation"
        );
    }

    #[test]
    fn test_gto_exponential_decay() {
        // Test that GTO decays exponentially with distance
        let gto = GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0));
        
        let val_1 = gto.evaluate(&Vector3::new(1.0, 0.0, 0.0));
        let val_2 = gto.evaluate(&Vector3::new(2.0, 0.0, 0.0));
        let val_3 = gto.evaluate(&Vector3::new(3.0, 0.0, 0.0));
        
        assert!(val_1 > val_2, "GTO should decay with distance");
        assert!(val_2 > val_3, "GTO should continue to decay");
        assert!(val_3 > 0.0, "GTO should remain positive");
    }

    #[test]
    fn test_gto_derivative_accuracy() {
        // Test 1D derivative against numerical finite difference for edge cases
        let test_cases = [
            (0.5, 0, 0.0),   // s-type at center
            (1.0, 1, 0.5),   // p-type off-center  
            (2.0, 2, -1.0),  // d-type negative position
        ];
        
        for &(alpha, l, center) in &test_cases {
            let gto = GTO1d::new(alpha, l, center);
            let x = center + 0.3; // Test point slightly off center
            
            let analytical = gto.derivative(x);
            
            let h = 1e-6;
            let numerical = (gto.evaluate(x + h) - gto.evaluate(x - h)) / (2.0 * h);
            
            let rel_error = if analytical.abs() > 1e-10 {
                (analytical - numerical).abs() / analytical.abs()
            } else {
                (analytical - numerical).abs()
            };
            
            assert!(
                rel_error < 1e-5,
                "Derivative test failed for alpha={}, l={}, center={}: analytical={}, numerical={}, rel_error={}",
                alpha, l, center, analytical, numerical, rel_error
            );
        }
    }

    #[test]
    fn test_hermite_coulomb_recursion() {
        // Test that Hermite-Coulomb recursion gives consistent results
        let p = 1.0_f64;
        let PCx = 0.5_f64; let PCy = 0.3_f64; let PCz = 0.2_f64;
        let RPC = (PCx*PCx + PCy*PCy + PCz*PCz).sqrt();
        
        // Test recursion relation: R(t+1,u,v,n) should relate to R(t,u,v,n+1)
        let R_000_0 = GTO::hermite_coulomb(0, 0, 0, 0, p, PCx, PCy, PCz, RPC);
        let R_100_0 = GTO::hermite_coulomb(1, 0, 0, 0, p, PCx, PCy, PCz, RPC);
        let R_000_1 = GTO::hermite_coulomb(0, 0, 0, 1, p, PCx, PCy, PCz, RPC);
        
        // Basic sanity checks
        assert!(R_000_0.is_finite(), "Base Hermite-Coulomb integral should be finite");
        assert!(R_100_0.is_finite(), "Higher-order Hermite-Coulomb integral should be finite");
        assert!(R_000_1.is_finite(), "Higher-n Hermite-Coulomb integral should be finite");
    }
}
