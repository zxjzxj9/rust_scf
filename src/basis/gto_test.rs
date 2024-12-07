use nalgebra::Vector3;
use crate::basis::gto::{GTO1d, GTO};
use crate::basis::helper::{simpson_integration, simpson_integration_3d};

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
        let integrand = |x, y, z| gto1.evaluate(&Vector3::new(x, y, z)) * gto2.evaluate(&Vector3::new(x, y, z));

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
}