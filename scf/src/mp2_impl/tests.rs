//! Tests for MP2 implementation

#[cfg(test)]
mod tests {
    use super::super::MP2;
    use basis::basis::Basis;
    use nalgebra::{DMatrix, DVector, Vector3};
    use periodic_table_on_an_enum::Element;
    use std::sync::Arc;

    /// Mock basis for testing
    #[derive(Clone)]
    struct MockBasis {
        center: Vector3<f64>,
    }

    impl Basis for MockBasis {
        fn evaluate(&self, _r: &Vector3<f64>) -> f64 {
            1.0
        }

        fn Sab(_a: &Self, _b: &Self) -> f64 {
            1.0
        }

        fn Tab(_a: &Self, _b: &Self) -> f64 {
            0.5
        }

        fn Vab(_a: &Self, _b: &Self, _R: Vector3<f64>, _Z: u32) -> f64 {
            -1.0
        }

        fn dVab_dR(_a: &Self, _b: &Self, _R: Vector3<f64>, _Z: u32) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn JKabcd(_a: &Self, _b: &Self, _c: &Self, _d: &Self) -> f64 {
            // Simple mock two-electron integral
            0.1
        }

        fn dJKabcd_dR(
            _a: &Self,
            _b: &Self,
            _c: &Self,
            _d: &Self,
            _R: Vector3<f64>,
        ) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dSab_dR(_a: &Self, _b: &Self, _atom_idx: usize) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dTab_dR(_a: &Self, _b: &Self, _atom_idx: usize) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dVab_dRbasis(
            _a: &Self,
            _b: &Self,
            _R: Vector3<f64>,
            _Z: u32,
            _atom_idx: usize,
        ) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dJKabcd_dRbasis(
            _a: &Self,
            _b: &Self,
            _c: &Self,
            _d: &Self,
            _atom_idx: usize,
        ) -> Vector3<f64> {
            Vector3::zeros()
        }
    }

    #[test]
    fn test_mp2_initialization() {
        // Create a simple system with 2 basis functions and 2 electrons (1 occupied, 1 virtual)
        let num_basis = 2;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-0.5, 0.5]);

        let mut mo_basis = Vec::new();
        for i in 0..num_basis {
            let basis = MockBasis {
                center: Vector3::new(0.0, 0.0, i as f64),
            };
            mo_basis.push(Arc::new(basis));
        }

        let elems = vec![
            Element::from_symbol("H").unwrap(),
            Element::from_symbol("H").unwrap(),
        ];

        let mp2 = MP2::new(mo_coeffs, orbital_energies, mo_basis, elems);

        assert_eq!(mp2.num_basis, 2);
        assert_eq!(mp2.num_occ, 1);
        assert_eq!(mp2.num_virt, 1);
        assert!(mp2.correlation_energy.is_none());
    }

    #[test]
    fn test_mp2_zero_virtual_orbitals() {
        // Test case with no virtual orbitals - should return 0 correlation energy
        let num_basis = 1;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-0.5]);

        let mut mo_basis = Vec::new();
        let basis = MockBasis {
            center: Vector3::zeros(),
        };
        mo_basis.push(Arc::new(basis));

        let elems = vec![
            Element::from_symbol("H").unwrap(),
            Element::from_symbol("H").unwrap(),
        ];

        let mut mp2 = MP2::new(mo_coeffs, orbital_energies, mo_basis, elems);
        let correlation_energy = mp2.calculate_mp2_energy();

        assert_eq!(correlation_energy, 0.0);
        assert_eq!(mp2.correlation_energy, Some(0.0));
    }

    #[test]
    fn test_mp2_energy_calculation() {
        // Create a minimal system to test energy calculation
        // 2 basis functions, 2 electrons -> 1 occupied, 1 virtual
        let num_basis = 2;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, 1.0]); // One below, one above Fermi level

        let mut mo_basis = Vec::new();
        for i in 0..num_basis {
            let basis = MockBasis {
                center: Vector3::new(0.0, 0.0, i as f64),
            };
            mo_basis.push(Arc::new(basis));
        }

        let elems = vec![
            Element::from_symbol("H").unwrap(),
            Element::from_symbol("H").unwrap(),
        ];

        let mut mp2 = MP2::new(mo_coeffs, orbital_energies, mo_basis, elems);
        let correlation_energy = mp2.calculate_mp2_energy();

        // With our mock basis, all integrals are 0.1
        // E_MP2 = (ia|jb) * [2*(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)
        // For i=j=0, a=b=1: 0.1 * (0.1 - 0.1) / (-1 + -1 - 1 - 1) = 0
        // The calculation should complete without panic

        // Correlation energy should be finite
        assert!(correlation_energy.is_finite());
        assert!(mp2.correlation_energy.is_some());
    }

    #[test]
    fn test_mp2_energy_direct_method() {
        // Test the direct MP2 energy calculation method
        let num_basis = 2;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, 1.0]);

        let mut mo_basis = Vec::new();
        for i in 0..num_basis {
            let basis = MockBasis {
                center: Vector3::new(0.0, 0.0, i as f64),
            };
            mo_basis.push(Arc::new(basis));
        }

        let elems = vec![
            Element::from_symbol("H").unwrap(),
            Element::from_symbol("H").unwrap(),
        ];

        let mut mp2 = MP2::new(mo_coeffs, orbital_energies, mo_basis, elems);

        // Calculate using direct method
        let correlation_energy_direct = mp2.calculate_mp2_energy_direct();

        // Should be finite
        assert!(correlation_energy_direct.is_finite());
    }
}
