//! Tests for CI implementation

#[cfg(test)]
mod tests {
    use super::super::CI;
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
    fn test_ci_initialization() {
        // Create a simple system with 4 basis functions and 2 electrons (1 occupied, 3 virtual)
        let num_basis = 4;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-0.5, 0.3, 0.5, 0.8]);

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
        let hf_energy = -1.0;

        let ci = CI::new(
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            hf_energy,
            5,
            1e-6,
        );

        assert_eq!(ci.num_basis, 4);
        assert_eq!(ci.num_occ, 1);
        assert_eq!(ci.num_virt, 3);
        assert_eq!(ci.max_states, 5);
        assert!((ci.convergence_threshold - 1e-6).abs() < 1e-10);
        assert!((ci.hf_energy - hf_energy).abs() < 1e-10);
        assert!(ci.correlation_energy.is_none());
        assert!(ci.excitation_energies.is_empty());
    }

    #[test]
    fn test_cis_calculation() {
        // Test CIS calculation with mock system
        let num_basis = 4;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, 0.5, 1.0, 1.5]);

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
        let hf_energy = -1.0;

        let mut ci = CI::new(
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            hf_energy,
            3,
            1e-6,
        );

        // Calculate CIS energies
        let excitation_energies = ci.calculate_cis_energies(3);

        // Should have some excited states
        assert!(!excitation_energies.is_empty());
        assert!(excitation_energies.len() <= 3);

        // Excitation energies should be positive
        for &energy in &excitation_energies {
            assert!(
                energy > 0.0,
                "Excitation energy should be positive: {}",
                energy
            );
        }

        // Excited states should be ordered (lowest first)
        for i in 1..excitation_energies.len() {
            assert!(
                excitation_energies[i] >= excitation_energies[i - 1],
                "Excitation energies should be ordered"
            );
        }

        // Check that excitation energies were stored
        assert_eq!(
            ci.get_excitation_energies().len(),
            excitation_energies.len()
        );
    }

    #[test]
    fn test_cisd_calculation() {
        // Test CISD calculation with mock system
        let num_basis = 4;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, 0.5, 1.0, 1.5]);

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
        let hf_energy = -1.0;

        let mut ci = CI::new(
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            hf_energy,
            1,
            1e-6,
        );

        // Calculate CISD energy
        let correlation_energy = ci.calculate_cisd_energy();

        // Correlation energy should be negative or zero
        assert!(
            correlation_energy <= 0.0,
            "Correlation energy should be negative or zero: {}",
            correlation_energy
        );

        // Total CISD energy should be lower than or equal to HF
        let cisd_energy = hf_energy + correlation_energy;
        assert!(
            cisd_energy <= hf_energy,
            "CISD energy should be lower than or equal to HF energy"
        );

        // Check that correlation energy was stored
        assert!(ci.get_correlation_energy().is_some());
        assert!((ci.get_correlation_energy().unwrap() - correlation_energy).abs() < 1e-10);
    }

    #[test]
    fn test_cis_no_virtual_orbitals() {
        // Test CIS with no virtual orbitals - should return empty vector
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
        let hf_energy = -1.0;

        let mut ci = CI::new(
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            hf_energy,
            5,
            1e-6,
        );
        let excitation_energies = ci.calculate_cis_energies(5);

        assert!(excitation_energies.is_empty());
    }

    #[test]
    fn test_cisd_no_virtual_orbitals() {
        // Test CISD with no virtual orbitals - should return 0 correlation energy
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
        let hf_energy = -1.0;

        let mut ci = CI::new(
            mo_coeffs,
            orbital_energies,
            mo_basis,
            elems,
            hf_energy,
            1,
            1e-6,
        );
        let correlation_energy = ci.calculate_cisd_energy();

        assert_eq!(correlation_energy, 0.0);
        assert_eq!(ci.get_correlation_energy(), Some(0.0));
    }
}
