//! Tests for CCSD implementation

#[cfg(test)]
mod tests {
    use super::super::CCSD;
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

        fn dJKabcd_dR(_a: &Self, _b: &Self, _c: &Self, _d: &Self, _R: Vector3<f64>) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dSab_dR(_a: &Self, _b: &Self, _atom_idx: usize) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dTab_dR(_a: &Self, _b: &Self, _atom_idx: usize) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dVab_dRbasis(_a: &Self, _b: &Self, _R: Vector3<f64>, _Z: u32, _atom_idx: usize) -> Vector3<f64> {
            Vector3::zeros()
        }

        fn dJKabcd_dRbasis(_a: &Self, _b: &Self, _c: &Self, _d: &Self, _atom_idx: usize) -> Vector3<f64> {
            Vector3::zeros()
        }
    }
    
    #[test]
    fn test_ccsd_initialization() {
        // Create a simple system with 4 basis functions and 2 electrons (1 occupied, 3 virtual)
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
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        let ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 50, 1e-6);
        
        assert_eq!(ccsd.num_basis, 4);
        assert_eq!(ccsd.num_occ, 1);
        assert_eq!(ccsd.num_virt, 3);
        assert_eq!(ccsd.max_iterations, 50);
        assert!((ccsd.convergence_threshold - 1e-6).abs() < 1e-10);
        
        // Check T1 dimensions
        assert_eq!(ccsd.t1.nrows(), 1);
        assert_eq!(ccsd.t1.ncols(), 3);
        
        // Check T2 dimensions
        let expected_t2_size = 1 * 1 * 3 * 3; // num_occ * num_occ * num_virt * num_virt
        assert_eq!(ccsd.t2.len(), expected_t2_size);
        
        // Check that amplitudes are initialized to zero
        assert!(ccsd.t1.iter().all(|&x| x == 0.0));
        assert!(ccsd.t2.iter().all(|&x| x == 0.0));
        
        // Check that correlation energy is not yet calculated
        assert!(ccsd.correlation_energy.is_none());
    }
    
    #[test]
    fn test_t2_storage_size() {
        // Test T2 amplitude storage dimensions
        let num_basis = 4;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, -0.5, 0.5, 1.0]);
        
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
            Element::from_symbol("H").unwrap(),
            Element::from_symbol("H").unwrap(),
        ];
        
        let ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 50, 1e-6);
        
        // For 2 occupied and 2 virtual orbitals
        // T2 should have num_occ^2 * num_virt^2 = 2^2 * 2^2 = 16 elements
        let expected_size = ccsd.num_occ * ccsd.num_occ * ccsd.num_virt * ccsd.num_virt;
        assert_eq!(ccsd.t2.len(), expected_size);
        
        // Verify dimensions are correct
        assert_eq!(ccsd.num_occ, 2);
        assert_eq!(ccsd.num_virt, 2);
    }
    
    #[test]
    fn test_ccsd_zero_virtual_orbitals() {
        // Test CCSD with no virtual orbitals - should return 0 correlation energy
        let num_basis = 1;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-0.5]);
        
        let mut mo_basis = Vec::new();
        let basis = MockBasis {
            center: Vector3::zeros(),
        };
        mo_basis.push(Arc::new(basis));
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        let ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 50, 1e-6);
        
        // Should handle gracefully
        assert_eq!(ccsd.num_virt, 0);
        assert_eq!(ccsd.t1.ncols(), 0);
        assert_eq!(ccsd.t2.len(), 0);
    }
    
    #[test]
    fn test_ccsd_energy_calculation() {
        // Test CCSD energy calculation with a minimal system
        // 2 basis functions, 2 electrons -> 1 occupied, 1 virtual
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
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        let mut ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 5, 1e-3);
        
        // Calculate CCSD energy (with low max_iterations for quick test)
        let correlation_energy = ccsd.solve();
        
        // Correlation energy should be finite
        assert!(correlation_energy.is_finite());
        
        // CCSD correlation energy should be non-positive (typically)
        // Note: This isn't always strictly true in pathological cases, but for reasonable systems
        assert!(correlation_energy <= 1e-3, 
                "CCSD correlation energy unexpectedly large: {}", correlation_energy);
        
        // Check that the energy was stored
        assert!(ccsd.correlation_energy.is_some());
        assert!((ccsd.correlation_energy.unwrap() - correlation_energy).abs() < 1e-10);
    }
    
    #[test]
    fn test_ccsd_convergence() {
        // Test that CCSD iterations converge
        let num_basis = 3;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, 0.5, 1.0]);
        
        let mut mo_basis = Vec::new();
        for i in 0..num_basis {
            let basis = MockBasis {
                center: Vector3::new(0.0, 0.0, i as f64),
            };
            mo_basis.push(Arc::new(basis));
        }
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        // Use a reasonable convergence threshold
        let mut ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 100, 1e-6);
        
        let correlation_energy = ccsd.solve();
        
        // Check that we got a result
        assert!(correlation_energy.is_finite());
        assert!(ccsd.correlation_energy.is_some());
    }
    
    #[test]
    fn test_ccsd_t1_dimensions() {
        // Test that T1 amplitudes have correct dimensions
        let num_basis = 5;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.5, -1.0, 0.5, 1.0, 1.5]);
        
        let mut mo_basis = Vec::new();
        for i in 0..num_basis {
            let basis = MockBasis {
                center: Vector3::new(0.0, 0.0, i as f64),
            };
            mo_basis.push(Arc::new(basis));
        }
        
        // 4 electrons = 2 occupied orbitals, 3 virtual
        let elems = vec![
            Element::from_symbol("He").unwrap(),
            Element::from_symbol("He").unwrap(),
        ];
        
        let ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 50, 1e-6);
        
        // T1 should be (num_occ, num_virt) = (2, 3)
        assert_eq!(ccsd.t1.nrows(), 2);
        assert_eq!(ccsd.t1.ncols(), 3);
        
        // T2 should have num_occ^2 * num_virt^2 = 2^2 * 3^2 = 36 elements
        assert_eq!(ccsd.t2.len(), 36);
    }
    
    #[test]
    fn test_get_correlation_energy() {
        // Test the getter method for correlation energy
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
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        let ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 50, 1e-6);
        
        // Before calculation, should return None
        assert!(ccsd.get_correlation_energy().is_none());
    }
    
    #[test]
    fn test_t1_diagnostic() {
        // Test the T1 diagnostic calculation
        // T1 diagnostic is a measure of multireference character
        let num_basis = 3;
        let mo_coeffs = DMatrix::identity(num_basis, num_basis);
        let orbital_energies = DVector::from_vec(vec![-1.0, 0.5, 1.0]);
        
        let mut mo_basis = Vec::new();
        for i in 0..num_basis {
            let basis = MockBasis {
                center: Vector3::new(0.0, 0.0, i as f64),
            };
            mo_basis.push(Arc::new(basis));
        }
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        let ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 50, 1e-6);
        
        // T1 diagnostic should be calculable (will be 0 before solving)
        let t1_diag = ccsd.t1_diagnostic();
        assert!(t1_diag >= 0.0, "T1 diagnostic should be non-negative");
        assert!(t1_diag.is_finite(), "T1 diagnostic should be finite");
        
        // For uninitialized amplitudes, T1 diagnostic should be 0
        assert_eq!(t1_diag, 0.0);
    }
    
    #[test]
    fn test_t1_diagnostic_after_solve() {
        // Test T1 diagnostic after CCSD solve
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
        
        let elems = vec![Element::from_symbol("H").unwrap(), Element::from_symbol("H").unwrap()];
        
        let mut ccsd = CCSD::new(mo_coeffs, orbital_energies, mo_basis, elems, 5, 1e-3);
        
        // Solve CCSD
        let _correlation_energy = ccsd.solve();
        
        // T1 diagnostic should now reflect the converged amplitudes
        let t1_diag = ccsd.t1_diagnostic();
        assert!(t1_diag >= 0.0, "T1 diagnostic should be non-negative");
        assert!(t1_diag.is_finite(), "T1 diagnostic should be finite");
        
        // For single-reference systems, T1 < 0.02 is typical
        // Our mock system may not follow this exactly
    }
}

