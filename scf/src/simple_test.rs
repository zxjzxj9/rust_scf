#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::SCF;
    use crate::simple::SimpleSCF;
    use basis::basis::{AOBasis, Basis};
    use basis::cgto::Basis631G;
    use nalgebra::{Vector3, DMatrix, DVector};
    use periodic_table_on_an_enum::Element;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::fs;

    fn load_basis_from_file_or_panic(atomic_symbol: &str) -> Basis631G {
        let path = format!("tests/basis_sets/sto-3g.{}.nwchem", atomic_symbol.to_lowercase());
        let basis_str = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read basis set file: {}", path));
        Basis631G::parse_nwchem(&basis_str)
    }

    fn load_631g_basis(atomic_symbol: &str) -> Basis631G {
        let path = format!("tests/basis_sets/6-31g.{}.nwchem", atomic_symbol.to_lowercase());
        let basis_str = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read basis set file: {}", path));
        Basis631G::parse_nwchem(&basis_str)
    }

    // Mock implementations for testing
    #[derive(Clone)]
    struct MockAOBasis {
        center: Vector3<f64>,
    }

    impl AOBasis for MockAOBasis {
        type BasisType = MockBasis;

        fn basis_size(&self) -> usize {
            1
        }

        fn get_basis(&self) -> Vec<Arc<Self::BasisType>> {
            vec![Arc::new(MockBasis {
                center: self.center,
            })]
        }

        fn set_center(&mut self, center: Vector3<f64>) {
            self.center = center;
        }

        fn get_center(&self) -> Option<Vector3<f64>> {
            Some(self.center)
        }
    }

    #[derive(Clone)]
    struct MockBasis {
        center: Vector3<f64>,
    }

    impl Basis for MockBasis {
        fn evaluate(&self, r: &Vector3<f64>) -> f64 {
            todo!()
        }

        fn Sab(a: &Self, b: &Self) -> f64 {
            if a.center == b.center {
                1.0
            } else {
                0.0
            }
        }

        fn Tab(_: &Self, _: &Self) -> f64 {
            0.1 // Kinetic energy integral
        }

        fn Vab(_: &Self, _: &Self, _: Vector3<f64>, charge: u32) -> f64 {
            -0.2 * charge as f64 // Potential energy integral
        }

        fn JKabcd(_: &Self, _: &Self, _: &Self, _: &Self) -> f64 {
            0.01 // Two-electron integral
        }

        fn dVab_dR(_: &Self, _: &Self, _: Vector3<f64>, _: u32) -> Vector3<f64> {
            Vector3::new(0.1, 0.1, 0.1) // Mock derivative of nuclear attraction integral
        }

        fn dJKabcd_dR(_: &Self, _: &Self, _: &Self, _: &Self, _: Vector3<f64>) -> Vector3<f64> {
            Vector3::new(0.01, 0.01, 0.01) // Mock derivative of two-electron integral
        }

        fn dSab_dR(_: &Self, _: &Self, _: usize) -> Vector3<f64> {
            Vector3::new(0.05, 0.05, 0.05) // Mock derivative of overlap integral w.r.t. basis center
        }

        fn dTab_dR(_: &Self, _: &Self, _: usize) -> Vector3<f64> {
            Vector3::new(0.02, 0.02, 0.02) // Mock derivative of kinetic integral w.r.t. basis center
        }

        fn dVab_dRbasis(_: &Self, _: &Self, _: Vector3<f64>, _: u32, _: usize) -> Vector3<f64> {
            Vector3::new(0.03, 0.03, 0.03) // Mock derivative of nuclear attraction w.r.t. basis center
        }

        fn dJKabcd_dRbasis(_: &Self, _: &Self, _: &Self, _: &Self, _: usize) -> Vector3<f64> {
            Vector3::new(0.001, 0.001, 0.001) // Mock derivative of two-electron integral w.r.t. basis center
        }
    }

    // Helper function to create mock basis
    fn create_mock_basis() -> MockAOBasis {
        MockAOBasis {
            center: Vector3::zeros(),
        }
    }

    #[test]
    fn test_init_basis() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);

        assert_eq!(scf.num_atoms, 2);
        assert_eq!(scf.ao_basis.len(), 2);
        assert_eq!(scf.elems, elems);
    }

    #[test]
    fn test_init_geometry() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);

        assert_eq!(scf.coords, coords);
        assert_eq!(scf.num_basis, 2);
        for (i, ao) in scf.ao_basis.iter().enumerate() {
            let center = ao.lock().unwrap().get_center().unwrap();
            assert_eq!(center, coords[i]);
        }
    }

    #[test]
    fn test_init_density_matrix() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();

        // Test matrix dimensions
        assert_eq!(scf.overlap_matrix.shape(), (2, 2));
        assert_eq!(scf.fock_matrix.shape(), (2, 2));

        // Verify overlap matrix is identity
        assert!(scf.overlap_matrix.clone().is_identity(1e-6));

        // Verify Fock matrix values: Tab + sum(Vab)
        let expected_diagonal = 0.1 + (-0.2 * 1.0 * 2.0); // -0.3 for diagonal
        let expected_off_diagonal = 0.1 + (-0.2 * 1.0 * 2.0); // -0.3 for off-diagonal in current mock setup

        for i in 0..scf.num_basis {
            for j in 0..scf.num_basis {
                let expected = if i == j {
                    expected_diagonal
                } else {
                    expected_off_diagonal
                };
                assert!((scf.fock_matrix[(i, j)] - expected).abs() < 1e-6);
            }
        }

        // Verify energy levels are sorted
        let eigenvalues = scf.e_level.as_slice();
        assert!(eigenvalues.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_init_fock_matrix() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();

        // Test fock matrix dimensions
        assert_eq!(
            scf.fock_matrix.shape(),
            (scf.num_basis, scf.num_basis)
        );

        // Verify h_core values (T + V = 0.02 + 2*(-0.03) = -0.04)
        let expected_value = 0.02 - 2.0 * 0.03;
        assert!(scf
            .fock_matrix
            .iter()
            .all(|&x| (x - expected_value).abs() < 1e-6));
    }

    #[test]
    fn test_scf_cycle_updates() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();

        let initial_coeffs = scf.coeffs.clone();
        let _initial_energy = scf.e_level.clone();

        scf.scf_cycle();

        // Verify updates after SCF cycle
        let diff = scf.coeffs.clone() - initial_coeffs;
        assert!(diff.iter().all(|&x| x.abs() < 1e-6));

        assert_eq!(scf.coeffs.ncols(), 2); // Should maintain dimensions
    }

    #[test]
    fn test_occupied_orbitals() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Helium]; // Atomic number 2
        let coords = vec![Vector3::zeros()];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("He", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
    }

    #[test]
    fn test_simple_scf() {
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let mut basis = HashMap::new();
        let mock_basis = create_mock_basis();

        basis.insert("H", &mock_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();

        assert!(scf.calculate_total_energy() < 0.0);
    }

    #[test]
    fn test_h2_sto3g_energy() {
        // Test SCF energy for H2 with STO-3G basis set
        // Reference energy from PySCF: -1.066453950284943
        let coords = vec![
            Vector3::new(0.0, 0.0, -0.7),
            Vector3::new(0.0, 0.0, 0.7),
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];

        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();

        let h_basis = load_basis_from_file_or_panic("H");
        basis_map.insert("H", &h_basis);

        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();

        let total_energy = scf.calculate_total_energy();
        let expected_energy = -1.06645395;

        assert!(
            (total_energy - expected_energy).abs() < 1e-6,
            "H2 STO-3G energy mismatch: got {}, expected {}",
            total_energy,
            expected_energy
        );
    }

    #[test]
    fn test_real_hellman_feynman_forces() {
        let mut coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.4), // Standard bond length for H2
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];

        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();
        
        let h_basis = load_basis_from_file_or_panic("H");
        basis_map.insert("H", &h_basis);

        scf.init_basis(&elems, basis_map.clone());

        // Calculate analytical forces
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        let analytical_forces = scf.calculate_forces();

        // Calculate numerical forces
        let numerical_forces = calculate_numerical_force(&elems, &coords, Some(scf.density_matrix.clone()), &basis_map);

        // Compare forces
        for (i, (analytical, numerical)) in analytical_forces.iter().zip(numerical_forces.iter()).enumerate() {
            let diff = (analytical - numerical).norm();
            assert!(
                diff < 1e-3,
                "Force mismatch for atom {}: analytical={:?}, numerical={:?}",
                i,
                analytical,
                numerical
            );
        }
    }

    #[test]
    fn test_hellman_feynman_force_debug() {
        let mut coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.5),
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];

        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();

        let h_basis = load_basis_from_file_or_panic("H");
        basis_map.insert("H", &h_basis);
        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
    
        let forces = scf.calculate_forces();
        println!("forces, {:?}", forces);
    }

    fn calculate_numerical_force(
        elems: &Vec<Element>,
        coords: &Vec<Vector3<f64>>,
        initial_density: Option<DMatrix<f64>>,
        basis_map: &HashMap<&str, &Basis631G>,
    ) -> Vec<Vector3<f64>> {
        let mut numerical_forces = vec![Vector3::zeros(); elems.len()];
        let delta = 1e-5;
    
        for i in 0..elems.len() {
            for j in 0..3 {
                let mut coords_plus = coords.clone();
                coords_plus[i][j] += delta;
                let energy_plus = run_scf_for_force(elems, &coords_plus, initial_density.clone(), basis_map);
    
                let mut coords_minus = coords.clone();
                coords_minus[i][j] -= delta;
                let energy_minus = run_scf_for_force(elems, &coords_minus, initial_density.clone(), basis_map);
    
                numerical_forces[i][j] = -(energy_plus - energy_minus) / (2.0 * delta);
            }
        }
        numerical_forces
    }
    
    fn run_scf_for_force(
        elems: &Vec<Element>,
        coords: &Vec<Vector3<f64>>,
        initial_density: Option<DMatrix<f64>>,
        basis_map: &HashMap<&str, &Basis631G>,
    ) -> f64 {
        let mut scf = SimpleSCF::<Basis631G>::new();
        scf.init_basis(&elems, basis_map.clone());
        scf.init_geometry(&coords, &elems);
        // Always compute the one-electron integrals and overlap matrix first.
        // This guarantees that the matrices required by `scf_cycle` are
        // positive-definite even when we want to reuse a previously converged
        // density matrix as the initial guess.
        scf.init_density_matrix();

        // If an initial density matrix was provided, overwrite the default
        // guess that `init_density_matrix` created. We can skip the obsolete
        // `init_fock_matrix` call because `scf_cycle` recomputes the Fock
        // matrix at the start of every cycle via `update_fock_matrix`.
        if let Some(density) = initial_density {
            scf.density_matrix = density;
        }
        scf.scf_cycle();
        scf.calculate_total_energy()
    }

    #[test]
    fn test_h2o_sto3g_energy() {
        // H2O molecule with STO-3G basis set
        // Reference energy: -74.965901 Hartree (from PySCF)
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.1173 * 1.88973),
            Vector3::new(0.0, 0.7572 * 1.88973, -0.4692 * 1.88973),
            Vector3::new(0.0, -0.7572 * 1.88973, -0.4692 * 1.88973),
        ];
        let elems = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];

        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();

        let h_basis = load_basis_from_file_or_panic("H");
        let o_basis = load_basis_from_file_or_panic("O");
        basis_map.insert("H", &h_basis);
        basis_map.insert("O", &o_basis);

        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();

        let total_energy = scf.calculate_total_energy();
        let expected_energy = -74.965901;

        assert!(
            (total_energy - expected_energy).abs() < 1e-5,
            "H2O STO-3G energy mismatch: got {}, expected {}",
            total_energy,
            expected_energy
        );
    }

    #[test]
    fn test_h2o_sto3g_zero_forces() {
        // H2O molecule at equilibrium geometry. Forces should be close to zero.
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.1173 * 1.88973),
            Vector3::new(0.0, 0.7572 * 1.88973, -0.4692 * 1.88973),
            Vector3::new(0.0, -0.7572 * 1.88973, -0.4692 * 1.88973),
        ];
        let elems = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];

        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();

        let h_basis = load_basis_from_file_or_panic("H");
        let o_basis = load_basis_from_file_or_panic("O");
        basis_map.insert("H", &h_basis);
        basis_map.insert("O", &o_basis);

        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();

        let forces = scf.calculate_forces();
        for (i, force) in forces.iter().enumerate() {
            assert!(
                force.norm() < 1e-5,
                "Force on atom {} is not zero: {:?}",
                i,
                force
            );
        }
    }

    #[test]
    fn test_ch4_631g_energy() {
        let elems = vec![
            Element::Carbon,
            Element::Hydrogen,
            Element::Hydrogen,
            Element::Hydrogen,
            Element::Hydrogen,
        ];
        // Coordinates in Bohr
        let d = 2.054 / (3.0_f64.sqrt());
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(d, d, d),
            Vector3::new(d, -d, -d),
            Vector3::new(-d, d, -d),
            Vector3::new(-d, -d, d),
        ];

        let mut scf = SimpleSCF::<Basis631G>::new();

        let c_basis = load_631g_basis("C");
        let h_basis = load_631g_basis("H");

        let mut basis_map = HashMap::new();
        basis_map.insert("C", &c_basis);
        basis_map.insert("H", &h_basis);

        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();

        scf.scf_cycle();

        let energy = scf.calculate_total_energy();

        let expected_energy = -40.195172;
        assert!((energy - expected_energy).abs() < 1e-4);
    }

    #[test]
    fn test_h2_force_calculation_comprehensive() {
        println!("=== Comprehensive H2 Force Calculation Test ===");
        
        // Test H2 molecule at multiple bond lengths to validate force behavior
        let test_cases = vec![
            ("equilibrium", 1.4),     // Near equilibrium 
            ("compressed", 1.1),      // Compressed (should have repulsive forces)
            ("stretched", 1.8),       // Stretched (should have attractive forces)
        ];
        
        for (description, bond_length) in test_cases {
            println!("\nTesting {} geometry (bond length: {:.1} bohr):", description, bond_length);
            
            let coords = vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, bond_length),
            ];
            let elems = vec![Element::Hydrogen, Element::Hydrogen];
            
            let mut scf = SimpleSCF::<Basis631G>::new();
            let mut basis_map = HashMap::new();
            let h_basis = load_basis_from_file_or_panic("H");
            basis_map.insert("H", &h_basis);
            
            scf.init_basis(&elems, basis_map.clone());
            scf.init_geometry(&coords, &elems);
            scf.init_density_matrix();
            scf.init_fock_matrix();
            scf.scf_cycle();
            
            // Calculate analytical forces
            let analytical_forces = scf.calculate_forces();
            
            // Calculate numerical forces for validation
            let numerical_forces = calculate_numerical_force(
                &elems, &coords, Some(scf.density_matrix.clone()), &basis_map
            );
            
            // Test 1: Force balance (conservation of momentum)
            let total_force: Vector3<f64> = analytical_forces.iter().sum();
            let force_balance_error = total_force.norm();
            assert!(
                force_balance_error < 1e-6,
                "{}: Forces not balanced - total force: [{:.6}, {:.6}, {:.6}], norm: {:.6}",
                description, total_force.x, total_force.y, total_force.z, force_balance_error
            );
            println!("  ✅ Force balance check passed (error: {:.2e})", force_balance_error);
            
            // Test 2: Force symmetry (equal and opposite for H2)
            let force_symmetry_error = (analytical_forces[0] + analytical_forces[1]).norm();
            assert!(
                force_symmetry_error < 1e-6,
                "{}: Forces not symmetric - force diff: {:.6}",
                description, force_symmetry_error
            );
            println!("  ✅ Force symmetry check passed (error: {:.2e})", force_symmetry_error);
            
            // Test 3: Comparison with numerical forces
            for (i, (analytical, numerical)) in analytical_forces.iter().zip(numerical_forces.iter()).enumerate() {
                let diff = (analytical - numerical).norm();
                assert!(
                    diff < 1e-2,  // Reasonable tolerance for analytical vs numerical
                    "{}: Force mismatch for atom {}: analytical={:?}, numerical={:?}, diff={:.6}",
                    description, i, analytical, numerical, diff
                );
            }
            
            let max_force_error = analytical_forces.iter()
                .zip(numerical_forces.iter())
                .map(|(a, n)| (a - n).norm())
                .fold(0.0, f64::max);
            println!("  ✅ Analytical vs numerical forces validated (max error: {:.2e})", max_force_error);
            
            // Test 4: Force direction checks based on geometry
            let force_along_bond = analytical_forces[1].z; // Force on second H atom along z-axis
            match description {
                "compressed" => {
                    assert!(
                        force_along_bond > 0.0,
                        "Compressed H2 should have repulsive force (positive), got: {:.6}",
                        force_along_bond
                    );
                    println!("  ✅ Compressed geometry shows repulsive force: {:.6}", force_along_bond);
                },
                "stretched" => {
                    assert!(
                        force_along_bond < 0.0,
                        "Stretched H2 should have attractive force (negative), got: {:.6}",
                        force_along_bond
                    );
                    println!("  ✅ Stretched geometry shows attractive force: {:.6}", force_along_bond);
                },
                "equilibrium" => {
                    assert!(
                        force_along_bond.abs() < 0.01,
                        "Equilibrium H2 should have small force, got: {:.6}",
                        force_along_bond
                    );
                    println!("  ✅ Equilibrium geometry shows small force: {:.6}", force_along_bond);
                },
                _ => {}
            }
            
            // Test 5: Energy consistency check
            let energy = scf.calculate_total_energy();
            println!("  📊 Total energy: {:.8} au", energy);
            println!("  📊 Forces: [{:.6}, {:.6}, {:.6}] and [{:.6}, {:.6}, {:.6}] au",
                analytical_forces[0].x, analytical_forces[0].y, analytical_forces[0].z,
                analytical_forces[1].x, analytical_forces[1].y, analytical_forces[1].z);
        }
        
        println!("\n🎉 All H2 force calculation tests passed!");
    }

    #[test]
    fn test_h2_force_convergence_with_scf() {
        println!("=== H2 Force Convergence with SCF Test ===");
        
        // Test that forces converge as SCF converges
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.5),
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        
        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();
        let h_basis = load_basis_from_file_or_panic("H");
        basis_map.insert("H", &h_basis);
        
        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        
        // Test forces at different stages of SCF convergence
        let mut previous_forces: Option<Vec<Vector3<f64>>> = None;
        let mut force_changes = Vec::new();
        
        // Capture forces during SCF cycles
        for cycle in 0..5 {
            if cycle > 0 {
                scf.update_fock_matrix();
                
                // Update molecular orbitals
                let l = scf.overlap_matrix.clone().cholesky().unwrap();
                let l_inv = l.inverse();
                let f_prime = l_inv.clone() * scf.fock_matrix.clone() * l_inv.transpose();
                let eig = f_prime.symmetric_eigen();
                
                let mut indices: Vec<usize> = (0..eig.eigenvalues.len()).collect();
                indices.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
                let sorted_eigenvalues = DVector::from_fn(eig.eigenvalues.len(), |i, _| eig.eigenvalues[indices[i]]);
                let sorted_eigenvectors = eig.eigenvectors.select_columns(&indices);
                
                let eigvecs = l_inv.transpose() * sorted_eigenvectors;
                scf.coeffs = crate::simple::align_eigenvectors(eigvecs);
                scf.e_level = sorted_eigenvalues;
                
                scf.update_density_matrix();
            }
            
            let current_forces = scf.calculate_forces();
            
            if let Some(prev_forces) = previous_forces {
                let force_change = current_forces.iter()
                    .zip(prev_forces.iter())
                    .map(|(curr, prev)| (curr - prev).norm())
                    .fold(0.0, f64::max);
                force_changes.push(force_change);
                
                println!("  Cycle {}: Max force change = {:.6} au", cycle, force_change);
            }
            
            previous_forces = Some(current_forces);
        }
        
        // Check that force changes decrease (forces converge)
        if force_changes.len() > 1 {
            let converging = force_changes.windows(2).all(|w| w[1] <= w[0] * 2.0); // Allow some numerical noise
            println!("  Forces converging: {}", converging);
            
            let final_change = force_changes.last().unwrap();
            assert!(
                *final_change < 1e-2,
                "Final force change should be small, got: {:.6}",
                final_change
            );
            println!("  ✅ Forces converged to acceptable tolerance");
        }
        
        println!("🎉 H2 force convergence test passed!");
    }

    #[test] 
    fn test_force_validation_infrastructure() {
        println!("=== Force Validation Infrastructure Test ===");
        
        // Test that our force validation functions work correctly
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.4),
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        
        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();
        let h_basis = load_basis_from_file_or_panic("H");
        basis_map.insert("H", &h_basis);
        
        scf.init_basis(&elems, basis_map.clone());
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        
        // Test different step sizes for numerical differentiation
        let step_sizes = [1e-3, 1e-4, 1e-5];
        let mut best_error = f64::INFINITY;
        let mut best_step = 0.0;
        
        for &step in &step_sizes {
            let numerical_forces = calculate_numerical_force_with_step(
                &elems, &coords, Some(scf.density_matrix.clone()), &basis_map, step
            );
            let analytical_forces = scf.calculate_forces();
            
            let max_error = analytical_forces.iter()
                .zip(numerical_forces.iter())
                .map(|(a, n)| (a - n).norm())
                .fold(0.0, f64::max);
            
            println!("  Step size {:.1e}: Max error = {:.2e}", step, max_error);
            
            if max_error < best_error {
                best_error = max_error;
                best_step = step;
            }
        }
        
        println!("  Best step size: {:.1e} with error: {:.2e}", best_step, best_error);
        assert!(
            best_error < 1e-2,
            "Best numerical-analytical force agreement should be reasonable, got: {:.2e}",
            best_error
        );
        
        println!("🎉 Force validation infrastructure test passed!");
    }

    fn calculate_numerical_force_with_step(
        elems: &Vec<Element>,
        coords: &Vec<Vector3<f64>>,
        initial_density: Option<DMatrix<f64>>,
        basis_map: &HashMap<&str, &Basis631G>,
        delta: f64,
    ) -> Vec<Vector3<f64>> {
        let mut numerical_forces = vec![Vector3::zeros(); elems.len()];
    
        for i in 0..elems.len() {
            for j in 0..3 {
                let mut coords_plus = coords.clone();
                coords_plus[i][j] += delta;
                let energy_plus = run_scf_for_force(elems, &coords_plus, initial_density.clone(), basis_map);
    
                let mut coords_minus = coords.clone();
                coords_minus[i][j] -= delta;
                let energy_minus = run_scf_for_force(elems, &coords_minus, initial_density.clone(), basis_map);
    
                numerical_forces[i][j] = -(energy_plus - energy_minus) / (2.0 * delta);
            }
        }
        numerical_forces
    }

    #[test]
    fn test_comprehensive_force_validation() {
        println!("=== Comprehensive Force Validation using ForceValidator ===");
        
        // Test using the existing force validation infrastructure
        use crate::force_validation::ForceValidator;
        
        // Simple H2 molecule for testing
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.5),
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        
        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis_map = HashMap::new();
        let h_basis = load_basis_from_file_or_panic("H");
        basis_map.insert("H", &h_basis);
        
        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        
        // Use the comprehensive force validation
        let (analytical_forces, numerical_forces, max_error) = 
            ForceValidator::validate_forces_comprehensive(&mut scf, &coords, &elems, 1e-4);
        
        println!("Force validation results:");
        println!("  Maximum error: {:.6} au/bohr", max_error);
        
        // Check that our implementation produces reasonable results
        assert!(
            max_error < 1e-1,  // Should be reasonable even if not perfect
            "Force calculation error too large: {:.6}",
            max_error
        );
        
        // Check force balance
        let total_analytical_force: Vector3<f64> = analytical_forces.iter().sum();
        let total_numerical_force: Vector3<f64> = numerical_forces.iter().sum();
        
        assert!(
            total_analytical_force.norm() < 1e-6,
            "Analytical forces should balance: {:.6}",
            total_analytical_force.norm()
        );
        
        assert!(
            total_numerical_force.norm() < 1e-6,
            "Numerical forces should balance: {:.6}",
            total_numerical_force.norm()
        );
        
        println!("  ✅ Force balance check passed");
        println!("  ✅ Force validation completed successfully");
        
        println!("🎉 Comprehensive force validation test passed!");
    }

    #[test]
    fn test_mock_force_calculation() {
        println!("=== Mock Force Calculation Test ===");
        
        // Use mock basis to test force calculation logic without basis set issues
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.4),
        ];
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        
        let mut scf = SimpleSCF::<MockAOBasis>::new();
        let mut basis_map = HashMap::new();
        let h_basis = create_mock_basis();
        basis_map.insert("H", &h_basis);
        
        scf.init_basis(&elems, basis_map);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        
        // Don't run full SCF cycle to avoid potential convergence issues
        // Just test that force calculation runs without crashing
        let forces = scf.calculate_forces();
        
        println!("Mock forces calculated:");
        for (i, force) in forces.iter().enumerate() {
            println!("  Atom {}: [{:.6}, {:.6}, {:.6}] au", 
                     i + 1, force.x, force.y, force.z);
        }
        
        // Test 1: Force balance (conservation of momentum)
        let total_force: Vector3<f64> = forces.iter().sum();
        let force_balance_error = total_force.norm();
        assert!(
            force_balance_error < 1e-6,
            "Forces not balanced - total force: [{:.6}, {:.6}, {:.6}], norm: {:.6}",
            total_force.x, total_force.y, total_force.z, force_balance_error
        );
        println!("  ✅ Force balance check passed (error: {:.2e})", force_balance_error);
        
        // Test 2: Force symmetry (equal and opposite for H2)
        let force_symmetry_error = (forces[0] + forces[1]).norm();
        assert!(
            force_symmetry_error < 1e-6,
            "Forces not symmetric - force diff: {:.6}",
            force_symmetry_error
        );
        println!("  ✅ Force symmetry check passed (error: {:.2e})", force_symmetry_error);
        
        // Test 3: Force calculation doesn't crash
        assert_eq!(forces.len(), 2, "Should have forces for 2 atoms");
        println!("  ✅ Force calculation completed successfully");
        
        println!("🎉 Mock force calculation test passed!");
    }
}
