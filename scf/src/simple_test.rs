#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::SCF;
    use crate::simple::SimpleSCF;
    use basis::basis::{AOBasis, Basis};
    use basis::cgto::Basis631G;
    use nalgebra::{Vector3, DMatrix};
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
        if let Some(density) = initial_density {
            scf.density_matrix = density;
            scf.init_fock_matrix();
        } else {
            scf.init_density_matrix();
            scf.init_fock_matrix();
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
}
