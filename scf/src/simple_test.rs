#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::SCF;
    use crate::simple::SimpleSCF;
    use basis::basis::{AOBasis, Basis};
    use basis::cgto::Basis631G;
    use nalgebra::Vector3;
    use periodic_table_on_an_enum::Element;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn fetch_basis(atomic_symbol: &str) -> Basis631G {
        let url = format!(
            "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
            atomic_symbol
        );
        let basis_str = reqwest::blocking::get(url).unwrap().text().unwrap();
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

        // Test integral matrix dimensions
        assert_eq!(
            scf.integral_matrix.shape(),
            (scf.num_basis * scf.num_basis, scf.num_basis * scf.num_basis)
        );

        // Verify integral values (0.01 - 0.5*0.01 = 0.005)
        let expected_value = 0.01 - 0.5 * 0.01;
        assert!(scf
            .integral_matrix
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
        let initial_energy = scf.e_level.clone();

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

        // Verify occupied orbitals count
        let total_electrons: usize = elems.iter().map(|e| e.get_atomic_number() as usize).sum();
        assert_eq!(total_electrons, 2);
        let n_occ = total_electrons / 2;
        assert_eq!(n_occ, 1);
    }

    #[test]
    fn test_simple_scf() {
        let mut scf = SimpleSCF::new();
        let h2o_coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.809),
            Vector3::new(1.443, 0.0, -0.453),
        ];
        let h2o_elems = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];

        let mut basis = HashMap::new();
        let h_basis = fetch_basis("H");
        let o_basis = fetch_basis("O");
        basis.insert("H", &h_basis);
        basis.insert("O", &o_basis);

        scf.init_basis(&h2o_elems, basis);
        scf.init_geometry(&h2o_coords, &h2o_elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
    }

    #[test]
    fn test_hellman_feynman_forces() {
        // Set up a simple H2 molecule with mock basis
        let mut scf = SimpleSCF::<Basis631G>::new();
        let elems = vec![Element::Hydrogen, Element::Hydrogen];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0)];
        let mut basis = HashMap::new();

        let h_basis = fetch_basis("H");
        let o_basis = fetch_basis("O");
        basis.insert("H", &h_basis);
        basis.insert("O", &o_basis);
        scf.init_basis(&elems, basis);
        scf.init_geometry(&coords, &elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();

        // Calculate forces
        let forces = scf.calculate_forces();

        // Check that we got the right number of forces
        assert_eq!(forces.len(), 2);

        // For H2, forces should be equal and opposite (Newton's third law)
        assert!((forces[0] + forces[1]).norm() < 1e-6);

        // Check values match expected (based on our mock implementation)
        let expected_force = Vector3::new(0.1, 0.1, 0.1);
        assert!((forces[0] - expected_force).norm() < 1e-6);
        assert!((forces[1] + expected_force).norm() < 1e-6);
    }

    #[test]
    fn test_real_hellman_feynman_forces() {
        // Set up H2 molecule with real basis
        let mut scf = SimpleSCF::<Basis631G>::new();

        // H2O coordinates
        let h2o_coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.809),
            Vector3::new(1.443, 0.0, -0.453),
        ];
        let h2o_elems = vec![Element::Hydrogen, Element::Hydrogen, Element::Oxygen];

        let mut basis = HashMap::new();
        let h_basis = fetch_basis("H");
        let o_basis = fetch_basis("O");
        basis.insert("H", &h_basis);
        basis.insert("O", &o_basis);

        scf.init_basis(&h2o_elems, basis);
        scf.init_geometry(&h2o_coords, &h2o_elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();

        // Calculate analytical forces
        let analytical_forces = scf.calculate_forces();

        // For symmetric H2, forces should be equal and opposite
        assert!((analytical_forces[0] + analytical_forces[1]).norm() < 1e-5);

        // Forces should be along the bond axis (z-axis in this case)
        assert!(analytical_forces[0].x.abs() < 1e-5);
        assert!(analytical_forces[0].y.abs() < 1e-5);
        assert!(analytical_forces[1].x.abs() < 1e-5);
        assert!(analytical_forces[1].y.abs() < 1e-5);

        // Validate with numerical derivative
        let numerical_force = calculate_numerical_force(&h2o_elems, &h2o_coords);

        // Compare analytical and numerical forces (z-component)
        println!("Analytical force: {:?}", analytical_forces[0]);
        println!("Numerical force: {:?}", numerical_force);

        // Allow reasonable tolerance since numerical derivatives are approximate
        assert!((analytical_forces[0].z - numerical_force.z).abs() < 1e-3);
    }

    // Helper function for numerical force calculation via finite difference
    fn calculate_numerical_force(elems: &Vec<Element>, coords: &Vec<Vector3<f64>>) -> Vector3<f64> {
        const DELTA: f64 = 1e-4;
        let mut numerical_force = Vector3::<f64>::zeros();

        // Calculate initial energy
        let mut scf = SimpleSCF::<Basis631G>::new();
        let mut basis = HashMap::new();

        let h_basis = fetch_basis("H");
        let o_basis = fetch_basis("O");
        basis.insert("H", &h_basis);
        basis.insert("O", &o_basis);

        scf.init_basis(elems, basis.clone());
        scf.init_geometry(coords, elems);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        let initial_energy = scf.calculate_total_energy();

        // Calculate force components using finite difference
        for dim in 0..3 {
            // Displace first atom in positive direction
            let mut pos_coords = coords.clone();
            match dim {
                0 => pos_coords[0].x += DELTA,
                1 => pos_coords[0].y += DELTA,
                2 => pos_coords[0].z += DELTA,
                _ => unreachable!(),
            }

            // Recalculate with displacement
            let mut scf_pos = SimpleSCF::<Basis631G>::new();
            scf_pos.init_basis(elems, basis.clone());
            scf_pos.init_geometry(&pos_coords, elems);
            scf_pos.init_density_matrix();
            scf_pos.init_fock_matrix();
            scf_pos.scf_cycle();
            let pos_energy = scf_pos.calculate_total_energy();

            // Calculate numerical force (negative gradient)
            let force_component = -(pos_energy - initial_energy) / DELTA;
            match dim {
                0 => numerical_force.x = force_component,
                1 => numerical_force.y = force_component,
                2 => numerical_force.z = force_component,
                _ => unreachable!(),
            }
        }

        numerical_force
    }
}
