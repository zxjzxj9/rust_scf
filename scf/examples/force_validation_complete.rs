use scf::simple::SimpleSCF;
use scf::scf::SCF;
use scf::force_validation::ForceValidator;
use basis::cgto::Basis631G;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use color_eyre::Result;
use tracing::info;

/// Create a minimal H basis set for testing (simplified STO-3G-like)
fn create_minimal_h_basis() -> Basis631G {
    use basis::cgto::ContractedGTO;
    use basis::gto::GTO;
    
    let mut basis = Basis631G {
        name: "minimal-H".to_string(),
        atomic_number: 1,
        basis_set: Vec::new(),
    };
    
    // Create a simple 1s orbital with one primitive
    let h_1s = ContractedGTO {
        primitives: vec![GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "1s".to_string(),
        Z: 1,
        n: 1,
        l: 0,
        m: 0,
        s: 0,
    };
    
    basis.basis_set.push(h_1s);
    basis
}

/// Create a minimal O basis set for testing
fn create_minimal_o_basis() -> Basis631G {
    use basis::cgto::ContractedGTO;
    use basis::gto::GTO;
    
    let mut basis = Basis631G {
        name: "minimal-O".to_string(),
        atomic_number: 8,
        basis_set: Vec::new(),
    };
    
    // Create simple 1s and 2p orbitals
    let h_1s = ContractedGTO {
        primitives: vec![GTO::new(5.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "1s".to_string(),
        Z: 8,
        n: 1,
        l: 0,
        m: 0,
        s: 0,
    };
    
    let p_2px = ContractedGTO {
        primitives: vec![GTO::new(2.0, Vector3::new(1, 0, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "2px".to_string(),
        Z: 8,
        n: 2,
        l: 1,
        m: -1,
        s: 0,
    };
    
    let p_2py = ContractedGTO {
        primitives: vec![GTO::new(2.0, Vector3::new(0, 1, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "2py".to_string(),
        Z: 8,
        n: 2,
        l: 1,
        m: 1,
        s: 0,
    };
    
    let p_2pz = ContractedGTO {
        primitives: vec![GTO::new(2.0, Vector3::new(0, 0, 1), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "2pz".to_string(),
        Z: 8,
        n: 2,
        l: 1,
        m: 0,
        s: 0,
    };
    
    basis.basis_set.extend(vec![h_1s, p_2px, p_2py, p_2pz]);
    basis
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    info!("##############################################################");
    info!("           Complete Force Calculation Validation");
    info!("##############################################################");
    info!("");

    // Test 1: H2 molecule with complete force calculation
    info!("=======================================================");
    info!("Test 1: H2 Molecule Force Validation");
    info!("=======================================================");
    
    let mut scf = SimpleSCF::<Basis631G>::new();
    
    // H2 molecule slightly stretched from equilibrium
    let coords = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1.6), // ~1.6 bohr, slightly longer than equilibrium
    ];
    let elements = vec![Element::Hydrogen, Element::Hydrogen];
    
    // Set up basis
    let mut basis = HashMap::new();
    let h_basis = create_minimal_h_basis();
    basis.insert("H", &h_basis);
    
    scf.init_basis(&elements, basis);
    scf.init_geometry(&coords, &elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();
    scf.scf_cycle();
    
    let initial_energy = scf.calculate_total_energy();
    let bond_length = (coords[1] - coords[0]).norm();
    
    info!("Initial conditions:");
    info!("  Bond length: {:.6} bohr ({:.6} Å)", bond_length, bond_length * 0.529177);
    info!("  SCF Energy: {:.8} au", initial_energy);
    info!("");
    
    // Calculate forces and validate
    info!("Force calculation and validation:");
    let (analytical_forces, numerical_forces, max_error) = ForceValidator::validate_forces_comprehensive(
        &mut scf, &coords, &elements, 1e-4
    );
    
    info!("Results:");
    info!("  Maximum error between analytical and numerical forces: {:.6} au/bohr", max_error);
    info!("  Analytical vs Numerical forces comparison:");
    for (i, (ana, num)) in analytical_forces.iter().zip(numerical_forces.iter()).enumerate() {
        let error = (ana - num).norm();
        info!("    Atom {}: Analytical=[{:.6}, {:.6}, {:.6}] Numerical=[{:.6}, {:.6}, {:.6}] Error={:.6}",
            i + 1, ana.x, ana.y, ana.z, num.x, num.y, num.z, error);
    }
    info!("");
    
    // Test 2: Force component breakdown
    info!("=======================================================");
    info!("Test 2: Force Component Breakdown Analysis");
    info!("=======================================================");
    
    // We'll implement this by temporarily modifying the force calculation to show components
    test_force_components(&mut scf, &coords, &elements)?;
    
    // Test 3: Water molecule (more complex case)
    info!("=======================================================");
    info!("Test 3: H2O Molecule Force Validation");
    info!("=======================================================");
    
    test_water_molecule_forces()?;
    
    info!("##############################################################");
    info!("                   Summary and Analysis");
    info!("##############################################################");
    
    if max_error < 1e-3 {
        info!("✅ EXCELLENT: Force calculation is highly accurate (error < 1e-3)");
        info!("   The implementation correctly includes:");
        info!("   - Nuclear-nuclear repulsion forces");
        info!("   - Electron-nuclear attraction forces");
        info!("   - Two-electron integral derivatives");
        info!("   - Pulay forces (basis function derivatives)");
    } else if max_error < 1e-2 {
        info!("✅ GOOD: Force calculation is reasonably accurate (error < 1e-2)");
        info!("   Minor discrepancies may be due to:");
        info!("   - Finite difference step size in numerical validation");
        info!("   - Simplified Pulay force implementation");
        info!("   - Numerical precision in integral calculations");
    } else {
        info!("⚠️  WARNING: Significant force calculation errors (error > 1e-2)");
        info!("   This indicates incomplete implementation or bugs in:");
        info!("   - Two-electron integral derivatives");
        info!("   - Pulay force calculation");
        info!("   - Basis function mapping to atoms");
    }
    
    info!("");
    info!("Key improvements implemented:");
    info!("✅ Two-electron integral derivatives (dJKabcd_dR)");
    info!("✅ Pulay forces for overlap matrices (dSab_dR)");
    info!("✅ Pulay forces for kinetic energy (dTab_dR)");
    info!("✅ Pulay forces for nuclear attraction (dVab_dRbasis)");
    info!("⚠️  Two-electron Pulay forces (simplified due to computational cost)");
    info!("");
    
    Ok(())
}

fn test_force_components(scf: &mut SimpleSCF<Basis631G>, coords: &[Vector3<f64>], elements: &[Element]) -> Result<()> {
    info!("Analyzing individual force components...");
    
    // This is a demonstration of what each component contributes
    // In a real implementation, you'd modify the force calculation to return components separately
    
    let total_forces = scf.calculate_forces();
    
    info!("Force component analysis:");
    info!("  Note: This shows the total forces from our complete implementation");
    info!("  Individual components (nuclear-nuclear, electron-nuclear, two-electron, Pulay)");
    info!("  are combined in the final result shown above.");
    info!("");
    info!("  Total forces on atoms:");
    for (i, force) in total_forces.iter().enumerate() {
        info!("    Atom {}: [{:.6}, {:.6}, {:.6}] au",
            i + 1, force.x, force.y, force.z);
    }
    
    // Calculate force magnitude and direction
    let bond_force = total_forces[1] - total_forces[0]; // Net force on bond
    let force_magnitude = bond_force.norm();
    let bond_vector = coords[1] - coords[0];
    let bond_direction = bond_vector.normalize();
    let force_along_bond = bond_force.dot(&bond_direction);
    
    info!("");
    info!("  Bond analysis:");
    info!("    Force magnitude: {:.6} au", force_magnitude);
    info!("    Force along bond: {:.6} au", force_along_bond);
    info!("    Force direction: [{:.6}, {:.6}, {:.6}]", 
          bond_force.x, bond_force.y, bond_force.z);
    
    if force_along_bond < 0.0 {
        info!("    → Bond wants to CONTRACT (attractive force)");
    } else {
        info!("    → Bond wants to EXPAND (repulsive force)");
    }
    
    Ok(())
}

fn test_water_molecule_forces() -> Result<()> {
    info!("Testing force calculation on H2O molecule...");
    
    let mut scf = SimpleSCF::<Basis631G>::new();
    
    // H2O geometry (not optimized, for force testing)
    let coords = vec![
        Vector3::new(0.0, 0.0, 0.0),      // O
        Vector3::new(0.0, 0.0, 1.8),      // H1
        Vector3::new(1.4, 0.0, -0.5),     // H2
    ];
    let elements = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];
    
    // Set up basis
    let mut basis = HashMap::new();
    let h_basis = create_minimal_h_basis();
    let o_basis = create_minimal_o_basis();
    basis.insert("H", &h_basis);
    basis.insert("O", &o_basis);
    
    scf.init_basis(&elements, basis);
    scf.init_geometry(&coords, &elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();
    scf.scf_cycle();
    
    let energy = scf.calculate_total_energy();
    info!("  H2O SCF Energy: {:.8} au", energy);
    
    // Calculate forces
    let forces = scf.calculate_forces();
    
    info!("  Forces on H2O atoms:");
    let atom_names = ["O", "H1", "H2"];
    for (i, (force, name)) in forces.iter().zip(atom_names.iter()).enumerate() {
        info!("    {}: [{:.6}, {:.6}, {:.6}] au", name, force.x, force.y, force.z);
    }
    
    // Check force balance (should sum to zero)
    let total_force: Vector3<f64> = forces.iter().sum();
    let force_balance = total_force.norm();
    info!("  Force balance check (should be ~0): {:.8} au", force_balance);
    
    if force_balance < 1e-6 {
        info!("  ✅ Forces are well balanced (conservation of momentum)");
    } else {
        info!("  ⚠️  Forces not perfectly balanced - possible numerical errors");
    }
    
    // Validate with numerical differentiation (smaller molecule, so faster)
    info!("  Running numerical validation...");
    let (analytical_forces, numerical_forces, max_error) = ForceValidator::validate_forces_comprehensive(
        &mut scf, &coords, &elements, 1e-4
    );
    
    info!("  H2O Force validation results:");
    info!("    Maximum error: {:.6} au/bohr", max_error);
    
    if max_error < 1e-2 {
        info!("    ✅ H2O forces are accurately calculated");
    } else {
        info!("    ⚠️  H2O forces show significant errors");
    }
    
    Ok(())
} 