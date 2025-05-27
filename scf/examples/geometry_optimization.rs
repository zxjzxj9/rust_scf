use scf::optim::{create_optimizer, GeometryOptimizer};
use scf::simple::SimpleSCF;
use scf::scf::SCF;
use basis::cgto::Basis631G;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("=======================================================");
    info!("         Geometry Optimization Example");
    info!("=======================================================");

    // Set up H2 molecule with non-equilibrium geometry
    let coords = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 2.0), // Slightly longer than equilibrium bond
    ];
    let elements = vec![Element::Hydrogen, Element::Hydrogen];

    info!("Initial H2 geometry:");
    for (i, (coord, elem)) in coords.iter().zip(elements.iter()).enumerate() {
        info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}] bohr", 
              i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
    }

    // Set up SCF calculation
    let mut scf = SimpleSCF::<Basis631G>::new();
    
    let mut basis = HashMap::new();
    let h_basis = create_minimal_h_basis();
    basis.insert("H", &h_basis);
    
    scf.init_basis(&elements, basis);
    scf.init_geometry(&coords, &elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();

    info!("\nRunning initial SCF calculation...");
    scf.scf_cycle();
    let initial_energy = scf.calculate_total_energy();
    info!("Initial energy: {:.8} au", initial_energy);

    // Create and run conjugate gradient optimizer
    info!("\n=======================================================");
    info!("       Conjugate Gradient Optimization");
    info!("=======================================================");
    
    let mut cg_optimizer = create_optimizer("cg", &mut scf, 20, 1e-4)?;
    cg_optimizer.init(coords.clone(), elements.clone());
    
    // Run optimization
    let (optimized_coords, final_energy) = cg_optimizer.optimize();
    
    info!("\nOptimization completed!");
    info!("Final geometry:");
    for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
        info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}] bohr", 
              i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
    }
    
    let bond_length = (optimized_coords[1] - optimized_coords[0]).norm();
    info!("Final H-H bond length: {:.6} bohr ({:.6} Å)", 
          bond_length, bond_length * 0.529177);
    
    info!("Energy change: {:.8} au", final_energy - initial_energy);

    // Validate forces at final geometry
    info!("\n=======================================================");
    info!("            Force Validation");
    info!("=======================================================");
    
    let numerical_forces = cg_optimizer.validate_forces(1e-4);
    let analytical_forces = cg_optimizer.get_forces();
    
    info!("Note: Force validation shows discrepancies due to incomplete");
    info!("implementation of two-electron integral derivatives.");

    Ok(())
} 