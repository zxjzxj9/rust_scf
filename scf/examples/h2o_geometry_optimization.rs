use scf::optim::create_optimizer;
use scf::simple::SimpleSCF;
use scf::scf::SCF;
use basis::cgto::Basis631G;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use tracing::info;

/// Load basis set from file or fetch from Basis Set Exchange
fn load_basis_for_element(element: &str) -> Result<Basis631G, Box<dyn std::error::Error>> {
    // Try to load from local file first
    let filename = format!("tests/basis_sets/6-31g.{}.nwchem", element.to_lowercase());
    if std::path::Path::new(&filename).exists() {
        info!("Loading {} basis set from local file: {}", element, filename);
        let basis_content = std::fs::read_to_string(&filename)?;
        return Ok(Basis631G::parse_nwchem(&basis_content));
    }

    // If local file doesn't exist, fetch from Basis Set Exchange
    info!("Fetching {} basis set from Basis Set Exchange...", element);
    let url = format!(
        "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
        element
    );
    
    let response = reqwest::blocking::get(&url)?;
    let basis_str = response.text()?;
    Ok(Basis631G::parse_nwchem(&basis_str))
}

/// Calculate bond distances and angles for H2O
fn analyze_h2o_geometry(coords: &[Vector3<f64>]) -> (f64, f64, f64) {
    // Assume order: O, H1, H2
    let o = coords[0];
    let h1 = coords[1];
    let h2 = coords[2];
    
    // Bond distances (in bohr)
    let oh1_distance = (h1 - o).norm();
    let oh2_distance = (h2 - o).norm();
    
    // H-O-H angle (in degrees)
    let v1 = (h1 - o).normalize();
    let v2 = (h2 - o).normalize();
    let cos_angle = v1.dot(&v2);
    let angle_rad = cos_angle.acos();
    let angle_deg = angle_rad.to_degrees();
    
    (oh1_distance, oh2_distance, angle_deg)
}

/// Convert bohr to angstrom
fn bohr_to_angstrom(bohr: f64) -> f64 {
    bohr * 0.529177210903
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("=======================================================");
    info!("         H2O Geometry Optimization Example");
    info!("=======================================================");

    // Set up H2O molecule with a distorted initial geometry
    // Starting from a geometry that's not at equilibrium
    let initial_coords = vec![
        Vector3::new(0.0, 0.0, 0.0),      // Oxygen at origin
        Vector3::new(0.0, 1.0, 0.8),      // H1 - longer bond, wider angle
        Vector3::new(0.0, -1.0, 0.8),     // H2 - longer bond, wider angle
    ];
    let elements = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];

    info!("Initial H2O geometry:");
    for (i, (coord, elem)) in initial_coords.iter().zip(elements.iter()).enumerate() {
        info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}] bohr", 
              i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
    }

    // Analyze initial geometry
    let (oh1_init, oh2_init, angle_init) = analyze_h2o_geometry(&initial_coords);
    info!("\nInitial geometric parameters:");
    info!("  O-H1 distance: {:.6} bohr ({:.6} Å)", oh1_init, bohr_to_angstrom(oh1_init));
    info!("  O-H2 distance: {:.6} bohr ({:.6} Å)", oh2_init, bohr_to_angstrom(oh2_init));
    info!("  H-O-H angle: {:.2}°", angle_init);

    // Load basis sets for O and H
    let o_basis = load_basis_for_element("O")?;
    let h_basis = load_basis_for_element("H")?;
    
    let mut basis_map = HashMap::new();
    basis_map.insert("O", &o_basis);
    basis_map.insert("H", &h_basis);

    // Set up SCF calculation
    let mut scf = SimpleSCF::<Basis631G>::new();
    
    // Adjust SCF parameters for better convergence with H2O
    scf.density_mixing = 0.3;  // Lower mixing for better stability
    scf.max_cycle = 150;       // More cycles for H2O convergence
    
    scf.init_basis(&elements, basis_map);
    scf.init_geometry(&initial_coords, &elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();

    info!("\nRunning initial SCF calculation...");
    scf.scf_cycle();
    let initial_energy = scf.calculate_total_energy();
    info!("Initial energy: {:.8} au", initial_energy);

    // Calculate initial forces to see how far from equilibrium we are
    let initial_forces = scf.calculate_forces();
    let max_initial_force = initial_forces.iter()
        .map(|f| f.norm())
        .fold(0.0, f64::max);
    info!("Maximum initial force: {:.6} au/bohr", max_initial_force);

    // Create and run conjugate gradient optimizer
    info!("\n=======================================================");
    info!("       Conjugate Gradient Optimization");
    info!("=======================================================");
    
    let max_iterations = 50;
    let convergence_threshold = 1e-4; // Force convergence threshold
    
    let mut cg_optimizer = create_optimizer("cg", &mut scf, max_iterations, convergence_threshold)?;
    cg_optimizer.init(initial_coords.clone(), elements.clone());
    
    info!("Starting optimization with:");
    info!("  Max iterations: {}", max_iterations);
    info!("  Force convergence threshold: {:.1e} au/bohr", convergence_threshold);
    
    // Run optimization
    let (optimized_coords, final_energy) = cg_optimizer.optimize();
    
    info!("\nOptimization completed!");
    info!("Final geometry:");
    for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
        info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}] bohr", 
              i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
    }
    
    // Analyze final geometry
    let (oh1_final, oh2_final, angle_final) = analyze_h2o_geometry(&optimized_coords);
    info!("\nFinal geometric parameters:");
    info!("  O-H1 distance: {:.6} bohr ({:.6} Å)", oh1_final, bohr_to_angstrom(oh1_final));
    info!("  O-H2 distance: {:.6} bohr ({:.6} Å)", oh2_final, bohr_to_angstrom(oh2_final));
    info!("  H-O-H angle: {:.2}°", angle_final);
    
    // Compare with experimental values
    info!("\nComparison with experimental values:");
    info!("  Experimental O-H distance: ~0.96 Å");
    info!("  Experimental H-O-H angle: ~104.5°");
    
    let avg_oh_distance = (oh1_final + oh2_final) / 2.0;
    info!("  Calculated average O-H distance: {:.6} Å", bohr_to_angstrom(avg_oh_distance));
    info!("  Calculated H-O-H angle: {:.2}°", angle_final);
    
    // Energy analysis
    info!("\nEnergy analysis:");
    info!("  Initial energy: {:.8} au", initial_energy);
    info!("  Final energy: {:.8} au", final_energy);
    info!("  Energy lowering: {:.8} au ({:.6} kcal/mol)", 
          initial_energy - final_energy, 
          (initial_energy - final_energy) * 627.5094);

    // Validate forces at final geometry
    info!("\n=======================================================");
    info!("            Final Force Analysis");
    info!("=======================================================");
    
    let final_forces = cg_optimizer.get_forces();
    info!("Final forces on each atom:");
    for (force, elem) in final_forces.iter().zip(elements.iter()) {
        info!("  {}: [{:.6}, {:.6}, {:.6}] au/bohr (magnitude: {:.6})", 
              elem.get_symbol(), force.x, force.y, force.z, force.norm());
    }
    
    let max_final_force = final_forces.iter()
        .map(|f| f.norm())
        .fold(0.0, f64::max);
    info!("Maximum final force: {:.6} au/bohr", max_final_force);
    
    if max_final_force < convergence_threshold {
        info!("✓ Optimization converged successfully!");
    } else {
        info!("⚠ Optimization did not fully converge. Consider more iterations.");
    }

    // Note about force accuracy
    info!("\nNote: Force calculations use approximations in the current implementation.");
    info!("For production calculations, consider using more sophisticated methods.");

    Ok(())
}
