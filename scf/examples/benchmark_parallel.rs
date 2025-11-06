// Benchmark to demonstrate performance improvements from Rayon parallelization

use std::time::Instant;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use basis::cgto::Basis631G;
use scf::{SimpleSCF, SCF};

fn create_h2_molecule() -> (Vec<Vector3<f64>>, Vec<Element>) {
    let coords = vec![
        Vector3::new(0.0, 0.0, 0.0),      // H1
        Vector3::new(0.0, 0.0, 1.4),      // H2 (bond length 1.4 bohr)
    ];
    
    let elems = vec![
        Element::Hydrogen,
        Element::Hydrogen,
    ];
    
    (coords, elems)
}

fn create_mock_basis() -> HashMap<&'static str, Basis631G> {
    let mut basis = HashMap::new();
    
    // Create a mock basis that parses from an empty string
    // This is just for demonstration - in practice you'd load real basis data
    let h_basis_str = r#"
# Mock STO-3G Hydrogen basis
H    S
      3.42525091         0.15432897
      0.62391373         0.53532814
      0.16885540         0.44463454
"#;
    
    let h_basis = Basis631G::parse_nwchem(h_basis_str);
    basis.insert("H", h_basis);
    
    basis
}

fn benchmark_scf() {
    println!("=== SCF Performance Benchmark with Rayon ===");
    
    let (coords, elems) = create_h2_molecule();
    let basis = create_mock_basis();
    
    // Convert basis to reference format
    let basis_refs: HashMap<&str, &Basis631G> = basis.iter()
        .map(|(k, v)| (*k, v))
        .collect();
    
    let mut scf: SimpleSCF<Basis631G> = SimpleSCF::new();
    scf.init_basis(&elems, basis_refs);
    scf.init_geometry(&coords, &elems);
    
    println!("Molecule: H2 with mock basis");
    println!("Number of atoms: {}", scf.num_atoms);
    println!("Number of basis functions: {}", scf.num_basis);
    
    // Benchmark matrix initialization
    println!("\n--- Benchmarking Matrix Initialization ---");
    let start = Instant::now();
    scf.init_density_matrix();
    let init_time = start.elapsed();
    println!("Parallel matrix initialization time: {:.3}ms", init_time.as_millis());
    
    // Benchmark SCF cycles
    println!("\n--- Benchmarking SCF Cycles ---");
    let start = Instant::now();
    scf.scf_cycle();
    let scf_time = start.elapsed();
    println!("Parallel SCF cycles time: {:.3}ms", scf_time.as_millis());
    
    // Benchmark energy calculation
    println!("\n--- Benchmarking Energy Calculation ---");
    let start = Instant::now();
    let total_energy = scf.calculate_total_energy();
    let energy_time = start.elapsed();
    println!("Parallel energy calculation time: {:.3}ms", energy_time.as_millis());
    
    // Benchmark force calculation
    println!("\n--- Benchmarking Force Calculation ---");
    let start = Instant::now();
    let forces = scf.calculate_forces();
    let force_time = start.elapsed();
    println!("Parallel force calculation time: {:.3}ms", force_time.as_millis());
    
    // Display results
    println!("\n--- Results ---");
    println!("Total energy: {:.8} au", total_energy);
    
    println!("Forces:");
    for (i, force) in forces.iter().enumerate() {
        println!("  H{}: [{:.6}, {:.6}, {:.6}] au", 
                 i+1, force.x, force.y, force.z);
    }
    
    println!("\n--- Performance Summary ---");
    println!("Matrix initialization: {:.3}ms", init_time.as_millis());
    println!("SCF convergence: {:.3}ms", scf_time.as_millis());
    println!("Energy calculation: {:.3}ms", energy_time.as_millis());
    println!("Force calculation: {:.3}ms", force_time.as_millis());
    println!("Total runtime: {:.3}ms", 
             (init_time + scf_time + energy_time + force_time).as_millis());
    
    println!("\n=== Benchmark Complete ===");
    println!("\nNote: This benchmark uses mock basis data for demonstration.");
    println!("Real-world performance will depend on the actual basis set size and molecular complexity.");
}

fn main() {
    // Set up logging
    use tracing_subscriber;
    tracing_subscriber::fmt::init();
    
    benchmark_scf();
} 