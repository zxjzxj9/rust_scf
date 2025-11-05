// Quick Start: GCMC for Lennard-Jones Particles
// 
// This is a minimal example to get started with GCMC simulations.
// Run with: cargo run --example gcmc_quickstart

use md::GCMC;

fn main() {
    println!("GCMC Quick Start Example\n");

    // Create a GCMC simulation in a cubic box
    // Parameters: epsilon, sigma, box_length, temperature, chemical_potential
    let mut gcmc = GCMC::new(
        1.0,    // ε (energy parameter)
        1.0,    // σ (length parameter) 
        10.0,   // L (box side length)
        1.5,    // T (temperature in reduced units)
        -3.0,   // μ (chemical potential in reduced units)
    );

    println!("System setup:");
    println!("  Box volume: {} σ³", gcmc.lj.volume());
    println!("  Temperature: {} ε/k_B", gcmc.temperature);
    println!("  Chemical potential: {} ε\n", gcmc.chemical_potential);

    // Initialize with random particles at target density
    gcmc.initialize_random(0.3);  // ρ = 0.3 σ⁻³
    println!("Initial configuration:");
    println!("  N = {}", gcmc.n_particles());
    println!("  ρ = {:.4} σ⁻³\n", gcmc.density());

    // Equilibration phase
    println!("Equilibration (10,000 steps)...");
    gcmc.run(10_000);
    println!("  N = {}", gcmc.n_particles());
    println!("  ρ = {:.4} σ⁻³\n", gcmc.density());

    // Production phase with sampling
    println!("Production (50,000 steps)...");
    for step in 0..50_000 {
        gcmc.monte_carlo_step();
        
        // Sample every 100 steps
        if step % 100 == 0 {
            gcmc.sample();
        }
    }

    // Print results
    println!("\n=== Results ===");
    println!("Average N: {:.1}", gcmc.stats.avg_n_particles);
    println!("Average ρ: {:.4} σ⁻³", gcmc.stats.avg_n_particles / gcmc.lj.volume());
    println!("Average E: {:.2} ε", gcmc.stats.avg_energy);
    println!("Average E/N: {:.3} ε", 
             gcmc.stats.avg_energy / gcmc.stats.avg_n_particles);
    
    println!("\nAcceptance rates:");
    println!("  Displacement: {:.1}%", 
             100.0 * gcmc.stats.displacement_acceptance_rate());
    println!("  Insertion:    {:.1}%", 
             100.0 * gcmc.stats.insertion_acceptance_rate());
    println!("  Deletion:     {:.1}%", 
             100.0 * gcmc.stats.deletion_acceptance_rate());

    println!("\nSimulation complete!");
    println!("Try changing the chemical potential to see different densities:");
    println!("  μ = -5.0 → low density (gas)");
    println!("  μ = -3.0 → moderate density");
    println!("  μ = -1.0 → high density (liquid)");
}

