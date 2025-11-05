// Example: Grand Canonical Monte Carlo (GCMC) for Lennard-Jones particles
// 
// This example demonstrates how to use GCMC to simulate LJ particles
// at fixed chemical potential, temperature, and volume.
// 
// The simulation will sample the equilibrium number of particles and
// density as a function of chemical potential.

use md::{GCMC, parallel_gcmc_sweep};

fn main() {
    println!("=== Grand Canonical Monte Carlo for Lennard-Jones Particles ===\n");

    // System parameters (reduced units)
    let epsilon = 1.0;  // LJ energy parameter
    let sigma = 1.0;    // LJ length parameter
    let box_length = 10.0f64;  // Cubic box side length
    let temperature = 1.5;  // Reduced temperature

    println!("System parameters:");
    println!("  ε = {}", epsilon);
    println!("  σ = {}", sigma);
    println!("  Box length = {}", box_length);
    println!("  Temperature = {}", temperature);
    println!("  Volume = {}\n", box_length.powi(3));

    // ========================================
    // Part 1: Single chemical potential simulation
    // ========================================
    println!("=== Part 1: Single GCMC Simulation ===\n");
    
    let mut gcmc = GCMC::new(epsilon, sigma, box_length, temperature, -3.0);
    
    // Initialize with some particles
    gcmc.initialize_random(0.3);
    println!("Initial N particles: {}", gcmc.n_particles());
    println!("Initial density: {:.4}\n", gcmc.density());

    // Equilibration phase
    println!("Running equilibration (10000 steps)...");
    let equilibration_steps = 10_000;
    gcmc.run(equilibration_steps);
    println!("After equilibration: N = {}, ρ = {:.4}\n", 
             gcmc.n_particles(), gcmc.density());

    // Production phase with sampling
    println!("Running production (50000 steps)...");
    let production_steps = 50_000;
    let sample_interval = 100;
    
    let mut n_samples = Vec::new();
    let mut density_samples = Vec::new();
    let mut energy_samples = Vec::new();

    for step in 0..production_steps {
        gcmc.monte_carlo_step();
        
        if step % sample_interval == 0 {
            gcmc.sample();
            n_samples.push(gcmc.n_particles() as f64);
            density_samples.push(gcmc.density());
            energy_samples.push(gcmc.potential_energy());
        }
    }

    // Print results
    println!("\nProduction phase results:");
    println!("  Average N: {:.2}", gcmc.stats.avg_n_particles);
    println!("  Average ρ: {:.4}", gcmc.stats.avg_n_particles / gcmc.lj.volume());
    println!("  Average E: {:.2}", gcmc.stats.avg_energy);
    println!("  Average E/N: {:.4}", 
             gcmc.stats.avg_energy / gcmc.stats.avg_n_particles);

    gcmc.stats.print_summary();

    // ========================================
    // Part 2: Chemical potential sweep
    // ========================================
    println!("\n\n=== Part 2: Chemical Potential Sweep ===\n");
    
    // Create GCMC simulations at different chemical potentials
    let mu_values: Vec<f64> = (-6..=0)
        .map(|i| i as f64)
        .collect();
    
    println!("Chemical potentials to scan: {:?}\n", mu_values);
    
    let gcmc_configs: Vec<GCMC> = mu_values.iter()
        .map(|&mu| {
            let mut gcmc = GCMC::new(epsilon, sigma, box_length, temperature, mu);
            gcmc.initialize_random(0.3);
            gcmc
        })
        .collect();

    println!("Running parallel GCMC sweep...");
    let results = parallel_gcmc_sweep(
        &gcmc_configs,
        10_000,  // equilibration steps
        30_000,  // production steps
        100,     // sample interval
    );

    // Print results table
    println!("\n{:>8} {:>12} {:>12} {:>15} {:>10} {:>10} {:>10}",
             "μ", "⟨N⟩", "⟨ρ⟩", "⟨E/N⟩", "Acc(D)%", "Acc(I)%", "Acc(Del)%");
    println!("{}", "-".repeat(85));
    
    for result in &results {
        println!("{:8.2} {:12.2} {:12.6} {:15.4} {:10.1} {:10.1} {:10.1}",
                 result.chemical_potential,
                 result.avg_n_particles,
                 result.avg_density,
                 result.avg_energy_per_particle,
                 100.0 * result.displacement_acceptance,
                 100.0 * result.insertion_acceptance,
                 100.0 * result.deletion_acceptance);
    }

    // ========================================
    // Part 3: High-density simulation
    // ========================================
    println!("\n\n=== Part 3: High Density Simulation ===\n");
    
    let high_mu = -1.0;
    let mut gcmc_high = GCMC::new(epsilon, sigma, box_length, temperature, high_mu);
    gcmc_high.initialize_random(0.5);
    
    println!("High chemical potential: μ = {}", high_mu);
    println!("Initial state: N = {}, ρ = {:.4}\n", 
             gcmc_high.n_particles(), gcmc_high.density());

    // Adjust move probabilities for high density
    // More displacement, less insertion/deletion
    gcmc_high.set_move_probabilities(0.7, 0.15, 0.15);
    println!("Move probabilities adjusted for high density:");
    println!("  Displacement: 70%");
    println!("  Insertion: 15%");
    println!("  Deletion: 15%\n");

    println!("Running equilibration...");
    gcmc_high.run(10_000);
    
    println!("Running production...");
    for step in 0..30_000 {
        gcmc_high.monte_carlo_step();
        if step % 100 == 0 {
            gcmc_high.sample();
        }
    }

    println!("\nHigh density results:");
    println!("  Average N: {:.2}", gcmc_high.stats.avg_n_particles);
    println!("  Average ρ: {:.4}", gcmc_high.stats.avg_n_particles / gcmc_high.lj.volume());
    println!("  Average E/N: {:.4}", 
             gcmc_high.stats.avg_energy / gcmc_high.stats.avg_n_particles);
    
    gcmc_high.stats.print_summary();

    // ========================================
    // Part 4: Low-density (gas phase) simulation
    // ========================================
    println!("\n\n=== Part 4: Low Density (Gas Phase) Simulation ===\n");
    
    let low_mu = -5.0;
    let mut gcmc_low = GCMC::new(epsilon, sigma, box_length, temperature, low_mu);
    gcmc_low.initialize_random(0.05);
    
    println!("Low chemical potential: μ = {}", low_mu);
    println!("Initial state: N = {}, ρ = {:.4}\n", 
             gcmc_low.n_particles(), gcmc_low.density());

    // Adjust move probabilities for low density
    // Less displacement, more insertion/deletion
    gcmc_low.set_move_probabilities(0.4, 0.3, 0.3);
    println!("Move probabilities adjusted for low density:");
    println!("  Displacement: 40%");
    println!("  Insertion: 30%");
    println!("  Deletion: 30%\n");

    println!("Running equilibration...");
    gcmc_low.run(10_000);
    
    println!("Running production...");
    for step in 0..30_000 {
        gcmc_low.monte_carlo_step();
        if step % 100 == 0 {
            gcmc_low.sample();
        }
    }

    println!("\nLow density results:");
    println!("  Average N: {:.2}", gcmc_low.stats.avg_n_particles);
    println!("  Average ρ: {:.6}", gcmc_low.stats.avg_n_particles / gcmc_low.lj.volume());
    println!("  Average E/N: {:.4}", 
             if gcmc_low.stats.avg_n_particles > 0.0 {
                 gcmc_low.stats.avg_energy / gcmc_low.stats.avg_n_particles
             } else {
                 0.0
             });
    
    gcmc_low.stats.print_summary();

    // ========================================
    // Summary
    // ========================================
    println!("\n\n=== Summary ===\n");
    println!("GCMC simulations completed successfully!");
    println!("\nKey observations:");
    println!("1. Particle number fluctuates based on chemical potential");
    println!("2. Higher μ → more particles, higher density");
    println!("3. Lower μ → fewer particles, lower density");
    println!("4. Acceptance rates depend on density and move probabilities");
    println!("\nThe simulation samples the grand canonical ensemble (μVT)");
    println!("where N can fluctuate while μ, V, and T are fixed.");
}

