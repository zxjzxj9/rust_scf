// Example: GCMC Phase Diagram Analysis for Lennard-Jones Fluid
//
// This example demonstrates how to use GCMC to explore the phase behavior
// of a Lennard-Jones fluid by scanning chemical potential at different temperatures.
// 
// The simulation computes density vs. chemical potential isotherms, which
// show gas-liquid phase transitions at low temperatures.

use md::{GCMC, parallel_gcmc_sweep};

fn main() {
    println!("=== GCMC Phase Diagram Analysis for LJ Fluid ===\n");

    // System parameters
    let epsilon = 1.0;
    let sigma = 1.0;
    let box_length = 12.0f64;  // Larger box for better phase behavior
    
    println!("System parameters:");
    println!("  ε = {}", epsilon);
    println!("  σ = {}", sigma);
    println!("  Box length = {} σ", box_length);
    println!("  Volume = {:.1} σ³\n", box_length.powi(3));

    // Temperature values to scan
    let temperatures = vec![
        0.8,   // Low temperature (below critical)
        1.0,   // Moderate temperature
        1.3,   // Near critical temperature (T_c ≈ 1.3 for LJ)
        1.5,   // Above critical temperature
        2.0,   // High temperature
    ];

    // Chemical potential range
    let mu_min = -6.0;
    let mu_max = 0.0;
    let mu_step = 0.3;
    let n_mu = ((mu_max - mu_min) / mu_step) as usize + 1;

    println!("Scanning {} temperatures:", temperatures.len());
    for &t in &temperatures {
        println!("  T = {:.2}", t);
    }
    println!("\nChemical potential range: [{:.1}, {:.1}] with step {:.1}", 
             mu_min, mu_max, mu_step);
    println!("Total simulations: {}\n", temperatures.len() * n_mu);

    // Run simulations for each temperature
    for &temperature in temperatures.iter() {
        println!("========================================");
        println!("Temperature: T = {:.2} ε/k_B", temperature);
        println!("========================================\n");

        // Generate chemical potential values
        let mu_values: Vec<f64> = (0..n_mu)
            .map(|i| mu_min + i as f64 * mu_step)
            .collect();

        // Create GCMC configurations
        let gcmc_configs: Vec<GCMC> = mu_values.iter()
            .map(|&mu| {
                let mut gcmc = GCMC::new(epsilon, sigma, box_length, temperature, mu);
                // Initialize with density estimate based on temperature
                let init_density = if temperature < 1.0 {
                    0.3  // Lower temperature, start with moderate density
                } else {
                    0.2  // Higher temperature, start with lower density
                };
                gcmc.initialize_random(init_density);
                
                // Adjust move probabilities based on expected density
                if mu > -2.0 {
                    // High density expected
                    gcmc.set_move_probabilities(0.7, 0.15, 0.15);
                } else if mu < -4.0 {
                    // Low density expected
                    gcmc.set_move_probabilities(0.4, 0.3, 0.3);
                } else {
                    // Moderate density
                    gcmc.set_move_probabilities(0.5, 0.25, 0.25);
                }
                
                gcmc
            })
            .collect();

        println!("Running {} parallel GCMC simulations...", mu_values.len());
        
        // Run parallel sweep
        let results = parallel_gcmc_sweep(
            &gcmc_configs,
            15_000,  // equilibration steps
            50_000,  // production steps
            100,     // sample interval
        );

        // Print results for this temperature
        println!("\nResults for T = {:.2}:", temperature);
        println!("{:>10} {:>12} {:>12} {:>12} {:>15} {:>10}",
                 "μ/ε", "⟨N⟩", "⟨ρ⟩σ³", "σ(ρ)", "⟨E/N⟩/ε", "Acc(D)%");
        println!("{}", "-".repeat(75));

        for result in &results {
            let rho_std = result.density_std();
            println!("{:10.2} {:12.2} {:12.6} {:12.6} {:15.4} {:10.1}",
                     result.chemical_potential,
                     result.avg_n_particles,
                     result.avg_density,
                     rho_std,
                     result.avg_energy_per_particle,
                     100.0 * result.displacement_acceptance);
        }

        // Analyze phase behavior
        println!("\nPhase analysis for T = {:.2}:", temperature);
        analyze_phase_behavior(&results, temperature);
        
        // Save data to file
        let filename = format!("gcmc_isotherm_T_{:.2}.dat", temperature);
        save_isotherm(&results, &filename);
        println!("Data saved to: {}\n", filename);
    }

    println!("\n=== Analysis Complete ===\n");
    println!("Summary:");
    println!("- At low T (< 1.3), you may observe a density jump indicating phase transition");
    println!("- At high T (> 1.3), smooth density increase with μ (supercritical fluid)");
    println!("- Large density fluctuations (σ(ρ)) indicate critical region");
    println!("\nFor phase diagram:");
    println!("1. Plot ρ vs. μ for each temperature");
    println!("2. Look for S-shaped curves at low T (van der Waals loop)");
    println!("3. Identify coexistence densities from Maxwell construction");
}

fn analyze_phase_behavior(results: &[md::GCMCResults], temperature: f64) {
    // Find regions of different densities
    let mut low_density_region = Vec::new();
    let mut high_density_region = Vec::new();
    
    for result in results {
        if result.avg_density < 0.1 {
            low_density_region.push((result.chemical_potential, result.avg_density));
        } else if result.avg_density > 0.4 {
            high_density_region.push((result.chemical_potential, result.avg_density));
        }
    }

    println!("  Low density (gas-like) points: {}", low_density_region.len());
    println!("  High density (liquid-like) points: {}", high_density_region.len());

    // Find maximum density fluctuation (indicator of critical region)
    let max_fluct = results.iter()
        .map(|r| r.density_std())
        .fold(0.0_f64, |a, b| a.max(b));
    
    if let Some(critical_point) = results.iter()
        .find(|r| (r.density_std() - max_fluct).abs() < 1e-6) {
        println!("  Maximum density fluctuation σ(ρ) = {:.4} at μ = {:.2}", 
                 max_fluct, critical_point.chemical_potential);
        
        if max_fluct > 0.05 && temperature < 1.5 {
            println!("  → Strong fluctuations suggest critical region or phase transition");
        }
    }

    // Estimate compressibility from density fluctuations
    // χ_T = (⟨N²⟩ - ⟨N⟩²) / (⟨N⟩ k_B T)
    println!("  Density fluctuations indicate compressibility:");
    for result in results.iter().filter(|r| r.avg_n_particles > 5.0) {
        let n_std = result.n_particles_std();
        if n_std > 0.0 {
            let compressibility = n_std.powi(2) / (result.avg_n_particles * temperature);
            if compressibility > 1.0 {
                println!("    μ = {:.2}: χ_T ≈ {:.2} (enhanced near critical point)",
                         result.chemical_potential, compressibility);
            }
        }
    }
}

fn save_isotherm(results: &[md::GCMCResults], filename: &str) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename).expect("Unable to create file");
    
    writeln!(file, "# GCMC Isotherm Data").unwrap();
    writeln!(file, "# Temperature: {:.6}", results[0].temperature).unwrap();
    writeln!(file, "# Columns: μ, ⟨N⟩, ⟨ρ⟩, σ(ρ), σ(N), ⟨E⟩, ⟨E/N⟩, Acc_disp, Acc_ins, Acc_del").unwrap();
    writeln!(file, "#").unwrap();

    for result in results {
        writeln!(file, "{:.6} {:.6} {:.8} {:.8} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                 result.chemical_potential,
                 result.avg_n_particles,
                 result.avg_density,
                 result.density_std(),
                 result.n_particles_std(),
                 result.avg_energy,
                 result.avg_energy_per_particle,
                 result.displacement_acceptance,
                 result.insertion_acceptance,
                 result.deletion_acceptance).unwrap();
    }
}

