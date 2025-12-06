// Example: Grand Canonical Monte Carlo (GCMC) for Lennard-Jones particles
//
// This example demonstrates how to use GCMC to simulate LJ particles
// at fixed chemical potential, temperature, and volume.
//
// The simulation will sample the equilibrium number of particles and
// density as a function of chemical potential.

use md::{parallel_gcmc_sweep, GCMCResults, GCMC};
use nalgebra::Vector3;
use std::f64::{INFINITY, NEG_INFINITY};

/// Build an FCC lattice with the requested number of unit cells and spacing.
fn create_fcc_lattice(n_cells: usize, spacing: f64) -> Vec<Vector3<f64>> {
    let basis = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5, 0.5, 0.0),
        Vector3::new(0.5, 0.0, 0.5),
        Vector3::new(0.0, 0.5, 0.5),
    ];

    let mut positions = Vec::with_capacity(4 * n_cells.pow(3));
    for i in 0..n_cells {
        for j in 0..n_cells {
            for k in 0..n_cells {
                let cell_origin = Vector3::new(i as f64, j as f64, k as f64) * spacing;
                for &b in &basis {
                    positions.push(cell_origin + b * spacing);
                }
            }
        }
    }
    positions
}

/// Estimate a simple Landau-like free-energy profile from density samples.
fn estimate_density_free_energy(results: &[GCMCResults], bin_width: f64) -> Vec<(f64, f64)> {
    if results.is_empty() {
        return Vec::new();
    }

    let mut densities = Vec::new();
    for r in results {
        densities.extend(r.density_samples.iter().copied());
    }

    if densities.is_empty() {
        return Vec::new();
    }

    let mut min_rho = INFINITY;
    let mut max_rho = NEG_INFINITY;
    for rho in &densities {
        if *rho < min_rho {
            min_rho = *rho;
        }
        if *rho > max_rho {
            max_rho = *rho;
        }
    }

    if (max_rho - min_rho).abs() < 1e-12 {
        return vec![(min_rho, 0.0)];
    }

    let bin_count = (((max_rho - min_rho) / bin_width).ceil() as usize).max(1);
    let mut counts = vec![0usize; bin_count];

    for rho in densities {
        let mut idx = ((rho - min_rho) / bin_width).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= bin_count {
            idx = (bin_count - 1) as isize;
        }
        counts[idx as usize] += 1;
    }

    let total_samples = counts.iter().sum::<usize>() as f64;
    if total_samples == 0.0 {
        return Vec::new();
    }

    let mut profile = Vec::new();
    for (i, count) in counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        let prob = *count as f64 / total_samples;
        let rho_center = min_rho + (i as f64 + 0.5) * bin_width;
        let delta_f_over_kbt = -prob.ln(); // dimensionless (ΔF / kT)
        profile.push((rho_center, delta_f_over_kbt));
    }

    if profile.is_empty() {
        return profile;
    }

    let min_val = profile.iter().map(|(_, f)| *f).fold(INFINITY, f64::min);

    profile
        .into_iter()
        .map(|(rho, f)| (rho, f - min_val))
        .collect()
}

fn closest_mu_for_density(results: &[GCMCResults], target_density: f64) -> Option<(f64, f64)> {
    results
        .iter()
        .min_by(|a, b| {
            let da = (a.avg_density - target_density).abs();
            let db = (b.avg_density - target_density).abs();
            da.partial_cmp(&db).unwrap()
        })
        .map(|res| (res.chemical_potential, res.avg_density))
}

fn main() {
    println!("=== Grand Canonical Monte Carlo for Lennard-Jones Particles ===\n");

    // System parameters (reduced units)
    let epsilon = 1.0; // LJ energy parameter
    let sigma = 1.0; // LJ length parameter
    let box_length = 10.0f64; // Cubic box side length
    let temperature = 1.5; // Reduced temperature

    println!("System parameters:");
    println!("  ε = {}", epsilon);
    println!("  σ = {}", sigma);
    println!("  Box length = {}", box_length);
    println!("  Temperature = {}", temperature);
    println!("  Volume = {}\n", box_length.powi(3));

    // ========================================
    // Part 1: Single chemical potential simulation
    // ========================================
    println!("=== Part 1: Single GCMC Simulation (Random Start) ===\n");

    let mut gcmc = GCMC::new(epsilon, sigma, box_length, temperature, -3.0);

    // Initialize with some particles
    gcmc.initialize_random(0.3);
    println!("Initial N particles: {}", gcmc.n_particles());
    println!("Initial density: {:.4}\n", gcmc.density());

    // Equilibration phase
    println!("Running equilibration (10000 steps)...");
    let equilibration_steps = 10_000;
    gcmc.run(equilibration_steps);
    println!(
        "After equilibration: N = {}, ρ = {:.4}\n",
        gcmc.n_particles(),
        gcmc.density()
    );

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
    println!(
        "  Average ρ: {:.4}",
        gcmc.stats.avg_n_particles / gcmc.lj.volume()
    );
    println!("  Average E: {:.2}", gcmc.stats.avg_energy);
    println!(
        "  Average E/N: {:.4}",
        gcmc.stats.avg_energy / gcmc.stats.avg_n_particles
    );

    gcmc.stats.print_summary();

    // ========================================
    // Part 2: Lattice-initialized cluster
    // ========================================
    println!("\n\n=== Part 2: Lattice-Initialized LJ Cluster ===\n");

    let lattice_cells: usize = 4;
    let lattice_spacing = box_length / lattice_cells as f64;
    let fcc_positions = create_fcc_lattice(lattice_cells, lattice_spacing);

    let mut gcmc_lattice = GCMC::new(epsilon, sigma, box_length, temperature, -2.5);
    gcmc_lattice.positions = fcc_positions;
    println!(
        "FCC cells: {} (spacing {:.3})",
        lattice_cells, lattice_spacing
    );
    println!("Initial lattice atoms: {}", gcmc_lattice.n_particles());
    println!("Initial lattice density: {:.4}\n", gcmc_lattice.density());

    println!(
        "Relaxing lattice configuration ({} steps)...",
        equilibration_steps
    );
    gcmc_lattice.run(equilibration_steps);
    println!(
        "After relaxation: N = {}, ρ = {:.4}",
        gcmc_lattice.n_particles(),
        gcmc_lattice.density()
    );

    println!("Running lattice production ({} steps)...", production_steps);
    for step in 0..production_steps {
        gcmc_lattice.monte_carlo_step();
        if step % sample_interval == 0 {
            gcmc_lattice.sample();
        }
    }

    println!("\nLattice-based results:");
    println!("  Average N: {:.2}", gcmc_lattice.stats.avg_n_particles);
    println!(
        "  Average ρ: {:.4}",
        gcmc_lattice.stats.avg_n_particles / gcmc_lattice.lj.volume()
    );
    let lattice_avg_e_per_particle = if gcmc_lattice.stats.avg_n_particles > 0.0 {
        gcmc_lattice.stats.avg_energy / gcmc_lattice.stats.avg_n_particles
    } else {
        0.0
    };
    println!("  Average E/N: {:.4}", lattice_avg_e_per_particle);
    gcmc_lattice.stats.print_summary();

    // ========================================
    // Part 3: Chemical potential sweep
    // ========================================
    println!("\n\n=== Part 3: Chemical Potential Sweep ===\n");

    // Create GCMC simulations at different chemical potentials
    let mu_values: Vec<f64> = (-6..=0).map(|i| i as f64).collect();

    println!("Chemical potentials to scan: {:?}\n", mu_values);

    let gcmc_configs: Vec<GCMC> = mu_values
        .iter()
        .map(|&mu| {
            let mut gcmc = GCMC::new(epsilon, sigma, box_length, temperature, mu);
            gcmc.initialize_random(0.3);
            gcmc
        })
        .collect();

    println!("Running parallel GCMC sweep...");
    let mut results = parallel_gcmc_sweep(
        &gcmc_configs,
        10_000, // equilibration steps
        30_000, // production steps
        100,    // sample interval
    );
    results.sort_by(|a, b| {
        a.chemical_potential
            .partial_cmp(&b.chemical_potential)
            .unwrap()
    });

    // Print results table
    println!(
        "\n{:>8} {:>12} {:>12} {:>15} {:>10} {:>10} {:>10}",
        "μ", "⟨N⟩", "⟨ρ⟩", "⟨E/N⟩", "Acc(D)%", "Acc(I)%", "Acc(Del)%"
    );
    println!("{}", "-".repeat(85));

    for result in &results {
        println!(
            "{:8.2} {:12.2} {:12.6} {:15.4} {:10.1} {:10.1} {:10.1}",
            result.chemical_potential,
            result.avg_n_particles,
            result.avg_density,
            result.avg_energy_per_particle,
            100.0 * result.displacement_acceptance,
            100.0 * result.insertion_acceptance,
            100.0 * result.deletion_acceptance
        );
    }

    // ========================================
    // Part 4: Density landscape via free energy
    // ========================================
    println!("\n\n=== Part 4: Density Free-Energy Estimate ===\n");
    let bin_width = 0.02;
    let free_energy_profile = estimate_density_free_energy(&results, bin_width);
    if free_energy_profile.is_empty() {
        println!("Not enough density samples to build a free-energy profile.");
    } else {
        println!("{:>12} {:>15}", "ρ", "ΔF/kT");
        println!("{}", "-".repeat(30));
        for (rho, delta) in &free_energy_profile {
            println!("{:12.4} {:15.6}", rho, delta);
        }

        if let Some((best_density, _)) = free_energy_profile
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            let best_density = *best_density;
            println!(
                "\nMinimum free energy observed near ρ = {:.4}",
                best_density
            );
            if let Some((mu_match, avg_density)) = closest_mu_for_density(&results, best_density) {
                println!(
                    "Closest simulated chemical potential: μ = {:.2} (⟨ρ⟩ ≈ {:.4})",
                    mu_match, avg_density
                );
            }
        }
    }

    // ========================================
    // Part 5: High-density simulation
    // ========================================
    println!("\n\n=== Part 5: High Density Simulation ===\n");

    let high_mu = -1.0;
    let mut gcmc_high = GCMC::new(epsilon, sigma, box_length, temperature, high_mu);
    gcmc_high.initialize_random(0.5);

    println!("High chemical potential: μ = {}", high_mu);
    println!(
        "Initial state: N = {}, ρ = {:.4}\n",
        gcmc_high.n_particles(),
        gcmc_high.density()
    );

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
    println!(
        "  Average ρ: {:.4}",
        gcmc_high.stats.avg_n_particles / gcmc_high.lj.volume()
    );
    println!(
        "  Average E/N: {:.4}",
        gcmc_high.stats.avg_energy / gcmc_high.stats.avg_n_particles
    );

    gcmc_high.stats.print_summary();

    // ========================================
    // Part 6: Low-density (gas phase) simulation
    // ========================================
    println!("\n\n=== Part 6: Low Density (Gas Phase) Simulation ===\n");

    let low_mu = -5.0;
    let mut gcmc_low = GCMC::new(epsilon, sigma, box_length, temperature, low_mu);
    gcmc_low.initialize_random(0.05);

    println!("Low chemical potential: μ = {}", low_mu);
    println!(
        "Initial state: N = {}, ρ = {:.4}\n",
        gcmc_low.n_particles(),
        gcmc_low.density()
    );

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
    println!(
        "  Average ρ: {:.6}",
        gcmc_low.stats.avg_n_particles / gcmc_low.lj.volume()
    );
    println!(
        "  Average E/N: {:.4}",
        if gcmc_low.stats.avg_n_particles > 0.0 {
            gcmc_low.stats.avg_energy / gcmc_low.stats.avg_n_particles
        } else {
            0.0
        }
    );

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
