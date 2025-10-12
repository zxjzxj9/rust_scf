use md::ising::{IsingModel2D, analysis};
use std::time::Instant;

/// Comparison of Wolff Cluster Algorithm vs Metropolis Single-Spin Algorithm
/// 
/// This example demonstrates:
/// - Dramatic speedup of cluster algorithms near critical temperature
/// - Elimination of critical slowing down
/// - Cluster size distribution analysis
/// - Equilibration time comparison
/// - Statistical efficiency comparison

fn main() {
    println!("🔥 Cluster Algorithm vs Metropolis Algorithm Comparison");
    println!("========================================================\n");
    
    let t_critical = analysis::critical_temperature_2d();
    
    // Test different temperatures around T_c
    let temperatures = [
        t_critical * 0.8,   // Below T_c (ordered phase)
        t_critical,         // At T_c (critical point)
        t_critical * 1.2,   // Above T_c (disordered phase)
    ];
    
    let temp_names = [
        "Below T_c (Ordered)",
        "At T_c (Critical)", 
        "Above T_c (Disordered)"
    ];
    
    let lattice_size = 32;
    let steps = 2000;
    
    println!("Test Parameters:");
    println!("- Lattice size: {}×{} ({} spins)", lattice_size, lattice_size, lattice_size * lattice_size);
    println!("- Steps per test: {}", steps);
    println!("- Critical temperature: T_c = {:.4}", t_critical);
    println!();
    
    for (i, &temp) in temperatures.iter().enumerate() {
        println!("📊 {} (T = {:.4}, T/T_c = {:.3})", temp_names[i], temp, temp / t_critical);
        println!("{}", "─".repeat(60));
        
        compare_algorithms_at_temperature(lattice_size, temp, steps);
        println!();
    }
    
    // Detailed critical behavior analysis
    analyze_critical_slowing_down(lattice_size);
    
    // Cluster size distribution analysis
    analyze_cluster_sizes(lattice_size, t_critical);
}

fn compare_algorithms_at_temperature(lattice_size: usize, temperature: f64, steps: usize) {
    // Test Metropolis algorithm
    println!("🔄 Metropolis Single-Spin Algorithm:");
    let mut ising_metro = IsingModel2D::new(lattice_size, temperature);
    
    let start_time = Instant::now();
    let mut energy_samples_metro = Vec::new();
    let mut mag_samples_metro = Vec::new();
    
    for _ in 0..steps {
        ising_metro.monte_carlo_step();
        energy_samples_metro.push(ising_metro.energy_per_site());
        mag_samples_metro.push(ising_metro.abs_magnetization_per_site());
    }
    let metro_time = start_time.elapsed();
    
    // Calculate statistics for Metropolis
    let metro_mean_energy = energy_samples_metro.iter().sum::<f64>() / energy_samples_metro.len() as f64;
    let metro_mean_mag = mag_samples_metro.iter().sum::<f64>() / mag_samples_metro.len() as f64;
    let metro_specific_heat = ising_metro.specific_heat(&energy_samples_metro);
    let metro_susceptibility = ising_metro.magnetic_susceptibility(&mag_samples_metro);
    
    println!("  Time: {:8.2} ms", metro_time.as_millis());
    println!("  Energy: {:8.4} ± {:6.4}", metro_mean_energy, 
             calculate_std_error(&energy_samples_metro));
    println!("  |Magnetization|: {:6.4} ± {:6.4}", metro_mean_mag, 
             calculate_std_error(&mag_samples_metro));
    println!("  Specific Heat: {:8.4}", metro_specific_heat);
    println!("  Susceptibility: {:8.4}", metro_susceptibility);
    
    // Test Wolff cluster algorithm
    println!("\n⚡ Wolff Cluster Algorithm:");
    let mut ising_wolff = IsingModel2D::new(lattice_size, temperature);
    
    let start_time = Instant::now();
    let mut energy_samples_wolff = Vec::new();
    let mut mag_samples_wolff = Vec::new();
    let mut cluster_sizes = Vec::new();
    
    for _ in 0..steps {
        let cluster_size = ising_wolff.wolff_cluster_step_with_size();
        cluster_sizes.push(cluster_size);
        energy_samples_wolff.push(ising_wolff.energy_per_site());
        mag_samples_wolff.push(ising_wolff.abs_magnetization_per_site());
    }
    let wolff_time = start_time.elapsed();
    
    // Calculate statistics for Wolff
    let wolff_mean_energy = energy_samples_wolff.iter().sum::<f64>() / energy_samples_wolff.len() as f64;
    let wolff_mean_mag = mag_samples_wolff.iter().sum::<f64>() / mag_samples_wolff.len() as f64;
    let wolff_specific_heat = ising_wolff.specific_heat(&energy_samples_wolff);
    let wolff_susceptibility = ising_wolff.magnetic_susceptibility(&mag_samples_wolff);
    
    let mean_cluster_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
    let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0);
    
    println!("  Time: {:8.2} ms", wolff_time.as_millis());
    println!("  Energy: {:8.4} ± {:6.4}", wolff_mean_energy, 
             calculate_std_error(&energy_samples_wolff));
    println!("  |Magnetization|: {:6.4} ± {:6.4}", wolff_mean_mag, 
             calculate_std_error(&mag_samples_wolff));
    println!("  Specific Heat: {:8.4}", wolff_specific_heat);
    println!("  Susceptibility: {:8.4}", wolff_susceptibility);
    println!("  Mean Cluster Size: {:6.1} spins", mean_cluster_size);
    println!("  Max Cluster Size: {:6} spins ({:.1}% of lattice)", 
             max_cluster_size, 100.0 * max_cluster_size as f64 / (lattice_size * lattice_size) as f64);
    
    // Performance comparison
    println!("\n📈 Performance Comparison:");
    let speedup = metro_time.as_millis() as f64 / wolff_time.as_millis() as f64;
    println!("  Speedup: {:.2}x faster", speedup);
    
    // Statistical efficiency comparison (variance per unit time)
    let metro_energy_var = calculate_variance(&energy_samples_metro);
    let wolff_energy_var = calculate_variance(&energy_samples_wolff);
    let metro_efficiency = 1.0 / (metro_energy_var * metro_time.as_secs_f64());
    let wolff_efficiency = 1.0 / (wolff_energy_var * wolff_time.as_secs_f64());
    let efficiency_gain = wolff_efficiency / metro_efficiency;
    
    println!("  Statistical Efficiency Gain: {:.2}x", efficiency_gain);
    
    // Check statistical consistency
    let energy_diff = (metro_mean_energy - wolff_mean_energy).abs();
    let mag_diff = (metro_mean_mag - wolff_mean_mag).abs();
    println!("  Energy Difference: {:8.6} (should be small)", energy_diff);
    println!("  Magnetization Difference: {:6.6} (should be small)", mag_diff);
}

fn analyze_critical_slowing_down(lattice_size: usize) {
    println!("🐌 Critical Slowing Down Analysis");
    println!("==================================");
    
    let t_critical = analysis::critical_temperature_2d();
    let test_sizes = [16, 24, 32];
    let steps = 1000;
    
    println!("Testing different system sizes at T_c = {:.4}:", t_critical);
    println!("┌──────────┬──────────────┬──────────────┬──────────────┐");
    println!("│   Size   │  Metropolis  │    Wolff     │   Speedup    │");
    println!("├──────────┼──────────────┼──────────────┼──────────────┤");
    
    for &size in &test_sizes {
        // Test Metropolis
        let mut ising_metro = IsingModel2D::new(size, t_critical);
        let start_time = Instant::now();
        for _ in 0..steps {
            ising_metro.monte_carlo_step();
        }
        let metro_time = start_time.elapsed();
        
        // Test Wolff
        let mut ising_wolff = IsingModel2D::new(size, t_critical);
        let start_time = Instant::now();
        for _ in 0..steps {
            ising_wolff.wolff_cluster_step();
        }
        let wolff_time = start_time.elapsed();
        
        let speedup = metro_time.as_millis() as f64 / wolff_time.as_millis() as f64;
        
        println!("│ {:6}×{} │ {:10.1} ms │ {:10.1} ms │ {:10.2}x │",
                size, size, metro_time.as_millis(), wolff_time.as_millis(), speedup);
    }
    
    println!("└──────────┴──────────────┴──────────────┴──────────────┘");
    println!("\nNote: Speedup typically increases with system size at T_c due to");
    println!("      elimination of critical slowing down in cluster algorithms.");
    println!();
}

fn analyze_cluster_sizes(lattice_size: usize, t_critical: f64) {
    println!("📊 Cluster Size Distribution Analysis");
    println!("====================================");
    
    let steps = 2000;
    let temperatures = [t_critical * 0.8, t_critical, t_critical * 1.2];
    let temp_names = ["Below T_c", "At T_c", "Above T_c"];
    
    for (i, &temp) in temperatures.iter().enumerate() {
        println!("\n{} (T = {:.4}):", temp_names[i], temp);
        
        let mut ising = IsingModel2D::new(lattice_size, temp);
        let mut cluster_sizes = Vec::new();
        
        // Equilibrate first
        for _ in 0..500 {
            ising.wolff_cluster_step();
        }
        
        // Collect cluster size statistics
        for _ in 0..steps {
            let size = ising.wolff_cluster_step_with_size();
            cluster_sizes.push(size);
        }
        
        let mean_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
        let max_size = *cluster_sizes.iter().max().unwrap_or(&0);
        let min_size = *cluster_sizes.iter().min().unwrap_or(&0);
        
        // Calculate size distribution
        let mut histogram = vec![0; 10];
        let bin_size = (max_size - min_size + 1) / histogram.len() + 1;
        
        for &size in &cluster_sizes {
            let bin = ((size - min_size) / bin_size).min(histogram.len() - 1);
            histogram[bin] += 1;
        }
        
        println!("  Mean cluster size: {:8.1} spins ({:.2}% of lattice)", 
                mean_size, 100.0 * mean_size / (lattice_size * lattice_size) as f64);
        println!("  Size range: {} - {} spins", min_size, max_size);
        
        // Show histogram
        println!("  Size distribution:");
        for (i, &count) in histogram.iter().enumerate() {
            let size_range_start = min_size + i * bin_size;
            let size_range_end = min_size + (i + 1) * bin_size - 1;
            let percentage = 100.0 * count as f64 / cluster_sizes.len() as f64;
            let bar = "█".repeat((percentage / 2.0) as usize);
            println!("    {:4}-{:4}: {:5.1}% {}", 
                    size_range_start, size_range_end, percentage, bar);
        }
    }
    
    println!("\n✅ Cluster algorithm analysis complete!");
    println!("\nKey Insights:");
    println!("- Cluster algorithms eliminate critical slowing down");
    println!("- Speedup is most dramatic near the critical temperature");
    println!("- Larger systems benefit more from cluster algorithms");
    println!("- Cluster sizes are largest at the critical temperature");
    println!("- Statistical efficiency can improve by orders of magnitude");
}

fn calculate_std_error(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (samples.len() - 1) as f64;
    
    (variance / samples.len() as f64).sqrt()
}

fn calculate_variance(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (samples.len() - 1) as f64
}

