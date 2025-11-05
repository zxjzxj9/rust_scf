use md::ising::{IsingModel4D, analysis};

/// 4D Ising Model Monte Carlo Simulation
/// 
/// This example demonstrates:
/// - Phase transition detection around the 4D critical temperature
/// - Upper critical dimension behavior (d = 4 is special!)
/// - Mean field theory validity in 4D with logarithmic corrections
/// - Temperature annealing from high to low temperatures
/// - Statistical property calculations (magnetization, energy, susceptibility, specific heat)
/// - Comparison with theoretical predictions for 4D systems
/// - Visualization through lower-dimensional slices of the 4D lattice

fn main() {
    println!("🧲 4D Ising Model Monte Carlo Simulation");
    println!("=========================================\n");
    
    // Simulation parameters - much smaller lattice for 4D due to N^4 scaling
    let lattice_size: usize = 8;  // 8x8x8x8 lattice (4096 spins total)
    let equilibration_steps = 3000;  // More steps needed for 4D equilibration
    let sampling_steps = 4000;       // Steps for statistical sampling
    let temp_steps = 12;             // Number of temperature points
    
    // Temperature range around critical temperature
    let t_critical = analysis::critical_temperature_4d();
    let temp_min = 4.0;
    let temp_max = 9.0;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;
    
    println!("🌟 Upper Critical Dimension Analysis");
    println!("4D is the upper critical dimension where mean field theory becomes exact!");
    println!("Expected features:");
    println!("- Mean field critical exponents");
    println!("- Logarithmic corrections to scaling");
    println!("- Enhanced critical fluctuations");
    println!();
    
    println!("Simulation Parameters:");
    println!("- Lattice size: {}⁴ = {} spins", lattice_size, lattice_size.pow(4));
    println!("- Coordination number: 8 (±x, ±y, ±z, ±w)");
    println!("- Equilibration steps: {}", equilibration_steps);
    println!("- Sampling steps: {}", sampling_steps);
    println!("- Critical temperature (4D): T_c = {:.4}", t_critical);
    println!("- Mean field prediction: T_c ≈ {:.1}", analysis::mean_field_critical_temperature(4));
    println!("- Temperature range: {:.2} → {:.2}", temp_min, temp_max);
    println!();
    
    // Results storage
    let mut results = Vec::new();
    
    // Temperature sweep
    for i in 0..temp_steps {
        let temperature = temp_min + i as f64 * temp_step;
        
        println!("🌡️  T = {:.4} (T/T_c = {:.4})", temperature, temperature / t_critical);
        
        // Initialize 4D Ising model
        let mut ising = IsingModel4D::new(lattice_size, temperature);
        
        // Equilibration phase
        print!("   Equilibrating... ");
        for step in 0..equilibration_steps {
            ising.monte_carlo_step();
            
            // Progress indicator for longer simulations
            if step % (equilibration_steps / 6) == 0 && step > 0 {
                print!("{}%.. ", (step * 100) / equilibration_steps);
            }
        }
        println!("✓");
        
        // Sampling phase
        print!("   Sampling... ");
        let mut energy_samples = Vec::new();
        let mut magnetization_samples = Vec::new();
        let mut abs_magnetization_samples = Vec::new();
        
        for step in 0..sampling_steps {
            ising.monte_carlo_step();
            
            energy_samples.push(ising.energy_per_site());
            magnetization_samples.push(ising.magnetization_per_site());
            abs_magnetization_samples.push(ising.abs_magnetization_per_site());
            
            // Progress indicator
            if step % (sampling_steps / 6) == 0 && step > 0 {
                print!("{}%.. ", (step * 100) / sampling_steps);
            }
        }
        println!("✓");
        
        // Calculate statistical properties
        let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
        let mean_magnetization = magnetization_samples.iter().sum::<f64>() / magnetization_samples.len() as f64;
        let mean_abs_magnetization = abs_magnetization_samples.iter().sum::<f64>() / abs_magnetization_samples.len() as f64;
        
        let specific_heat = ising.specific_heat(&energy_samples);
        let susceptibility = ising.magnetic_susceptibility(&magnetization_samples);
        
        results.push(SimulationResult {
            temperature,
            mean_energy,
            mean_magnetization,
            mean_abs_magnetization,
            specific_heat,
            susceptibility,
        });
        
        println!("   E = {:8.4}, |M| = {:6.4}, C = {:8.4}, χ = {:8.4}", 
                mean_energy, mean_abs_magnetization, specific_heat, susceptibility);
        println!();
    }
    
    // Print summary table
    print_results_table(&results, t_critical);
    
    // Demonstrate 4D phase transition with visualization
    demonstrate_4d_phase_transition(lattice_size);
    
    // Upper critical dimension analysis
    analyze_upper_critical_dimension(&results, t_critical);
}

#[derive(Debug, Clone)]
struct SimulationResult {
    temperature: f64,
    mean_energy: f64,
    mean_magnetization: f64,
    mean_abs_magnetization: f64,
    specific_heat: f64,
    susceptibility: f64,
}

fn print_results_table(results: &[SimulationResult], t_critical: f64) {
    println!("\n📊 4D Ising Model Results Summary");
    println!("=================================");
    println!("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│    T     │   T/Tc   │    E     │   |M|    │    C     │    χ     │");
    println!("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤");
    
    for result in results {
        println!("│ {:8.4} │ {:8.4} │ {:8.4} │ {:8.4} │ {:8.4} │ {:8.4} │",
                result.temperature,
                result.temperature / t_critical,
                result.mean_energy,
                result.mean_abs_magnetization,
                result.specific_heat,
                result.susceptibility);
    }
    
    println!("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘");
    println!();
    println!("Legend:");
    println!("- T: Temperature");
    println!("- T/Tc: Reduced temperature (Tc = {:.4})", t_critical);
    println!("- E: Energy per site");
    println!("- |M|: Absolute magnetization per site");
    println!("- C: Specific heat");
    println!("- χ: Magnetic susceptibility");
    println!();
    
    // Find peak positions
    let max_specific_heat = results.iter()
        .max_by(|a, b| a.specific_heat.partial_cmp(&b.specific_heat).unwrap())
        .unwrap();
    
    let max_susceptibility = results.iter()
        .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap())
        .unwrap();
    
    println!("📈 Critical Behavior Analysis (4D):");
    println!("- Specific heat peak at T = {:.4} (T/Tc = {:.4})", 
             max_specific_heat.temperature, max_specific_heat.temperature / t_critical);
    println!("- Susceptibility peak at T = {:.4} (T/Tc = {:.4})", 
             max_susceptibility.temperature, max_susceptibility.temperature / t_critical);
    println!("- Theoretical critical temperature: T_c = {:.4}", t_critical);
    println!("- 4D is the upper critical dimension - expect mean field behavior!");
}

fn demonstrate_4d_phase_transition(_lattice_size: usize) {
    println!("\n🔥 4D Phase Transition Demonstration");
    println!("====================================");
    
    let t_critical = analysis::critical_temperature_4d();
    
    // Show configurations at different temperatures
    let temperatures = [3.0, t_critical, 9.0];
    let temp_names = ["Low T (Ordered)", "Critical T", "High T (Disordered)"];
    
    for (i, &temp) in temperatures.iter().enumerate() {
        println!("\n{} (T = {:.4}):", temp_names[i], temp);
        
        let mut ising = IsingModel4D::new(6, temp);  // Smaller system for visualization
        
        // Equilibrate
        for _ in 0..1500 {
            ising.monte_carlo_step();
        }
        
        // Show slices of the 4D configuration
        println!("Showing 2D slices through the 4D lattice:");
        show_4d_slices(&ising);
        
        // Sample statistics over short run
        let mut energy_samples = Vec::new();
        let mut mag_samples = Vec::new();
        
        for _ in 0..400 {
            ising.monte_carlo_step();
            energy_samples.push(ising.energy_per_site());
            mag_samples.push(ising.abs_magnetization_per_site());
        }
        
        let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
        let mean_mag = mag_samples.iter().sum::<f64>() / mag_samples.len() as f64;
        
        println!("Average E = {:.4}, Average |M| = {:.4}", mean_energy, mean_mag);
        
        // Show correlation function
        let correlations = ising.correlation_function(5);
        println!("Correlation function: {:?}", 
                correlations.iter().map(|&x| format!("{:.3}", x)).collect::<Vec<_>>());
        
        println!("{}", "─".repeat(60));
    }
}

fn show_4d_slices(ising: &IsingModel4D) {
    // Show a few representative 2D slices from the 4D lattice
    let size = ising.size;
    let mid = size / 2;
    
    // Slice through the middle of the 4D lattice
    println!("2D slice at z={}, w={} (middle of 4D lattice):", mid, mid);
    ising.print_2d_slice(mid, mid);
    
    // Corner slice
    println!("2D slice at z=0, w=0 (corner of 4D lattice):");
    ising.print_2d_slice(0, 0);
}

fn analyze_upper_critical_dimension(results: &[SimulationResult], t_critical: f64) {
    println!("\n🔬 Upper Critical Dimension Analysis");
    println!("====================================");
    
    println!("In 4D, the Ising model is at the upper critical dimension where:");
    println!("1. Mean field theory becomes exact");
    println!("2. Critical exponents take mean field values");
    println!("3. Logarithmic corrections appear");
    println!("4. Finite size effects are enhanced");
    println!();
    
    // Compare critical temperatures across dimensions
    let t_c_2d = analysis::critical_temperature_2d();
    let t_c_3d = analysis::critical_temperature_3d();
    let t_c_4d = analysis::critical_temperature_4d();
    let t_c_mf = analysis::mean_field_critical_temperature(4);
    
    println!("📊 Dimensional Comparison:");
    println!("- 2D Critical Temperature: T_c = {:.4}", t_c_2d);
    println!("- 3D Critical Temperature: T_c = {:.4}", t_c_3d);
    println!("- 4D Critical Temperature: T_c = {:.4}", t_c_4d);
    println!("- Mean Field Prediction:   T_c = {:.1}", t_c_mf);
    println!("- Ratio T_c(4D)/T_c(3D) = {:.4}", t_c_4d / t_c_3d);
    println!("- Deviation from MF: {:.1}%", 100.0 * (t_c_mf - t_c_4d).abs() / t_c_4d);
    println!();
    
    println!("⚡ Ground State Energies:");
    println!("- 2D: E₀ = {:.1} J per site", analysis::energy_per_site_at_zero_temp_2d());
    println!("- 3D: E₀ = {:.1} J per site", analysis::energy_per_site_at_zero_temp_3d());
    println!("- 4D: E₀ = {:.1} J per site", analysis::energy_per_site_at_zero_temp_4d());
    println!();
    
    // Find the critical region results
    let critical_region: Vec<_> = results.iter()
        .filter(|r| (r.temperature / t_critical - 1.0).abs() < 0.2)
        .collect();
    
    if !critical_region.is_empty() {
        println!("🎯 Critical Region Behavior:");
        for result in critical_region {
            let reduced_temp = result.temperature / t_critical;
            println!("  T/Tc = {:.3}: C = {:.3}, χ = {:.3}, |M| = {:.4}",
                    reduced_temp, result.specific_heat, result.susceptibility, 
                    result.mean_abs_magnetization);
        }
        println!();
    }
    
    println!("✅ 4D Simulation complete!");
    println!("\nKey Insights:");
    println!("- 4D is the upper critical dimension for the Ising model");
    println!("- Mean field theory becomes exact with logarithmic corrections");
    println!("- Each spin has 8 neighbors vs 6 in 3D and 4 in 2D");
    println!("- Critical behavior shows enhanced fluctuations");
    println!("- Finite size effects are more pronounced than in lower dimensions");
    println!("- Applications: Testing field theories, quantum phase transitions");
    println!("- The 4D model helps understand the crossover to mean field behavior");
}

// Performance demonstration
fn performance_comparison() {
    println!("\n⚡ Performance Scaling Analysis");
    println!("==============================");
    
    let sizes = [4, 6, 8];
    let steps = 100;
    
    for &size in &sizes {
        let start = std::time::Instant::now();
        
        let mut ising = IsingModel4D::new(size, 5.0);
        for _ in 0..steps {
            ising.monte_carlo_step();
        }
        
        let duration = start.elapsed();
        let total_spins = size.pow(4);
        let flips_per_step = total_spins;
        let total_flips = flips_per_step * steps;
        
        println!("Size {}⁴ ({:6} spins): {:8.2} ms ({:6.0} flips/ms)", 
                size, total_spins, duration.as_millis(), 
                total_flips as f64 / duration.as_millis() as f64);
    }
    
    println!("\nNote: 4D simulations scale as O(N⁴) and require significant computational resources");
    println!("For production runs, consider parallel algorithms and GPU acceleration");
}

#[cfg(feature = "extended_analysis")]
fn extended_analysis() {
    println!("\n🔬 Extended 4D Analysis");
    println!("=======================");
    
    // This would include:
    // - Finite size scaling analysis
    // - Critical exponent determination
    // - Logarithmic correction analysis
    // - Correlation length measurements
    // - Binder cumulant analysis
    
    println!("Extended analysis would include:");
    println!("- Finite size scaling: ξ/L scaling");
    println!("- Critical exponents: β, γ, ν with mean field values");
    println!("- Logarithmic corrections: ln(L) factors");
    println!("- Binder cumulant: U₄ universality");
    println!("- Correlation functions: G(r) ~ r^(-(d-2+η))");
}
