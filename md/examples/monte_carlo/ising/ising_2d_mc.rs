use md::ising::{analysis, IsingModel2D};
use std::collections::VecDeque;

/// 2D Ising Model Monte Carlo Simulation
///
/// This example demonstrates:
/// - Phase transition detection around the critical temperature
/// - Temperature annealing from high to low temperatures
/// - Statistical property calculations (magnetization, energy, susceptibility, specific heat)
/// - Comparison with theoretical predictions

fn main() {
    println!("🧲 2D Ising Model Monte Carlo Simulation");
    println!("=========================================\n");

    // Simulation parameters
    let lattice_size = 32; // 32x32 lattice (1024 spins total)
    let equilibration_steps = 1000; // Steps to reach equilibrium
    let sampling_steps = 2000; // Steps for statistical sampling
    let temp_steps = 20; // Number of temperature points

    // Temperature range around critical temperature
    let t_critical = analysis::critical_temperature();
    let temp_min = 1.5;
    let temp_max = 3.5;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;

    println!("Simulation Parameters:");
    println!(
        "- Lattice size: {}×{} ({} spins)",
        lattice_size,
        lattice_size,
        lattice_size * lattice_size
    );
    println!("- Equilibration steps: {}", equilibration_steps);
    println!("- Sampling steps: {}", sampling_steps);
    println!(
        "- Critical temperature (theoretical): T_c = {:.4}",
        t_critical
    );
    println!("- Temperature range: {:.2} → {:.2}", temp_min, temp_max);
    println!();

    // Results storage
    let mut results = Vec::new();

    // Temperature sweep
    for i in 0..temp_steps {
        let temperature = temp_min + i as f64 * temp_step;

        println!(
            "🌡️  T = {:.4} (T/T_c = {:.4})",
            temperature,
            temperature / t_critical
        );

        // Initialize Ising model
        let mut ising = IsingModel2D::new(lattice_size, temperature);

        // Equilibration phase
        print!("   Equilibrating... ");
        for _ in 0..equilibration_steps {
            ising.monte_carlo_step();
        }
        println!("✓");

        // Sampling phase
        print!("   Sampling... ");
        let mut energy_samples = Vec::new();
        let mut magnetization_samples = Vec::new();
        let mut abs_magnetization_samples = Vec::new();

        for _ in 0..sampling_steps {
            ising.monte_carlo_step();

            energy_samples.push(ising.energy_per_site());
            magnetization_samples.push(ising.magnetization_per_site());
            abs_magnetization_samples.push(ising.abs_magnetization_per_site());
        }
        println!("✓");

        // Calculate statistical properties
        let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
        let mean_magnetization =
            magnetization_samples.iter().sum::<f64>() / magnetization_samples.len() as f64;
        let mean_abs_magnetization =
            abs_magnetization_samples.iter().sum::<f64>() / abs_magnetization_samples.len() as f64;

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

        println!(
            "   E = {:8.4}, |M| = {:6.4}, C = {:8.4}, χ = {:8.4}",
            mean_energy, mean_abs_magnetization, specific_heat, susceptibility
        );
        println!();
    }

    // Print summary table
    print_results_table(&results, t_critical);

    // Demonstrate single temperature detailed analysis
    demonstrate_phase_transition(lattice_size);
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
    println!("\n📊 Results Summary");
    println!("==================");
    println!("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│    T     │   T/Tc   │    E     │   |M|    │    C     │    χ     │");
    println!("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤");

    for result in results {
        println!(
            "│ {:8.4} │ {:8.4} │ {:8.4} │ {:8.4} │ {:8.4} │ {:8.4} │",
            result.temperature,
            result.temperature / t_critical,
            result.mean_energy,
            result.mean_abs_magnetization,
            result.specific_heat,
            result.susceptibility
        );
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
    let max_specific_heat = results
        .iter()
        .max_by(|a, b| a.specific_heat.partial_cmp(&b.specific_heat).unwrap())
        .unwrap();

    let max_susceptibility = results
        .iter()
        .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap())
        .unwrap();

    println!("📈 Critical Behavior Analysis:");
    println!(
        "- Specific heat peak at T = {:.4} (T/Tc = {:.4})",
        max_specific_heat.temperature,
        max_specific_heat.temperature / t_critical
    );
    println!(
        "- Susceptibility peak at T = {:.4} (T/Tc = {:.4})",
        max_susceptibility.temperature,
        max_susceptibility.temperature / t_critical
    );
    println!(
        "- Theoretical critical temperature: T_c = {:.4}",
        t_critical
    );
}

fn demonstrate_phase_transition(lattice_size: usize) {
    println!("\n🔥 Phase Transition Demonstration");
    println!("=================================");

    let t_critical = analysis::critical_temperature();

    // Show configurations at different temperatures
    let temperatures = [1.5, t_critical, 3.0];
    let temp_names = ["Low T (Ordered)", "Critical T", "High T (Disordered)"];

    for (i, &temp) in temperatures.iter().enumerate() {
        println!("\n{} (T = {:.4}):", temp_names[i], temp);

        let mut ising = IsingModel2D::new(8, temp); // Smaller system for visualization

        // Equilibrate
        for _ in 0..500 {
            ising.monte_carlo_step();
        }

        // Show configuration
        ising.print_configuration();

        // Sample statistics over short run
        let mut energy_samples = Vec::new();
        let mut mag_samples = Vec::new();

        for _ in 0..200 {
            ising.monte_carlo_step();
            energy_samples.push(ising.energy_per_site());
            mag_samples.push(ising.abs_magnetization_per_site());
        }

        let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
        let mean_mag = mag_samples.iter().sum::<f64>() / mag_samples.len() as f64;

        println!(
            "Average E = {:.4}, Average |M| = {:.4}",
            mean_energy, mean_mag
        );
    }

    // Time evolution demonstration
    println!("\n⏰ Time Evolution at Critical Temperature");
    println!("========================================");

    let mut ising = IsingModel2D::new_ordered(lattice_size, t_critical);

    let time_points = [0, 100, 500, 1000, 2000];
    let mut magnetization_history = VecDeque::new();
    let window_size = 50;

    for &target_step in &time_points {
        // Run to target step
        while ising.step < target_step as u64 {
            ising.monte_carlo_step();

            // Track recent magnetization for smoothing
            magnetization_history.push_back(ising.abs_magnetization_per_site());
            if magnetization_history.len() > window_size {
                magnetization_history.pop_front();
            }
        }

        let smoothed_mag = if !magnetization_history.is_empty() {
            magnetization_history.iter().sum::<f64>() / magnetization_history.len() as f64
        } else {
            ising.abs_magnetization_per_site()
        };

        println!(
            "Step {:4}: E = {:8.4}, |M| = {:6.4} (smoothed: {:6.4})",
            ising.step,
            ising.energy_per_site(),
            ising.abs_magnetization_per_site(),
            smoothed_mag
        );
    }

    println!("\n✅ Simulation complete!");
    println!("\nNotes:");
    println!("- The 2D Ising model undergoes a continuous phase transition at T_c ≈ 2.269");
    println!("- Below T_c: Ordered phase with spontaneous magnetization");
    println!("- Above T_c: Disordered phase with zero net magnetization");
    println!("- Critical behavior shows peaks in specific heat and susceptibility");
    println!("- Finite size effects may shift the apparent critical temperature");
}
