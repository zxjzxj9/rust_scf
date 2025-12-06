use md::ising::{analysis, IsingModel2D, IsingModel3D, IsingModel4D};
use std::time::Instant;

/// Comprehensive Guide to Applying Wolff Cluster Algorithm
///
/// This example demonstrates:
/// - When to use Wolff vs Metropolis
/// - Performance benefits
/// - Best practices for different scenarios
/// - Cluster size analysis
/// - Temperature-dependent behavior

fn main() {
    println!("🔥 Wolff Cluster Algorithm Application Guide");
    println!("============================================\n");

    // Demonstrate Wolff effectiveness across dimensions
    demonstrate_wolff_effectiveness();

    // Show optimal usage scenarios
    show_optimal_usage_scenarios();

    // Performance analysis
    performance_analysis();

    // Advanced techniques
    advanced_wolff_techniques();
}

fn demonstrate_wolff_effectiveness() {
    println!("📊 Wolff Algorithm Effectiveness");
    println!("================================\n");

    let dimensions = ["2D", "3D", "4D"];
    let critical_temps = [
        analysis::critical_temperature_2d(),
        analysis::critical_temperature_3d(),
        analysis::critical_temperature_4d(),
    ];

    for (i, (dim, &t_c)) in dimensions.iter().zip(critical_temps.iter()).enumerate() {
        println!("🌡️  {} Ising Model (T_c = {:.4})", dim, t_c);
        println!("{}", "─".repeat(40));

        match i {
            0 => compare_2d_algorithms(t_c),
            1 => compare_3d_algorithms(t_c),
            2 => compare_4d_algorithms(t_c),
            _ => unreachable!(),
        }
        println!();
    }
}

fn compare_2d_algorithms(t_critical: f64) {
    let lattice_size = 32;
    let steps = 1000;

    // Test at critical temperature (where Wolff shines most)
    let temperature = t_critical;

    // Metropolis timing
    let mut ising_metro = IsingModel2D::new(lattice_size, temperature);
    let start = Instant::now();
    for _ in 0..steps {
        ising_metro.monte_carlo_step();
    }
    let metro_time = start.elapsed();

    // Wolff timing
    let mut ising_wolff = IsingModel2D::new(lattice_size, temperature);
    let mut cluster_sizes = Vec::new();
    let start = Instant::now();
    for _ in 0..steps {
        let size = ising_wolff.wolff_cluster_step_with_size();
        cluster_sizes.push(size);
    }
    let wolff_time = start.elapsed();

    let speedup = metro_time.as_millis() as f64 / wolff_time.as_millis() as f64;
    let mean_cluster_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
    let max_cluster_size = *cluster_sizes.iter().max().unwrap();

    println!("  Metropolis: {:6} ms", metro_time.as_millis());
    println!(
        "  Wolff:      {:6} ms ({:.1}x faster)",
        wolff_time.as_millis(),
        speedup
    );
    println!(
        "  Mean cluster size: {:.1} spins ({:.1}% of lattice)",
        mean_cluster_size,
        100.0 * mean_cluster_size / (lattice_size * lattice_size) as f64
    );
    println!(
        "  Max cluster size:  {} spins ({:.1}% of lattice)",
        max_cluster_size,
        100.0 * max_cluster_size as f64 / (lattice_size * lattice_size) as f64
    );
}

fn compare_3d_algorithms(t_critical: f64) {
    let lattice_size = 16; // Smaller for 3D
    let steps = 500;

    let temperature = t_critical;

    // Metropolis timing
    let mut ising_metro = IsingModel3D::new(lattice_size, temperature);
    let start = Instant::now();
    for _ in 0..steps {
        ising_metro.monte_carlo_step();
    }
    let metro_time = start.elapsed();

    // Wolff timing
    let mut ising_wolff = IsingModel3D::new(lattice_size, temperature);
    let mut cluster_sizes = Vec::new();
    let start = Instant::now();
    for _ in 0..steps {
        let size = ising_wolff.wolff_cluster_step_with_size();
        cluster_sizes.push(size);
    }
    let wolff_time = start.elapsed();

    let speedup = metro_time.as_millis() as f64 / (wolff_time.as_millis() as f64).max(1.0);
    let mean_cluster_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
    let total_spins = lattice_size * lattice_size * lattice_size;

    println!("  Metropolis: {:6} ms", metro_time.as_millis());
    println!(
        "  Wolff:      {:6} ms ({:.1}x faster)",
        wolff_time.as_millis(),
        speedup
    );
    println!(
        "  Mean cluster size: {:.1} spins ({:.2}% of lattice)",
        mean_cluster_size,
        100.0 * mean_cluster_size / total_spins as f64
    );
}

fn compare_4d_algorithms(t_critical: f64) {
    let lattice_size = 8; // Much smaller for 4D
    let steps = 100;

    let temperature = t_critical;

    // Metropolis timing
    let mut ising_metro = IsingModel4D::new(lattice_size, temperature);
    let start = Instant::now();
    for _ in 0..steps {
        ising_metro.monte_carlo_step();
    }
    let metro_time = start.elapsed();

    // Wolff timing
    let mut ising_wolff = IsingModel4D::new(lattice_size, temperature);
    let mut cluster_sizes = Vec::new();
    let start = Instant::now();
    for _ in 0..steps {
        let size = ising_wolff.wolff_cluster_step_with_size();
        cluster_sizes.push(size);
    }
    let wolff_time = start.elapsed();

    let speedup = metro_time.as_millis() as f64 / (wolff_time.as_millis() as f64).max(1.0);
    let mean_cluster_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
    let total_spins = (lattice_size as f64).powi(4) as usize;

    println!("  Metropolis: {:6} ms", metro_time.as_millis());
    println!(
        "  Wolff:      {:6} ms ({:.1}x faster)",
        wolff_time.as_millis(),
        speedup
    );
    println!(
        "  Mean cluster size: {:.1} spins ({:.3}% of lattice)",
        mean_cluster_size,
        100.0 * mean_cluster_size / total_spins as f64
    );
}

fn show_optimal_usage_scenarios() {
    println!("💡 When to Use Wolff Algorithm");
    println!("==============================\n");

    println!("✅ **USE WOLFF WHEN**:");
    println!("   🎯 Near critical temperature (T ≈ T_c)");
    println!("   📏 Large system sizes (N > 1000 spins)");
    println!("   🐌 Critical slowing down is problematic");
    println!("   📊 Need good statistics quickly");
    println!("   🔄 Long equilibration times with Metropolis");
    println!();

    println!("❌ **USE METROPOLIS WHEN**:");
    println!("   🌡️  Very high temperatures (T >> T_c)");
    println!("   🧊 Very low temperatures (T << T_c)");
    println!("   📦 Small system sizes (N < 100 spins)");
    println!("   💻 Memory is very limited");
    println!("   🎲 Want simple, local dynamics");
    println!();

    // Demonstrate temperature dependence
    demonstrate_temperature_dependence();
}

fn demonstrate_temperature_dependence() {
    println!("🌡️  Temperature Dependence of Wolff Efficiency");
    println!("==============================================\n");

    let lattice_size = 24;
    let steps = 200;
    let t_critical = analysis::critical_temperature_2d();

    let temperatures = [
        ("Low T", t_critical * 0.6),
        ("Critical T", t_critical),
        ("High T", t_critical * 1.6),
    ];

    println!("┌─────────────┬────────────┬────────────┬──────────────┬────────────┐");
    println!("│ Temperature │ Metropolis │    Wolff   │   Speedup    │ Cluster %  │");
    println!("├─────────────┼────────────┼────────────┼──────────────┼────────────┤");

    for (name, temp) in temperatures {
        // Metropolis
        let mut ising_metro = IsingModel2D::new(lattice_size, temp);
        let start = Instant::now();
        for _ in 0..steps {
            ising_metro.monte_carlo_step();
        }
        let metro_time = start.elapsed();

        // Wolff
        let mut ising_wolff = IsingModel2D::new(lattice_size, temp);
        let mut cluster_sizes = Vec::new();
        let start = Instant::now();
        for _ in 0..steps {
            let size = ising_wolff.wolff_cluster_step_with_size();
            cluster_sizes.push(size);
        }
        let wolff_time = start.elapsed();

        let speedup = metro_time.as_millis() as f64 / (wolff_time.as_millis() as f64).max(1.0);
        let mean_cluster_percent = cluster_sizes.iter().sum::<usize>() as f64
            / (cluster_sizes.len() * lattice_size * lattice_size) as f64
            * 100.0;

        println!(
            "│ {:11} │ {:8} ms │ {:8} ms │ {:10.1}x │ {:8.2}% │",
            name,
            metro_time.as_millis(),
            wolff_time.as_millis(),
            speedup,
            mean_cluster_percent
        );
    }

    println!("└─────────────┴────────────┴────────────┴──────────────┴────────────┘");
    println!("\n📈 **Key Insight**: Wolff is most effective near T_c where clusters are largest!");
}

fn performance_analysis() {
    println!("\n⚡ Performance Analysis");
    println!("======================\n");

    println!("🔍 **Scaling Behavior**:");
    println!("- Metropolis: O(N) per sweep, but τ ~ N^z with z ≈ 2.1 at T_c");
    println!("- Wolff:      O(N) per step, but τ ~ 1 (no critical slowing down!)");
    println!();

    println!("📊 **Statistical Efficiency**:");
    println!("- Wolff eliminates autocorrelation between successive configurations");
    println!("- Each Wolff step is nearly independent");
    println!("- Can achieve same statistical accuracy with ~100x fewer samples");
    println!();

    println!("💾 **Memory Usage**:");
    println!("- Wolff needs temporary arrays for visited sites and cluster queue");
    println!("- Memory overhead: O(N) for tracking visited sites");
    println!("- Worth it for the massive speedup!");
    println!();
}

fn advanced_wolff_techniques() {
    println!("🚀 Advanced Wolff Techniques");
    println!("============================\n");

    println!("1. **Hybrid Algorithm**:");
    println!("   - Use Wolff near T_c, Metropolis elsewhere");
    println!("   - Switch based on temperature or cluster size");
    println!();

    println!("2. **Cluster Size Control**:");
    println!("   - Monitor cluster sizes to detect phase transitions");
    println!("   - Large clusters → near critical point");
    println!("   - Small clusters → off-critical regime");
    println!();

    println!("3. **Multiple Clusters**:");
    println!("   - Can flip multiple non-overlapping clusters per step");
    println!("   - Parallel cluster identification");
    println!();

    println!("4. **Improved Estimators**:");
    println!("   - Use cluster information for better observables");
    println!("   - Cluster-based correlation functions");
    println!();

    // Example of hybrid approach
    demonstrate_hybrid_approach();
}

fn demonstrate_hybrid_approach() {
    println!("🔧 Hybrid Algorithm Example");
    println!("===========================\n");

    let lattice_size = 20;
    let mut ising = IsingModel2D::new(lattice_size, analysis::critical_temperature_2d());

    println!("Running hybrid simulation:");
    for step in 0..100 {
        // Use cluster size to decide algorithm
        let cluster_size = ising.wolff_cluster_step_with_size();
        let cluster_fraction = cluster_size as f64 / (lattice_size * lattice_size) as f64;

        // If clusters are small, consider using Metropolis for next few steps
        let algorithm = if cluster_fraction < 0.01 {
            "Metro (next)"
        } else {
            "Wolff"
        };

        if step % 20 == 0 {
            println!(
                "Step {}: Cluster size = {} ({:.2}%), using {}",
                step,
                cluster_size,
                cluster_fraction * 100.0,
                algorithm
            );
        }

        // In practice, you might switch algorithms here based on cluster_fraction
    }

    println!("\n✅ Algorithm Guide Complete!");
    println!("\n🎯 **Quick Start Recipe**:");
    println!("1. For critical temperature studies → Use Wolff");
    println!("2. For high/low temperature → Use Metropolis");
    println!("3. For large systems → Use Wolff");
    println!("4. For small systems → Either works fine");
    println!("5. When in doubt → Try both and compare!");
}
