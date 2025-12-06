use md::ising::{analysis, IsingModel2D, IsingModel3D};
use std::collections::HashMap;

/// Comprehensive Critical Temperature Analysis for 2D and 3D Ising Models
///
/// This program demonstrates multiple methods for determining critical temperature:
/// 1. Peak finding in specific heat and magnetic susceptibility
/// 2. Binder cumulant method
/// 3. Finite size scaling analysis
/// 4. Comparison with theoretical values

fn main() {
    println!("🌡️ Critical Temperature Analysis for Ising Models");
    println!("==================================================\n");

    // Run analysis for both 2D and 3D models
    analyze_critical_temperature_2d();
    println!("\n{}\n", "=".repeat(60));
    analyze_critical_temperature_3d();

    // Compare theoretical vs numerical results
    println!("\n{}", "=".repeat(60));
    println!("\n📊 Theoretical vs Numerical Comparison");
    println!("======================================");
    compare_theoretical_values();
}

fn analyze_critical_temperature_2d() {
    println!("📐 2D Ising Model Critical Temperature Analysis");
    println!("===============================================");

    let theoretical_tc = analysis::critical_temperature_2d();
    println!("Theoretical T_c = {:.4}\n", theoretical_tc);

    // Method 1: Peak finding with single system size
    println!("🔍 Method 1: Peak Finding (32×32 system)");
    let tc_peaks = find_critical_temp_peaks_2d(32);
    println!(
        "- Specific heat peak: T_c = {:.4} (ΔT = {:.4})",
        tc_peaks.specific_heat,
        tc_peaks.specific_heat - theoretical_tc
    );
    println!(
        "- Susceptibility peak: T_c = {:.4} (ΔT = {:.4})",
        tc_peaks.susceptibility,
        tc_peaks.susceptibility - theoretical_tc
    );

    // Method 2: Binder cumulant method
    println!("\n🎯 Method 2: Binder Cumulant Analysis");
    let tc_binder = binder_cumulant_analysis_2d();
    println!(
        "- Binder cumulant crossing: T_c ≈ {:.4} (ΔT = {:.4})",
        tc_binder,
        tc_binder - theoretical_tc
    );

    // Method 3: Finite size scaling
    println!("\n📏 Method 3: Finite Size Scaling");
    let sizes = vec![8, 12, 16, 24, 32];
    let tc_scaling = finite_size_scaling_2d(&sizes);
    println!(
        "- Extrapolated T_c = {:.4} (ΔT = {:.4})",
        tc_scaling,
        tc_scaling - theoretical_tc
    );

    // Summary for 2D
    println!("\n📋 2D Summary:");
    println!("┌─────────────────────────┬──────────┬──────────┐");
    println!("│ Method                  │   T_c    │   Error  │");
    println!("├─────────────────────────┼──────────┼──────────┤");
    println!(
        "│ Theoretical (exact)     │ {:8.4} │    --    │",
        theoretical_tc
    );
    println!(
        "│ Specific Heat Peak      │ {:8.4} │ {:8.4} │",
        tc_peaks.specific_heat,
        tc_peaks.specific_heat - theoretical_tc
    );
    println!(
        "│ Susceptibility Peak     │ {:8.4} │ {:8.4} │",
        tc_peaks.susceptibility,
        tc_peaks.susceptibility - theoretical_tc
    );
    println!(
        "│ Binder Cumulant         │ {:8.4} │ {:8.4} │",
        tc_binder,
        tc_binder - theoretical_tc
    );
    println!(
        "│ Finite Size Scaling     │ {:8.4} │ {:8.4} │",
        tc_scaling,
        tc_scaling - theoretical_tc
    );
    println!("└─────────────────────────┴──────────┴──────────┘");
}

fn analyze_critical_temperature_3d() {
    println!("🧊 3D Ising Model Critical Temperature Analysis");
    println!("===============================================");

    let theoretical_tc = analysis::critical_temperature_3d();
    println!("Theoretical T_c = {:.4}\n", theoretical_tc);

    // Method 1: Peak finding with single system size
    println!("🔍 Method 1: Peak Finding (16³ system)");
    let tc_peaks = find_critical_temp_peaks_3d(16);
    println!(
        "- Specific heat peak: T_c = {:.4} (ΔT = {:.4})",
        tc_peaks.specific_heat,
        tc_peaks.specific_heat - theoretical_tc
    );
    println!(
        "- Susceptibility peak: T_c = {:.4} (ΔT = {:.4})",
        tc_peaks.susceptibility,
        tc_peaks.susceptibility - theoretical_tc
    );

    // Method 2: Binder cumulant method
    println!("\n🎯 Method 2: Binder Cumulant Analysis");
    let tc_binder = binder_cumulant_analysis_3d();
    println!(
        "- Binder cumulant crossing: T_c ≈ {:.4} (ΔT = {:.4})",
        tc_binder,
        tc_binder - theoretical_tc
    );

    // Method 3: Finite size scaling (reduced sizes for 3D due to computational cost)
    println!("\n📏 Method 3: Finite Size Scaling");
    let sizes = vec![6, 8, 10, 12];
    let tc_scaling = finite_size_scaling_3d(&sizes);
    println!(
        "- Extrapolated T_c = {:.4} (ΔT = {:.4})",
        tc_scaling,
        tc_scaling - theoretical_tc
    );

    // Summary for 3D
    println!("\n📋 3D Summary:");
    println!("┌─────────────────────────┬──────────┬──────────┐");
    println!("│ Method                  │   T_c    │   Error  │");
    println!("├─────────────────────────┼──────────┼──────────┤");
    println!(
        "│ Theoretical (numerical) │ {:8.4} │    --    │",
        theoretical_tc
    );
    println!(
        "│ Specific Heat Peak      │ {:8.4} │ {:8.4} │",
        tc_peaks.specific_heat,
        tc_peaks.specific_heat - theoretical_tc
    );
    println!(
        "│ Susceptibility Peak     │ {:8.4} │ {:8.4} │",
        tc_peaks.susceptibility,
        tc_peaks.susceptibility - theoretical_tc
    );
    println!(
        "│ Binder Cumulant         │ {:8.4} │ {:8.4} │",
        tc_binder,
        tc_binder - theoretical_tc
    );
    println!(
        "│ Finite Size Scaling     │ {:8.4} │ {:8.4} │",
        tc_scaling,
        tc_scaling - theoretical_tc
    );
    println!("└─────────────────────────┴──────────┴──────────┘");
}

#[derive(Debug, Clone)]
struct CriticalTemperatures {
    specific_heat: f64,
    susceptibility: f64,
}

fn find_critical_temp_peaks_2d(size: usize) -> CriticalTemperatures {
    let temp_range = (1.8, 2.8, 21); // (min, max, steps)
    let (temp_min, temp_max, temp_steps) = temp_range;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;

    let mut results = Vec::new();

    println!(
        "   Scanning temperature range {:.2} to {:.2} with {} points...",
        temp_min, temp_max, temp_steps
    );

    for i in 0..temp_steps {
        let temperature = temp_min + i as f64 * temp_step;

        let mut ising = IsingModel2D::new(size, temperature);

        // Equilibration
        for _ in 0..2000 {
            ising.monte_carlo_step();
        }

        // Sampling
        let mut energy_samples = Vec::new();
        let mut mag_samples = Vec::new();

        for _ in 0..3000 {
            ising.monte_carlo_step();
            energy_samples.push(ising.energy_per_site());
            mag_samples.push(ising.magnetization_per_site());
        }

        let specific_heat = ising.specific_heat(&energy_samples);
        let susceptibility = ising.magnetic_susceptibility(&mag_samples);

        results.push((temperature, specific_heat, susceptibility));
        print!(".");
    }
    println!(" Done!");

    // Find peaks
    let max_heat = results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let max_susceptibility = results
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    CriticalTemperatures {
        specific_heat: max_heat.0,
        susceptibility: max_susceptibility.0,
    }
}

fn find_critical_temp_peaks_3d(size: usize) -> CriticalTemperatures {
    let temp_range = (3.8, 5.2, 15); // (min, max, steps) - fewer steps for 3D due to cost
    let (temp_min, temp_max, temp_steps) = temp_range;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;

    let mut results = Vec::new();

    println!(
        "   Scanning temperature range {:.2} to {:.2} with {} points...",
        temp_min, temp_max, temp_steps
    );

    for i in 0..temp_steps {
        let temperature = temp_min + i as f64 * temp_step;

        let mut ising = IsingModel3D::new(size, temperature);

        // Equilibration (fewer steps for 3D due to computational cost)
        for _ in 0..1000 {
            ising.monte_carlo_step();
        }

        // Sampling
        let mut energy_samples = Vec::new();
        let mut mag_samples = Vec::new();

        for _ in 0..1500 {
            ising.monte_carlo_step();
            energy_samples.push(ising.energy_per_site());
            mag_samples.push(ising.magnetization_per_site());
        }

        let specific_heat = ising.specific_heat(&energy_samples);
        let susceptibility = ising.magnetic_susceptibility(&mag_samples);

        results.push((temperature, specific_heat, susceptibility));
        print!(".");
    }
    println!(" Done!");

    // Find peaks
    let max_heat = results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let max_susceptibility = results
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    CriticalTemperatures {
        specific_heat: max_heat.0,
        susceptibility: max_susceptibility.0,
    }
}

fn binder_cumulant_analysis_2d() -> f64 {
    println!("   Calculating Binder cumulant for multiple system sizes...");

    let sizes = vec![12, 16, 24, 32];
    let temp_range = (2.0, 2.5, 11); // Narrow range around critical temperature
    let (temp_min, temp_max, temp_steps) = temp_range;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;

    let mut binder_data: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

    for &size in &sizes {
        let mut size_data = Vec::new();

        for i in 0..temp_steps {
            let temperature = temp_min + i as f64 * temp_step;
            let mut ising = IsingModel2D::new(size, temperature);

            // Equilibration
            for _ in 0..1500 {
                ising.monte_carlo_step();
            }

            // Calculate Binder cumulant
            let mut m2_samples = Vec::new();
            let mut m4_samples = Vec::new();

            for _ in 0..2000 {
                ising.monte_carlo_step();
                let m = ising.magnetization_per_site();
                let m2 = m * m;
                let m4 = m2 * m2;

                m2_samples.push(m2);
                m4_samples.push(m4);
            }

            let mean_m2 = m2_samples.iter().sum::<f64>() / m2_samples.len() as f64;
            let mean_m4 = m4_samples.iter().sum::<f64>() / m4_samples.len() as f64;

            let binder = 1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2);
            size_data.push((temperature, binder));
        }

        binder_data.insert(size, size_data);
        print!(".");
    }
    println!(" Done!");

    // Find crossing point (simplified - look for temperature where curves are closest)
    let mut min_spread = f64::INFINITY;
    let mut crossing_temp = 0.0;

    for i in 0..temp_steps {
        let temp = temp_min + i as f64 * temp_step;
        let values: Vec<f64> = sizes.iter().map(|&size| binder_data[&size][i].1).collect();

        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let spread = max_val - min_val;

        if spread < min_spread {
            min_spread = spread;
            crossing_temp = temp;
        }
    }

    crossing_temp
}

fn binder_cumulant_analysis_3d() -> f64 {
    println!("   Calculating Binder cumulant for multiple system sizes...");

    let sizes = vec![8, 10, 12]; // Smaller sizes for 3D
    let temp_range = (4.2, 4.8, 9); // Narrow range around critical temperature
    let (temp_min, temp_max, temp_steps) = temp_range;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;

    let mut binder_data: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

    for &size in &sizes {
        let mut size_data = Vec::new();

        for i in 0..temp_steps {
            let temperature = temp_min + i as f64 * temp_step;
            let mut ising = IsingModel3D::new(size, temperature);

            // Equilibration
            for _ in 0..800 {
                ising.monte_carlo_step();
            }

            // Calculate Binder cumulant
            let mut m2_samples = Vec::new();
            let mut m4_samples = Vec::new();

            for _ in 0..1200 {
                ising.monte_carlo_step();
                let m = ising.magnetization_per_site();
                let m2 = m * m;
                let m4 = m2 * m2;

                m2_samples.push(m2);
                m4_samples.push(m4);
            }

            let mean_m2 = m2_samples.iter().sum::<f64>() / m2_samples.len() as f64;
            let mean_m4 = m4_samples.iter().sum::<f64>() / m4_samples.len() as f64;

            let binder = 1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2);
            size_data.push((temperature, binder));
        }

        binder_data.insert(size, size_data);
        print!(".");
    }
    println!(" Done!");

    // Find crossing point
    let mut min_spread = f64::INFINITY;
    let mut crossing_temp = 0.0;

    for i in 0..temp_steps {
        let temp = temp_min + i as f64 * temp_step;
        let values: Vec<f64> = sizes.iter().map(|&size| binder_data[&size][i].1).collect();

        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let spread = max_val - min_val;

        if spread < min_spread {
            min_spread = spread;
            crossing_temp = temp;
        }
    }

    crossing_temp
}

fn finite_size_scaling_2d(sizes: &[usize]) -> f64 {
    println!("   Performing finite size scaling analysis...");

    let mut tc_estimates = Vec::new();

    for &size in sizes {
        let tc = find_critical_temp_peaks_2d(size).specific_heat;
        tc_estimates.push((size, tc));
        println!("   Size {}: T_c ≈ {:.4}", size, tc);
    }

    // Simple linear extrapolation to infinite size
    // In practice, would use more sophisticated finite size scaling theory
    let largest_size = *sizes.last().unwrap();
    let largest_tc = tc_estimates.last().unwrap().1;

    // For 2D Ising, finite size scaling: T_c(L) = T_c(∞) + A/ln(L) + B/L
    // Use linear regression on 1/ln(L) for better extrapolation
    if tc_estimates.len() >= 3 {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let n = tc_estimates.len() as f64;

        for &(size, tc) in &tc_estimates {
            let x = 1.0 / (size as f64).ln(); // 1/ln(L)
            sum_x += x;
            sum_y += tc;
            sum_xy += x * tc;
            sum_x2 += x * x;
        }

        // Linear regression: y = a + b*x, extrapolate to x=0 (infinite size)
        let b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let a = (sum_y - b * sum_x) / n;
        a // Extrapolated T_c at infinite size
    } else {
        // Fallback for insufficient data
        let correction = 0.02 / (largest_size as f64).ln();
        largest_tc + correction
    }
}

fn finite_size_scaling_3d(sizes: &[usize]) -> f64 {
    println!("   Performing finite size scaling analysis...");

    let mut tc_estimates = Vec::new();

    for &size in sizes {
        let tc = find_critical_temp_peaks_3d(size).specific_heat;
        tc_estimates.push((size, tc));
        println!("   Size {}: T_c ≈ {:.4}", size, tc);
    }

    // For 3D Ising, finite size scaling: T_c(L) = T_c(∞) + A/L^(1/ν) where ν ≈ 0.63
    // Use linear regression on 1/L for simplicity (approximate)
    if tc_estimates.len() >= 3 {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let n = tc_estimates.len() as f64;

        for &(size, tc) in &tc_estimates {
            let x = 1.0 / size as f64; // 1/L (simplified from 1/L^(1/ν))
            sum_x += x;
            sum_y += tc;
            sum_xy += x * tc;
            sum_x2 += x * x;
        }

        // Linear regression: y = a + b*x, extrapolate to x=0 (infinite size)
        let b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let a = (sum_y - b * sum_x) / n;
        a // Extrapolated T_c at infinite size
    } else {
        // Fallback for insufficient data
        let largest_size = *sizes.last().unwrap();
        let largest_tc = tc_estimates.last().unwrap().1;
        let correction = 0.1 / largest_size as f64;
        largest_tc + correction
    }
}

fn compare_theoretical_values() {
    let tc_2d = analysis::critical_temperature_2d();
    let tc_3d = analysis::critical_temperature_3d();

    println!("📐 2D Ising Model:");
    println!("   T_c = {:.6} (exact analytical result)", tc_2d);
    println!("   Formula: T_c = 2J / (k_B * ln(1 + √2))");
    println!("   ln(1 + √2) = {:.6}", (1.0 + 2.0_f64.sqrt()).ln());

    println!("\n🧊 3D Ising Model:");
    println!("   T_c = {:.4} (high-precision numerical result)", tc_3d);
    println!("   From: Series expansions and Monte Carlo simulations");
    println!("   No exact analytical solution exists");

    println!("\n🔢 Key Differences:");
    println!("   T_c(3D) / T_c(2D) = {:.4}", tc_3d / tc_2d);
    println!("   Coordination numbers: 2D = 4, 3D = 6");
    println!("   Phase transition: Both continuous (second-order)");

    println!("\n💡 Physical Interpretation:");
    println!("   - Higher coordination in 3D → stronger interactions → higher T_c");
    println!("   - 2D is at the lower critical dimension for Ising model");
    println!("   - Both show universal critical behavior");

    println!("\n🎯 Numerical Methods for T_c Determination:");
    println!("   1. Specific Heat Peak: Local maximum of C_V(T)");
    println!("   2. Susceptibility Peak: Local maximum of χ(T)");
    println!("   3. Binder Cumulant: Size-independent crossing at T_c");
    println!("   4. Finite Size Scaling: Extrapolation to thermodynamic limit");
    println!("   5. Critical Slowing Down: Correlation time divergence");
}
