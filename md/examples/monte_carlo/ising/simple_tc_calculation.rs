use md::ising::{IsingModel2D, IsingModel3D, analysis};

/// Simple Critical Temperature Calculation for Ising Models
/// 
/// This program demonstrates basic methods for finding critical temperature
/// by comparing theoretical values with simulation results.

fn main() {
    println!("🌡️ Simple Critical Temperature Calculator");
    println!("=========================================\n");
    
    // Show theoretical values
    display_theoretical_values();
    
    // Quick verification with peak finding
    println!("\n📊 Quick Numerical Verification");
    println!("===============================");
    
    // 2D verification
    println!("\n📐 2D Ising Model (24×24):");
    let tc_2d_numerical = quick_find_tc_2d();
    let tc_2d_theory = analysis::critical_temperature_2d();
    println!("   Theoretical: T_c = {:.4}", tc_2d_theory);
    println!("   Numerical:   T_c = {:.4}", tc_2d_numerical);
    println!("   Difference:  ΔT = {:.4} ({:.2}%)", 
             tc_2d_numerical - tc_2d_theory,
             100.0 * (tc_2d_numerical - tc_2d_theory) / tc_2d_theory);
    
    // 3D verification  
    println!("\n🧊 3D Ising Model (12³):");
    let tc_3d_numerical = quick_find_tc_3d();
    let tc_3d_theory = analysis::critical_temperature_3d();
    println!("   Theoretical: T_c = {:.4}", tc_3d_theory);
    println!("   Numerical:   T_c = {:.4}", tc_3d_numerical);
    println!("   Difference:  ΔT = {:.4} ({:.2}%)", 
             tc_3d_numerical - tc_3d_theory,
             100.0 * (tc_3d_numerical - tc_3d_theory) / tc_3d_theory);
    
    // Demonstrate temperature-dependent behavior
    demonstrate_temperature_behavior();
}

fn display_theoretical_values() {
    println!("📚 Theoretical Critical Temperatures");
    println!("===================================");
    
    let tc_2d = analysis::critical_temperature_2d();
    let tc_3d = analysis::critical_temperature_3d();
    
    println!("📐 2D Ising Model:");
    println!("   T_c = {:.6}", tc_2d);
    println!("   Formula: T_c = 2J / (k_B × ln(1 + √2))");
    println!("   Status: Exact analytical solution (Onsager, 1944)");
    
    println!("\n🧊 3D Ising Model:");
    println!("   T_c = {:.4}", tc_3d);
    println!("   Status: High-precision numerical result");
    println!("   Method: Series expansions + Monte Carlo");
    
    println!("\n🔍 Key Facts:");
    println!("   • Ratio T_c(3D)/T_c(2D) = {:.3}", tc_3d / tc_2d);
    println!("   • Both are continuous (2nd order) phase transitions");
    println!("   • 2D is at lower critical dimension (d_c = 2)");
    println!("   • 3D shows mean-field-like behavior (d > d_upper = 4)");
}

fn quick_find_tc_2d() -> f64 {
    println!("   Scanning temperatures near theoretical T_c...");
    
    let size = 24;
    let tc_theory = analysis::critical_temperature_2d();
    
    // Scan around theoretical value
    let temp_min = tc_theory - 0.3;
    let temp_max = tc_theory + 0.3;
    let temp_steps = 13;
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;
    
    let mut max_susceptibility = 0.0;
    let mut tc_estimate = tc_theory;
    
    for i in 0..temp_steps {
        let temperature = temp_min + i as f64 * temp_step;
        let mut ising = IsingModel2D::new(size, temperature);
        
        // Quick equilibration and sampling
        for _ in 0..1000 { ising.monte_carlo_step(); }
        
        let mut mag_samples = Vec::new();
        for _ in 0..1500 {
            ising.monte_carlo_step();
            mag_samples.push(ising.magnetization_per_site());
        }
        
        let susceptibility = ising.magnetic_susceptibility(&mag_samples);
        
        if susceptibility > max_susceptibility {
            max_susceptibility = susceptibility;
            tc_estimate = temperature;
        }
        
        print!(".");
    }
    println!(" Done!");
    
    tc_estimate
}

fn quick_find_tc_3d() -> f64 {
    println!("   Scanning temperatures near theoretical T_c...");
    
    let size = 12;
    let tc_theory = analysis::critical_temperature_3d();
    
    // Scan around theoretical value
    let temp_min = tc_theory - 0.4;
    let temp_max = tc_theory + 0.4;
    let temp_steps = 9; // Fewer points for 3D due to computational cost
    let temp_step = (temp_max - temp_min) / (temp_steps - 1) as f64;
    
    let mut max_susceptibility = 0.0;
    let mut tc_estimate = tc_theory;
    
    for i in 0..temp_steps {
        let temperature = temp_min + i as f64 * temp_step;
        let mut ising = IsingModel3D::new(size, temperature);
        
        // Quick equilibration and sampling
        for _ in 0..600 { ising.monte_carlo_step(); }
        
        let mut mag_samples = Vec::new();
        for _ in 0..1000 {
            ising.monte_carlo_step();
            mag_samples.push(ising.magnetization_per_site());
        }
        
        let susceptibility = ising.magnetic_susceptibility(&mag_samples);
        
        if susceptibility > max_susceptibility {
            max_susceptibility = susceptibility;
            tc_estimate = temperature;
        }
        
        print!(".");
    }
    println!(" Done!");
    
    tc_estimate
}

fn demonstrate_temperature_behavior() {
    println!("\n🌡️ Temperature-Dependent Behavior Demonstration");
    println!("================================================");
    
    // 2D demonstration
    println!("\n📐 2D System Behavior:");
    let tc_2d = analysis::critical_temperature_2d();
    let temps_2d = [tc_2d * 0.7, tc_2d, tc_2d * 1.3];
    let labels = ["Below T_c (Ordered)", "At T_c (Critical)", "Above T_c (Disordered)"];
    
    for (i, &temp) in temps_2d.iter().enumerate() {
        println!("\n   {} (T = {:.3}):", labels[i], temp);
        
        let mut ising = IsingModel2D::new(16, temp);
        
        // Equilibrate
        for _ in 0..1000 { ising.monte_carlo_step(); }
        
        // Sample properties
        let mut energy_sum = 0.0;
        let mut mag_sum = 0.0;
        let samples = 500;
        
        for _ in 0..samples {
            ising.monte_carlo_step();
            energy_sum += ising.energy_per_site();
            mag_sum += ising.abs_magnetization_per_site();
        }
        
        let avg_energy = energy_sum / samples as f64;
        let avg_mag = mag_sum / samples as f64;
        
        println!("      Energy per site: {:7.4}", avg_energy);
        println!("      |Magnetization|:  {:7.4}", avg_mag);
        println!("      Order parameter: {:7.4}", if temp < tc_2d { avg_mag } else { 0.0 });
    }
    
    // 3D demonstration
    println!("\n🧊 3D System Behavior:");
    let tc_3d = analysis::critical_temperature_3d();
    let temps_3d = [tc_3d * 0.7, tc_3d, tc_3d * 1.3];
    
    for (i, &temp) in temps_3d.iter().enumerate() {
        println!("\n   {} (T = {:.3}):", labels[i], temp);
        
        let mut ising = IsingModel3D::new(10, temp);
        
        // Equilibrate
        for _ in 0..600 { ising.monte_carlo_step(); }
        
        // Sample properties
        let mut energy_sum = 0.0;
        let mut mag_sum = 0.0;
        let samples = 400;
        
        for _ in 0..samples {
            ising.monte_carlo_step();
            energy_sum += ising.energy_per_site();
            mag_sum += ising.abs_magnetization_per_site();
        }
        
        let avg_energy = energy_sum / samples as f64;
        let avg_mag = mag_sum / samples as f64;
        
        println!("      Energy per site: {:7.4}", avg_energy);
        println!("      |Magnetization|:  {:7.4}", avg_mag);
        println!("      Order parameter: {:7.4}", if temp < tc_3d { avg_mag } else { 0.0 });
    }
    
    println!("\n💡 Key Observations:");
    println!("   • Below T_c: High magnetization (ordered phase)");
    println!("   • At T_c: Fluctuating magnetization (critical point)");  
    println!("   • Above T_c: Near-zero magnetization (disordered phase)");
    println!("   • Energy increases monotonically with temperature");
    println!("   • 3D systems show stronger ordering than 2D at same T/T_c");
}

/// Additional utility function to calculate critical exponents (basic version)
#[allow(dead_code)]
fn estimate_critical_exponents() {
    println!("\n🔬 Critical Exponents Estimation (Advanced)");
    println!("==========================================");
    
    println!("For the 2D Ising model, exact critical exponents:");
    println!("   α (specific heat):     α = 0 (logarithmic)");
    println!("   β (magnetization):     β = 1/8 = 0.125");
    println!("   γ (susceptibility):    γ = 7/4 = 1.75");
    println!("   ν (correlation length): ν = 1");
    println!("   η (correlation decay):  η = 1/4 = 0.25");
    
    println!("\nFor the 3D Ising model, numerical critical exponents:");
    println!("   α (specific heat):     α ≈ 0.11");
    println!("   β (magnetization):     β ≈ 0.326");
    println!("   γ (susceptibility):    γ ≈ 1.237");
    println!("   ν (correlation length): ν ≈ 0.63");
    println!("   η (correlation decay):  η ≈ 0.036");
    
    println!("\n🎯 These satisfy scaling relations:");
    println!("   α + 2β + γ = 2 (Rushbrooke)");
    println!("   γ = ν(2 - η) (Fisher)"); 
    println!("   dν = 2 - α (Josephson hyperscaling)");
}

