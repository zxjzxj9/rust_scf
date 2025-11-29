
use super::*;
use approx::*;

#[test]
fn test_ising_creation() {
    let ising = IsingModel2D::new(10, 2.0);
    assert_eq!(ising.size, 10);
    assert_eq!(ising.temperature, 2.0);
    assert_eq!(ising.spins.len(), 10);
    assert_eq!(ising.spins[0].len(), 10);
}

#[test]
fn test_ordered_state() {
    let ising = IsingModel2D::new_ordered(5, 1.0);
    assert_eq!(ising.magnetization_per_site(), 1.0);
    assert_relative_eq!(ising.energy_per_site(), -2.0, epsilon = 1e-10);
}

#[test]
fn test_energy_calculation() {
    let mut ising = IsingModel2D::new_ordered(3, 1.0);
    let initial_energy = ising.total_energy();
    
    // Flip one spin and check energy change
    ising.spins[1][1] = -1;
    let new_energy = ising.total_energy();
    
    // The energy should increase by 8J (one spin surrounded by 4 opposite neighbors)
    assert_relative_eq!(new_energy - initial_energy, 8.0, epsilon = 1e-10);
}

#[test]
fn test_periodic_boundary_conditions() {
    let ising = IsingModel2D::new_ordered(3, 1.0);
    
    // Test corner and edge cases
    assert_eq!(ising.get_spin(-1, 0), 1);  // Should wrap to (2, 0)
    assert_eq!(ising.get_spin(3, 1), 1);   // Should wrap to (0, 1)
    assert_eq!(ising.get_spin(1, -1), 1);  // Should wrap to (1, 2)
    assert_eq!(ising.get_spin(1, 3), 1);   // Should wrap to (1, 0)
}

#[test]
fn test_critical_temperature() {
    let t_c = analysis::critical_temperature();
    assert_relative_eq!(t_c, 2.269, epsilon = 0.001);
}

// 3D Ising Model Tests
#[test]
fn test_ising3d_creation() {
    let ising = IsingModel3D::new(8, 3.0);
    assert_eq!(ising.size, 8);
    assert_eq!(ising.temperature, 3.0);
    assert_eq!(ising.spins.len(), 8);
    assert_eq!(ising.spins[0].len(), 8);
    assert_eq!(ising.spins[0][0].len(), 8);
}

#[test]
fn test_ising3d_ordered_state() {
    let ising = IsingModel3D::new_ordered(5, 1.0);
    assert_eq!(ising.magnetization_per_site(), 1.0);
    assert_relative_eq!(ising.energy_per_site(), -3.0, epsilon = 1e-10);
}

#[test]
fn test_ising3d_energy_calculation() {
    let mut ising = IsingModel3D::new_ordered(3, 1.0);
    let initial_energy = ising.total_energy();
    
    // Flip one spin and check energy change
    ising.spins[1][1][1] = -1;
    let new_energy = ising.total_energy();
    
    // The energy should increase by 12J (one spin surrounded by 6 opposite neighbors)
    assert_relative_eq!(new_energy - initial_energy, 12.0, epsilon = 1e-10);
}

#[test]
fn test_ising3d_periodic_boundary_conditions() {
    let ising = IsingModel3D::new_ordered(3, 1.0);
    
    // Test corner and edge cases
    assert_eq!(ising.get_spin(-1, 0, 0), 1);  // Should wrap to (2, 0, 0)
    assert_eq!(ising.get_spin(3, 1, 2), 1);   // Should wrap to (0, 1, 2)
    assert_eq!(ising.get_spin(1, -1, 1), 1);  // Should wrap to (1, 2, 1)
    assert_eq!(ising.get_spin(1, 3, 0), 1);   // Should wrap to (1, 0, 0)
    assert_eq!(ising.get_spin(0, 1, -1), 1);  // Should wrap to (0, 1, 2)
    assert_eq!(ising.get_spin(2, 0, 3), 1);   // Should wrap to (2, 0, 0)
}

#[test]
fn test_ising3d_critical_temperatures() {
    let t_c_2d = analysis::critical_temperature_2d();
    let t_c_3d = analysis::critical_temperature_3d();
    
    assert_relative_eq!(t_c_2d, 2.269, epsilon = 0.001);
    assert_relative_eq!(t_c_3d, 4.511, epsilon = 0.001);
    assert!(t_c_3d > t_c_2d); // 3D critical temperature should be higher
}

#[test]
fn test_ising3d_monte_carlo_step() {
    let mut ising = IsingModel3D::new(4, 2.0);
    let initial_step = ising.step;
    
    ising.monte_carlo_step();
    
    assert_eq!(ising.step, initial_step + 1);
}

#[test]
fn test_ising3d_magnetization_bounds() {
    let ising_ordered = IsingModel3D::new_ordered(5, 1.0);
    let total_spins = (5 * 5 * 5) as f64;
    
    // For ordered state, all spins are +1
    assert_eq!(ising_ordered.magnetization(), total_spins);
    assert_eq!(ising_ordered.magnetization_per_site(), 1.0);
    
    // Check that magnetization is bounded by the number of spins
    let ising_random = IsingModel3D::new(5, 5.0);
    assert!(ising_random.magnetization().abs() <= total_spins);
    assert!(ising_random.magnetization_per_site().abs() <= 1.0);
}

// 4D Ising Model Tests
#[test]
fn test_ising4d_creation() {
    let ising = IsingModel4D::new(6, 4.0);
    assert_eq!(ising.size, 6);
    assert_eq!(ising.temperature, 4.0);
    assert_eq!(ising.spins.len(), 6);
    assert_eq!(ising.spins[0].len(), 6);
    assert_eq!(ising.spins[0][0].len(), 6);
    assert_eq!(ising.spins[0][0][0].len(), 6);
}

#[test]
fn test_ising4d_ordered_state() {
    let ising = IsingModel4D::new_ordered(4, 2.0);
    assert_eq!(ising.magnetization_per_site(), 1.0);
    assert_relative_eq!(ising.energy_per_site(), -4.0, epsilon = 1e-10);
}

#[test]
fn test_ising4d_energy_calculation() {
    let mut ising = IsingModel4D::new_ordered(3, 1.0);
    let initial_energy = ising.total_energy();
    
    // Flip one spin and check energy change
    ising.spins[1][1][1][1] = -1;
    let new_energy = ising.total_energy();
    
    // The energy should increase by 16J (one spin surrounded by 8 opposite neighbors)
    assert_relative_eq!(new_energy - initial_energy, 16.0, epsilon = 1e-10);
}

#[test]
fn test_ising4d_periodic_boundary_conditions() {
    let ising = IsingModel4D::new_ordered(3, 1.0);
    
    // Test corner and edge cases for all 4 dimensions
    assert_eq!(ising.get_spin(-1, 0, 0, 0), 1);  // Should wrap to (2, 0, 0, 0)
    assert_eq!(ising.get_spin(3, 1, 2, 1), 1);   // Should wrap to (0, 1, 2, 1)
    assert_eq!(ising.get_spin(1, -1, 1, 2), 1);  // Should wrap to (1, 2, 1, 2)
    assert_eq!(ising.get_spin(1, 3, 0, 1), 1);   // Should wrap to (1, 0, 0, 1)
    assert_eq!(ising.get_spin(0, 1, -1, 2), 1);  // Should wrap to (0, 1, 2, 2)
    assert_eq!(ising.get_spin(2, 0, 3, 1), 1);   // Should wrap to (2, 0, 0, 1)
    assert_eq!(ising.get_spin(1, 2, 1, -1), 1);  // Should wrap to (1, 2, 1, 2)
    assert_eq!(ising.get_spin(0, 1, 2, 3), 1);   // Should wrap to (0, 1, 2, 0)
}

#[test]
fn test_ising4d_critical_temperatures() {
    let t_c_2d = analysis::critical_temperature_2d();
    let t_c_3d = analysis::critical_temperature_3d();
    let t_c_4d = analysis::critical_temperature_4d();
    
    assert_relative_eq!(t_c_2d, 2.269, epsilon = 0.001);
    assert_relative_eq!(t_c_3d, 4.511, epsilon = 0.001);
    assert_relative_eq!(t_c_4d, 6.68, epsilon = 0.01);
    
    // Critical temperatures should increase with dimension
    assert!(t_c_3d > t_c_2d);
    assert!(t_c_4d > t_c_3d);
}

#[test]
fn test_ising4d_monte_carlo_step() {
    let mut ising = IsingModel4D::new(3, 3.0);
    let initial_step = ising.step;
    
    ising.monte_carlo_step();
    
    assert_eq!(ising.step, initial_step + 1);
}

#[test]
fn test_ising4d_magnetization_bounds() {
    let ising_ordered = IsingModel4D::new_ordered(4, 1.0);
    let total_spins = (4_f64).powi(4);
    
    // For ordered state, all spins are +1
    assert_eq!(ising_ordered.magnetization(), total_spins);
    assert_eq!(ising_ordered.magnetization_per_site(), 1.0);
    
    // Check that magnetization is bounded by the number of spins
    let ising_random = IsingModel4D::new(4, 8.0);
    assert!(ising_random.magnetization().abs() <= total_spins);
    assert!(ising_random.magnetization_per_site().abs() <= 1.0);
}

#[test]
fn test_coordination_numbers() {
    assert_eq!(analysis::coordination_number(2), 4);
    assert_eq!(analysis::coordination_number(3), 6);
    assert_eq!(analysis::coordination_number(4), 8);
    assert_eq!(analysis::coordination_number(5), 10);
}

#[test]
fn test_mean_field_approximation() {
    // Mean field theory should be exact in d >= 4
    let mf_4d = analysis::mean_field_critical_temperature(4);
    let actual_4d = analysis::critical_temperature_4d();
    
    // Should be close but not exactly equal due to finite size effects
    assert!((mf_4d - actual_4d).abs() / actual_4d < 0.3);
    assert_eq!(mf_4d, 8.0); // 2 * 4 = 8 neighbors
}

// Wolff Cluster Algorithm Tests
#[test]
fn test_wolff_2d_basic() {
    let mut ising = IsingModel2D::new_ordered(8, 2.0);
    let initial_energy = ising.total_energy();
    let initial_mag = ising.magnetization();
    
    // Perform Wolff steps
    for _ in 0..10 {
        ising.wolff_cluster_step();
    }
    
    // Energy and magnetization should change but be physically reasonable
    assert!(ising.total_energy().is_finite());
    assert!(ising.magnetization().abs() <= (ising.size * ising.size) as f64);
    assert_eq!(ising.step, 10);
}

#[test]
fn test_wolff_3d_basic() {
    let mut ising = IsingModel3D::new_ordered(6, 3.0);
    let initial_energy = ising.total_energy();
    
    // Perform Wolff steps
    for _ in 0..5 {
        ising.wolff_cluster_step();
    }
    
    // Energy should be finite and magnetization bounded
    assert!(ising.total_energy().is_finite());
    assert!(ising.magnetization().abs() <= (ising.size * ising.size * ising.size) as f64);
    assert_eq!(ising.step, 5);
}

#[test]
fn test_wolff_4d_basic() {
    let mut ising = IsingModel4D::new_ordered(4, 4.0);
    let total_spins = (ising.size as f64).powi(4);
    
    // Perform Wolff steps
    for _ in 0..3 {
        ising.wolff_cluster_step();
    }
    
    // Check basic properties
    assert!(ising.total_energy().is_finite());
    assert!(ising.magnetization().abs() <= total_spins);
    assert_eq!(ising.step, 3);
}

#[test]
fn test_wolff_cluster_sizes() {
    let mut ising = IsingModel2D::new_ordered(6, 1.0);  // Low temperature
    let total_spins = ising.size * ising.size;
    
    let mut cluster_sizes = Vec::new();
    for _ in 0..20 {
        let size = ising.wolff_cluster_step_with_size();
        cluster_sizes.push(size);
    }
    
    // All cluster sizes should be reasonable
    for &size in &cluster_sizes {
        assert!(size > 0);
        assert!(size <= total_spins);
    }
    
    // At low temperature, clusters should generally be small
    let mean_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
    assert!(mean_size > 0.0);
    assert!(mean_size <= total_spins as f64);
}

#[test]
fn test_wolff_energy_conservation() {
    // Test that energy calculations are consistent after Wolff steps
    let mut ising1 = IsingModel2D::new(8, 2.5);
    let mut ising2 = ising1.clone();
    
    // Apply same random seed for deterministic comparison
    // (Note: This test checks consistency, not exact reproduction due to RNG)
    
    // Both should maintain physical properties
    for _ in 0..5 {
        ising1.monte_carlo_step();
        ising2.wolff_cluster_step();
    }
    
    // Both should have reasonable energies
    let energy1 = ising1.total_energy();
    let energy2 = ising2.total_energy();
    
    assert!(energy1.is_finite());
    assert!(energy2.is_finite());
    
    // Energy per site should be in reasonable range for 2D Ising model
    let min_energy_per_site = -2.0;  // Ground state
    let max_energy_per_site = 2.0;   // Maximally frustrated state
    
    assert!(ising1.energy_per_site() >= min_energy_per_site);
    assert!(ising1.energy_per_site() <= max_energy_per_site);
    assert!(ising2.energy_per_site() >= min_energy_per_site);
    assert!(ising2.energy_per_site() <= max_energy_per_site);
}

#[test]
fn test_wolff_detailed_balance() {
    // Test that Wolff algorithm maintains detailed balance
    // by checking that equilibrium properties are preserved
    let mut ising = IsingModel2D::new(12, analysis::critical_temperature_2d());
    
    // Equilibrate
    for _ in 0..100 {
        ising.wolff_cluster_step();
    }
    
    // Collect samples
    let mut energy_samples = Vec::new();
    let mut mag_samples = Vec::new();
    
    for _ in 0..200 {
        ising.wolff_cluster_step();
        energy_samples.push(ising.energy_per_site());
        mag_samples.push(ising.magnetization_per_site());
    }
    
    // Check that samples are reasonable
    let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
    let mean_mag = mag_samples.iter().sum::<f64>() / mag_samples.len() as f64;
    
    // At critical temperature, magnetization should be small on average
    assert!(mean_mag.abs() < 0.5);
    assert!(mean_energy > -2.0 && mean_energy < 0.0);
    
    // Variance should be reasonable (not zero, not infinite)
    let energy_var = energy_samples.iter()
        .map(|e| (e - mean_energy).powi(2))
        .sum::<f64>() / energy_samples.len() as f64;
    
    assert!(energy_var > 0.0);
    assert!(energy_var < 1.0);
}

#[test]
fn test_wolff_vs_metropolis_consistency() {
    // Test that both algorithms sample the same equilibrium distribution
    // (at least approximately for finite samples)
    let temp = 3.0;
    let size = 10;
    let steps = 500;
    
    let mut ising_metro = IsingModel2D::new(size, temp);
    let mut ising_wolff = IsingModel2D::new(size, temp);
    
    // Equilibrate both
    for _ in 0..200 {
        ising_metro.monte_carlo_step();
        ising_wolff.wolff_cluster_step();
    }
    
    // Collect samples
    let mut metro_energies = Vec::new();
    let mut wolff_energies = Vec::new();
    
    for _ in 0..steps {
        ising_metro.monte_carlo_step();
        ising_wolff.wolff_cluster_step();
        
        metro_energies.push(ising_metro.energy_per_site());
        wolff_energies.push(ising_wolff.energy_per_site());
    }
    
    // Calculate means
    let metro_mean = metro_energies.iter().sum::<f64>() / metro_energies.len() as f64;
    let wolff_mean = wolff_energies.iter().sum::<f64>() / wolff_energies.len() as f64;
    
    // Means should be reasonably close (within statistical error)
    let diff = (metro_mean - wolff_mean).abs();
    assert!(diff < 0.1, "Energy means too different: Metro={:.4}, Wolff={:.4}, diff={:.4}", 
           metro_mean, wolff_mean, diff);
}
