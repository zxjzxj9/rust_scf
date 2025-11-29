/// Utility functions for analyzing Ising model results

/// Critical temperature for 2D Ising model (exact result)
/// T_c = 2J / (k_B * ln(1 + √2)) ≈ 2.269 J/k_B
pub fn critical_temperature_2d() -> f64 {
    2.0 / (1.0 + 2.0_f64.sqrt()).ln()
}

/// Critical temperature for 3D Ising model (numerical result)
/// T_c ≈ 4.511 J/k_B (from series expansions and Monte Carlo simulations)
pub fn critical_temperature_3d() -> f64 {
    4.511
}

/// Critical temperature for 4D Ising model (numerical/mean field result)
/// T_c ≈ 6.68 J/k_B
/// In d=4, the Ising model is at the upper critical dimension where
/// mean field theory becomes exact and logarithmic corrections appear
pub fn critical_temperature_4d() -> f64 {
    6.68
}

/// Keep backwards compatibility
pub fn critical_temperature() -> f64 {
    critical_temperature_2d()
}

/// Theoretical magnetization at T=0 (all spins aligned)
pub fn magnetization_at_zero_temp() -> f64 {
    1.0
}

/// Theoretical energy per site at T=0 for 2D (all spins aligned)
pub fn energy_per_site_at_zero_temp_2d() -> f64 {
    // Each spin has 4 aligned neighbors, E = -J * 4 / 2 = -2J per site
    -2.0
}

/// Theoretical energy per site at T=0 for 3D (all spins aligned)
pub fn energy_per_site_at_zero_temp_3d() -> f64 {
    // Each spin has 6 aligned neighbors, E = -J * 6 / 2 = -3J per site
    -3.0
}

/// Theoretical energy per site at T=0 for 4D (all spins aligned)
pub fn energy_per_site_at_zero_temp_4d() -> f64 {
    // Each spin has 8 aligned neighbors, E = -J * 8 / 2 = -4J per site
    -4.0
}

/// Keep backwards compatibility
pub fn energy_per_site_at_zero_temp() -> f64 {
    energy_per_site_at_zero_temp_2d()
}

/// Calculate coordination number (number of nearest neighbors) for each dimension
pub fn coordination_number(dimension: usize) -> usize {
    2 * dimension
}

/// Estimate critical temperature using mean field approximation
/// T_c ≈ z * J / k_B where z is the coordination number
/// This becomes exact in the limit d → ∞
pub fn mean_field_critical_temperature(dimension: usize) -> f64 {
    coordination_number(dimension) as f64
}
