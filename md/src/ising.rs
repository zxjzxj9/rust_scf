use rand::prelude::*;

/// 2D Ising Model for Monte Carlo simulation
/// 
/// The Ising model is a mathematical model of ferromagnetism in statistical mechanics.
/// It consists of discrete variables representing magnetic dipole moments of atomic spins
/// that can be in one of two states (+1 or -1). The spins are arranged on a lattice.
#[derive(Debug, Clone)]
pub struct IsingModel2D {
    /// Lattice size (L x L)
    pub size: usize,
    /// Spin configuration: +1 or -1 for each site
    pub spins: Vec<Vec<i8>>,
    /// Temperature in units of J/k_B (reduced temperature)
    pub temperature: f64,
    /// Coupling constant J (typically set to 1)
    pub coupling: f64,
    /// External magnetic field h
    pub magnetic_field: f64,
    /// Random number generator
    rng: ThreadRng,
    /// Monte Carlo step counter
    pub step: u64,
}

impl IsingModel2D {
    /// Create a new 2D Ising model with random initial configuration
    pub fn new(size: usize, temperature: f64) -> Self {
        let mut rng = thread_rng();
        let mut spins = vec![vec![0i8; size]; size];
        
        // Initialize with random spins
        for i in 0..size {
            for j in 0..size {
                spins[i][j] = if rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }
        
        Self {
            size,
            spins,
            temperature,
            coupling: 1.0,
            magnetic_field: 0.0,
            rng,
            step: 0,
        }
    }
    
    /// Create a new 2D Ising model with all spins up (ordered state)
    pub fn new_ordered(size: usize, temperature: f64) -> Self {
        let spins = vec![vec![1i8; size]; size];
        
        Self {
            size,
            spins,
            temperature,
            coupling: 1.0,
            magnetic_field: 0.0,
            rng: thread_rng(),
            step: 0,
        }
    }
    
    /// Set the external magnetic field
    pub fn set_magnetic_field(&mut self, field: f64) {
        self.magnetic_field = field;
    }
    
    /// Set the coupling constant
    pub fn set_coupling(&mut self, coupling: f64) {
        self.coupling = coupling;
    }
    
    /// Get the spin at position (i, j) with periodic boundary conditions
    fn get_spin(&self, i: i32, j: i32) -> i8 {
        let i = ((i % self.size as i32) + self.size as i32) % self.size as i32;
        let j = ((j % self.size as i32) + self.size as i32) % self.size as i32;
        self.spins[i as usize][j as usize]
    }
    
    /// Calculate the local energy change if we flip spin at (i, j)
    fn delta_energy(&self, i: usize, j: usize) -> f64 {
        let current_spin = self.spins[i][j] as f64;
        let i = i as i32;
        let j = j as i32;
        
        // Sum of neighboring spins (with periodic boundary conditions)
        let neighbors_sum = self.get_spin(i-1, j) as f64 
                          + self.get_spin(i+1, j) as f64
                          + self.get_spin(i, j-1) as f64
                          + self.get_spin(i, j+1) as f64;
        
        // ΔE = 2 * J * s_i * (Σ neighbors) + 2 * h * s_i
        // Factor of 2 because we're flipping the spin
        2.0 * (self.coupling * current_spin * neighbors_sum + self.magnetic_field * current_spin)
    }
    
    /// Perform one Monte Carlo step (Metropolis algorithm)
    pub fn monte_carlo_step(&mut self) {
        // Try to flip N^2 spins per MC step (one sweep)
        for _ in 0..self.size * self.size {
            let i = self.rng.gen_range(0..self.size);
            let j = self.rng.gen_range(0..self.size);
            
            let delta_e = self.delta_energy(i, j);
            
            // Metropolis acceptance criterion
            if delta_e <= 0.0 || self.rng.gen::<f64>() < (-delta_e / self.temperature).exp() {
                // Accept the flip
                self.spins[i][j] *= -1;
            }
        }
        
        self.step += 1;
    }
    
    /// Calculate the total energy of the system
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        
        for i in 0..self.size {
            for j in 0..self.size {
                let spin = self.spins[i][j] as f64;
                
                // Nearest neighbors (avoid double counting by only counting right and down)
                let right = self.get_spin(i as i32, (j + 1) as i32) as f64;
                let down = self.get_spin((i + 1) as i32, j as i32) as f64;
                
                energy -= self.coupling * spin * (right + down);
                energy -= self.magnetic_field * spin;
            }
        }
        
        energy
    }
    
    /// Calculate the magnetization (sum of all spins)
    pub fn magnetization(&self) -> f64 {
        self.spins.iter()
            .flat_map(|row| row.iter())
            .map(|&s| s as f64)
            .sum()
    }
    
    /// Calculate the magnetization per site
    pub fn magnetization_per_site(&self) -> f64 {
        self.magnetization() / (self.size * self.size) as f64
    }
    
    /// Calculate the absolute magnetization per site
    pub fn abs_magnetization_per_site(&self) -> f64 {
        self.magnetization_per_site().abs()
    }
    
    /// Calculate the energy per site
    pub fn energy_per_site(&self) -> f64 {
        self.total_energy() / (self.size * self.size) as f64
    }
    
    /// Set the temperature for annealing
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }
    
    /// Get the current temperature
    pub fn get_temperature(&self) -> f64 {
        self.temperature
    }
    
    /// Calculate specific heat (requires sampling over multiple configurations)
    pub fn specific_heat(&self, energy_samples: &[f64]) -> f64 {
        if energy_samples.len() < 2 {
            return 0.0;
        }
        
        let mean_energy: f64 = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
        let mean_energy_squared: f64 = energy_samples.iter()
            .map(|e| e * e)
            .sum::<f64>() / energy_samples.len() as f64;
        
        let variance = mean_energy_squared - mean_energy * mean_energy;
        variance / (self.temperature * self.temperature * (self.size * self.size) as f64)
    }
    
    /// Calculate magnetic susceptibility (requires sampling over multiple configurations)
    pub fn magnetic_susceptibility(&self, magnetization_samples: &[f64]) -> f64 {
        if magnetization_samples.len() < 2 {
            return 0.0;
        }
        
        let mean_mag: f64 = magnetization_samples.iter().sum::<f64>() / magnetization_samples.len() as f64;
        let mean_mag_squared: f64 = magnetization_samples.iter()
            .map(|m| m * m)
            .sum::<f64>() / magnetization_samples.len() as f64;
        
        let variance = mean_mag_squared - mean_mag * mean_mag;
        variance / (self.temperature * (self.size * self.size) as f64)
    }
    
    /// Get a copy of the current spin configuration for visualization
    pub fn get_spins(&self) -> &Vec<Vec<i8>> {
        &self.spins
    }
    
    /// Print the current spin configuration (useful for small systems)
    pub fn print_configuration(&self) {
        for row in &self.spins {
            for &spin in row {
                print!("{:2}", if spin == 1 { "↑" } else { "↓" });
            }
            println!();
        }
    }
}

/// Utility functions for analyzing Ising model results
pub mod analysis {
    use super::*;
    
    /// Critical temperature for 2D Ising model (exact result)
    /// T_c = 2J / (k_B * ln(1 + √2)) ≈ 2.269 J/k_B
    pub fn critical_temperature() -> f64 {
        2.0 / (1.0 + 2.0_f64.sqrt()).ln()
    }
    
    /// Theoretical magnetization at T=0 (all spins aligned)
    pub fn magnetization_at_zero_temp() -> f64 {
        1.0
    }
    
    /// Theoretical energy per site at T=0 (all spins aligned)
    pub fn energy_per_site_at_zero_temp() -> f64 {
        -2.0  // Each spin has 4 aligned neighbors, E = -J * 4 / 2 = -2J per site
    }
}

#[cfg(test)]
mod tests {
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
}
