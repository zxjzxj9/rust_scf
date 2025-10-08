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

/// 3D Ising Model for Monte Carlo simulation
/// 
/// Extension of the 2D Ising model to three dimensions.
/// In 3D, each spin has 6 nearest neighbors (±x, ±y, ±z directions).
#[derive(Debug, Clone)]
pub struct IsingModel3D {
    /// Lattice size (L x L x L)
    pub size: usize,
    /// Spin configuration: +1 or -1 for each site
    pub spins: Vec<Vec<Vec<i8>>>,
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

impl IsingModel3D {
    /// Create a new 3D Ising model with random initial configuration
    pub fn new(size: usize, temperature: f64) -> Self {
        let mut rng = thread_rng();
        let mut spins = vec![vec![vec![0i8; size]; size]; size];
        
        // Initialize with random spins
        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    spins[i][j][k] = if rng.gen_bool(0.5) { 1 } else { -1 };
                }
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
    
    /// Create a new 3D Ising model with all spins up (ordered state)
    pub fn new_ordered(size: usize, temperature: f64) -> Self {
        let spins = vec![vec![vec![1i8; size]; size]; size];
        
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
    
    /// Get the spin at position (i, j, k) with periodic boundary conditions
    fn get_spin(&self, i: i32, j: i32, k: i32) -> i8 {
        let i = ((i % self.size as i32) + self.size as i32) % self.size as i32;
        let j = ((j % self.size as i32) + self.size as i32) % self.size as i32;
        let k = ((k % self.size as i32) + self.size as i32) % self.size as i32;
        self.spins[i as usize][j as usize][k as usize]
    }
    
    /// Calculate the local energy change if we flip spin at (i, j, k)
    fn delta_energy(&self, i: usize, j: usize, k: usize) -> f64 {
        let current_spin = self.spins[i][j][k] as f64;
        let i = i as i32;
        let j = j as i32;
        let k = k as i32;
        
        // Sum of 6 neighboring spins (with periodic boundary conditions)
        let neighbors_sum = self.get_spin(i-1, j, k) as f64 
                          + self.get_spin(i+1, j, k) as f64
                          + self.get_spin(i, j-1, k) as f64
                          + self.get_spin(i, j+1, k) as f64
                          + self.get_spin(i, j, k-1) as f64
                          + self.get_spin(i, j, k+1) as f64;
        
        // ΔE = 2 * J * s_i * (Σ neighbors) + 2 * h * s_i
        // Factor of 2 because we're flipping the spin
        2.0 * (self.coupling * current_spin * neighbors_sum + self.magnetic_field * current_spin)
    }
    
    /// Perform one Monte Carlo step (Metropolis algorithm)
    pub fn monte_carlo_step(&mut self) {
        // Try to flip N^3 spins per MC step (one sweep)
        for _ in 0..self.size * self.size * self.size {
            let i = self.rng.gen_range(0..self.size);
            let j = self.rng.gen_range(0..self.size);
            let k = self.rng.gen_range(0..self.size);
            
            let delta_e = self.delta_energy(i, j, k);
            
            // Metropolis acceptance criterion
            if delta_e <= 0.0 || self.rng.gen::<f64>() < (-delta_e / self.temperature).exp() {
                // Accept the flip
                self.spins[i][j][k] *= -1;
            }
        }
        
        self.step += 1;
    }
    
    /// Calculate the total energy of the system
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        
        for i in 0..self.size {
            for j in 0..self.size {
                for k in 0..self.size {
                    let spin = self.spins[i][j][k] as f64;
                    
                    // Nearest neighbors (avoid double counting by only counting positive directions)
                    let right = self.get_spin(i as i32, j as i32, (k + 1) as i32) as f64;
                    let down = self.get_spin(i as i32, (j + 1) as i32, k as i32) as f64;
                    let forward = self.get_spin((i + 1) as i32, j as i32, k as i32) as f64;
                    
                    energy -= self.coupling * spin * (right + down + forward);
                    energy -= self.magnetic_field * spin;
                }
            }
        }
        
        energy
    }
    
    /// Calculate the magnetization (sum of all spins)
    pub fn magnetization(&self) -> f64 {
        self.spins.iter()
            .flat_map(|plane| plane.iter())
            .flat_map(|row| row.iter())
            .map(|&s| s as f64)
            .sum()
    }
    
    /// Calculate the magnetization per site
    pub fn magnetization_per_site(&self) -> f64 {
        self.magnetization() / (self.size * self.size * self.size) as f64
    }
    
    /// Calculate the absolute magnetization per site
    pub fn abs_magnetization_per_site(&self) -> f64 {
        self.magnetization_per_site().abs()
    }
    
    /// Calculate the energy per site
    pub fn energy_per_site(&self) -> f64 {
        self.total_energy() / (self.size * self.size * self.size) as f64
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
        variance / (self.temperature * self.temperature * (self.size * self.size * self.size) as f64)
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
        variance / (self.temperature * (self.size * self.size * self.size) as f64)
    }
    
    /// Get a copy of the current spin configuration for visualization
    pub fn get_spins(&self) -> &Vec<Vec<Vec<i8>>> {
        &self.spins
    }
    
    /// Print a 2D slice of the 3D configuration at a given z-level (useful for visualization)
    pub fn print_slice(&self, z_level: usize) {
        if z_level >= self.size {
            println!("Error: z_level {} exceeds lattice size {}", z_level, self.size);
            return;
        }
        
        println!("Z-slice at level {} (size {}×{}):", z_level, self.size, self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                print!("{:2}", if self.spins[i][j][z_level] == 1 { "↑" } else { "↓" });
            }
            println!();
        }
    }
    
    /// Print multiple slices for better 3D visualization
    pub fn print_multiple_slices(&self, num_slices: usize) {
        let slice_spacing = self.size.max(1) / num_slices.max(1);
        
        for slice in 0..num_slices.min(self.size) {
            let z_level = slice * slice_spacing;
            if z_level < self.size {
                self.print_slice(z_level);
                println!();
            }
        }
    }
}

/// 4D Ising Model for Monte Carlo simulation
/// 
/// Extension of the Ising model to four dimensions.
/// In 4D, each spin has 8 nearest neighbors (±x, ±y, ±z, ±w directions).
/// This is useful for studying higher-dimensional phase transitions and
/// testing theories in dimensions where mean field theory becomes exact (d ≥ 4).
#[derive(Debug, Clone)]
pub struct IsingModel4D {
    /// Lattice size (L x L x L x L)
    pub size: usize,
    /// Spin configuration: +1 or -1 for each site
    pub spins: Vec<Vec<Vec<Vec<i8>>>>,
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

impl IsingModel4D {
    /// Create a new 4D Ising model with random initial configuration
    pub fn new(size: usize, temperature: f64) -> Self {
        let mut rng = thread_rng();
        let mut spins = vec![vec![vec![vec![0i8; size]; size]; size]; size];
        
        // Initialize with random spins
        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    for l in 0..size {
                        spins[i][j][k][l] = if rng.gen_bool(0.5) { 1 } else { -1 };
                    }
                }
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
    
    /// Create a new 4D Ising model with all spins up (ordered state)
    pub fn new_ordered(size: usize, temperature: f64) -> Self {
        let spins = vec![vec![vec![vec![1i8; size]; size]; size]; size];
        
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
    
    /// Get the spin at position (i, j, k, l) with periodic boundary conditions
    fn get_spin(&self, i: i32, j: i32, k: i32, l: i32) -> i8 {
        let i = ((i % self.size as i32) + self.size as i32) % self.size as i32;
        let j = ((j % self.size as i32) + self.size as i32) % self.size as i32;
        let k = ((k % self.size as i32) + self.size as i32) % self.size as i32;
        let l = ((l % self.size as i32) + self.size as i32) % self.size as i32;
        self.spins[i as usize][j as usize][k as usize][l as usize]
    }
    
    /// Calculate the local energy change if we flip spin at (i, j, k, l)
    fn delta_energy(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let current_spin = self.spins[i][j][k][l] as f64;
        let i = i as i32;
        let j = j as i32;
        let k = k as i32;
        let l = l as i32;
        
        // Sum of 8 neighboring spins (with periodic boundary conditions)
        let neighbors_sum = self.get_spin(i-1, j, k, l) as f64 
                          + self.get_spin(i+1, j, k, l) as f64
                          + self.get_spin(i, j-1, k, l) as f64
                          + self.get_spin(i, j+1, k, l) as f64
                          + self.get_spin(i, j, k-1, l) as f64
                          + self.get_spin(i, j, k+1, l) as f64
                          + self.get_spin(i, j, k, l-1) as f64
                          + self.get_spin(i, j, k, l+1) as f64;
        
        // ΔE = 2 * J * s_i * (Σ neighbors) + 2 * h * s_i
        // Factor of 2 because we're flipping the spin
        2.0 * (self.coupling * current_spin * neighbors_sum + self.magnetic_field * current_spin)
    }
    
    /// Perform one Monte Carlo step (Metropolis algorithm)
    pub fn monte_carlo_step(&mut self) {
        // Try to flip N^4 spins per MC step (one sweep)
        for _ in 0..self.size * self.size * self.size * self.size {
            let i = self.rng.gen_range(0..self.size);
            let j = self.rng.gen_range(0..self.size);
            let k = self.rng.gen_range(0..self.size);
            let l = self.rng.gen_range(0..self.size);
            
            let delta_e = self.delta_energy(i, j, k, l);
            
            // Metropolis acceptance criterion
            if delta_e <= 0.0 || self.rng.gen::<f64>() < (-delta_e / self.temperature).exp() {
                // Accept the flip
                self.spins[i][j][k][l] *= -1;
            }
        }
        
        self.step += 1;
    }
    
    /// Calculate the total energy of the system
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        
        for i in 0..self.size {
            for j in 0..self.size {
                for k in 0..self.size {
                    for l in 0..self.size {
                        let spin = self.spins[i][j][k][l] as f64;
                        
                        // Nearest neighbors (avoid double counting by only counting positive directions)
                        let neighbors = [
                            self.get_spin(i as i32, j as i32, k as i32, (l + 1) as i32) as f64,
                            self.get_spin(i as i32, j as i32, (k + 1) as i32, l as i32) as f64,
                            self.get_spin(i as i32, (j + 1) as i32, k as i32, l as i32) as f64,
                            self.get_spin((i + 1) as i32, j as i32, k as i32, l as i32) as f64,
                        ];
                        
                        for neighbor in neighbors {
                            energy -= self.coupling * spin * neighbor;
                        }
                        energy -= self.magnetic_field * spin;
                    }
                }
            }
        }
        
        energy
    }
    
    /// Calculate the magnetization (sum of all spins)
    pub fn magnetization(&self) -> f64 {
        self.spins.iter()
            .flat_map(|vol| vol.iter())
            .flat_map(|plane| plane.iter())
            .flat_map(|row| row.iter())
            .map(|&s| s as f64)
            .sum()
    }
    
    /// Calculate the magnetization per site
    pub fn magnetization_per_site(&self) -> f64 {
        let total_sites = (self.size as f64).powi(4);
        self.magnetization() / total_sites
    }
    
    /// Calculate the absolute magnetization per site
    pub fn abs_magnetization_per_site(&self) -> f64 {
        self.magnetization_per_site().abs()
    }
    
    /// Calculate the energy per site
    pub fn energy_per_site(&self) -> f64 {
        let total_sites = (self.size as f64).powi(4);
        self.total_energy() / total_sites
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
        let total_sites = (self.size as f64).powi(4);
        variance / (self.temperature * self.temperature * total_sites)
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
        let total_sites = (self.size as f64).powi(4);
        variance / (self.temperature * total_sites)
    }
    
    /// Get a copy of the current spin configuration for analysis
    pub fn get_spins(&self) -> &Vec<Vec<Vec<Vec<i8>>>> {
        &self.spins
    }
    
    /// Print a 3D slice of the 4D configuration at a given w-level
    pub fn print_3d_slice(&self, w_level: usize) {
        if w_level >= self.size {
            println!("Error: w_level {} exceeds lattice size {}", w_level, self.size);
            return;
        }
        
        println!("4D → 3D slice at w={} (showing {} z-slices):", w_level, self.size.min(4));
        
        let num_z_slices = self.size.min(4);
        let z_spacing = self.size.max(1) / num_z_slices.max(1);
        
        for z_slice in 0..num_z_slices {
            let z_level = z_slice * z_spacing;
            if z_level < self.size {
                println!("  Z-slice {} (w={}, z={}):", z_slice, w_level, z_level);
                for i in 0..self.size.min(8) {  // Limit display size
                    print!("    ");
                    for j in 0..self.size.min(8) {
                        print!("{:2}", if self.spins[i][j][z_level][w_level] == 1 { "↑" } else { "↓" });
                    }
                    println!();
                }
                println!();
            }
        }
    }
    
    /// Print a 2D slice through the 4D configuration
    pub fn print_2d_slice(&self, k_level: usize, l_level: usize) {
        if k_level >= self.size || l_level >= self.size {
            println!("Error: slice indices exceed lattice size");
            return;
        }
        
        println!("4D → 2D slice at (z={}, w={}):", k_level, l_level);
        for i in 0..self.size.min(12) {
            for j in 0..self.size.min(12) {
                print!("{:2}", if self.spins[i][j][k_level][l_level] == 1 { "↑" } else { "↓" });
            }
            println!();
        }
    }
    
    /// Calculate correlation function between spins at different separations
    pub fn correlation_function(&self, max_distance: usize) -> Vec<f64> {
        let mut correlations = vec![0.0; max_distance + 1];
        let mut counts = vec![0; max_distance + 1];
        
        // Sample a subset of spins for efficiency
        let sample_size = (self.size * self.size).min(1000);
        
        for _ in 0..sample_size {
            let i0 = self.rng.gen_range(0..self.size);
            let j0 = self.rng.gen_range(0..self.size);
            let k0 = self.rng.gen_range(0..self.size);
            let l0 = self.rng.gen_range(0..self.size);
            
            let spin0 = self.spins[i0][j0][k0][l0] as f64;
            
            for distance in 0..=max_distance {
                if distance < self.size {
                    let i1 = (i0 + distance) % self.size;
                    let spin1 = self.spins[i1][j0][k0][l0] as f64;
                    
                    correlations[distance] += spin0 * spin1;
                    counts[distance] += 1;
                }
            }
        }
        
        // Normalize
        for i in 0..correlations.len() {
            if counts[i] > 0 {
                correlations[i] /= counts[i] as f64;
            }
        }
        
        correlations
    }
}

/// Utility functions for analyzing Ising model results
pub mod analysis {
    
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
        -2.0  // Each spin has 4 aligned neighbors, E = -J * 4 / 2 = -2J per site
    }
    
    /// Theoretical energy per site at T=0 for 3D (all spins aligned)
    pub fn energy_per_site_at_zero_temp_3d() -> f64 {
        -3.0  // Each spin has 6 aligned neighbors, E = -J * 6 / 2 = -3J per site
    }
    
    /// Theoretical energy per site at T=0 for 4D (all spins aligned)
    pub fn energy_per_site_at_zero_temp_4d() -> f64 {
        -4.0  // Each spin has 8 aligned neighbors, E = -J * 8 / 2 = -4J per site
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
}
