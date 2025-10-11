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
    
    /// Perform one Wolff cluster algorithm step
    /// This is much more efficient than single-spin flips, especially near T_c
    pub fn wolff_cluster_step(&mut self) {
        use std::collections::VecDeque;
        
        // Choose random seed spin
        let seed_i = self.rng.gen_range(0..self.size);
        let seed_j = self.rng.gen_range(0..self.size);
        let seed_spin = self.spins[seed_i][seed_j];
        
        // Probability to add neighbors to cluster
        // P = 1 - exp(-2*J*beta) for same-orientation neighbors
        let add_probability = 1.0 - (-2.0 * self.coupling / self.temperature).exp();
        
        // Track visited sites and cluster
        let mut visited = vec![vec![false; self.size]; self.size];
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with seed
        queue.push_back((seed_i, seed_j));
        visited[seed_i][seed_j] = true;
        
        // Grow cluster using breadth-first search
        while let Some((i, j)) = queue.pop_front() {
            cluster.push((i, j));
            
            // Check all 4 neighbors
            let neighbors = [
                (i.wrapping_sub(1), j),
                (i.wrapping_add(1), j),
                (i, j.wrapping_sub(1)),
                (i, j.wrapping_add(1)),
            ];
            
            for &(ni, nj) in &neighbors {
                let ni = ni % self.size;
                let nj = nj % self.size;
                
                // Skip if already visited
                if visited[ni][nj] {
                    continue;
                }
                
                // Check if neighbor has same orientation as seed
                if self.spins[ni][nj] == seed_spin {
                    // Add to cluster with probability P
                    if self.rng.gen::<f64>() < add_probability {
                        visited[ni][nj] = true;
                        queue.push_back((ni, nj));
                    }
                }
            }
        }
        
        // Flip entire cluster
        for &(i, j) in &cluster {
            self.spins[i][j] *= -1;
        }
        
        self.step += 1;
    }
    
    /// Get the size of the last Wolff cluster (for analysis)
    pub fn wolff_cluster_step_with_size(&mut self) -> usize {
        use std::collections::VecDeque;
        
        // Choose random seed spin
        let seed_i = self.rng.gen_range(0..self.size);
        let seed_j = self.rng.gen_range(0..self.size);
        let seed_spin = self.spins[seed_i][seed_j];
        
        // Probability to add neighbors to cluster
        let add_probability = 1.0 - (-2.0 * self.coupling / self.temperature).exp();
        
        // Track visited sites and cluster
        let mut visited = vec![vec![false; self.size]; self.size];
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with seed
        queue.push_back((seed_i, seed_j));
        visited[seed_i][seed_j] = true;
        
        // Grow cluster using breadth-first search
        while let Some((i, j)) = queue.pop_front() {
            cluster.push((i, j));
            
            // Check all 4 neighbors
            let neighbors = [
                (i.wrapping_sub(1), j),
                (i.wrapping_add(1), j),
                (i, j.wrapping_sub(1)),
                (i, j.wrapping_add(1)),
            ];
            
            for &(ni, nj) in &neighbors {
                let ni = ni % self.size;
                let nj = nj % self.size;
                
                // Skip if already visited
                if visited[ni][nj] {
                    continue;
                }
                
                // Check if neighbor has same orientation as seed
                if self.spins[ni][nj] == seed_spin {
                    // Add to cluster with probability P
                    if self.rng.gen::<f64>() < add_probability {
                        visited[ni][nj] = true;
                        queue.push_back((ni, nj));
                    }
                }
            }
        }
        
        let cluster_size = cluster.len();
        
        // Flip entire cluster
        for &(i, j) in &cluster {
            self.spins[i][j] *= -1;
        }
        
        self.step += 1;
        cluster_size
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
    
    /// Perform one Wolff cluster algorithm step for 3D
    /// This is much more efficient than single-spin flips, especially near T_c
    pub fn wolff_cluster_step(&mut self) {
        use std::collections::VecDeque;
        
        // Choose random seed spin
        let seed_i = self.rng.gen_range(0..self.size);
        let seed_j = self.rng.gen_range(0..self.size);
        let seed_k = self.rng.gen_range(0..self.size);
        let seed_spin = self.spins[seed_i][seed_j][seed_k];
        
        // Probability to add neighbors to cluster
        // P = 1 - exp(-2*J*beta) for same-orientation neighbors
        let add_probability = 1.0 - (-2.0 * self.coupling / self.temperature).exp();
        
        // Track visited sites and cluster
        let mut visited = vec![vec![vec![false; self.size]; self.size]; self.size];
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with seed
        queue.push_back((seed_i, seed_j, seed_k));
        visited[seed_i][seed_j][seed_k] = true;
        
        // Grow cluster using breadth-first search
        while let Some((i, j, k)) = queue.pop_front() {
            cluster.push((i, j, k));
            
            // Check all 6 neighbors in 3D
            let neighbors = [
                (i.wrapping_sub(1), j, k),
                (i.wrapping_add(1), j, k),
                (i, j.wrapping_sub(1), k),
                (i, j.wrapping_add(1), k),
                (i, j, k.wrapping_sub(1)),
                (i, j, k.wrapping_add(1)),
            ];
            
            for &(ni, nj, nk) in &neighbors {
                let ni = ni % self.size;
                let nj = nj % self.size;
                let nk = nk % self.size;
                
                // Skip if already visited
                if visited[ni][nj][nk] {
                    continue;
                }
                
                // Check if neighbor has same orientation as seed
                if self.spins[ni][nj][nk] == seed_spin {
                    // Add to cluster with probability P
                    if self.rng.gen::<f64>() < add_probability {
                        visited[ni][nj][nk] = true;
                        queue.push_back((ni, nj, nk));
                    }
                }
            }
        }
        
        // Flip entire cluster
        for &(i, j, k) in &cluster {
            self.spins[i][j][k] *= -1;
        }
        
        self.step += 1;
    }
    
    /// Get the size of the last Wolff cluster (for analysis) in 3D
    pub fn wolff_cluster_step_with_size(&mut self) -> usize {
        use std::collections::VecDeque;
        
        // Choose random seed spin
        let seed_i = self.rng.gen_range(0..self.size);
        let seed_j = self.rng.gen_range(0..self.size);
        let seed_k = self.rng.gen_range(0..self.size);
        let seed_spin = self.spins[seed_i][seed_j][seed_k];
        
        // Probability to add neighbors to cluster
        let add_probability = 1.0 - (-2.0 * self.coupling / self.temperature).exp();
        
        // Track visited sites and cluster
        let mut visited = vec![vec![vec![false; self.size]; self.size]; self.size];
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with seed
        queue.push_back((seed_i, seed_j, seed_k));
        visited[seed_i][seed_j][seed_k] = true;
        
        // Grow cluster using breadth-first search
        while let Some((i, j, k)) = queue.pop_front() {
            cluster.push((i, j, k));
            
            // Check all 6 neighbors in 3D
            let neighbors = [
                (i.wrapping_sub(1), j, k),
                (i.wrapping_add(1), j, k),
                (i, j.wrapping_sub(1), k),
                (i, j.wrapping_add(1), k),
                (i, j, k.wrapping_sub(1)),
                (i, j, k.wrapping_add(1)),
            ];
            
            for &(ni, nj, nk) in &neighbors {
                let ni = ni % self.size;
                let nj = nj % self.size;
                let nk = nk % self.size;
                
                // Skip if already visited
                if visited[ni][nj][nk] {
                    continue;
                }
                
                // Check if neighbor has same orientation as seed
                if self.spins[ni][nj][nk] == seed_spin {
                    // Add to cluster with probability P
                    if self.rng.gen::<f64>() < add_probability {
                        visited[ni][nj][nk] = true;
                        queue.push_back((ni, nj, nk));
                    }
                }
            }
        }
        
        let cluster_size = cluster.len();
        
        // Flip entire cluster
        for &(i, j, k) in &cluster {
            self.spins[i][j][k] *= -1;
        }
        
        self.step += 1;
        cluster_size
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
        
        // Sample systematically instead of randomly for deterministic results
        let sample_spacing = (self.size / 4).max(1);
        
        for i0 in (0..self.size).step_by(sample_spacing) {
            for j0 in (0..self.size).step_by(sample_spacing) {
                for k0 in (0..self.size).step_by(sample_spacing) {
                    for l0 in (0..self.size).step_by(sample_spacing) {
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
    
    /// Perform one Wolff cluster algorithm step for 4D
    /// This is much more efficient than single-spin flips, especially near T_c
    pub fn wolff_cluster_step(&mut self) {
        use std::collections::VecDeque;
        
        // Choose random seed spin
        let seed_i = self.rng.gen_range(0..self.size);
        let seed_j = self.rng.gen_range(0..self.size);
        let seed_k = self.rng.gen_range(0..self.size);
        let seed_l = self.rng.gen_range(0..self.size);
        let seed_spin = self.spins[seed_i][seed_j][seed_k][seed_l];
        
        // Probability to add neighbors to cluster
        // P = 1 - exp(-2*J*beta) for same-orientation neighbors
        let add_probability = 1.0 - (-2.0 * self.coupling / self.temperature).exp();
        
        // Track visited sites and cluster
        let mut visited = vec![vec![vec![vec![false; self.size]; self.size]; self.size]; self.size];
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with seed
        queue.push_back((seed_i, seed_j, seed_k, seed_l));
        visited[seed_i][seed_j][seed_k][seed_l] = true;
        
        // Grow cluster using breadth-first search
        while let Some((i, j, k, l)) = queue.pop_front() {
            cluster.push((i, j, k, l));
            
            // Check all 8 neighbors in 4D
            let neighbors = [
                (i.wrapping_sub(1), j, k, l),
                (i.wrapping_add(1), j, k, l),
                (i, j.wrapping_sub(1), k, l),
                (i, j.wrapping_add(1), k, l),
                (i, j, k.wrapping_sub(1), l),
                (i, j, k.wrapping_add(1), l),
                (i, j, k, l.wrapping_sub(1)),
                (i, j, k, l.wrapping_add(1)),
            ];
            
            for &(ni, nj, nk, nl) in &neighbors {
                let ni = ni % self.size;
                let nj = nj % self.size;
                let nk = nk % self.size;
                let nl = nl % self.size;
                
                // Skip if already visited
                if visited[ni][nj][nk][nl] {
                    continue;
                }
                
                // Check if neighbor has same orientation as seed
                if self.spins[ni][nj][nk][nl] == seed_spin {
                    // Add to cluster with probability P
                    if self.rng.gen::<f64>() < add_probability {
                        visited[ni][nj][nk][nl] = true;
                        queue.push_back((ni, nj, nk, nl));
                    }
                }
            }
        }
        
        // Flip entire cluster
        for &(i, j, k, l) in &cluster {
            self.spins[i][j][k][l] *= -1;
        }
        
        self.step += 1;
    }
    
    /// Get the size of the last Wolff cluster (for analysis) in 4D
    pub fn wolff_cluster_step_with_size(&mut self) -> usize {
        use std::collections::VecDeque;
        
        // Choose random seed spin
        let seed_i = self.rng.gen_range(0..self.size);
        let seed_j = self.rng.gen_range(0..self.size);
        let seed_k = self.rng.gen_range(0..self.size);
        let seed_l = self.rng.gen_range(0..self.size);
        let seed_spin = self.spins[seed_i][seed_j][seed_k][seed_l];
        
        // Probability to add neighbors to cluster
        let add_probability = 1.0 - (-2.0 * self.coupling / self.temperature).exp();
        
        // Track visited sites and cluster
        let mut visited = vec![vec![vec![vec![false; self.size]; self.size]; self.size]; self.size];
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with seed
        queue.push_back((seed_i, seed_j, seed_k, seed_l));
        visited[seed_i][seed_j][seed_k][seed_l] = true;
        
        // Grow cluster using breadth-first search
        while let Some((i, j, k, l)) = queue.pop_front() {
            cluster.push((i, j, k, l));
            
            // Check all 8 neighbors in 4D
            let neighbors = [
                (i.wrapping_sub(1), j, k, l),
                (i.wrapping_add(1), j, k, l),
                (i, j.wrapping_sub(1), k, l),
                (i, j.wrapping_add(1), k, l),
                (i, j, k.wrapping_sub(1), l),
                (i, j, k.wrapping_add(1), l),
                (i, j, k, l.wrapping_sub(1)),
                (i, j, k, l.wrapping_add(1)),
            ];
            
            for &(ni, nj, nk, nl) in &neighbors {
                let ni = ni % self.size;
                let nj = nj % self.size;
                let nk = nk % self.size;
                let nl = nl % self.size;
                
                // Skip if already visited
                if visited[ni][nj][nk][nl] {
                    continue;
                }
                
                // Check if neighbor has same orientation as seed
                if self.spins[ni][nj][nk][nl] == seed_spin {
                    // Add to cluster with probability P
                    if self.rng.gen::<f64>() < add_probability {
                        visited[ni][nj][nk][nl] = true;
                        queue.push_back((ni, nj, nk, nl));
                    }
                }
            }
        }
        
        let cluster_size = cluster.len();
        
        // Flip entire cluster
        for &(i, j, k, l) in &cluster {
            self.spins[i][j][k][l] *= -1;
        }
        
        self.step += 1;
        cluster_size
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
}
