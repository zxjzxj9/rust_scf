use rand::prelude::*;
use rayon::prelude::*;

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
    pub(crate) fn get_spin(&self, i: i32, j: i32) -> i8 {
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

    // ===== PARALLEL MONTE CARLO METHODS =====

    /// Perform multiple Monte Carlo steps in parallel
    /// This parallelizes the execution of multiple independent MC steps
    pub fn parallel_monte_carlo_steps(&mut self, num_steps: usize) {
        // Create thread-safe models by initializing new RNGs
        let spins = &self.spins;
        let size = self.size;
        let temperature = self.temperature;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        
        let results: Vec<Vec<Vec<i8>>> = (0..num_steps)
            .into_par_iter()
            .map(|_| {
                let mut model = IsingModel2D {
                    size,
                    spins: spins.clone(),
                    temperature,
                    coupling,
                    magnetic_field,
                    rng: thread_rng(),
                    step: 0,
                };
                model.monte_carlo_step();
                model.spins
            })
            .collect();

        // Use result from a random thread to avoid bias
        let final_spins = &results[thread_rng().gen_range(0..results.len())];
        self.spins = final_spins.clone();
        self.step += num_steps as u64;
    }

    /// Perform parallel ensemble sampling: run multiple independent simulations
    /// Returns statistics (energy, magnetization) from all runs
    pub fn parallel_ensemble_sampling(
        &self,
        num_runs: usize,
        steps_per_run: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let size = self.size;
        let temperature = self.temperature;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        let spins = self.spins.clone();
        
        let results: Vec<(f64, f64)> = (0..num_runs)
            .into_par_iter()
            .map(|_| {
                let mut model = IsingModel2D {
                    size,
                    spins: spins.clone(),
                    temperature,
                    coupling,
                    magnetic_field,
                    rng: thread_rng(),
                    step: 0,
                };
                
                // Equilibrate
                for _ in 0..steps_per_run / 4 {
                    model.monte_carlo_step();
                }
                
                // Sample observables
                let mut energies = Vec::new();
                let mut magnetizations = Vec::new();
                
                for _ in 0..steps_per_run {
                    model.monte_carlo_step();
                    energies.push(model.energy_per_site());
                    magnetizations.push(model.abs_magnetization_per_site());
                }
                
                let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
                let mean_mag = magnetizations.iter().sum::<f64>() / magnetizations.len() as f64;
                
                (mean_energy, mean_mag)
            })
            .collect();

        let energies = results.iter().map(|(e, _)| *e).collect();
        let magnetizations = results.iter().map(|(_, m)| *m).collect();
        
        (energies, magnetizations)
    }

    /// Parallel calculation of total energy using Rayon
    /// More efficient for large systems
    pub fn parallel_total_energy(&self) -> f64 {
        let size = self.size;
        let spins = &self.spins;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        
        let energy: f64 = (0..size)
            .into_par_iter()
            .map(|i| {
                let mut row_energy = 0.0;
                for j in 0..size {
                    let spin = spins[i][j] as f64;
                    
                    // Only count right and down neighbors to avoid double counting
                    let right_neighbor = spins[i][(j + 1) % size] as f64;
                    let down_neighbor = spins[(i + 1) % size][j] as f64;
                    
                    row_energy -= coupling * spin * (right_neighbor + down_neighbor);
                    row_energy -= magnetic_field * spin;
                }
                row_energy
            })
            .sum();
        
        energy
    }

    /// Parallel calculation of magnetization components
    pub fn parallel_magnetization_stats(&self) -> (f64, f64, f64) {
        let size = self.size;
        let spins = &self.spins;
        
        let (total_mag, total_abs_mag, total_squared_mag): (f64, f64, f64) = (0..size)
            .into_par_iter()
            .map(|i| {
                let mut row_mag = 0.0;
                let mut row_abs_mag = 0.0;
                let mut row_sq_mag = 0.0;
                
                for j in 0..size {
                    let spin = spins[i][j] as f64;
                    row_mag += spin;
                    row_abs_mag += spin.abs();
                    row_sq_mag += spin * spin;
                }
                
                (row_mag, row_abs_mag, row_sq_mag)
            })
            .reduce(
                || (0.0, 0.0, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            );

        let total_sites = (size * size) as f64;
        (
            total_mag / total_sites,
            total_abs_mag / total_sites,
            total_squared_mag / total_sites,
        )
    }

    /// Parallel Wolff cluster steps for improved sampling efficiency
    pub fn parallel_wolff_ensemble(&self, num_clusters: usize) -> Vec<usize> {
        let size = self.size;
        let temperature = self.temperature;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        let spins = self.spins.clone();
        
        (0..num_clusters)
            .into_par_iter()
            .map(|_| {
                let mut model = IsingModel2D {
                    size,
                    spins: spins.clone(),
                    temperature,
                    coupling,
                    magnetic_field,
                    rng: thread_rng(),
                    step: 0,
                };
                model.wolff_cluster_step_with_size()
            })
            .collect()
    }
}

