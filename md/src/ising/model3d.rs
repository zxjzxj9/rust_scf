use rand::prelude::*;
use rayon::prelude::*;

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
    pub(crate) fn get_spin(&self, i: i32, j: i32, k: i32) -> i8 {
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

    // ===== PARALLEL MONTE CARLO METHODS =====

    /// Perform parallel ensemble sampling for 3D Ising model
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
                let mut model = IsingModel3D {
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

    /// Parallel calculation of total energy for 3D model
    pub fn parallel_total_energy(&self) -> f64 {
        let size = self.size;
        let spins = &self.spins;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        
        let energy: f64 = (0..size)
            .into_par_iter()
            .map(|i| {
                let mut plane_energy = 0.0;
                for j in 0..size {
                    for k in 0..size {
                        let spin = spins[i][j][k] as f64;
                        
                        // Only count positive direction neighbors to avoid double counting
                        let x_neighbor = spins[(i + 1) % size][j][k] as f64;
                        let y_neighbor = spins[i][(j + 1) % size][k] as f64;
                        let z_neighbor = spins[i][j][(k + 1) % size] as f64;
                        
                        plane_energy -= coupling * spin * (x_neighbor + y_neighbor + z_neighbor);
                        plane_energy -= magnetic_field * spin;
                    }
                }
                plane_energy
            })
            .sum();
        
        energy
    }

    /// Parallel Wolff cluster ensemble for 3D model
    pub fn parallel_wolff_ensemble(&self, num_clusters: usize) -> Vec<usize> {
        let size = self.size;
        let temperature = self.temperature;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        let spins = self.spins.clone();
        
        (0..num_clusters)
            .into_par_iter()
            .map(|_| {
                let mut model = IsingModel3D {
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

