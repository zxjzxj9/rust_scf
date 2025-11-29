use rand::prelude::*;
use rayon::prelude::*;

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
    pub(crate) fn get_spin(&self, i: i32, j: i32, k: i32, l: i32) -> i8 {
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

    // ===== PARALLEL MONTE CARLO METHODS =====

    /// Perform parallel ensemble sampling for 4D Ising model
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
                let mut model = IsingModel4D {
                    size,
                    spins: spins.clone(),
                    temperature,
                    coupling,
                    magnetic_field,
                    rng: thread_rng(),
                    step: 0,
                };
                
                // Equilibrate (shorter for 4D due to computational cost)
                for _ in 0..steps_per_run / 8 {
                    model.monte_carlo_step();
                }
                
                // Sample observables
                let mut energies = Vec::new();
                let mut magnetizations = Vec::new();
                
                for _ in 0..steps_per_run / 2 {  // Fewer steps for 4D
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

    /// Parallel Wolff cluster ensemble for 4D model
    pub fn parallel_wolff_ensemble(&self, num_clusters: usize) -> Vec<usize> {
        let size = self.size;
        let temperature = self.temperature;
        let coupling = self.coupling;
        let magnetic_field = self.magnetic_field;
        let spins = self.spins.clone();
        
        (0..num_clusters)
            .into_par_iter()
            .map(|_| {
                let mut model = IsingModel4D {
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

