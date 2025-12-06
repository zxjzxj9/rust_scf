// file: `md/src/gcmc.rs`
use crate::lj_pot::LennardJones;
use nalgebra::{Matrix3, Vector3};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Grand Canonical Monte Carlo (GCMC) simulator for Lennard-Jones particles
///
/// In the grand canonical ensemble (μVT), the following are constant:
/// - Chemical potential (μ)
/// - Volume (V)
/// - Temperature (T)
///
/// The number of particles (N) fluctuates according to the chemical potential.
///
/// GCMC uses three types of Monte Carlo moves:
/// 1. Displacement: Move a particle to a new position (Metropolis criterion)
/// 2. Insertion: Add a particle at a random position
/// 3. Deletion: Remove a randomly chosen particle
#[derive(Debug)]
pub struct GCMC {
    /// Lennard-Jones potential with PBC support
    pub lj: LennardJones,
    /// Current particle positions
    pub positions: Vec<Vector3<f64>>,
    /// Temperature (in reduced units where k_B = 1)
    pub temperature: f64,
    /// Chemical potential (in reduced units)
    pub chemical_potential: f64,
    /// Random number generator
    rng: StdRng,
    /// Step counter
    pub step: u64,
    /// Statistics
    pub stats: GCMCStatistics,
    /// Maximum displacement for particle moves
    pub max_displacement: f64,
    /// Move probabilities: [displacement, insertion, deletion]
    pub move_probabilities: [f64; 3],
}

impl Clone for GCMC {
    fn clone(&self) -> Self {
        Self {
            lj: self.lj.clone(),
            positions: self.positions.clone(),
            temperature: self.temperature,
            chemical_potential: self.chemical_potential,
            rng: StdRng::from_entropy(), // Create new RNG for the clone
            step: self.step,
            stats: self.stats.clone(),
            max_displacement: self.max_displacement,
            move_probabilities: self.move_probabilities,
        }
    }
}

/// Statistics for tracking GCMC simulation
#[derive(Debug, Clone)]
pub struct GCMCStatistics {
    /// Number of displacement moves attempted
    pub displacement_attempts: u64,
    /// Number of displacement moves accepted
    pub displacement_accepted: u64,
    /// Number of insertion moves attempted
    pub insertion_attempts: u64,
    /// Number of insertion moves accepted
    pub insertion_accepted: u64,
    /// Number of deletion moves attempted
    pub deletion_attempts: u64,
    /// Number of deletion moves accepted
    pub deletion_accepted: u64,
    /// Running average of number of particles
    pub avg_n_particles: f64,
    /// Running average of energy
    pub avg_energy: f64,
    /// Number of samples for averages
    pub n_samples: u64,
}

impl GCMCStatistics {
    pub fn new() -> Self {
        Self {
            displacement_attempts: 0,
            displacement_accepted: 0,
            insertion_attempts: 0,
            insertion_accepted: 0,
            deletion_attempts: 0,
            deletion_accepted: 0,
            avg_n_particles: 0.0,
            avg_energy: 0.0,
            n_samples: 0,
        }
    }

    /// Get displacement acceptance rate
    pub fn displacement_acceptance_rate(&self) -> f64 {
        if self.displacement_attempts == 0 {
            0.0
        } else {
            self.displacement_accepted as f64 / self.displacement_attempts as f64
        }
    }

    /// Get insertion acceptance rate
    pub fn insertion_acceptance_rate(&self) -> f64 {
        if self.insertion_attempts == 0 {
            0.0
        } else {
            self.insertion_accepted as f64 / self.insertion_attempts as f64
        }
    }

    /// Get deletion acceptance rate
    pub fn deletion_acceptance_rate(&self) -> f64 {
        if self.deletion_attempts == 0 {
            0.0
        } else {
            self.deletion_accepted as f64 / self.deletion_attempts as f64
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Print summary statistics
    pub fn print_summary(&self) {
        println!("\n=== GCMC Statistics ===");
        println!(
            "Displacement moves: {} / {} ({:.2}% accepted)",
            self.displacement_accepted,
            self.displacement_attempts,
            100.0 * self.displacement_acceptance_rate()
        );
        println!(
            "Insertion moves:    {} / {} ({:.2}% accepted)",
            self.insertion_accepted,
            self.insertion_attempts,
            100.0 * self.insertion_acceptance_rate()
        );
        println!(
            "Deletion moves:     {} / {} ({:.2}% accepted)",
            self.deletion_accepted,
            self.deletion_attempts,
            100.0 * self.deletion_acceptance_rate()
        );
        println!("Average N particles: {:.2}", self.avg_n_particles);
        println!("Average energy:      {:.6}", self.avg_energy);
        println!("Number of samples:   {}", self.n_samples);
    }
}

impl GCMC {
    /// Create a new GCMC simulation in a cubic box
    ///
    /// # Arguments
    /// * `epsilon` - LJ energy parameter
    /// * `sigma` - LJ length parameter
    /// * `box_length` - Side length of cubic box
    /// * `temperature` - Temperature (in reduced units, k_B = 1)
    /// * `chemical_potential` - Chemical potential (in reduced units)
    ///
    /// # Returns
    /// A new GCMC simulator with no initial particles
    pub fn new(
        epsilon: f64,
        sigma: f64,
        box_length: f64,
        temperature: f64,
        chemical_potential: f64,
    ) -> Self {
        let box_lengths = Vector3::new(box_length, box_length, box_length);
        let lj = LennardJones::new(epsilon, sigma, box_lengths);

        Self {
            lj,
            positions: Vec::new(),
            temperature,
            chemical_potential,
            rng: StdRng::from_entropy(),
            step: 0,
            stats: GCMCStatistics::new(),
            max_displacement: 0.5 * sigma,
            move_probabilities: [0.5, 0.25, 0.25], // displacement, insertion, deletion
        }
    }

    /// Create a new GCMC simulation with a non-cubic lattice
    ///
    /// # Arguments
    /// * `epsilon` - LJ energy parameter
    /// * `sigma` - LJ length parameter
    /// * `lattice` - 3x3 matrix defining the simulation box
    /// * `temperature` - Temperature (in reduced units, k_B = 1)
    /// * `chemical_potential` - Chemical potential (in reduced units)
    pub fn from_lattice(
        epsilon: f64,
        sigma: f64,
        lattice: Matrix3<f64>,
        temperature: f64,
        chemical_potential: f64,
    ) -> Self {
        let lj = LennardJones::from_lattice(epsilon, sigma, lattice);

        Self {
            lj,
            positions: Vec::new(),
            temperature,
            chemical_potential,
            rng: StdRng::from_entropy(),
            step: 0,
            stats: GCMCStatistics::new(),
            max_displacement: 0.5 * sigma,
            move_probabilities: [0.5, 0.25, 0.25],
        }
    }

    /// Initialize with random particles at a target density
    ///
    /// # Arguments
    /// * `target_density` - Desired number density (particles per unit volume)
    pub fn initialize_random(&mut self, target_density: f64) {
        let volume = self.lj.volume();
        let n_particles = (target_density * volume) as usize;

        self.positions.clear();
        for _ in 0..n_particles {
            let pos = self.generate_random_position();
            self.positions.push(pos);
        }
    }

    /// Generate a random position within the simulation box
    fn generate_random_position(&mut self) -> Vector3<f64> {
        // Generate fractional coordinates [0, 1) and convert to Cartesian
        let frac = Vector3::new(
            self.rng.gen::<f64>(),
            self.rng.gen::<f64>(),
            self.rng.gen::<f64>(),
        );
        self.lj.lattice * frac
    }

    /// Compute the potential energy of a single particle with all others
    ///
    /// # Arguments
    /// * `pos` - Position of the test particle
    /// * `skip_index` - Optional index to skip (for displacement moves)
    fn compute_particle_energy(&self, pos: Vector3<f64>, skip_index: Option<usize>) -> f64 {
        let sigma2 = self.lj.sigma * self.lj.sigma;
        let r_cut = 2.5 * self.lj.sigma;
        let r_cut2 = r_cut * r_cut;
        let min_r2 = 0.01 * sigma2;

        let mut energy = 0.0;
        for (i, &other_pos) in self.positions.iter().enumerate() {
            // Skip if this is the particle we're moving
            if let Some(skip) = skip_index {
                if i == skip {
                    continue;
                }
            }

            let rij = self.lj.minimum_image(pos - other_pos);
            let mut r2 = rij.norm_squared();

            if r2 > r_cut2 {
                continue;
            }
            if r2 < min_r2 {
                r2 = min_r2;
            }

            energy += self.lj.lj_potential(r2);
        }

        energy
    }

    /// Attempt a displacement move (Metropolis algorithm)
    fn attempt_displacement(&mut self) -> bool {
        if self.positions.is_empty() {
            return false;
        }

        self.stats.displacement_attempts += 1;

        // Select random particle
        let idx = self.rng.gen_range(0..self.positions.len());
        let old_pos = self.positions[idx];

        // Compute old energy
        let old_energy = self.compute_particle_energy(old_pos, Some(idx));

        // Generate new position by random displacement
        let displacement = Vector3::new(
            self.max_displacement * (2.0 * self.rng.gen::<f64>() - 1.0),
            self.max_displacement * (2.0 * self.rng.gen::<f64>() - 1.0),
            self.max_displacement * (2.0 * self.rng.gen::<f64>() - 1.0),
        );
        let new_pos = old_pos + displacement;

        // Apply PBC to new position
        let new_pos = self.wrap_position(new_pos);

        // Compute new energy
        let new_energy = self.compute_particle_energy(new_pos, Some(idx));

        // Metropolis acceptance criterion
        let delta_e = new_energy - old_energy;
        let accept = if delta_e <= 0.0 {
            true
        } else {
            self.rng.gen::<f64>() < (-delta_e / self.temperature).exp()
        };

        if accept {
            self.positions[idx] = new_pos;
            self.stats.displacement_accepted += 1;
        }

        accept
    }

    /// Attempt an insertion move
    fn attempt_insertion(&mut self) -> bool {
        self.stats.insertion_attempts += 1;

        // Generate random position
        let new_pos = self.generate_random_position();

        // Compute energy of new particle
        let energy = self.compute_particle_energy(new_pos, None);

        // GCMC insertion acceptance criterion
        // acc = min(1, (V/(N+1)) * exp(-β*ΔU + β*μ))
        let n = self.positions.len();
        let volume = self.lj.volume();
        let beta = 1.0 / self.temperature;

        let log_acc =
            (volume / (n as f64 + 1.0)).ln() - beta * energy + beta * self.chemical_potential;

        let accept = if log_acc >= 0.0 {
            true
        } else {
            self.rng.gen::<f64>() < log_acc.exp()
        };

        if accept {
            self.positions.push(new_pos);
            self.stats.insertion_accepted += 1;
        }

        accept
    }

    /// Attempt a deletion move
    fn attempt_deletion(&mut self) -> bool {
        if self.positions.is_empty() {
            return false;
        }

        self.stats.deletion_attempts += 1;

        // Select random particle to delete
        let idx = self.rng.gen_range(0..self.positions.len());
        let pos = self.positions[idx];

        // Compute energy of particle to be deleted
        let energy = self.compute_particle_energy(pos, Some(idx));

        // GCMC deletion acceptance criterion
        // acc = min(1, (N/V) * exp(-β*ΔU - β*μ))
        let n = self.positions.len();
        let volume = self.lj.volume();
        let beta = 1.0 / self.temperature;

        let log_acc = (n as f64 / volume).ln() + beta * energy - beta * self.chemical_potential;

        let accept = if log_acc >= 0.0 {
            true
        } else {
            self.rng.gen::<f64>() < log_acc.exp()
        };

        if accept {
            self.positions.swap_remove(idx);
            self.stats.deletion_accepted += 1;
        }

        accept
    }

    /// Wrap position to stay within the simulation box (fractional coordinates [0,1))
    fn wrap_position(&self, pos: Vector3<f64>) -> Vector3<f64> {
        // Convert to fractional coordinates
        let frac = self.lj.lattice_inv * pos;

        // Wrap to [0, 1)
        let wrapped = Vector3::new(
            frac.x - frac.x.floor(),
            frac.y - frac.y.floor(),
            frac.z - frac.z.floor(),
        );

        // Convert back to Cartesian
        self.lj.lattice * wrapped
    }

    /// Perform one Monte Carlo step (randomly choose move type)
    pub fn monte_carlo_step(&mut self) {
        // Choose move type based on probabilities
        let r = self.rng.gen::<f64>();

        if r < self.move_probabilities[0] {
            // Displacement
            self.attempt_displacement();
        } else if r < self.move_probabilities[0] + self.move_probabilities[1] {
            // Insertion
            self.attempt_insertion();
        } else {
            // Deletion
            self.attempt_deletion();
        }

        self.step += 1;
    }

    /// Perform multiple Monte Carlo steps
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.monte_carlo_step();
        }
    }

    /// Sample current state for statistics
    pub fn sample(&mut self) {
        let n = self.positions.len() as f64;
        let energy = if self.positions.is_empty() {
            0.0
        } else {
            self.lj.compute_potential_energy(&self.positions)
        };

        // Update running averages
        let count = self.stats.n_samples as f64;
        self.stats.avg_n_particles = (self.stats.avg_n_particles * count + n) / (count + 1.0);
        self.stats.avg_energy = (self.stats.avg_energy * count + energy) / (count + 1.0);
        self.stats.n_samples += 1;
    }

    /// Get current number of particles
    pub fn n_particles(&self) -> usize {
        self.positions.len()
    }

    /// Get current number density
    pub fn density(&self) -> f64 {
        self.positions.len() as f64 / self.lj.volume()
    }

    /// Get current potential energy
    pub fn potential_energy(&self) -> f64 {
        if self.positions.is_empty() {
            0.0
        } else {
            self.lj.compute_potential_energy(&self.positions)
        }
    }

    /// Get current potential energy per particle
    pub fn potential_energy_per_particle(&self) -> f64 {
        if self.positions.is_empty() {
            0.0
        } else {
            self.potential_energy() / self.positions.len() as f64
        }
    }

    /// Set move probabilities (must sum to 1.0)
    ///
    /// # Arguments
    /// * `displacement` - Probability of displacement move
    /// * `insertion` - Probability of insertion move
    /// * `deletion` - Probability of deletion move
    pub fn set_move_probabilities(&mut self, displacement: f64, insertion: f64, deletion: f64) {
        let sum = displacement + insertion + deletion;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Move probabilities must sum to 1.0"
        );
        self.move_probabilities = [displacement, insertion, deletion];
    }

    /// Set maximum displacement for particle moves
    pub fn set_max_displacement(&mut self, max_disp: f64) {
        self.max_displacement = max_disp;
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Set chemical potential
    pub fn set_chemical_potential(&mut self, mu: f64) {
        self.chemical_potential = mu;
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats.reset();
    }

    /// Get a copy of the positions
    pub fn get_positions(&self) -> &Vec<Vector3<f64>> {
        &self.positions
    }
}

/// Parallel GCMC sampling for computing averages at multiple chemical potentials
pub fn parallel_gcmc_sweep(
    gcmc_configs: &[GCMC],
    equilibration_steps: usize,
    production_steps: usize,
    sample_interval: usize,
) -> Vec<GCMCResults> {
    gcmc_configs
        .par_iter()
        .map(|gcmc_base| {
            let mut gcmc = gcmc_base.clone();

            // Equilibration
            gcmc.run(equilibration_steps);

            // Production with sampling
            let mut n_particles_samples = Vec::new();
            let mut energy_samples = Vec::new();
            let mut density_samples = Vec::new();

            for step in 0..production_steps {
                gcmc.monte_carlo_step();

                if step % sample_interval == 0 {
                    n_particles_samples.push(gcmc.n_particles() as f64);
                    energy_samples.push(gcmc.potential_energy());
                    density_samples.push(gcmc.density());
                    gcmc.sample();
                }
            }

            GCMCResults {
                chemical_potential: gcmc.chemical_potential,
                temperature: gcmc.temperature,
                avg_n_particles: gcmc.stats.avg_n_particles,
                avg_density: gcmc.stats.avg_n_particles / gcmc.lj.volume(),
                avg_energy: gcmc.stats.avg_energy,
                avg_energy_per_particle: if gcmc.stats.avg_n_particles > 0.0 {
                    gcmc.stats.avg_energy / gcmc.stats.avg_n_particles
                } else {
                    0.0
                },
                displacement_acceptance: gcmc.stats.displacement_acceptance_rate(),
                insertion_acceptance: gcmc.stats.insertion_acceptance_rate(),
                deletion_acceptance: gcmc.stats.deletion_acceptance_rate(),
                n_particles_samples,
                energy_samples,
                density_samples,
            }
        })
        .collect()
}

/// Results from a GCMC simulation
#[derive(Debug, Clone)]
pub struct GCMCResults {
    pub chemical_potential: f64,
    pub temperature: f64,
    pub avg_n_particles: f64,
    pub avg_density: f64,
    pub avg_energy: f64,
    pub avg_energy_per_particle: f64,
    pub displacement_acceptance: f64,
    pub insertion_acceptance: f64,
    pub deletion_acceptance: f64,
    pub n_particles_samples: Vec<f64>,
    pub energy_samples: Vec<f64>,
    pub density_samples: Vec<f64>,
}

impl GCMCResults {
    /// Calculate standard deviation of density
    pub fn density_std(&self) -> f64 {
        if self.density_samples.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_density;
        let variance: f64 = self
            .density_samples
            .iter()
            .map(|&rho| (rho - mean).powi(2))
            .sum::<f64>()
            / (self.density_samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate standard deviation of particle number
    pub fn n_particles_std(&self) -> f64 {
        if self.n_particles_samples.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_n_particles;
        let variance: f64 = self
            .n_particles_samples
            .iter()
            .map(|&n| (n - mean).powi(2))
            .sum::<f64>()
            / (self.n_particles_samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== GCMC Results ===");
        println!("Chemical potential: {:.6}", self.chemical_potential);
        println!("Temperature:        {:.6}", self.temperature);
        println!(
            "Avg N particles:    {:.2} ± {:.2}",
            self.avg_n_particles,
            self.n_particles_std()
        );
        println!(
            "Avg density:        {:.6} ± {:.6}",
            self.avg_density,
            self.density_std()
        );
        println!("Avg energy:         {:.6}", self.avg_energy);
        println!("Avg energy/particle:{:.6}", self.avg_energy_per_particle);
        println!("Acceptance rates:");
        println!(
            "  Displacement: {:.2}%",
            100.0 * self.displacement_acceptance
        );
        println!("  Insertion:    {:.2}%", 100.0 * self.insertion_acceptance);
        println!("  Deletion:     {:.2}%", 100.0 * self.deletion_acceptance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gcmc_creation() {
        let gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -5.0);
        assert_eq!(gcmc.positions.len(), 0);
        assert_eq!(gcmc.temperature, 1.0);
        assert_eq!(gcmc.chemical_potential, -5.0);
    }

    #[test]
    fn test_gcmc_initialization() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -5.0);
        gcmc.initialize_random(0.5);

        let expected_n = (0.5 * 1000.0) as usize;
        assert!((gcmc.n_particles() as i32 - expected_n as i32).abs() < 50);
    }

    #[test]
    fn test_gcmc_insertion_deletion() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, 5.0); // High μ favors insertion

        // Should tend to insert particles with high chemical potential
        for _ in 0..100 {
            gcmc.attempt_insertion();
        }

        assert!(gcmc.n_particles() > 0);
    }

    #[test]
    fn test_gcmc_displacement() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -5.0);
        gcmc.initialize_random(0.1);

        let initial_positions = gcmc.positions.clone();

        for _ in 0..100 {
            gcmc.attempt_displacement();
        }

        // At least some particles should have moved
        let mut moved_count = 0;
        for i in 0..gcmc.positions.len() {
            if (gcmc.positions[i] - initial_positions[i]).norm() > 1e-10 {
                moved_count += 1;
            }
        }

        assert!(moved_count > 0);
    }

    #[test]
    fn test_gcmc_statistics() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -2.0);
        gcmc.initialize_random(0.5);

        gcmc.run(1000);

        assert!(gcmc.stats.displacement_attempts > 0);
        assert!(gcmc.stats.displacement_acceptance_rate() >= 0.0);
        assert!(gcmc.stats.displacement_acceptance_rate() <= 1.0);
    }

    #[test]
    fn test_gcmc_sampling() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -3.0);
        gcmc.initialize_random(0.5);

        for _ in 0..10 {
            gcmc.monte_carlo_step();
            gcmc.sample();
        }

        assert_eq!(gcmc.stats.n_samples, 10);
        assert!(gcmc.stats.avg_n_particles > 0.0);
    }

    #[test]
    fn test_wrap_position() {
        let gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -5.0);

        // Position outside box
        let pos = Vector3::new(12.0, -3.0, 15.0);
        let wrapped = gcmc.wrap_position(pos);

        // Should be wrapped to [0, 10) in all dimensions
        assert!(wrapped.x >= 0.0 && wrapped.x < 10.0);
        assert!(wrapped.y >= 0.0 && wrapped.y < 10.0);
        assert!(wrapped.z >= 0.0 && wrapped.z < 10.0);
    }

    #[test]
    fn test_density_calculation() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -5.0);
        gcmc.initialize_random(0.5);

        let density = gcmc.density();
        assert!((density - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_move_probabilities() {
        let mut gcmc = GCMC::new(1.0, 1.0, 10.0, 1.0, -5.0);
        gcmc.set_move_probabilities(0.6, 0.2, 0.2);

        assert_relative_eq!(gcmc.move_probabilities[0], 0.6, epsilon = 1e-10);
        assert_relative_eq!(gcmc.move_probabilities[1], 0.2, epsilon = 1e-10);
        assert_relative_eq!(gcmc.move_probabilities[2], 0.2, epsilon = 1e-10);
    }
}
