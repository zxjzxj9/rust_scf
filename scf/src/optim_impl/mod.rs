//! Geometry optimization algorithms
//!
//! This module contains various optimization algorithms for molecular geometry
//! optimization, including conjugate gradient and steepest descent methods.

mod cg;
mod steepest_descent;

pub use cg::CGOptimizer;
pub use steepest_descent::SteepestDescentOptimizer;

use crate::scf_impl::SCF;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::str::FromStr;

enum OptimizationAlgorithm {
    ConjugateGradient,
    SteepestDescent,
}

impl FromStr for OptimizationAlgorithm {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cg" => Ok(Self::ConjugateGradient),
            "sd" => Ok(Self::SteepestDescent),
            _ => Err(color_eyre::eyre::eyre!("Unknown algorithm: {}", s)),
        }
    }
}

/// Create an optimizer based on algorithm choice
pub fn create_optimizer<S: SCF + Clone + 'static>(
    algorithm: &str,
    scf: &mut S,
    max_iterations: usize,
    convergence_threshold: f64,
) -> Result<Box<dyn GeometryOptimizer<SCFType = S>>, color_eyre::eyre::Error> {
    let algo = OptimizationAlgorithm::from_str(algorithm)?;

    match algo {
        OptimizationAlgorithm::ConjugateGradient => Ok(Box::new(CGOptimizer::new(
            scf,
            max_iterations,
            convergence_threshold,
        ))),
        OptimizationAlgorithm::SteepestDescent => Ok(Box::new(SteepestDescentOptimizer::new(
            scf,
            max_iterations,
            convergence_threshold,
        ))),
    }
}

/// Trait for geometry optimization algorithms
pub trait GeometryOptimizer {
    type SCFType: SCF;

    /// Create a new optimizer with SCF object and convergence settings
    fn new(scf: &mut Self::SCFType, max_iterations: usize, convergence_threshold: f64) -> Self
    where
        Self: Sized;

    /// Initialize with molecule data
    fn init(&mut self, coords: Vec<Vector3<f64>>, elements: Vec<Element>);

    /// Set optimization parameters
    fn set_max_iterations(&mut self, max_iter: usize);
    fn set_convergence_threshold(&mut self, threshold: f64);
    fn set_step_size(&mut self, step_size: f64);

    /// Run the optimization algorithm
    fn optimize(&mut self) -> (Vec<Vector3<f64>>, f64);

    /// Get optimization state information
    fn get_coordinates(&self) -> &Vec<Vector3<f64>>;
    fn get_energy(&self) -> f64;
    fn get_forces(&self) -> &Vec<Vector3<f64>>;

    /// Check convergence status
    fn is_converged(&self) -> bool;

    /// Log optimization progress
    fn log_progress(&self, iteration: usize);

    /// Update coordinates using forces - specific to each algorithm
    fn update_coordinates(&mut self);

    /// Calculate RMS and maximum force
    fn calculate_force_metrics(&self) -> (f64, f64) {
        let forces = self.get_forces();
        let mut max_force: f64 = 0.0;
        let mut sum_squared: f64 = 0.0;

        for force in forces {
            let norm = force.norm();
            max_force = max_force.max(norm);
            sum_squared += norm * norm;
        }

        let rms = (sum_squared / forces.len() as f64).sqrt();
        (rms, max_force)
    }

    /// Perform line search to find optimal step size
    fn line_search(&mut self, direction: &[Vector3<f64>]) -> f64 {
        let initial_coords = self.get_coordinates().clone();
        let initial_energy = self.get_energy();

        // Try different step sizes
        let step_sizes = [0.01, 0.05, 0.1, 0.2, 0.5];
        let mut best_step = 0.1;
        let mut best_energy = initial_energy;

        for &step in &step_sizes {
            // Test this step size
            let test_coords: Vec<Vector3<f64>> = initial_coords
                .iter()
                .zip(direction.iter())
                .map(|(coord, dir)| coord + step * dir)
                .collect();

            // Calculate energy at test position
            let test_energy = self.evaluate_energy_at(&test_coords);

            if test_energy < best_energy {
                best_energy = test_energy;
                best_step = step;
            }
        }

        best_step
    }

    /// Evaluate energy at given coordinates without updating the optimizer state
    fn evaluate_energy_at(&self, _coords: &[Vector3<f64>]) -> f64 {
        // This should be implemented by concrete types to avoid state changes
        // For now, return current energy as fallback
        self.get_energy()
    }

    /// Validate forces using numerical gradients (for debugging)
    fn validate_forces(&mut self, delta: f64) -> Vec<Vector3<f64>> {
        let current_coords = self.get_coordinates().clone();
        let _current_energy = self.get_energy();
        let analytical_forces = self.get_forces().clone();

        let mut numerical_forces = vec![Vector3::zeros(); current_coords.len()];

        // Calculate numerical gradients for each atom
        for atom_idx in 0..current_coords.len() {
            for dim in 0..3 {
                // Positive displacement
                let mut pos_coords = current_coords.clone();
                match dim {
                    0 => pos_coords[atom_idx].x += delta,
                    1 => pos_coords[atom_idx].y += delta,
                    2 => pos_coords[atom_idx].z += delta,
                    _ => unreachable!(),
                }
                let pos_energy = self.evaluate_energy_at(&pos_coords);

                // Negative displacement
                let mut neg_coords = current_coords.clone();
                match dim {
                    0 => neg_coords[atom_idx].x -= delta,
                    1 => neg_coords[atom_idx].y -= delta,
                    2 => neg_coords[atom_idx].z -= delta,
                    _ => unreachable!(),
                }
                let neg_energy = self.evaluate_energy_at(&neg_coords);

                // Central difference approximation
                let force_component = -(pos_energy - neg_energy) / (2.0 * delta);
                match dim {
                    0 => numerical_forces[atom_idx].x = force_component,
                    1 => numerical_forces[atom_idx].y = force_component,
                    2 => numerical_forces[atom_idx].z = force_component,
                    _ => unreachable!(),
                }
            }
        }

        // Log comparison
        tracing::info!("  Force validation (numerical vs analytical):");
        for (i, (num, ana)) in numerical_forces
            .iter()
            .zip(analytical_forces.iter())
            .enumerate()
        {
            let diff = (num - ana).norm();
            tracing::info!(
                "    Atom {}: num=[{:.6}, {:.6}, {:.6}] ana=[{:.6}, {:.6}, {:.6}] diff={:.6}",
                i + 1,
                num.x,
                num.y,
                num.z,
                ana.x,
                ana.y,
                ana.z,
                diff
            );
        }

        numerical_forces
    }
}
