//! Geometry optimization implementations
//!
//! This module provides various optimization algorithms for molecular geometry optimization.

use crate::scf_impl::SCF;
use color_eyre::eyre::Result;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use tracing::info;

/// Trait for geometry optimization algorithms
pub trait GeometryOptimizer {
    fn init(&mut self, coords: Vec<Vector3<f64>>, elements: Vec<Element>);
    fn optimize(&mut self) -> (Vec<Vector3<f64>>, f64);
    fn set_step_size(&mut self, step_size: f64);
}

/// Conjugate Gradient optimizer
pub struct CGOptimizer<'a, S: SCF> {
    scf: &'a mut S,
    max_iterations: usize,
    convergence_threshold: f64,
    step_size: f64,
    coords: Vec<Vector3<f64>>,
    elements: Vec<Element>,
}

impl<'a, S: SCF + Clone> CGOptimizer<'a, S> {
    pub fn new(scf: &'a mut S, max_iterations: usize, convergence_threshold: f64) -> Self {
        CGOptimizer {
            scf,
            max_iterations,
            convergence_threshold,
            step_size: 0.1,
            coords: Vec::new(),
            elements: Vec::new(),
        }
    }
}

impl<'a, S: SCF + Clone> GeometryOptimizer for CGOptimizer<'a, S> {
    fn init(&mut self, coords: Vec<Vector3<f64>>, elements: Vec<Element>) {
        self.coords = coords;
        self.elements = elements;
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn optimize(&mut self) -> (Vec<Vector3<f64>>, f64) {
        info!("Starting Conjugate Gradient optimization");
        info!("Max iterations: {}", self.max_iterations);
        info!("Convergence threshold: {:.6e}", self.convergence_threshold);

        let mut coords = self.coords.clone();
        let mut prev_gradient: Option<Vec<Vector3<f64>>> = None;
        let mut search_direction: Option<Vec<Vector3<f64>>> = None;

        for iteration in 0..self.max_iterations {
            // Re-initialize SCF with current geometry
            self.scf.init_geometry(&coords, &self.elements);
            self.scf.init_density_matrix();
            self.scf.init_fock_matrix();
            self.scf.scf_cycle();

            let energy = self.scf.calculate_total_energy();
            let forces = self.scf.calculate_forces();

            // Convert forces to gradients (negative of forces)
            let gradient: Vec<Vector3<f64>> = forces.iter().map(|f| -f).collect();
            let max_gradient = gradient.iter().map(|g| g.norm()).fold(0.0f64, f64::max);

            info!("Iteration {}: E = {:.10} au, max |grad| = {:.6e}",
                  iteration, energy, max_gradient);

            if max_gradient < self.convergence_threshold {
                info!("Optimization converged!");
                return (coords, energy);
            }

            // Compute search direction using conjugate gradient
            let new_search_direction = if let (Some(ref prev_grad), Some(ref prev_dir)) =
                (&prev_gradient, &search_direction) {
                // Fletcher-Reeves formula for beta
                let grad_dot_grad: f64 = gradient.iter().map(|g| g.dot(g)).sum();
                let prev_grad_dot: f64 = prev_grad.iter().map(|g| g.dot(g)).sum();
                let beta = grad_dot_grad / prev_grad_dot.max(1e-10);

                gradient.iter().zip(prev_dir.iter())
                    .map(|(g, d)| g + beta * d)
                    .collect()
            } else {
                // First iteration: use steepest descent direction
                gradient.clone()
            };

            // Update coordinates along search direction
            for (coord, dir) in coords.iter_mut().zip(new_search_direction.iter()) {
                *coord += self.step_size * dir;
            }

            prev_gradient = Some(gradient);
            search_direction = Some(new_search_direction);
        }

        info!("Optimization did not converge within {} iterations", self.max_iterations);
        let final_energy = self.scf.calculate_total_energy();
        (coords, final_energy)
    }
}

/// Steepest Descent optimizer
pub struct SteepestDescentOptimizer<'a, S: SCF> {
    scf: &'a mut S,
    max_iterations: usize,
    convergence_threshold: f64,
    step_size: f64,
    coords: Vec<Vector3<f64>>,
    elements: Vec<Element>,
}

impl<'a, S: SCF + Clone> SteepestDescentOptimizer<'a, S> {
    pub fn new(scf: &'a mut S, max_iterations: usize, convergence_threshold: f64) -> Self {
        SteepestDescentOptimizer {
            scf,
            max_iterations,
            convergence_threshold,
            step_size: 0.1,
            coords: Vec::new(),
            elements: Vec::new(),
        }
    }
}

impl<'a, S: SCF + Clone> GeometryOptimizer for SteepestDescentOptimizer<'a, S> {
    fn init(&mut self, coords: Vec<Vector3<f64>>, elements: Vec<Element>) {
        self.coords = coords;
        self.elements = elements;
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn optimize(&mut self) -> (Vec<Vector3<f64>>, f64) {
        info!("Starting Steepest Descent optimization");
        info!("Max iterations: {}", self.max_iterations);
        info!("Convergence threshold: {:.6e}", self.convergence_threshold);

        let mut coords = self.coords.clone();

        for iteration in 0..self.max_iterations {
            // Re-initialize SCF with current geometry
            self.scf.init_geometry(&coords, &self.elements);
            self.scf.init_density_matrix();
            self.scf.init_fock_matrix();
            self.scf.scf_cycle();

            let energy = self.scf.calculate_total_energy();
            let forces = self.scf.calculate_forces();
            let max_force = forces.iter().map(|f| f.norm()).fold(0.0f64, f64::max);

            info!("Iteration {}: E = {:.10} au, max |F| = {:.6e}",
                  iteration, energy, max_force);

            if max_force < self.convergence_threshold {
                info!("Optimization converged!");
                return (coords, energy);
            }

            // Update coordinates in the direction of the forces
            for (coord, force) in coords.iter_mut().zip(forces.iter()) {
                *coord += self.step_size * force.normalize();
            }
        }

        info!("Optimization did not converge within {} iterations", self.max_iterations);
        let final_energy = self.scf.calculate_total_energy();
        (coords, final_energy)
    }
}

/// Create an optimizer based on the algorithm name
pub fn create_optimizer<'a, S: SCF + Clone>(
    algorithm: &str,
    scf: &'a mut S,
    max_iterations: usize,
    convergence_threshold: f64,
) -> Result<Box<dyn GeometryOptimizer + 'a>> {
    match algorithm.to_lowercase().as_str() {
        "cg" => Ok(Box::new(CGOptimizer::new(scf, max_iterations, convergence_threshold))),
        "sd" => Ok(Box::new(SteepestDescentOptimizer::new(scf, max_iterations, convergence_threshold))),
        _ => Err(color_eyre::eyre::eyre!("Unknown optimization algorithm: {}", algorithm)),
    }
}
