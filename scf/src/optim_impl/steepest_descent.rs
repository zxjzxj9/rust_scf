//! Steepest Descent optimization algorithm

use super::GeometryOptimizer;
use crate::scf_impl::SCF;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;

/// Steepest Descent geometry optimizer
pub struct SteepestDescentOptimizer<S: SCF> {
    scf: S,
    coords: Vec<Vector3<f64>>,
    elements: Vec<Element>,
    forces: Vec<Vector3<f64>>,
    energy: f64,
    previous_energy: f64,
    step_size: f64,
    max_iterations: usize,
    convergence_threshold: f64,
}

impl<S: SCF + Clone> GeometryOptimizer for SteepestDescentOptimizer<S> {
    type SCFType = S;
    
    fn new(scf: &mut Self::SCFType, max_iterations: usize, convergence_threshold: f64) -> Self {
        SteepestDescentOptimizer {
            scf: scf.clone(),
            coords: Vec::new(),
            elements: Vec::new(),
            forces: Vec::new(),
            energy: 0.0,
            previous_energy: 0.0,
            step_size: 0.1,
            max_iterations,
            convergence_threshold,
        }
    }

    fn init(&mut self, coords: Vec<Vector3<f64>>, elements: Vec<Element>) {
        self.coords = coords;
        self.elements = elements;
        // Initialize SCF with geometry
        self.scf.init_geometry(&self.coords, &self.elements);
        // Recompute matrices for the new geometry
        self.scf.init_density_matrix();
        // Run initial SCF to get energy and forces
        self.scf.scf_cycle();
        self.energy = self.scf.calculate_total_energy();
        self.forces = self.scf.calculate_forces();
    }

    fn update_coordinates(&mut self) {
        // Steepest descent implementation
        for (i, force) in self.forces.iter().enumerate() {
            self.coords[i] += self.step_size * force;
        }

        // Update SCF with new geometry and recalculate
        self.previous_energy = self.energy;
        self.scf.init_geometry(&self.coords, &self.elements);
        self.scf.init_density_matrix();
        self.scf.scf_cycle();
        self.energy = self.scf.calculate_total_energy();
        self.forces = self.scf.calculate_forces();
    }

    fn optimize(&mut self) -> (Vec<Vector3<f64>>, f64) {
        // Log initial state
        tracing::info!("#####################################################");
        tracing::info!("---------- Starting Geometry Optimization ----------");
        tracing::info!("#####################################################");

        self.log_progress(0);

        // Main optimization loop
        for iteration in 1..=self.max_iterations {
            // Update coordinates (moves atoms according to forces)
            self.update_coordinates();

            // Log progress
            self.log_progress(iteration);

            // Check for convergence
            if self.is_converged() {
                tracing::info!("Optimization converged after {} iterations", iteration);
                tracing::info!("-----------------------------------------------------\n");
                break;
            }

            // Check if we've reached max iterations
            if iteration == self.max_iterations {
                tracing::info!("Optimization reached maximum number of iterations ({}) without converging",
                          self.max_iterations);
                tracing::info!("-----------------------------------------------------\n");
            }
        }

        // Return optimized coordinates and final energy
        (self.coords.clone(), self.energy)
    }

    fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    fn set_convergence_threshold(&mut self, threshold: f64) {
        self.convergence_threshold = threshold;
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn get_coordinates(&self) -> &Vec<Vector3<f64>> {
        &self.coords
    }

    fn get_energy(&self) -> f64 {
        self.energy
    }

    fn get_forces(&self) -> &Vec<Vector3<f64>> {
        &self.forces
    }

    fn is_converged(&self) -> bool {
        // Check convergence based on energy change and force metrics
        let energy_change = (self.energy - self.previous_energy).abs();
        let (rms_force, max_force) = self.calculate_force_metrics();

        // Log convergence criteria
        tracing::info!("  Convergence check:");
        tracing::info!("    Energy change: {:.8} au", energy_change);
        tracing::info!("    Max force: {:.8} au", max_force);
        tracing::info!("    RMS force: {:.8} au", rms_force);

        // Return true if both criteria are met
        energy_change < self.convergence_threshold && max_force < self.convergence_threshold
    }

    fn log_progress(&self, iteration: usize) {
        let (rms_force, max_force) = self.calculate_force_metrics();

        if iteration == 0 {
            tracing::info!("  Initial state:");
        } else {
            tracing::info!("  Iteration {}:", iteration);
            tracing::info!("    Energy change: {:.8} au", self.energy - self.previous_energy);
        }

        tracing::info!("    Energy: {:.8} au", self.energy);
        tracing::info!("    Max force: {:.8} au", max_force);
        tracing::info!("    RMS force: {:.8} au", rms_force);

        // Always log geometry and forces after each iteration
        tracing::info!("    Current geometry (atomic units):");
        for (i, (coord, elem)) in self.coords.iter().zip(self.elements.iter()).enumerate() {
            tracing::info!("      Atom {}: {} at [{:10.6}, {:10.6}, {:10.6}] bohr",
            i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
        }
        
        tracing::info!("    Current forces (atomic units):");
        for (i, (force, elem)) in self.forces.iter().zip(self.elements.iter()).enumerate() {
            let force_magnitude = force.norm();
            tracing::info!("      Atom {}: {} force [{:10.6}, {:10.6}, {:10.6}] au (|F| = {:10.6} au)",
            i + 1, elem.get_symbol(), force.x, force.y, force.z, force_magnitude);
        }
    }

    fn evaluate_energy_at(&self, coords: &[Vector3<f64>]) -> f64 {
        // Create a temporary SCF copy to evaluate energy without changing state
        let mut temp_scf = self.scf.clone();
        temp_scf.init_geometry(&coords.to_vec(), &self.elements);
        temp_scf.init_density_matrix();
        temp_scf.scf_cycle();
        temp_scf.calculate_total_energy()
    }
}

