//! Conjugate Gradient optimization algorithm

use super::GeometryOptimizer;
use crate::scf_impl::SCF;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;

/// Conjugate Gradient geometry optimizer
pub struct CGOptimizer<S: SCF> {
    scf: S,
    coords: Vec<Vector3<f64>>,
    elements: Vec<Element>,
    forces: Vec<Vector3<f64>>,
    energy: f64,
    previous_energy: f64,
    step_size: f64,
    max_iterations: usize,
    convergence_threshold: f64,
    // State needed for conjugate gradient algorithm
    previous_forces: Vec<Vector3<f64>>,
    directions: Vec<Vector3<f64>>,
    iteration: usize,
}

impl<S: SCF + Clone> GeometryOptimizer for CGOptimizer<S> {
    type SCFType = S;

    fn new(scf: &mut Self::SCFType, max_iterations: usize, convergence_threshold: f64) -> Self {
        CGOptimizer {
            scf: scf.clone(),
            coords: Vec::new(),
            elements: Vec::new(),
            forces: Vec::new(),
            energy: 0.0,
            previous_energy: 0.0,
            step_size: 0.1,
            max_iterations,
            convergence_threshold,
            previous_forces: Vec::new(),
            directions: Vec::new(),
            iteration: 0,
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

        // Initialize CG-specific state
        self.previous_forces = self.forces.clone();
        self.directions = self.forces.clone(); // First direction is just the forces
        self.iteration = 0;
    }

    fn update_coordinates(&mut self) {
        self.iteration += 1;

        // Use line search to find optimal step size (simplified version)
        let current_step = if self.iteration > 1 {
            self.adaptive_line_search()
        } else {
            self.step_size
        };

        // Move atoms according to current direction with adaptive step
        // Move in the direction to minimize energy
        for (i, direction) in self.directions.iter().enumerate() {
            self.coords[i] += current_step * direction;

            // Ensure coordinates remain finite to avoid propagating NaN/Inf
            if !self.coords[i].x.is_finite() {
                self.coords[i].x = 0.0;
            }
            if !self.coords[i].y.is_finite() {
                self.coords[i].y = 0.0;
            }
            if !self.coords[i].z.is_finite() {
                self.coords[i].z = 0.0;
            }

            // Clamp to reasonable bounds
            self.coords[i].x = self.coords[i].x.clamp(-50.0, 50.0);
            self.coords[i].y = self.coords[i].y.clamp(-50.0, 50.0);
            self.coords[i].z = self.coords[i].z.clamp(-50.0, 50.0);
        }

        // Update SCF with new geometry and recalculate
        self.previous_energy = self.energy;
        self.scf.init_geometry(&self.coords, &self.elements);
        self.scf.init_density_matrix();
        self.scf.scf_cycle();
        self.energy = self.scf.calculate_total_energy();

        // Store previous forces before updating
        self.previous_forces = self.forces.clone();
        self.forces = self.scf.calculate_forces();

        // Calculate beta for next iteration using Polak-Ribière+ formula
        if !self.is_converged() && self.iteration < self.max_iterations {
            let beta = self.calculate_beta_polak_ribiere_plus();

            // Update directions for next iteration: d = F + beta * d_prev
            for i in 0..self.directions.len() {
                self.directions[i] = self.forces[i] + beta * self.directions[i];
            }

            // Restart CG if direction becomes uphill
            if self.is_direction_uphill() {
                tracing::info!("    Restarting CG: direction became uphill");
                self.directions = self.forces.clone();
            }

            // Sanitize directions to avoid NaN/Inf propagation
            for dir in &mut self.directions {
                if !dir.x.is_finite() {
                    dir.x = 0.0;
                }
                if !dir.y.is_finite() {
                    dir.y = 0.0;
                }
                if !dir.z.is_finite() {
                    dir.z = 0.0;
                }
            }
        }
    }

    fn optimize(&mut self) -> (Vec<Vector3<f64>>, f64) {
        // Log initial state
        tracing::info!("#####################################################");
        tracing::info!("-------- Starting CG Geometry Optimization ----------");
        tracing::info!("#####################################################");

        self.log_progress(0);
        self.iteration = 0; // Reset iteration counter

        // Main optimization loop
        for iteration in 1..=self.max_iterations {
            // Use update_coordinates instead of direct implementation
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
                tracing::info!(
                    "Optimization reached maximum number of iterations ({}) without converging",
                    self.max_iterations
                );
                tracing::info!("-----------------------------------------------------\n");
            }
        }

        // Final sanitization to ensure coordinates are finite
        for coord in &mut self.coords {
            if !coord.x.is_finite() {
                coord.x = 0.0;
            }
            if !coord.y.is_finite() {
                coord.y = 0.0;
            }
            if !coord.z.is_finite() {
                coord.z = 0.0;
            }
            coord.x = coord.x.clamp(-50.0, 50.0);
            coord.y = coord.y.clamp(-50.0, 50.0);
            coord.z = coord.z.clamp(-50.0, 50.0);
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

        // Use both energy and force criteria for convergence
        let energy_converged = energy_change < self.convergence_threshold;
        let force_converged = max_force < self.convergence_threshold;

        energy_converged && force_converged
    }

    fn evaluate_energy_at(&self, coords: &[Vector3<f64>]) -> f64 {
        // Create a temporary SCF copy to evaluate energy without changing state
        let mut temp_scf = self.scf.clone();
        temp_scf.init_geometry(&coords.to_vec(), &self.elements);
        temp_scf.init_density_matrix();
        temp_scf.scf_cycle();
        temp_scf.calculate_total_energy()
    }

    fn log_progress(&self, iteration: usize) {
        let (rms_force, max_force) = self.calculate_force_metrics();

        if iteration == 0 {
            tracing::info!("  Initial state:");
        } else {
            tracing::info!("  Iteration {}:", iteration);
            tracing::info!(
                "    Energy change: {:.8} au",
                self.energy - self.previous_energy
            );
        }

        tracing::info!("    Energy: {:.8} au", self.energy);
        tracing::info!("    Max force: {:.8} au", max_force);
        tracing::info!("    RMS force: {:.8} au", rms_force);

        // Always log geometry and forces after each iteration
        tracing::info!("    Current geometry (atomic units):");
        for (i, (coord, elem)) in self.coords.iter().zip(self.elements.iter()).enumerate() {
            tracing::info!(
                "      Atom {}: {} at [{:10.6}, {:10.6}, {:10.6}] bohr",
                i + 1,
                elem.get_symbol(),
                coord.x,
                coord.y,
                coord.z
            );
        }

        tracing::info!("    Current forces (atomic units):");
        for (i, (force, elem)) in self.forces.iter().zip(self.elements.iter()).enumerate() {
            let force_magnitude = force.norm();
            tracing::info!(
                "      Atom {}: {} force [{:10.6}, {:10.6}, {:10.6}] au (|F| = {:10.6} au)",
                i + 1,
                elem.get_symbol(),
                force.x,
                force.y,
                force.z,
                force_magnitude
            );
        }
    }
}

impl<S: SCF + Clone> CGOptimizer<S> {
    /// Simplified adaptive line search for CG
    fn adaptive_line_search(&mut self) -> f64 {
        let mut step = self.step_size;

        // Energy went up → shrink step drastically.
        if self.energy > self.previous_energy {
            step *= 0.5;
        // Energy dropped a lot → only grow a bit.
        } else if self.previous_energy - self.energy > 1e-3 {
            step *= 1.05;
        }

        // Hard limits to avoid runaway geometries.
        step = step.clamp(0.001, 0.05);

        // Persist for next iteration so growth/decay is cumulative.
        self.step_size = step;

        step
    }

    /// Calculate beta using Polak-Ribière+ formula with restart capability
    fn calculate_beta_polak_ribiere_plus(&self) -> f64 {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..self.forces.len() {
            let current_force = self.forces[i];
            let prev_force = self.previous_forces[i];

            // Polak-Ribière: F_current · (F_current - F_prev)
            numerator += current_force.dot(&(current_force - prev_force));

            // F_prev · F_prev
            denominator += prev_force.dot(&prev_force);
        }

        // Avoid division by zero
        if denominator.abs() < 1e-12 {
            return 0.0;
        }

        // Polak-Ribière+ ensures beta >= 0 (automatic restart if negative)
        let beta = numerator / denominator;
        beta.max(0.0)
    }

    /// Check if current search direction is uphill (should restart CG)
    fn is_direction_uphill(&self) -> bool {
        let mut dot_product = 0.0;
        for i in 0..self.forces.len() {
            dot_product += self.forces[i].dot(&self.directions[i]);
        }
        dot_product < 0.0 // If negative, direction is uphill
    }
}
