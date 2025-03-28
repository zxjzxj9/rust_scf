use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use crate::scf::SCF;

pub trait GeometryOptimizer {
    type SCFType: SCF;

    /// Create a new optimizer with SCF object and convergence settings
    fn new(scf: &mut Self::SCFType, max_iterations: usize, convergence_threshold: f64) -> Self;

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
        let mut max_force:f64 = 0.0;
        let mut sum_squared:f64 = 0.0;

        for force in forces {
            let norm = force.norm();
            max_force = max_force.max(norm);
            sum_squared += norm * norm;
        }

        let rms = (sum_squared / forces.len() as f64).sqrt();
        (rms, max_force)
    }
}

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

impl<S: SCF  + Clone> GeometryOptimizer for SteepestDescentOptimizer<S> {
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
        // Run initial SCF to get energy and forces
        self.scf.scf_cycle();
        self.energy = self.scf.calculate_total_energy();
        self.forces = self.scf.calculate_forces();
    }

    // Implement other required methods...

    fn update_coordinates(&mut self) {
        // Steepest descent implementation
        for (i, force) in self.forces.iter().enumerate() {
            self.coords[i] += self.step_size * force;
        }

        // Update SCF with new geometry and recalculate
        self.previous_energy = self.energy;
        self.scf.init_geometry(&self.coords, &self.elements);
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

        // Log coordinates if needed
        if iteration == 0 || iteration % 5 == 0 {
            tracing::info!("    Current geometry:");
            for (i, (coord, elem)) in self.coords.iter().zip(self.elements.iter()).enumerate() {
                tracing::info!("      Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
                i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
            }
        }
    }
}

impl<S: SCF + Clone> GeometryOptimizer for CGOptimizer<S> {
    type SCFType = S;

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

        // Log coordinates if needed
        if iteration == 0 || iteration % 5 == 0 {
            tracing::info!("    Current geometry:");
            for (i, (coord, elem)) in self.coords.iter().zip(self.elements.iter()).enumerate() {
                tracing::info!("      Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
                i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
            }
        }
    }

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
        // Run initial SCF to get energy and forces
        self.scf.scf_cycle();
        self.energy = self.scf.calculate_total_energy();
        self.forces = self.scf.calculate_forces();

        // Initialize CG-specific state
        self.previous_forces = self.forces.clone();
        self.directions = self.forces.clone(); // First direction is just the forces
        self.iteration = 0;
    }

    // Other methods remain unchanged...

    fn update_coordinates(&mut self) {
        self.iteration += 1;

        // Move atoms according to current direction
        for (i, direction) in self.directions.iter().enumerate() {
            self.coords[i] += self.step_size * direction;
        }

        // Update SCF with new geometry and recalculate
        self.previous_energy = self.energy;
        self.scf.init_geometry(&self.coords, &self.elements);
        self.scf.scf_cycle();
        self.energy = self.scf.calculate_total_energy();
        self.forces = self.scf.calculate_forces();

        // Calculate beta using Polak-Ribière formula for next iteration
        if !self.is_converged() && self.iteration < self.max_iterations {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..self.forces.len() {
                let current_force = self.forces[i];
                let prev_force = self.previous_forces[i];

                // F_current · (F_current - F_prev)
                numerator += current_force.dot(&(current_force - prev_force));

                // F_prev · F_prev
                denominator += prev_force.dot(&prev_force);
            }

            // Avoid division by zero and ensure beta is non-negative
            let beta = if denominator.abs() < 1e-10 {
                0.0
            } else {
                (numerator / denominator).max(0.0)
            };

            // Update directions for next iteration: d = F + beta * d_prev
            for i in 0..self.directions.len() {
                self.directions[i] = self.forces[i] + beta * self.directions[i];
            }

            self.previous_forces = self.forces.clone();
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
                tracing::info!("Optimization reached maximum number of iterations ({}) without converging",
                           self.max_iterations);
                tracing::info!("-----------------------------------------------------\n");
            }
        }

        // Return optimized coordinates and final energy
        (self.coords.clone(), self.energy)
    }
}