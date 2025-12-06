// Argon Melting Simulation Example
//
// This example demonstrates a molecular dynamics simulation of argon atoms
// undergoing a solid-to-liquid phase transition using the Lennard-Jones potential.
//
// Physical parameters are based on realistic values for argon:
// - ε = 120 K × k_B ≈ 1.65 × 10^-21 J (well depth)
// - σ = 3.4 Å (atomic diameter)
// - Mass = 39.948 u (atomic mass of argon)
//
// The simulation starts with atoms arranged in an FCC lattice at low temperature
// and gradually heats the system while monitoring for melting signatures.

use md::{Integrator, LennardJones, NoseHooverVerlet};
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;
use std::collections::VecDeque;

// Physical constants and argon parameters
const K_B: f64 = 1.380649e-23; // Boltzmann constant (J/K)
const ARGON_MASS: f64 = 39.948 * 1.66054e-27; // kg (atomic mass unit to kg)
const ARGON_EPSILON: f64 = 120.0 * K_B; // J (well depth in energy units)
const ARGON_SIGMA: f64 = 3.4e-10; // m (collision diameter)

// Conversion factors for reduced units
const ENERGY_UNIT: f64 = ARGON_EPSILON; // J
const LENGTH_UNIT: f64 = ARGON_SIGMA; // m
const MASS_UNIT: f64 = ARGON_MASS; // kg

fn time_unit() -> f64 {
    (MASS_UNIT * LENGTH_UNIT * LENGTH_UNIT / ENERGY_UNIT).sqrt() // s
}

struct MeltingAnalyzer {
    position_history: VecDeque<Vec<Vector3<f64>>>,
    max_history: usize,
    last_diffusion_time: f64,
}

impl MeltingAnalyzer {
    fn new(max_history: usize) -> Self {
        Self {
            position_history: VecDeque::new(),
            max_history,
            last_diffusion_time: 0.0,
        }
    }

    fn update(&mut self, positions: &[Vector3<f64>], _time: f64) {
        self.position_history.push_back(positions.to_vec());

        if self.position_history.len() > self.max_history {
            self.position_history.pop_front();
        }
    }

    /// Calculate mean squared displacement (MSD) over the stored history
    fn calculate_msd(&self) -> Option<f64> {
        if self.position_history.len() < 2 {
            return None;
        }

        let initial_positions = self.position_history.front().unwrap();
        let final_positions = self.position_history.back().unwrap();

        let mut total_msd = 0.0;
        let n_atoms = initial_positions.len();

        for i in 0..n_atoms {
            let displacement = final_positions[i] - initial_positions[i];
            total_msd += displacement.dot(&displacement);
        }

        Some(total_msd / n_atoms as f64)
    }

    /// Estimate diffusion coefficient from MSD
    fn diffusion_coefficient(&self, dt: f64) -> Option<f64> {
        let msd = self.calculate_msd()?;
        let time_interval = (self.position_history.len() - 1) as f64 * dt;

        if time_interval > 0.0 {
            // D = MSD / (6 * t) for 3D diffusion
            Some(msd / (6.0 * time_interval))
        } else {
            None
        }
    }
}

fn create_fcc_lattice(n_cells: usize, a: f64) -> Vec<Vector3<f64>> {
    let basis = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5, 0.5, 0.0),
        Vector3::new(0.5, 0.0, 0.5),
        Vector3::new(0.0, 0.5, 0.5),
    ];

    let mut positions = Vec::new();

    for i in 0..n_cells {
        for j in 0..n_cells {
            for k in 0..n_cells {
                let origin = Vector3::new(i as f64, j as f64, k as f64) * a;
                for &b in &basis {
                    positions.push(origin + b * a);
                }
            }
        }
    }

    positions
}

fn thermalize_velocities(n_atoms: usize, temperature: f64) -> Vec<Vector3<f64>> {
    let mut rng = rand::thread_rng();
    let mut velocities = Vec::with_capacity(n_atoms);

    for _ in 0..n_atoms {
        let v = Vector3::new(
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
        );
        velocities.push(v * (temperature as f64).sqrt());
    }

    // Remove center-of-mass motion
    let v_cm: Vector3<f64> = velocities.iter().sum::<Vector3<f64>>() / n_atoms as f64;
    for v in &mut velocities {
        *v -= v_cm;
    }

    velocities
}

fn temperature_to_kelvin(reduced_temp: f64) -> f64 {
    reduced_temp * ARGON_EPSILON / K_B
}

fn time_to_femtoseconds(reduced_time: f64) -> f64 {
    reduced_time * time_unit() * 1e15
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      Argon Melting Simulation                       ║");
    println!("║                     Using Lennard-Jones Potential                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Physical Parameters:                                                 ║");
    println!(
        "║   ε = {:.1} K × k_B = {:.2e} J                                ║",
        ARGON_EPSILON / K_B,
        ARGON_EPSILON
    );
    println!(
        "║   σ = {:.1} Å = {:.2e} m                                      ║",
        ARGON_SIGMA * 1e10,
        ARGON_SIGMA
    );
    println!(
        "║   Mass = {:.3} u = {:.2e} kg                                 ║",
        ARGON_MASS / 1.66054e-27,
        ARGON_MASS
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Simulation parameters (in reduced units)
    let n_cells = 4;
    let lattice_constant = 1.5; // Slightly larger than minimum energy distance
    let box_length = n_cells as f64 * lattice_constant;
    let box_lengths = Vector3::new(box_length, box_length, box_length);

    // Create FCC lattice
    let positions = create_fcc_lattice(n_cells, lattice_constant);
    let n_atoms = positions.len();
    let masses = vec![1.0; n_atoms]; // Reduced units

    println!("System Setup:");
    println!("  Lattice: FCC with {} unit cells", n_cells);
    println!("  Total atoms: {}", n_atoms);
    println!(
        "  Box dimensions: {:.2} × {:.2} × {:.2} σ",
        box_length, box_length, box_length
    );
    println!(
        "  Density: {:.3} atoms/σ³",
        n_atoms as f64 / box_length.powi(3)
    );
    println!();

    // Temperature ramping parameters
    let initial_temp = 0.5; // Reduced units (~60 K)
    let final_temp = 1.5; // Reduced units (~180 K) - above melting point
    let temp_ramp_steps = 20000;
    let equilibration_steps = 5000;
    let total_steps = temp_ramp_steps + equilibration_steps;

    let dt = 0.001; // Reduced time units

    println!("Temperature Schedule:");
    println!(
        "  Initial: {:.2} (reduced) = {:.1} K",
        initial_temp,
        temperature_to_kelvin(initial_temp)
    );
    println!(
        "  Final: {:.2} (reduced) = {:.1} K",
        final_temp,
        temperature_to_kelvin(final_temp)
    );
    println!("  Ramping steps: {}", temp_ramp_steps);
    println!("  Equilibration steps: {}", equilibration_steps);
    println!(
        "  Time step: {:.3} (reduced) = {:.1} fs",
        dt,
        time_to_femtoseconds(dt)
    );
    println!();

    // Initialize system
    let velocities = thermalize_velocities(n_atoms, initial_temp);
    let lj = LennardJones::new(1.0, 1.0, box_lengths); // Reduced units

    let mut integrator = NoseHooverVerlet::new(
        positions,
        velocities,
        masses,
        lj,
        100.0, // Q parameter for thermostat
        initial_temp,
        1.0, // k_B in reduced units
    );

    let mut analyzer = MeltingAnalyzer::new(100); // Keep last 100 snapshots for diffusion analysis

    // Output header
    println!("┌────────┬─────────┬──────────┬──────────┬──────────┬──────────┬───────────┐");
    println!(
        "│ {:>6} │ {:>7} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>9} │",
        "Step", "T (K)", "T_red", "KE_red", "PE_red", "Total_E", "Diff(σ²/τ)"
    );
    println!("├────────┼─────────┼──────────┼──────────┼──────────┼──────────┼───────────┤");

    for step in 0..total_steps {
        // Update target temperature (linear ramp)
        let target_temp = if step < temp_ramp_steps {
            let progress = step as f64 / temp_ramp_steps as f64;
            initial_temp + (final_temp - initial_temp) * progress
        } else {
            final_temp
        };

        // Update integrator target temperature
        integrator.set_target_temperature(target_temp);

        // Integration step
        integrator.step(dt);

        // Apply periodic boundary conditions
        for pos in &mut integrator.positions {
            for k in 0..3 {
                let box_l = box_lengths[k];
                pos[k] -= box_l * (pos[k] / box_l).floor();
            }
        }

        // Update analyzer
        analyzer.update(&integrator.positions, step as f64 * dt);

        // Output every 1000 steps
        if step % 1000 == 0 {
            let current_temp = integrator.temperature();
            let kinetic = integrator
                .velocities
                .iter()
                .zip(&integrator.masses)
                .map(|(v, &m)| 0.5 * m * v.dot(v))
                .sum::<f64>();
            let potential = integrator
                .provider
                .compute_potential_energy(&integrator.positions);
            let total_energy = kinetic + potential;

            let diffusion_coeff = analyzer.diffusion_coefficient(dt * 1000.0).unwrap_or(0.0);

            println!(
                "│ {:>6} │ {:>7.1} │ {:>8.3} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>9.6} │",
                step,
                temperature_to_kelvin(current_temp),
                current_temp,
                kinetic,
                potential,
                total_energy,
                diffusion_coeff
            );
        }
    }

    println!("└────────┴─────────┴──────────┴──────────┴──────────┴──────────┴───────────┘");
    println!();

    // Final analysis
    let final_temp = integrator.temperature();
    let final_diffusion = analyzer.diffusion_coefficient(dt * 1000.0).unwrap_or(0.0);

    println!("Final Analysis:");
    println!(
        "  Final temperature: {:.3} (reduced) = {:.1} K",
        final_temp,
        temperature_to_kelvin(final_temp)
    );
    println!("  Diffusion coefficient: {:.6} σ²/τ", final_diffusion);

    // Melting point estimation (rough heuristic based on diffusion)
    if final_diffusion > 0.01 {
        println!("  🔥 System appears to be in LIQUID state (high diffusion)");
        println!("     Estimated melting occurred during temperature ramp");
    } else if final_diffusion > 0.001 {
        println!("  🌡️  System shows INTERMEDIATE mobility (possible melting)");
    } else {
        println!("  🧊 System appears to be in SOLID state (low diffusion)");
    }

    println!();
    println!("Note: In reduced units, argon typically melts around T* ≈ 0.8-1.0");
    println!("      This corresponds to ~100-120 K in real units.");
    println!();
    println!("✅ Argon melting simulation completed!");
}
