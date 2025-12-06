// Multi-Atom NPT Simulation Example
//
// This example demonstrates NPT molecular dynamics with multiple interacting atoms
// using Lennard-Jones potential. Shows the key differences from single-atom systems:
//
// 1. Inter-atomic interactions (LJ potential)
// 2. Collective dynamics and structure formation
// 3. Proper system initialization for multiple atoms
// 4. Realistic phase behavior
//
// This is a direct comparison to single_atom_npt.rs to highlight the differences.

use md::{Integrator, LennardJones, NoseHooverParrinelloRahman};
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;
use std::collections::VecDeque;

// Physical constants and parameters
const K_B: f64 = 1.0; // Reduced units
const ARGON_MASS: f64 = 1.0;
const EPSILON: f64 = 1.0; // LJ well depth
const SIGMA: f64 = 1.0; // LJ collision diameter

/// Structure analyzer for multi-atom systems
struct StructureAnalyzer {
    position_history: VecDeque<Vec<Vector3<f64>>>,
    max_history: usize,
}

impl StructureAnalyzer {
    fn new(max_history: usize) -> Self {
        Self {
            position_history: VecDeque::new(),
            max_history,
        }
    }

    fn update(&mut self, positions: &[Vector3<f64>]) {
        self.position_history.push_back(positions.to_vec());
        if self.position_history.len() > self.max_history {
            self.position_history.pop_front();
        }
    }

    /// Calculate radial distribution function (simplified)
    fn radial_distribution_peak(&self, box_lengths: Vector3<f64>) -> f64 {
        if self.position_history.is_empty() {
            return 0.0;
        }

        let positions = self.position_history.back().unwrap();
        let n_atoms = positions.len();
        if n_atoms < 2 {
            return 0.0;
        }

        let mut distances = Vec::new();

        for i in 0..n_atoms {
            for j in (i + 1)..n_atoms {
                let mut dr = positions[i] - positions[j];

                // Apply minimum image convention
                for k in 0..3 {
                    let box_l = box_lengths[k];
                    dr[k] -= box_l * (dr[k] / box_l).round();
                }

                let distance = dr.norm();
                if distance > 0.1 && distance < box_lengths.x * 0.5 {
                    distances.push(distance);
                }
            }
        }

        if distances.is_empty() {
            0.0
        } else {
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            distances[distances.len() / 2] // Median distance as structure indicator
        }
    }

    /// Calculate mean squared displacement for diffusion
    fn calculate_diffusion_coefficient(&self, dt_interval: f64) -> f64 {
        if self.position_history.len() < 2 {
            return 0.0;
        }

        let initial = self.position_history.front().unwrap();
        let final_pos = self.position_history.back().unwrap();
        let n_atoms = initial.len();

        let mut total_msd = 0.0;
        for i in 0..n_atoms {
            let displacement = final_pos[i] - initial[i];
            total_msd += displacement.dot(&displacement);
        }

        let msd = total_msd / n_atoms as f64;
        let time_interval = (self.position_history.len() - 1) as f64 * dt_interval;

        if time_interval > 0.0 {
            msd / (6.0 * time_interval) // 3D diffusion coefficient
        } else {
            0.0
        }
    }
}

/// Create a simple cubic lattice of atoms
fn create_simple_cubic_lattice(n_per_side: usize, lattice_spacing: f64) -> Vec<Vector3<f64>> {
    let mut positions = Vec::new();

    for i in 0..n_per_side {
        for j in 0..n_per_side {
            for k in 0..n_per_side {
                let pos = Vector3::new(
                    i as f64 * lattice_spacing,
                    j as f64 * lattice_spacing,
                    k as f64 * lattice_spacing,
                );
                positions.push(pos);
            }
        }
    }

    positions
}

/// Initialize Maxwell-Boltzmann velocity distribution for multiple atoms
fn initialize_velocities(n_atoms: usize, temperature: f64) -> Vec<Vector3<f64>> {
    let mut rng = rand::thread_rng();
    let mut velocities = Vec::with_capacity(n_atoms);

    // Sample individual velocities
    for _ in 0..n_atoms {
        let v = Vector3::new(
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
        );
        velocities.push(v * temperature.sqrt());
    }

    // Remove center-of-mass motion
    let v_cm: Vector3<f64> = velocities.iter().sum::<Vector3<f64>>() / n_atoms as f64;
    for v in &mut velocities {
        *v -= v_cm;
    }

    // Scale to exact target temperature
    let current_temp = velocities.iter().map(|v| v.dot(v)).sum::<f64>() / (3.0 * n_atoms as f64);

    if current_temp > 0.0 {
        let scale_factor = (temperature / current_temp).sqrt();
        for v in &mut velocities {
            *v *= scale_factor;
        }
    }

    velocities
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                     Multi-Atom NPT Simulation                       ║");
    println!("║                   With Lennard-Jones Interactions                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Key Features vs Single Atom:                                        ║");
    println!("║   • Inter-atomic LJ interactions: V(r) = 4ε[(σ/r)¹²-(σ/r)⁶]      ║");
    println!("║   • Collective structural dynamics                                   ║");
    println!("║   • Realistic pressure from kinetic + virial contributions         ║");
    println!("║   • Phase transitions and structure formation                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // System parameters
    let n_per_side = 3; // 3x3x3 = 27 atoms
    let lattice_spacing = 1.2; // Slightly larger than σ
    let initial_temp = 0.8;
    let target_pressure = 0.5;

    // Create initial configuration
    let positions = create_simple_cubic_lattice(n_per_side, lattice_spacing);
    let n_atoms = positions.len();
    let velocities = initialize_velocities(n_atoms, initial_temp);
    let masses = vec![ARGON_MASS; n_atoms];

    // Box setup - start with lattice size + buffer
    let initial_box_length = (n_per_side as f64) * lattice_spacing * 1.2;
    let initial_box = Vector3::new(initial_box_length, initial_box_length, initial_box_length);

    println!("System Setup:");
    println!("  Number of atoms: {}", n_atoms);
    println!(
        "  Lattice: {}x{}x{} simple cubic",
        n_per_side, n_per_side, n_per_side
    );
    println!("  Lattice spacing: {:.2} σ", lattice_spacing);
    println!(
        "  Initial box: {:.2} × {:.2} × {:.2}",
        initial_box_length, initial_box_length, initial_box_length
    );
    println!(
        "  Initial density: {:.3} atoms/σ³",
        n_atoms as f64 / initial_box.x.powi(3)
    );
    println!("  Initial temperature: {:.3} (reduced)", initial_temp);
    println!("  Target pressure: {:.3} (reduced)", target_pressure);
    println!();

    // Lennard-Jones force provider
    let lj = LennardJones::new(EPSILON, SIGMA, initial_box);

    // NPT integrator with gentler coupling for multi-atom system
    let q_t = 200.0; // Thermostat coupling
    let q_p = 1000.0; // Barostat coupling

    let mut integrator = NoseHooverParrinelloRahman::new(
        positions,
        velocities,
        masses,
        lj,
        initial_box,
        q_t,
        q_p,
        initial_temp,
        target_pressure,
        K_B,
    );

    // Analysis tools
    let mut analyzer = StructureAnalyzer::new(50);

    // Simulation parameters
    let dt = 0.002;
    let total_steps = 20000;
    let output_interval = 400;
    let analysis_interval = 200;

    println!("Simulation Parameters:");
    println!("  Time step: {:.3} (reduced)", dt);
    println!("  Total steps: {}", total_steps);
    println!("  Thermostat coupling: {:.1}", q_t);
    println!("  Barostat coupling: {:.1}", q_p);
    println!();

    // Temperature ramping (optional)
    let temp_ramp_steps = 12000;
    let final_temp = 1.2;

    println!("Temperature Schedule:");
    println!(
        "  Steps 0-{}: {:.2} → {:.2} (gradual heating)",
        temp_ramp_steps, initial_temp, final_temp
    );
    println!(
        "  Steps {}-{}: {:.2} (equilibration)",
        temp_ramp_steps, total_steps, final_temp
    );
    println!();

    // Output header
    println!("┌────────┬─────────┬─────────┬─────────┬────────┬────────┬─────────┬──────────┐");
    println!(
        "│ {:>6} │ {:>7} │ {:>7} │ {:>7} │ {:>6} │ {:>6} │ {:>7} │ {:>8} │",
        "Step", "T_curr", "P_curr", "P_tgt", "Volume", "Box_L", "RDF_pk", "Diffusion"
    );
    println!("├────────┼─────────┼─────────┼─────────┼────────┼────────┼─────────┼──────────┤");

    // Main simulation loop
    for step in 0..total_steps {
        // Temperature ramping
        let current_target_temp = if step < temp_ramp_steps {
            let progress = step as f64 / temp_ramp_steps as f64;
            initial_temp + (final_temp - initial_temp) * progress
        } else {
            final_temp
        };

        integrator.set_target_temperature(current_target_temp);

        // Integration step
        integrator.step(dt);

        // Update LJ potential with current box size
        integrator.provider = LennardJones::new(EPSILON, SIGMA, integrator.box_lengths);

        // Apply periodic boundary conditions
        for pos in &mut integrator.positions {
            for k in 0..3 {
                let box_l = integrator.box_lengths[k];
                pos[k] -= box_l * (pos[k] / box_l).floor();
            }
        }

        // Update analysis
        if step % analysis_interval == 0 {
            analyzer.update(&integrator.positions);
        }

        // Output
        if step % output_interval == 0 {
            let current_temp = integrator.temperature();
            let current_pressure = integrator.get_pressure();
            let volume = integrator.get_volume();
            let box_length = integrator.box_lengths.x;
            let rdf_peak = analyzer.radial_distribution_peak(integrator.box_lengths);
            let diffusion = analyzer.calculate_diffusion_coefficient(dt * analysis_interval as f64);

            println!(
                "│ {:>6} │ {:>7.3} │ {:>7.3} │ {:>7.3} │ {:>6.2} │ {:>6.2} │ {:>7.3} │ {:>8.5} │",
                step,
                current_temp,
                current_pressure,
                target_pressure,
                volume,
                box_length,
                rdf_peak,
                diffusion
            );
        }
    }

    println!("└────────┴─────────┴─────────┴─────────┴────────┴────────┴─────────┴──────────┘");
    println!();

    // Final analysis
    let final_temp = integrator.temperature();
    let final_pressure = integrator.get_pressure();
    let final_volume = integrator.get_volume();
    let final_density = n_atoms as f64 / final_volume;
    let final_rdf_peak = analyzer.radial_distribution_peak(integrator.box_lengths);
    let final_diffusion = analyzer.calculate_diffusion_coefficient(dt * analysis_interval as f64);

    println!("Final Analysis:");
    println!(
        "  Final temperature: {:.4} (target: {:.3})",
        final_temp, final_temp
    );
    println!(
        "  Final pressure: {:.4} (target: {:.3})",
        final_pressure, target_pressure
    );
    println!("  Final volume: {:.4}", final_volume);
    println!("  Final density: {:.4} atoms/σ³", final_density);
    println!(
        "  Final box: {:.3} × {:.3} × {:.3}",
        integrator.box_lengths.x, integrator.box_lengths.y, integrator.box_lengths.z
    );
    println!("  Radial distribution peak: {:.3} σ", final_rdf_peak);
    println!("  Diffusion coefficient: {:.5} σ²/τ", final_diffusion);
    println!();

    // Phase identification
    println!("Phase Analysis:");
    if final_diffusion < 0.001 {
        println!("  🧊 SOLID-like behavior (low diffusion)");
        println!("     Atoms are localized, possibly crystalline structure");
    } else if final_diffusion < 0.01 {
        println!("  🌊 INTERMEDIATE behavior (moderate diffusion)");
        println!("     Possible liquid or soft solid state");
    } else {
        println!("  💨 LIQUID/GAS-like behavior (high diffusion)");
        println!("     High atomic mobility, fluid behavior");
    }

    if final_rdf_peak > 0.0 {
        if final_rdf_peak < 1.2 {
            println!("  📏 High density: atoms are closely packed");
        } else if final_rdf_peak < 2.0 {
            println!("  📏 Medium density: typical liquid-like spacing");
        } else {
            println!("  📏 Low density: gas-like or expanded structure");
        }
    }

    println!();
    println!("🎯 Multi-Atom NPT Key Insights:");
    println!("   • Inter-atomic forces create realistic pressure");
    println!("   • Collective behavior enables phase transitions");
    println!("   • Structure formation depends on density/temperature");
    println!("   • NPT ensemble allows natural volume evolution");
    println!();
    println!("💡 Compare with single_atom_npt.rs to see the difference!");
}
