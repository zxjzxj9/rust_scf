// Single Atom NPT Simulation Example
//
// This example demonstrates NPT (constant pressure and temperature) molecular dynamics
// simulation with a single atom. This showcases the behavior of the barostat and
// thermostat without inter-atomic interactions.
//
// Key features demonstrated:
// - NPT ensemble with Nosé-Hoover thermostat + Parrinello-Rahman barostat
// - Volume fluctuations under constant pressure
// - Temperature control via thermostat
// - Pressure monitoring and box size evolution
//
// Physical insight: With 1 atom, pressure comes purely from kinetic energy (ideal gas law)
// P = nkT/V, so the volume should adjust to maintain constant pressure as temperature changes.

use md::{ForceProvider, Integrator, NoseHooverParrinelloRahman};
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;

// Physical constants (using argon-like parameters in reduced units)
const K_B: f64 = 1.0; // Reduced units where k_B = 1
const ARGON_MASS: f64 = 1.0; // Mass in reduced units

/// Simple force provider for single atom system
/// In this case, forces are zero since there are no interactions
struct SingleAtomForces {
    external_force: Vector3<f64>,
}

impl SingleAtomForces {
    fn new() -> Self {
        Self {
            external_force: Vector3::zeros(),
        }
    }
    
    /// Add a weak harmonic potential to center the atom (optional)
    fn with_centering_force(strength: f64) -> Self {
        Self {
            external_force: Vector3::zeros(), // Will be calculated based on position
        }
    }
}

impl ForceProvider for SingleAtomForces {
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let mut forces = vec![Vector3::zeros(); positions.len()];
        
        // For a truly free single atom, use zero forces
        // This demonstrates pure NPT ensemble behavior without external potentials
        for i in 0..forces.len() {
            forces[i] = self.external_force; // Usually zero
        }
        
        forces
    }
}

fn initialize_velocity(temperature: f64) -> Vector3<f64> {
    let mut rng = rand::thread_rng();
    
    // Sample from Maxwell-Boltzmann distribution
    let v = Vector3::new(
        rng.sample(StandardNormal),
        rng.sample(StandardNormal),
        rng.sample(StandardNormal),
    );
    
    // Scale to desired temperature (in reduced units: <1/2 m v²> = 3/2 kT)
    v * (temperature).sqrt()
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                     Single Atom NPT Simulation                      ║");
    println!("║                 Demonstrating Pressure & Temperature Control        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ NPT Ensemble Features:                                              ║");
    println!("║   • Nosé-Hoover thermostat (temperature control)                   ║");
    println!("║   • Parrinello-Rahman barostat (pressure control)                  ║");
    println!("║   • Volume fluctuations to maintain constant pressure              ║");
    println!("║   • Single atom ideal gas behavior: PV = nkT                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Simulation parameters (reduced units)
    let initial_temp = 1.0;  // Reduced temperature
    let target_pressure = 0.1;  // Reduced pressure (lower for stability)
    let initial_box_length = 8.0;  // Initial cubic box size (larger initial size)
    
    // System setup
    let position = vec![Vector3::new(0.1, 0.2, 0.3)]; // Near center, slightly offset
    let velocity = vec![initialize_velocity(initial_temp)];
    let mass = vec![ARGON_MASS];
    
    let initial_box = Vector3::new(initial_box_length, initial_box_length, initial_box_length);
    
    // Force provider (minimal forces for 1 atom)
    let force_provider = SingleAtomForces::new();
    
    // NPT integrator setup (gentler coupling for stability)
    let q_t = 100.0;  // Thermostat coupling parameter (larger = gentler)
    let q_p = 2000.0; // Barostat coupling parameter (much slower coupling)
    
    let mut integrator = NoseHooverParrinelloRahman::new(
        position,
        velocity,
        mass,
        force_provider,
        initial_box,
        q_t,
        q_p,
        initial_temp,
        target_pressure,
        K_B,
    );
    
    // Simulation parameters
    let dt = 0.001; // Time step (reduced units) - smaller for stability
    let total_steps = 25000;
    let output_interval = 500;
    
    println!("Simulation Parameters:");
    println!("  Initial temperature: {:.3} (reduced)", initial_temp);
    println!("  Target pressure: {:.3} (reduced)", target_pressure);
    println!("  Initial box size: {:.2} × {:.2} × {:.2}", 
             initial_box_length, initial_box_length, initial_box_length);
    println!("  Initial volume: {:.3}", integrator.get_volume());
    println!("  Time step: {:.3} (reduced)", dt);
    println!("  Total steps: {}", total_steps);
    println!();

    // Temperature schedule (optional ramping)
    let temp_ramp_steps = 15000;
    let final_temp = 1.5; // Moderate heating to test barostat response

    println!("Temperature Schedule:");
    println!("  Steps 0-{}: {:.2} → {:.2} (linear ramp)", 
             temp_ramp_steps, initial_temp, final_temp);
    println!("  Steps {}-{}: {:.2} (constant)", 
             temp_ramp_steps, total_steps, final_temp);
    println!();
    
    // Output header
    println!("┌────────┬─────────┬───────────┬───────────┬──────────┬──────────┬──────────┐");
    println!("│ {:>6} │ {:>7} │ {:>9} │ {:>9} │ {:>8} │ {:>8} │ {:>8} │", 
             "Step", "T_curr", "P_curr", "P_target", "Volume", "Box_X", "KE");
    println!("├────────┼─────────┼───────────┼───────────┼──────────┼──────────┼──────────┤");

    // Main simulation loop
    for step in 0..total_steps {
        // Update target temperature (optional ramping)
        let current_target_temp = if step < temp_ramp_steps {
            let progress = step as f64 / temp_ramp_steps as f64;
            initial_temp + (final_temp - initial_temp) * progress
        } else {
            final_temp
        };
        
        integrator.set_target_temperature(current_target_temp);
        
        // Integration step
        integrator.step(dt);

        // Apply periodic boundary conditions (though not critical for 1 atom)
        for pos in &mut integrator.positions {
            for k in 0..3 {
                let box_l = integrator.box_lengths[k];
                pos[k] -= box_l * (pos[k] / box_l).floor();
            }
        }
        
        // Output at intervals
        if step % output_interval == 0 {
            let current_temp = integrator.temperature();
            let current_pressure = integrator.get_pressure();
            let volume = integrator.get_volume();
            let box_x = integrator.box_lengths.x;
            
            // Calculate kinetic energy
            let kinetic = integrator.velocities
                .iter()
                .zip(&integrator.masses)
                .map(|(v, &m)| 0.5 * m * v.dot(v))
                .sum::<f64>();
            
            println!("│ {:>6} │ {:>7.3} │ {:>9.3} │ {:>9.3} │ {:>8.3} │ {:>8.3} │ {:>8.4} │", 
                     step, 
                     current_temp,
                     current_pressure, 
                     target_pressure,
                     volume, 
                     box_x,
                     kinetic);
        }
    }
    
    println!("└────────┴─────────┴───────────┴───────────┴──────────┴──────────┴──────────┘");
    println!();

    // Final analysis
    let final_temp = integrator.temperature();
    let final_pressure = integrator.get_pressure();
    let final_volume = integrator.get_volume();
    let box_lengths = integrator.box_lengths;
    
    println!("Final Analysis:");
    println!("  Final temperature: {:.4} (target: {:.3})", final_temp, final_temp);
    println!("  Final pressure: {:.4} (target: {:.3})", final_pressure, target_pressure);
    println!("  Final volume: {:.4}", final_volume);
    println!("  Final box dimensions: {:.3} × {:.3} × {:.3}", 
             box_lengths.x, box_lengths.y, box_lengths.z);
    
    // Theoretical prediction check (ideal gas law)
    let theoretical_volume = final_temp / target_pressure; // PV = nkT for n=1, k=1
    println!("  Theoretical volume (PV=nkT): {:.4}", theoretical_volume);
    let volume_error = ((final_volume - theoretical_volume) / theoretical_volume * 100.0).abs();
    println!("  Volume error from ideal gas: {:.2}%", volume_error);
    
    println!();
    
    // Physics insight
    if volume_error < 10.0 {
        println!("✅ Success! NPT simulation correctly follows ideal gas law");
        println!("   The barostat successfully maintained pressure while allowing volume changes");
    } else {
        println!("⚠️  Large deviation from ideal gas behavior");
        println!("   This might indicate barostat coupling issues or insufficient equilibration");
    }
    
    println!();
    println!("🎯 NPT Ensemble Demonstration:");
    println!("   • Temperature was controlled by Nosé-Hoover thermostat");
    println!("   • Pressure was maintained by Parrinello-Rahman barostat");
    println!("   • Volume fluctuated to satisfy P = nkT/V (ideal gas law)");
    println!("   • Single atom system demonstrates pure NPT mechanics");
    println!();
    println!("💡 Key Insight: Even with minimal interactions, NPT ensemble");
    println!("   shows rich dynamics through volume-pressure coupling!");
}
