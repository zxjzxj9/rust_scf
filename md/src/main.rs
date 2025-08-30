mod run_md;
mod lj_pot;

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;
use run_md::{Integrator, NoseHooverVerlet};
use lj_pot::LennardJones;

fn main() {
    // fcc lattice parameters
    let n_cells = 4;
    let a = 5.0;
    let basis = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5, 0.5, 0.0),
        Vector3::new(0.5, 0.0, 0.5),
        Vector3::new(0.0, 0.5, 0.5),
    ];

    let box_lengths = Vector3::new(
        n_cells as f64 * a,
        n_cells as f64 * a,
        n_cells as f64 * a,
    );

    // build positions
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

    let n_atoms = positions.len();
    let mut velocities = vec![Vector3::zeros(); n_atoms];
    let temperature = 1.0;
    
    // thermalize velocities, using normal distribution
    let mut rng = rand::thread_rng();
    for i in 0..n_atoms {
        let v = Vector3::new(
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
        );
        velocities[i] = v * (temperature as f64).sqrt();
    }
    
    let masses = vec![1.0; n_atoms];

    // LJ parameters for argon in reduced units
    let lj = LennardJones::new(1.0, 1.0, box_lengths);
    let mut integrator = NoseHooverVerlet::new(
        positions,
        velocities,
        masses,
        lj,
        /* Q */ 100.0,
        temperature,
        /* k_B */ 1.0,
    );

    let dt = 0.001;
    let steps = 10_000;
    
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║               Molecular Dynamics Simulation                  ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Atoms: {:>6}                                               ║", n_atoms);
    println!("║ Box dimensions: [{:.2}, {:.2}, {:.2}]                       ║", 
             box_lengths.x, box_lengths.y, box_lengths.z);
    println!("║ Temperature: {:.2} K                                        ║", temperature);
    println!("║ Time step: {:.3} fs                                        ║", dt);
    println!("║ Total steps: {}                                         ║", steps);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Print table header with nice formatting
    println!("┌────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│ {:>6} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │", "Step", "Temp (K)", "KE", "PE", "Total E");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┤");

    for step in 0..steps {
        integrator.step(dt);

        // apply periodic boundary conditions
        for pos in &mut integrator.positions {
            for k in 0..3 {
                let box_l = box_lengths[k];
                pos[k] -= box_l * (pos[k] / box_l).floor();
            }
        }

        if step % 500 == 0 {
            let temp = integrator.temperature();
            let kinetic = integrator.velocities
                .iter()
                .zip(&integrator.masses)
                .map(|(v, &m)| 0.5 * m * v.dot(v))
                .sum::<f64>();
            let potential = integrator.provider.compute_potential_energy(&integrator.positions);
            let total_energy = kinetic + potential;
            
            println!("│ {:>6} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │", 
                     step, temp, kinetic, potential, total_energy);
        }
    }
    
    // Close the table with a nice border
    println!("└────────┴──────────┴──────────┴──────────┴──────────┘");
    println!();
    println!("✅ Simulation completed successfully!");
}
