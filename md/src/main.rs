mod run_md;
mod lj_pot;

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;
use run_md::{Integrator, NoseHooverVerlet};
use lj_pot::LennardJones;
use crate::run_md::ForceProvider;

fn main() {
    // fcc lattice parameters
    let n_cells = 4;
    let a = 5.0;
    let basis = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5, 0.0, 0.0),
        Vector3::new(0.0, 0.5, 0.0),
        Vector3::new(0.0, 0.0, 0.5),
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
    let box_lengths = Vector3::new(n_cells as f64 * a, n_cells as f64 * a, n_cells as f64 * a);

    // LJ parameters for argon in reduced units
    let lj = LennardJones::new(1.0, 1.0, box_lengths);
    let mut integrator = NoseHooverVerlet::new(
        positions,
        velocities,
        masses,
        lj,
        /* Q */ 1.0,
        temperature ,
        /* k_B */ 1.0,
    );

    let dt = 0.005;
    let steps = 10_000;
    for step in 0..steps {
        integrator.step(dt);

        // apply periodic boundary conditions
        for pos in &mut integrator.positions {
            for k in 0..3 {
                let L = a;
                pos[k] -= L * (pos[k] / L).floor();
            }
        }

        if step % 1 == 0 {
            let e0 = integrator.provider.compute_forces(&integrator.positions)[0];
            println!("Step {}: temperature={:?}", step, integrator.temperature());
            
        }
    }
}
