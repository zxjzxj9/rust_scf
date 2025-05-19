extern crate nalgebra as na;

use na::Vector3;

use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Initialize velocities for `n_atoms` with masses `masses`
/// at temperature `T` (with Boltzmann constant `k_B`),
/// subtracting the center‑of‑mass velocity.
///
/// Returns a Vec<Vector3<f64>> of velocities.
fn init_velocities(
    masses: &[f64],
    T: f64,
    k_B: f64,
) -> Vec<Vector3<f64>> {
    let mut rng = thread_rng();
    let n = masses.len();
    let mut velocities = Vec::with_capacity(n);

    // draw each component from N(0, sigma²)
    for &m in masses {
        let sigma = (k_B * T / m).sqrt();
        let dist = Normal::new(0.0, sigma).unwrap();
        let v = Vector3::new(
            dist.sample(&mut rng),
            dist.sample(&mut rng),
            dist.sample(&mut rng),
        );
        velocities.push(v);
    }

    // remove center‑of‑mass drift
    let v_cm = velocities.iter().sum::<Vector3<f64>>() / n as f64;
    velocities.iter_mut().for_each(|v| *v -= v_cm);

    velocities
}

/// Trait for providing forces given particle positions
pub trait ForceProvider {
    /// Given a slice of positions, returns a Vec of forces of equal length
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>>;
}

/// Trait for any integrator advancing the system in time
pub trait Integrator {
    /// Advance the state by dt
    fn step(&mut self, dt: f64);
    fn temperature(&self) -> f64;
}

/// Velocity Verlet integrator for molecular dynamics
pub struct VelocityVerlet<F: ForceProvider> {
    pub positions: Vec<Vector3<f64>>,
    pub velocities: Vec<Vector3<f64>>,
    pub masses: Vec<f64>,
    pub forces: Vec<Vector3<f64>>,
    provider: F,
}

impl<F: ForceProvider> VelocityVerlet<F> {
    /// Create a new VelocityVerlet integrator
    /// - `positions`, `velocities`, `masses` must have the same length
    /// - `provider` computes forces on the particles
    pub fn new(
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        masses: Vec<f64>,
        provider: F,
    ) -> Self {
        let forces = provider.compute_forces(&positions);
        VelocityVerlet { positions, velocities, masses, forces, provider }
    }
}

impl<F: ForceProvider> Integrator for VelocityVerlet<F> {
    fn step(&mut self, dt: f64) {
        let n = self.positions.len();
        // 1. Update positions
        for i in 0..n {
            let accel = self.forces[i] / self.masses[i];
            // print max accelerate abs component
            println!("max accel: {:?}", accel.abs());
            let disp = self.velocities[i] * dt + accel * (0.5 * dt * dt);
            self.positions[i] += disp;
        }

        // 2. Compute new forces
        let new_forces = self.provider.compute_forces(&self.positions);

        // 3. Update velocities
        for i in 0..n {
            let a_old = self.forces[i] / self.masses[i];
            let a_new = new_forces[i] / self.masses[i];
            let dv = (a_old + a_new) * (0.5 * dt);
            self.velocities[i] += dv;
        }

        // 4. Store new forces for next step
        self.forces = new_forces;
    }

    fn temperature(&self) -> f64 {
        let mut kinetic = 0.0;
        for i in 0..self.positions.len() {
            let v = &self.velocities[i];
            kinetic += 0.5 * self.masses[i] * v.dot(&v);
        }
        let dof = self.positions.len() * 3;
        let k_B = 1.0; // Boltzmann constant
        (2.0 * kinetic) / (dof as f64 * k_B)
    }
}

/// Nose–Hoover Velocity Verlet integrator for NVT molecular dynamics
pub struct NoseHooverVerlet<F: ForceProvider> {
    pub positions: Vec<Vector3<f64>>,
    pub velocities: Vec<Vector3<f64>>,
    pub masses: Vec<f64>,
    pub forces: Vec<Vector3<f64>>,
    pub(crate) provider: F,

    // Thermostat variables
    xi: f64,       // friction coefficient
    eta: f64,      // thermostat position
    Q: f64,        // thermostat mass parameter
    target_temp: f64,
    dof: usize,    // degrees of freedom for kinetic energy
    k_B: f64,      // Boltzmann constant
}


impl<F: ForceProvider> NoseHooverVerlet<F> {
    /// Create a new Nose-Hoover integrator in NVT ensemble
    /// - `Q`: thermostat mass parameter
    /// - `target_temp`: desired temperature
    /// - `k_B`: Boltzmann constant (set to 1.0 for reduced units)
    pub fn new(
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        masses: Vec<f64>,
        provider: F,
        Q: f64,
        target_temp: f64,
        k_B: f64,
    ) -> Self {
        let dof = positions.len() * 3;
        let forces = provider.compute_forces(&positions);



        NoseHooverVerlet { positions,
            velocities,
            masses,
            forces,
            provider,
            xi: 0.0,
            eta: 0.0,
            Q,
            target_temp,
            dof,
            k_B }
    }
}

impl<F: ForceProvider> Integrator for NoseHooverVerlet<F> {
    fn step(&mut self, dt: f64) {
        let half_dt = 0.5 * dt;
        let n = self.positions.len();

        // 1) Compute kinetic energy
        let mut kinetic = 0.0;
        for i in 0..n {
            let v = &self.velocities[i];
            kinetic += 0.5 * self.masses[i] * v.dot(&v);
        }
        let gkT = (self.dof as f64) * self.k_B * self.target_temp;

        // 2) Half-update thermostat xi
        let xi_dot = (2.0 * kinetic - gkT) / self.Q;
        self.xi += xi_dot * half_dt;

        // 3) Half-update velocities using current forces and xi
        for i in 0..n {
            let accel = self.forces[i] / self.masses[i];
            // Apply thermostat damping and acceleration in first half-step
            self.velocities[i] = self.velocities[i] * (1.0 - self.xi * half_dt) + accel * half_dt;
        }

        // 4) Full-update positions
        for i in 0..n {
            self.positions[i] += self.velocities[i] * dt;
        }

        // 5) Recompute forces
        let new_forces = self.provider.compute_forces(&self.positions);

        // 6) Compute new kinetic energy and update xi again
        kinetic = 0.0;
        for i in 0..n {
            let v = &self.velocities[i];
            kinetic += 0.5 * self.masses[i] * v.dot(&v);
        }
        let xi_dot_new = (2.0 * kinetic - gkT) / self.Q;
        self.xi += xi_dot_new * half_dt;

        // 7) Second half-update velocities
        for i in 0..n {
            let a_new = new_forces[i] / self.masses[i];
            // Correct implementation: apply thermostat damping and new acceleration
            self.velocities[i] = self.velocities[i] * (1.0 - self.xi * half_dt) + a_new * half_dt;
        }

        // 8) Update thermostat position eta
        self.eta += self.xi * dt;

        // 9) Store new forces
        self.forces = new_forces;
    }

    fn temperature(&self) -> f64 {
        let mut kinetic = 0.0;
        for i in 0..self.positions.len() {
            let v = &self.velocities[i];
            kinetic += 0.5 * self.masses[i] * v.dot(&v);
        }
        let dof = self.positions.len() * 3;
        let k_B = 1.0; // Boltzmann constant
        (2.0 * kinetic) / (dof as f64 * k_B)
    }
}

// Example usage:
//
// struct LennardJones;
// impl ForceProvider for LennardJones {
//     fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
//         // Compute pairwise LJ forces
//         unimplemented!()
//     }
// }
//
// let mut verlet = VelocityVerlet::new(
//     initial_positions.clone(),
//     initial_velocities.clone(),
//     masses.clone(),
//     LennardJones,
// );
// let mut nvt = NoseHooverVerlet::new(
//     initial_positions,
//     initial_velocities,
//     masses,
//     LennardJones,
//     /*Q=*/100.0,
//     /*target_temp=*/1.0,
//     /*k_B=*/1.0,
// );
// for _ in 0..steps {
//     nvt.step(dt);
// }
