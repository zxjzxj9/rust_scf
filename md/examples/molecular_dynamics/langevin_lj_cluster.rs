// Langevin dynamics example for a Lennard-Jones cluster
//
// This example shows how to couple a finite cluster of LJ particles to a
// Langevin thermostat. The dynamics keeps the cluster near a target
// temperature through friction + stochastic kicks while allowing the
// structure to explore its inherent-energy landscape.

use md::{ForceProvider, Integrator, LennardJones};
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::StandardNormal;

const K_B: f64 = 1.0; // Reduced units

struct LangevinDynamics<F: ForceProvider> {
    pub positions: Vec<Vector3<f64>>,
    pub velocities: Vec<Vector3<f64>>,
    pub masses: Vec<f64>,
    inv_masses: Vec<f64>,
    pub provider: F,
    pub forces: Vec<Vector3<f64>>,
    gamma: f64,
    target_temp: f64,
    k_b: f64,
    dof: usize,
    rng: StdRng,
}

impl<F: ForceProvider> LangevinDynamics<F> {
    fn new(
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        masses: Vec<f64>,
        provider: F,
        gamma: f64,
        target_temp: f64,
        k_b: f64,
        rng_seed: u64,
    ) -> Self {
        let forces = provider.compute_forces(&positions);
        let inv_masses = masses.iter().map(|&m| 1.0 / m).collect::<Vec<_>>();
        let dof = positions.len() * 3;
        Self {
            positions,
            velocities,
            masses,
            inv_masses,
            provider,
            forces,
            gamma,
            target_temp,
            k_b,
            dof,
            rng: StdRng::seed_from_u64(rng_seed),
        }
    }

    fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .zip(&self.masses)
            .map(|(v, &m)| 0.5 * m * v.dot(v))
            .sum()
    }

    fn set_target_temperature(&mut self, temp: f64) {
        self.target_temp = temp;
    }
}

impl<F: ForceProvider> Integrator for LangevinDynamics<F> {
    fn step(&mut self, dt: f64) {
        let sqrt_dt = dt.sqrt();
        for i in 0..self.positions.len() {
            let inv_m = self.inv_masses[i];
            let deterministic = self.forces[i] * inv_m - self.velocities[i] * self.gamma;
            let noise_scale = (2.0 * self.gamma * self.k_b * self.target_temp * inv_m).sqrt();
            let random_vec = Vector3::new(
                self.rng.sample(StandardNormal),
                self.rng.sample(StandardNormal),
                self.rng.sample(StandardNormal),
            );
            let stochastic = random_vec * (noise_scale * sqrt_dt);

            self.velocities[i] += deterministic * dt + stochastic;
            self.positions[i] += self.velocities[i] * dt;
        }
        self.forces = self.provider.compute_forces(&self.positions);
    }

    fn temperature(&self) -> f64 {
        2.0 * self.kinetic_energy() / (self.dof as f64 * self.k_b)
    }
}

fn create_compact_cluster(n_atoms: usize, spacing: f64) -> Vec<Vector3<f64>> {
    let mut candidates = Vec::new();
    for x in -4..=4 {
        for y in -4..=4 {
            for z in -4..=4 {
                let pos = Vector3::new(x as f64, y as f64, z as f64) * spacing;
                candidates.push((pos.norm(), pos));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    candidates
        .into_iter()
        .take(n_atoms)
        .map(|(_, pos)| pos)
        .collect()
}

fn shift_cluster_to_box(
    mut positions: Vec<Vector3<f64>>,
    box_lengths: Vector3<f64>,
) -> Vec<Vector3<f64>> {
    let com: Vector3<f64> = positions.iter().copied().sum::<Vector3<f64>>() / positions.len() as f64;
    let target_center = box_lengths * 0.5;
    for pos in &mut positions {
        *pos += target_center - com;
    }
    positions
}

fn initialize_velocities(
    n_atoms: usize,
    temperature: f64,
    mass: f64,
    rng: &mut StdRng,
) -> Vec<Vector3<f64>> {
    let sigma = (temperature / mass).sqrt();
    let mut velocities = Vec::with_capacity(n_atoms);
    for _ in 0..n_atoms {
        let v = Vector3::new(
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
        ) * sigma;
        velocities.push(v);
    }
    remove_linear_momentum(&mut velocities);
    velocities
}

fn remove_linear_momentum(velocities: &mut [Vector3<f64>]) {
    if velocities.is_empty() {
        return;
    }
    let v_cm: Vector3<f64> = velocities.iter().copied().sum::<Vector3<f64>>() / velocities.len() as f64;
    for v in velocities {
        *v -= v_cm;
    }
}

fn recenter_cluster(positions: &mut [Vector3<f64>], box_lengths: Vector3<f64>) {
    if positions.is_empty() {
        return;
    }
    let com: Vector3<f64> = positions.iter().copied().sum::<Vector3<f64>>() / positions.len() as f64;
    let shift = box_lengths * 0.5 - com;
    for pos in positions {
        *pos += shift;
        for dim in 0..3 {
            let extent = box_lengths[dim];
            pos[dim] = pos[dim].rem_euclid(extent);
        }
    }
}

fn cluster_metrics(positions: &[Vector3<f64>], center: Vector3<f64>) -> (f64, f64) {
    if positions.is_empty() {
        return (0.0, 0.0);
    }
    let mut avg = 0.0;
    let mut max_r = 0.0;
    for pos in positions {
        let r = (*pos - center).norm();
        avg += r;
        if r > max_r {
            max_r = r;
        }
    }
    (avg / positions.len() as f64, max_r)
}

fn main() {
    println!("===================================================");
    println!("     Langevin Dynamics for an LJ Cluster (NVT)     ");
    println!("===================================================\n");

    let n_atoms = 38;
    let target_temp = 0.6; // Reduced temperature
    let gamma = 1.5; // Friction coefficient (1/tau)
    let dt = 0.002;
    let total_steps = 25_000;
    let sample_interval = 250;
    let mass = 1.0;
    let box_lengths = Vector3::new(20.0, 20.0, 20.0);

    let raw_positions = create_compact_cluster(n_atoms, 1.1);
    let positions = shift_cluster_to_box(raw_positions, box_lengths);
    let mut vel_rng = StdRng::seed_from_u64(2025);
    let velocities = initialize_velocities(n_atoms, target_temp, mass, &mut vel_rng);
    let masses = vec![mass; n_atoms];
    let lj = LennardJones::new(1.0, 1.0, box_lengths);

    let mut integrator = LangevinDynamics::new(
        positions,
        velocities,
        masses,
        lj,
        gamma,
        target_temp,
        K_B,
        4242,
    );

    recenter_cluster(&mut integrator.positions, box_lengths);

    println!("Atoms             : {}", n_atoms);
    println!("Target T*         : {:.3}", target_temp);
    println!("Time step         : {:.4}", dt);
    println!("Friction gamma    : {:.2} 1/tau", gamma);
    println!("Box (sigma units) : {:.1} x {:.1} x {:.1}", box_lengths.x, box_lengths.y, box_lengths.z);
    println!("\nStep   |   T*   |   K/N  |   U/N  |  Avg R |  Max R |  E/N");
    println!("-----------------------------------------------------------");

    let mut temp_accum = 0.0;
    let mut e_accum = 0.0;
    let mut samples = 0usize;
    let center = box_lengths * 0.5;

    for step in 0..=total_steps {
        integrator.step(dt);
        recenter_cluster(&mut integrator.positions, box_lengths);
        remove_linear_momentum(&mut integrator.velocities);

        if step % sample_interval == 0 {
            let kinetic = integrator.kinetic_energy();
            let potential = integrator.provider.compute_potential_energy(&integrator.positions);
            let temp = integrator.temperature();
            let (avg_r, max_r) = cluster_metrics(&integrator.positions, center);

            temp_accum += temp;
            e_accum += kinetic + potential;
            samples += 1;

            println!(
                "{:5} | {:6.3} | {:6.3} | {:6.3} | {:6.3} | {:6.3} | {:6.3}",
                step,
                temp,
                kinetic / n_atoms as f64,
                potential / n_atoms as f64,
                avg_r,
                max_r,
                (kinetic + potential) / n_atoms as f64
            );
        }

        if step == total_steps / 2 {
            integrator.set_target_temperature(0.5 * target_temp + 0.5 * (target_temp + 0.2));
        }
    }

    println!("\nSimulation completed.");
    if samples > 0 {
        println!("Average T*  : {:.3}", temp_accum / samples as f64);
        println!("Average E/N : {:.3}", e_accum / samples as f64 / n_atoms as f64);
    }
    println!("Data columns show instantaneous reduced values every {} steps.", sample_interval);
}

