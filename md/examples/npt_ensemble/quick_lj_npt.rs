// Quick LJ Cluster NPT Test
//
// A minimal example for rapid testing of NPT ensemble with LJ clusters.
// Smaller system, shorter runtime, focused output.

use md::{ForceProvider, LennardJones};
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;

const K_B: f64 = 1.0;
const MASS: f64 = 1.0;

/// Simple NPT integrator for LJ clusters
struct SimpleNPT {
    pub positions: Vec<Vector3<f64>>,
    pub velocities: Vec<Vector3<f64>>,
    pub lj: LennardJones,
    pub box_size: f64,
    xi: f64,    // Thermostat variable
    v_box: f64, // Box velocity
    target_temp: f64,
    target_pres: f64,
    q_t: f64,
    q_p: f64,
}

impl SimpleNPT {
    fn new(
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        box_size: f64,
        target_temp: f64,
        target_pres: f64,
    ) -> Self {
        let box_vec = Vector3::new(box_size, box_size, box_size);
        let lj = LennardJones::new(1.0, 1.0, box_vec);

        Self {
            positions,
            velocities,
            lj,
            box_size,
            xi: 0.0,
            v_box: 0.0,
            target_temp,
            target_pres,
            q_t: 50.0,
            q_p: 1000.0,
        }
    }

    fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .map(|v| 0.5 * MASS * v.norm_squared())
            .sum()
    }

    fn temperature(&self) -> f64 {
        let n = self.positions.len();
        2.0 * self.kinetic_energy() / (3.0 * n as f64 * K_B)
    }

    fn pressure(&self) -> f64 {
        let ke = self.kinetic_energy();
        self.lj.compute_pressure(&self.positions, ke)
    }

    fn step(&mut self, dt: f64) {
        let half_dt = 0.5 * dt;
        let n = self.positions.len();

        // Current state
        let ke = self.kinetic_energy();
        let p = self.pressure();
        let v = self.box_size.powi(3);

        // Update thermostat
        let target_ke = 1.5 * n as f64 * K_B * self.target_temp;
        self.xi += ((2.0 * ke - 2.0 * target_ke) / self.q_t) * half_dt;
        self.xi = self.xi.clamp(-5.0, 5.0);

        // Update barostat
        let p_error = p - self.target_pres;
        self.v_box += (p_error * v / self.q_p) * half_dt;
        self.v_box = self.v_box.clamp(-0.05, 0.05);

        // Update box size
        let old_box = self.box_size;
        self.box_size *= 1.0 + self.v_box * dt;
        self.box_size = self.box_size.clamp(3.0, 50.0);

        // Scale positions
        let scale = self.box_size / old_box;
        for pos in &mut self.positions {
            *pos *= scale;
        }

        // Update LJ with new box
        let box_vec = Vector3::new(self.box_size, self.box_size, self.box_size);
        self.lj = LennardJones::new(1.0, 1.0, box_vec);

        // Compute forces
        let forces = self.lj.compute_forces(&self.positions);

        // Update velocities (half step)
        let damping = 1.0 / (1.0 + (self.xi + 3.0 * self.v_box) * half_dt);
        for i in 0..n {
            self.velocities[i] += forces[i] / MASS * half_dt;
            self.velocities[i] *= damping;
        }

        // Update positions
        for i in 0..n {
            self.positions[i] += self.velocities[i] * dt;
        }

        // PBC
        for pos in &mut self.positions {
            for k in 0..3 {
                pos[k] -= self.box_size * (pos[k] / self.box_size).floor();
            }
        }

        // Recompute forces
        let forces = self.lj.compute_forces(&self.positions);

        // Update velocities (second half)
        for i in 0..n {
            self.velocities[i] += forces[i] / MASS * half_dt;
            self.velocities[i] *= damping;
        }

        // Update thermostat (second half)
        let ke = self.kinetic_energy();
        self.xi += ((2.0 * ke - 2.0 * target_ke) / self.q_t) * half_dt;
        self.xi = self.xi.clamp(-5.0, 5.0);

        // Update barostat (second half)
        let p = self.pressure();
        let v = self.box_size.powi(3);
        let p_error = p - self.target_pres;
        self.v_box += (p_error * v / self.q_p) * half_dt;
        self.v_box = self.v_box.clamp(-0.05, 0.05);
    }
}

fn init_velocities(n: usize, temp: f64) -> Vec<Vector3<f64>> {
    let mut rng = rand::thread_rng();
    let mut vels: Vec<Vector3<f64>> = (0..n)
        .map(|_| {
            Vector3::new(
                rng.sample(StandardNormal),
                rng.sample(StandardNormal),
                rng.sample(StandardNormal),
            ) * temp.sqrt()
        })
        .collect();

    // Remove COM motion
    let v_cm: Vector3<f64> = vels.iter().sum::<Vector3<f64>>() / n as f64;
    for v in &mut vels {
        *v -= v_cm;
    }

    // Rescale to exact temperature
    let ke: f64 = vels.iter().map(|v| 0.5 * MASS * v.norm_squared()).sum();
    let t_curr = 2.0 * ke / (3.0 * n as f64 * K_B);
    let scale = (temp / t_curr).sqrt();
    for v in &mut vels {
        *v *= scale;
    }

    vels
}

fn create_cluster(n_side: usize) -> Vec<Vector3<f64>> {
    let mut pos = Vec::new();
    let spacing = 1.3;

    for i in 0..n_side {
        for j in 0..n_side {
            for k in 0..n_side {
                pos.push(Vector3::new(
                    i as f64 * spacing,
                    j as f64 * spacing,
                    k as f64 * spacing,
                ));
            }
        }
    }

    pos
}

fn main() {
    println!("═══════════════════════════════════════════════");
    println!("       Quick LJ Cluster NPT Test");
    println!("═══════════════════════════════════════════════\n");

    // Small system for quick testing
    let n_side = 2; // 2x2x2 = 8 atoms
    let positions = create_cluster(n_side);
    let n = positions.len();

    let temp = 1.0;
    let pres = 0.2;
    let velocities = init_velocities(n, temp);

    let box_size = 6.0;
    let mut npt = SimpleNPT::new(positions, velocities, box_size, temp, pres);

    println!("System: {} atoms", n);
    println!("Target T: {:.2}, P: {:.2}\n", temp, pres);

    let dt = 0.005;
    let steps = 5000;
    let interval = 250;

    println!("Step    T      P      Box    PE/atom");
    println!("─────────────────────────────────────");

    for step in 0..=steps {
        npt.step(dt);

        if step % interval == 0 {
            let t = npt.temperature();
            let p = npt.pressure();
            let pe = npt.lj.compute_potential_energy(&npt.positions) / n as f64;

            println!(
                "{:5}  {:5.2}  {:5.2}  {:5.2}  {:7.3}",
                step, t, p, npt.box_size, pe
            );
        }
    }

    println!("\n✅ Quick test complete!");
    println!("   Final T: {:.3} (target: {:.2})", npt.temperature(), temp);
    println!("   Final P: {:.3} (target: {:.2})", npt.pressure(), pres);
    println!("   Final box: {:.2}", npt.box_size);
}
