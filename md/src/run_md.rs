use itertools::izip;
use nalgebra::Vector3;

pub trait ForceProvider {
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>>;
}

pub trait Integrator {
    /// Advance the system by dt
    fn step(&mut self, dt: f64);

    /// Compute the instantaneous temperature
    fn temperature(&self) -> f64;
}

pub struct NoseHooverVerlet<F: ForceProvider> {
    pub positions: Vec<Vector3<f64>>,
    pub velocities: Vec<Vector3<f64>>,
    pub masses: Vec<f64>,
    inv_masses: Vec<f64>,
    pub forces: Vec<Vector3<f64>>,
    pub(crate) provider: F,
    xi: f64,
    eta: f64,
    Q: f64,
    target_temp: f64,
    dof: usize,
    k_B: f64,
    gkT: f64,
}

impl<F: ForceProvider> NoseHooverVerlet<F> {
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
        let gkT = dof as f64 * k_B * target_temp;
        let forces = provider.compute_forces(&positions);
        let inv_masses = masses.iter().map(|&m| 1.0 / m).collect();
        NoseHooverVerlet {
            positions,
            velocities,
            masses,
            inv_masses,
            forces,
            provider,
            xi: 0.0,
            eta: 0.0,
            Q,
            target_temp,
            dof,
            k_B,
            gkT,
        }
    }

    #[inline]
    fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .zip(&self.masses)
            .map(|(v, &m)| 0.5 * m * v.dot(v))
            .sum()
    }
}

impl<F: ForceProvider> Integrator for NoseHooverVerlet<F> {
    fn step(&mut self, dt: f64) {
        let half_dt = 0.5 * dt;
        // thermostat half‐step
        let kin = self.kinetic_energy();
        self.xi += (2.0 * kin - self.gkT) / self.Q * half_dt;

        // first half‐step velocities
        for (v, &f, &inv_m) in izip!(&mut self.velocities, &self.forces, &self.inv_masses)
        {
            *v = *v * (1.0 - self.xi * half_dt) + (f * inv_m) * half_dt;
        }

        // full‐step positions
        for (pos, &v) in self.positions.iter_mut().zip(&self.velocities) {
            *pos += v * dt;
        }

        // recompute forces
        let new_forces = self.provider.compute_forces(&self.positions);

        // thermostat half‐step (new)
        let kin = self.kinetic_energy();
        self.xi += (2.0 * kin - self.gkT) / self.Q * half_dt;

        // second half‐step velocities
        for (v, &f_new, &inv_m) in izip!(&mut self.velocities, &new_forces, &self.inv_masses)
        {
            *v = *v * (1.0 - self.xi * half_dt) + (f_new * inv_m) * half_dt;
        }

        self.forces = new_forces;
        self.eta += self.xi * dt;
    }

    fn temperature(&self) -> f64 {
        2.0 * self.kinetic_energy() / (self.dof as f64 * self.k_B)
    }
}