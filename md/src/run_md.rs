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
    q: f64,
    target_temp: f64,
    dof: usize,
    k_b: f64,
    gk_t: f64,
}

impl<F: ForceProvider> NoseHooverVerlet<F> {
    pub fn new(
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        masses: Vec<f64>,
        provider: F,
        q: f64,
        target_temp: f64,
        k_b: f64,
    ) -> Self {
        let dof = positions.len() * 3;
        let gk_t = dof as f64 * k_b * target_temp;
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
            q,
            target_temp,
            dof,
            k_b,
            gk_t,
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
        
        // Update xi (thermostat variable) first half-step
        let kin = self.kinetic_energy();
        let xi_dot = (2.0 * kin - self.gk_t) / self.q;
        self.xi += xi_dot * half_dt;
        
        // Limit xi to prevent runaway
        self.xi = self.xi.clamp(-10.0, 10.0);

        // Update velocities (first half-step)
        for (v, &f, &inv_m) in izip!(&mut self.velocities, &self.forces, &self.inv_masses) {
            let scaling_factor = 1.0 / (1.0 + self.xi * half_dt);
            let force_contrib = f * inv_m * half_dt;
            *v = (*v + force_contrib) * scaling_factor;
        }

        // Update positions (full step)
        for (pos, &v) in self.positions.iter_mut().zip(&self.velocities) {
            *pos += v * dt;
        }

        // Recompute forces
        self.forces = self.provider.compute_forces(&self.positions);

        // Update velocities (second half-step)
        for (v, &f_new, &inv_m) in izip!(&mut self.velocities, &self.forces, &self.inv_masses) {
            let force_contrib = f_new * inv_m * half_dt;
            *v += force_contrib;
            let scaling_factor = 1.0 / (1.0 + self.xi * half_dt);
            *v *= scaling_factor;
        }

        // Update xi (thermostat variable) second half-step
        let kin = self.kinetic_energy();
        let xi_dot = (2.0 * kin - self.gk_t) / self.q;
        self.xi += xi_dot * half_dt;
        
        // Limit xi to prevent runaway
        self.xi = self.xi.clamp(-10.0, 10.0);

        // Update eta (extended coordinate)
        self.eta += self.xi * dt;
    }

    fn temperature(&self) -> f64 {
        2.0 * self.kinetic_energy() / (self.dof as f64 * self.k_b)
    }
}