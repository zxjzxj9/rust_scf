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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Mock force provider for testing
    struct MockForceProvider {
        forces: Vec<Vector3<f64>>,
    }

    impl MockForceProvider {
        fn new(forces: Vec<Vector3<f64>>) -> Self {
            MockForceProvider { forces }
        }
    }

    impl ForceProvider for MockForceProvider {
        fn compute_forces(&self, _positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
            self.forces.clone()
        }
    }

    #[test]
    fn test_nose_hoover_verlet_new() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        let mock_forces = vec![Vector3::new(0.5, 0.0, 0.0), Vector3::new(-0.5, 0.0, 0.0)];
        let provider = MockForceProvider::new(mock_forces);
        let q = 100.0;
        let target_temp = 300.0;
        let k_b = 1.0;

        let integrator = NoseHooverVerlet::new(
            positions.clone(),
            velocities.clone(),
            masses.clone(),
            provider,
            q,
            target_temp,
            k_b,
        );

        assert_eq!(integrator.positions, positions);
        assert_eq!(integrator.velocities, velocities);
        assert_eq!(integrator.masses, masses);
        assert_eq!(integrator.q, q);
        assert_eq!(integrator.target_temp, target_temp);
        assert_eq!(integrator.k_b, k_b);
        assert_eq!(integrator.dof, 6); // 2 atoms * 3 dimensions
        assert_eq!(integrator.xi, 0.0);
        assert_eq!(integrator.eta, 0.0);
    }

    #[test]
    fn test_kinetic_energy_calculation() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 4.0, 0.0)];
        let masses = vec![1.0, 2.0];
        let mock_forces = vec![Vector3::zeros(), Vector3::zeros()];
        let provider = MockForceProvider::new(mock_forces);

        let integrator = NoseHooverVerlet::new(
            positions,
            velocities,
            masses,
            provider,
            100.0,
            300.0,
            1.0,
        );

        // KE = 0.5 * m1 * v1^2 + 0.5 * m2 * v2^2
        // = 0.5 * 1.0 * (2^2) + 0.5 * 2.0 * (3^2 + 4^2)
        // = 0.5 * 1.0 * 4 + 0.5 * 2.0 * 25
        // = 2.0 + 25.0 = 27.0
        let expected_ke = 27.0;
        assert_relative_eq!(integrator.kinetic_energy(), expected_ke, epsilon = 1e-10);
    }

    #[test]
    fn test_temperature_calculation() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(2.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        let mock_forces = vec![Vector3::zeros(), Vector3::zeros()];
        let provider = MockForceProvider::new(mock_forces);
        let k_b = 1.0;

        let integrator = NoseHooverVerlet::new(
            positions,
            velocities,
            masses,
            provider,
            100.0,
            300.0,
            k_b,
        );

        // KE = 0.5 * 1.0 * 4.0 = 2.0
        // T = 2 * KE / (dof * k_b) = 2 * 2.0 / (6 * 1.0) = 4.0 / 6.0 = 2/3
        let expected_temp = 2.0 / 3.0;
        assert_relative_eq!(integrator.temperature(), expected_temp, epsilon = 1e-10);
    }

    #[test]
    fn test_integration_step_conservation() {
        // Test that a single step preserves basic properties
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        
        // Zero forces for conservation test
        let mock_forces = vec![Vector3::zeros(), Vector3::zeros()];
        let provider = MockForceProvider::new(mock_forces);

        let mut integrator = NoseHooverVerlet::new(
            positions.clone(),
            velocities.clone(),
            masses,
            provider,
            100.0,
            300.0,
            1.0,
        );

        let initial_positions = integrator.positions.clone();
        let dt = 0.001;

        integrator.step(dt);

        // Check that positions have changed (integration is working)
        assert!((integrator.positions[0] - initial_positions[0]).norm() > 0.0);
        assert!((integrator.positions[1] - initial_positions[1]).norm() > 0.0);

        // Check that the number of particles is preserved
        assert_eq!(integrator.positions.len(), 2);
        assert_eq!(integrator.velocities.len(), 2);
    }

    #[test]
    fn test_step_with_constant_force() {
        // Test integration with a constant force
        let positions = vec![Vector3::new(0.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(0.0, 0.0, 0.0)];
        let masses = vec![1.0];
        
        // Constant force in x direction
        let constant_force = vec![Vector3::new(1.0, 0.0, 0.0)];
        let provider = MockForceProvider::new(constant_force);

        let mut integrator = NoseHooverVerlet::new(
            positions,
            velocities,
            masses,
            provider,
            1e10, // Large Q to minimize thermostat effect
            1e-10, // Very low temperature to minimize thermostat effect
            1.0,
        );

        let initial_position = integrator.positions[0];
        let dt = 0.01;

        // Take several steps
        for _ in 0..10 {
            integrator.step(dt);
        }

        // With constant force F=1, mass=1, we expect acceleration a=1
        // Position should have increased
        assert!(integrator.positions[0].x > initial_position.x);
        
        // Velocity should be positive (accelerating in +x direction)
        assert!(integrator.velocities[0].x > 0.0);
    }

    #[test]
    fn test_thermostat_temperature_control() {
        // Test that the thermostat tries to maintain target temperature
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        
        // Start with high velocity (high temperature)
        let velocities = vec![
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(-10.0, 0.0, 0.0),
            Vector3::new(0.0, 10.0, 0.0),
        ];
        let masses = vec![1.0, 1.0, 1.0];
        
        let mock_forces = vec![Vector3::zeros(), Vector3::zeros(), Vector3::zeros()];
        let provider = MockForceProvider::new(mock_forces);

        let mut integrator = NoseHooverVerlet::new(
            positions,
            velocities,
            masses,
            provider,
            10.0, // Moderate Q for reasonable thermostat response
            1.0,  // Target temperature
            1.0,
        );

        let initial_temp = integrator.temperature();
        let dt = 0.001;

        // Run for many steps to see thermostat effect
        for _ in 0..1000 {
            integrator.step(dt);
        }

        let final_temp = integrator.temperature();
        
        // The thermostat should reduce the temperature from its initial high value
        assert!(final_temp < initial_temp);
        
        // Temperature should be closer to target (1.0) than initial
        let target_temp = 1.0;
        assert!((final_temp - target_temp).abs() < (initial_temp - target_temp).abs());
    }

    #[test]
    fn test_xi_clamping() {
        // Test that xi is properly clamped to prevent runaway
        let positions = vec![Vector3::new(0.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(100.0, 0.0, 0.0)]; // Very high velocity
        let masses = vec![1.0];
        
        let mock_forces = vec![Vector3::zeros()];
        let provider = MockForceProvider::new(mock_forces);

        let mut integrator = NoseHooverVerlet::new(
            positions,
            velocities,
            masses,
            provider,
            0.01, // Very small Q to make thermostat aggressive
            0.1,  // Low target temperature
            1.0,
        );

        let dt = 0.1;
        
        // Take a few steps with aggressive thermostat
        for _ in 0..10 {
            integrator.step(dt);
            
            // xi should always be within bounds
            assert!(integrator.xi >= -10.0);
            assert!(integrator.xi <= 10.0);
        }
    }

    #[test]
    fn test_force_provider_interface() {
        // Test that our integrator works with the ForceProvider trait
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        
        let test_forces = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(-1.0, -2.0, -3.0)];
        let provider = MockForceProvider::new(test_forces.clone());

        let integrator = NoseHooverVerlet::new(
            positions,
            velocities,
            masses,
            provider,
            100.0,
            300.0,
            1.0,
        );

        // Check that forces were computed correctly during initialization
        assert_eq!(integrator.forces, test_forces);
    }

    #[test]
    fn test_degrees_of_freedom_calculation() {
        // Test different numbers of atoms
        let single_atom = vec![Vector3::new(0.0, 0.0, 0.0)];
        let velocities_1 = vec![Vector3::new(0.0, 0.0, 0.0)];
        let masses_1 = vec![1.0];
        let provider_1 = MockForceProvider::new(vec![Vector3::zeros()]);

        let integrator_1 = NoseHooverVerlet::new(
            single_atom,
            velocities_1,
            masses_1,
            provider_1,
            100.0,
            300.0,
            1.0,
        );
        assert_eq!(integrator_1.dof, 3); // 1 atom * 3 dimensions

        let three_atoms = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let velocities_3 = vec![Vector3::zeros(); 3];
        let masses_3 = vec![1.0; 3];
        let provider_3 = MockForceProvider::new(vec![Vector3::zeros(); 3]);

        let integrator_3 = NoseHooverVerlet::new(
            three_atoms,
            velocities_3,
            masses_3,
            provider_3,
            100.0,
            300.0,
            1.0,
        );
        assert_eq!(integrator_3.dof, 9); // 3 atoms * 3 dimensions
    }
}