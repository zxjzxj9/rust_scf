// file: `md/src/lj_pot.rs`
use crate::run_md::ForceProvider;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

/// Lennard-Jones potential with support for arbitrary lattice structures.
///
/// The lattice is represented by a 3x3 matrix where each column is a lattice vector:
/// - Column 0: a-vector (first lattice vector)
/// - Column 1: b-vector (second lattice vector)
/// - Column 2: c-vector (third lattice vector)
///
/// For orthogonal boxes, this is a diagonal matrix with box lengths on the diagonal.
#[derive(Debug, Clone)]
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
    /// Lattice matrix: columns are lattice vectors [a, b, c]
    pub lattice: Matrix3<f64>,
    /// Inverse lattice matrix for minimum image convention
    pub lattice_inv: Matrix3<f64>,
    /// Legacy box_lengths for backward compatibility (deprecated)
    #[deprecated(note = "Use lattice matrix instead")]
    pub box_lengths: Vector3<f64>,
}

impl LennardJones {
    /// Create a new Lennard-Jones potential with an orthogonal box (backward compatible).
    ///
    /// # Arguments
    /// * `epsilon` - Energy parameter
    /// * `sigma` - Length parameter
    /// * `box_lengths` - Box dimensions [Lx, Ly, Lz]
    #[allow(deprecated)]
    pub fn new(epsilon: f64, sigma: f64, box_lengths: Vector3<f64>) -> Self {
        let lattice = Matrix3::from_diagonal(&box_lengths);
        let lattice_inv = Matrix3::from_diagonal(&Vector3::new(
            1.0 / box_lengths.x,
            1.0 / box_lengths.y,
            1.0 / box_lengths.z,
        ));
        LennardJones {
            epsilon,
            sigma,
            lattice,
            lattice_inv,
            box_lengths,
        }
    }

    /// Create a new Lennard-Jones potential with an arbitrary lattice.
    ///
    /// # Arguments
    /// * `epsilon` - Energy parameter
    /// * `sigma` - Length parameter
    /// * `lattice` - 3x3 matrix where columns are lattice vectors [a, b, c]
    ///
    /// # Example
    /// ```
    /// use nalgebra::{Vector3, Matrix3};
    /// use md::lj_pot::LennardJones;
    ///
    /// // Triclinic box with 60-degree angles
    /// let a = Vector3::new(10.0, 0.0, 0.0);
    /// let b = Vector3::new(5.0, 8.66, 0.0);  // 60 degrees from a
    /// let c = Vector3::new(0.0, 0.0, 10.0);
    /// let lattice = Matrix3::from_columns(&[a, b, c]);
    ///
    /// let lj = LennardJones::from_lattice(1.0, 1.0, lattice);
    /// ```
    #[allow(deprecated)]
    pub fn from_lattice(epsilon: f64, sigma: f64, lattice: Matrix3<f64>) -> Self {
        let lattice_inv = lattice
            .try_inverse()
            .expect("Lattice matrix must be invertible (non-zero volume)");

        // For backward compatibility, extract diagonal as box_lengths
        let box_lengths = Vector3::new(
            lattice.column(0).norm(),
            lattice.column(1).norm(),
            lattice.column(2).norm(),
        );

        LennardJones {
            epsilon,
            sigma,
            lattice,
            lattice_inv,
            box_lengths,
        }
    }

    /// Update the lattice during simulation (e.g., for NPT ensemble).
    ///
    /// # Arguments
    /// * `new_lattice` - New 3x3 lattice matrix
    #[allow(deprecated)]
    pub fn set_lattice(&mut self, new_lattice: Matrix3<f64>) {
        self.lattice = new_lattice;
        self.lattice_inv = new_lattice
            .try_inverse()
            .expect("Lattice matrix must be invertible (non-zero volume)");

        // Update box_lengths for backward compatibility
        self.box_lengths = Vector3::new(
            new_lattice.column(0).norm(),
            new_lattice.column(1).norm(),
            new_lattice.column(2).norm(),
        );
    }

    /// Apply minimum image convention for arbitrary lattice.
    ///
    /// This works by:
    /// 1. Converting distance vector to fractional coordinates
    /// 2. Wrapping fractional coordinates to [-0.5, 0.5)
    /// 3. Converting back to Cartesian coordinates
    pub fn minimum_image(&self, d: Vector3<f64>) -> Vector3<f64> {
        // Convert to fractional coordinates
        let frac = self.lattice_inv * d;

        // Wrap to [-0.5, 0.5)
        let wrapped = Vector3::new(
            frac.x - frac.x.round(),
            frac.y - frac.y.round(),
            frac.z - frac.z.round(),
        );

        // Convert back to Cartesian
        self.lattice * wrapped
    }

    /// Get the volume of the simulation box
    pub fn volume(&self) -> f64 {
        self.lattice.determinant().abs()
    }

    pub fn lj_potential(&self, r2: f64) -> f64 {
        let inv_r2 = self.sigma * self.sigma / r2;
        let inv_r6 = inv_r2 * inv_r2 * inv_r2;
        4.0 * self.epsilon * (inv_r6 * inv_r6 - inv_r6)
    }

    pub fn compute_potential_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        let n = positions.len();
        let sigma2 = self.sigma * self.sigma;

        // cutoff at 2.5 sigma
        let r_cut = 2.5 * self.sigma;
        let r_cut2 = r_cut * r_cut;

        // prevent singularity for r^2 < min_r2
        let min_r2 = 0.01 * sigma2;

        // Parallelize over the outer loop
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut local_potential = 0.0;
                for j in (i + 1)..n {
                    let rij = self.minimum_image(positions[i] - positions[j]);
                    let mut r2 = rij.norm_squared();

                    // skip outside cutoff, clamp below min_r2
                    if r2 > r_cut2 {
                        continue;
                    }
                    if r2 < min_r2 {
                        r2 = min_r2;
                    }

                    local_potential += self.lj_potential(r2);
                }
                local_potential
            })
            .sum()
    }
}

impl ForceProvider for LennardJones {
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = positions.len();
        let sigma2 = self.sigma * self.sigma;

        // cutoff at 2.5 sigma
        let r_cut = 2.5 * self.sigma;
        let r_cut2 = r_cut * r_cut;

        // prevent singularity for r^2 < min_r2
        let min_r2 = 0.01 * sigma2;

        // Parallelize over the outer loop, collecting local force contributions
        let force_contributions: Vec<Vec<Vector3<f64>>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut local_forces = vec![Vector3::zeros(); n];
                for j in (i + 1)..n {
                    let rij = self.minimum_image(positions[i] - positions[j]);
                    let mut r2 = rij.norm_squared();

                    // skip outside cutoff, clamp below min_r2
                    if r2 > r_cut2 {
                        continue;
                    }
                    if r2 < min_r2 {
                        r2 = min_r2;
                    }

                    let inv_r2 = sigma2 / r2;
                    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    let f_mag = 48.0 * self.epsilon * inv_r6 * (inv_r6 - 0.5) / r2;
                    let fij = rij * f_mag;

                    local_forces[i] += fij;
                    local_forces[j] -= fij;
                }
                local_forces
            })
            .collect();

        // Sum all force contributions
        let mut forces = vec![Vector3::zeros(); n];
        for local_forces in force_contributions {
            for i in 0..n {
                forces[i] += local_forces[i];
            }
        }

        forces
    }
}

impl LennardJones {
    /// Compute the virial contribution to pressure.
    ///
    /// The virial is calculated as: W = -1/3 * Σ_ij r_ij · F_ij
    ///
    /// This is used in the pressure calculation:
    /// P = ρkT + W/V = (N*k*T + W) / V
    ///
    /// # Arguments
    /// * `positions` - Particle positions
    ///
    /// # Returns
    /// The virial W (not divided by volume)
    pub fn compute_virial(&self, positions: &[Vector3<f64>]) -> f64 {
        let n = positions.len();
        let sigma2 = self.sigma * self.sigma;

        let r_cut = 2.5 * self.sigma;
        let r_cut2 = r_cut * r_cut;
        let min_r2 = 0.01 * sigma2;

        // Parallelize virial calculation
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut local_virial = 0.0;
                for j in (i + 1)..n {
                    let rij = self.minimum_image(positions[i] - positions[j]);
                    let mut r2 = rij.norm_squared();

                    if r2 > r_cut2 {
                        continue;
                    }
                    if r2 < min_r2 {
                        r2 = min_r2;
                    }

                    let inv_r2 = sigma2 / r2;
                    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    let f_mag = 48.0 * self.epsilon * inv_r6 * (inv_r6 - 0.5) / r2;

                    // Virial contribution: r · F
                    local_virial += rij.dot(&(rij * f_mag));
                }
                local_virial
            })
            .sum()
    }

    /// Compute the pressure tensor for the system.
    ///
    /// The pressure tensor is calculated as:
    /// P_αβ = (1/V) * [Σ_i m*v_α*v_β + Σ_ij r_ij,α * F_ij,β]
    ///
    /// This function computes only the configurational (virial) part.
    /// For the full pressure tensor, add the kinetic contribution:
    /// P_αβ += (1/V) * Σ_i m_i * v_i,α * v_i,β
    ///
    /// # Arguments
    /// * `positions` - Particle positions
    ///
    /// # Returns
    /// The configurational part of the pressure tensor (3x3 matrix)
    /// Divide by volume to get pressure in units of force/area
    pub fn compute_pressure_tensor(&self, positions: &[Vector3<f64>]) -> Matrix3<f64> {
        let n = positions.len();
        let sigma2 = self.sigma * self.sigma;

        let r_cut = 2.5 * self.sigma;
        let r_cut2 = r_cut * r_cut;
        let min_r2 = 0.01 * sigma2;

        // Parallelize pressure tensor calculation
        let tensor_contributions: Vec<Matrix3<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut local_tensor = Matrix3::zeros();
                for j in (i + 1)..n {
                    let rij = self.minimum_image(positions[i] - positions[j]);
                    let mut r2 = rij.norm_squared();

                    if r2 > r_cut2 {
                        continue;
                    }
                    if r2 < min_r2 {
                        r2 = min_r2;
                    }

                    let inv_r2 = sigma2 / r2;
                    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    let f_mag = 48.0 * self.epsilon * inv_r6 * (inv_r6 - 0.5) / r2;
                    let fij = rij * f_mag;

                    // Add r_ij ⊗ F_ij to tensor (outer product)
                    for alpha in 0..3 {
                        for beta in 0..3 {
                            local_tensor[(alpha, beta)] += rij[alpha] * fij[beta];
                        }
                    }
                }
                local_tensor
            })
            .collect();

        // Sum all contributions
        let mut pressure_tensor = Matrix3::zeros();
        for local_tensor in tensor_contributions {
            pressure_tensor += local_tensor;
        }

        pressure_tensor
    }

    /// Compute the scalar pressure from kinetic energy and positions.
    ///
    /// P = (2*K + W) / (3*V)
    ///
    /// where K is the kinetic energy, W is the virial, and V is the volume.
    ///
    /// # Arguments
    /// * `positions` - Particle positions
    /// * `kinetic_energy` - Total kinetic energy of the system
    ///
    /// # Returns
    /// Pressure in units of energy/volume
    pub fn compute_pressure(&self, positions: &[Vector3<f64>], kinetic_energy: f64) -> f64 {
        let virial = self.compute_virial(positions);
        let volume = self.volume();
        (2.0 * kinetic_energy + virial) / (3.0 * volume)
    }

    /// Compute scalar pressure from the pressure tensor.
    ///
    /// P = (P_xx + P_yy + P_zz) / 3
    ///
    /// # Arguments
    /// * `positions` - Particle positions
    /// * `velocities` - Particle velocities
    /// * `mass` - Particle mass (assumed uniform)
    ///
    /// # Returns
    /// Scalar pressure in units of mass*velocity²/volume
    pub fn compute_pressure_from_tensor(
        &self,
        positions: &[Vector3<f64>],
        velocities: &[Vector3<f64>],
        mass: f64,
    ) -> f64 {
        let n = positions.len();
        let volume = self.volume();

        // Configurational part from virial
        let config_tensor = self.compute_pressure_tensor(positions);

        // Kinetic part: Σ_i m * v_i ⊗ v_i
        let mut kinetic_tensor = Matrix3::zeros();
        for i in 0..n {
            for alpha in 0..3 {
                for beta in 0..3 {
                    kinetic_tensor[(alpha, beta)] +=
                        mass * velocities[i][alpha] * velocities[i][beta];
                }
            }
        }

        // Total pressure tensor
        let pressure_tensor = (kinetic_tensor + config_tensor) / volume;

        // Return trace / 3 for scalar pressure
        (pressure_tensor[(0, 0)] + pressure_tensor[(1, 1)] + pressure_tensor[(2, 2)]) / 3.0
    }

    /// Compute instantaneous temperature from kinetic energy.
    ///
    /// T = 2*K / (3*N*k_B)
    ///
    /// Note: Returns temperature in units where k_B = 1.
    /// For real units, multiply by your energy unit / k_B.
    ///
    /// # Arguments
    /// * `kinetic_energy` - Total kinetic energy
    /// * `n_particles` - Number of particles
    ///
    /// # Returns
    /// Temperature in reduced units (k_B = 1)
    pub fn temperature_from_kinetic_energy(kinetic_energy: f64, n_particles: usize) -> f64 {
        2.0 * kinetic_energy / (3.0 * n_particles as f64)
    }

    /// Compute kinetic energy from velocities.
    ///
    /// K = (1/2) * Σ_i m * v_i²
    ///
    /// # Arguments
    /// * `velocities` - Particle velocities
    /// * `mass` - Particle mass (assumed uniform)
    ///
    /// # Returns
    /// Total kinetic energy
    pub fn kinetic_energy(velocities: &[Vector3<f64>], mass: f64) -> f64 {
        velocities
            .iter()
            .map(|v| 0.5 * mass * v.norm_squared())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lennard_jones_new() {
        let epsilon = 1.5;
        let sigma = 2.0;
        let box_lengths = Vector3::new(10.0, 10.0, 10.0);

        let lj = LennardJones::new(epsilon, sigma, box_lengths);

        assert_eq!(lj.epsilon, epsilon);
        assert_eq!(lj.sigma, sigma);
        // Test that lattice is diagonal for orthogonal box
        assert_eq!(lj.lattice[(0, 0)], 10.0);
        assert_eq!(lj.lattice[(1, 1)], 10.0);
        assert_eq!(lj.lattice[(2, 2)], 10.0);
    }

    #[test]
    fn test_minimum_image_convention() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));

        // Test no wrapping needed
        let d1 = Vector3::new(1.0, 2.0, 3.0);
        let result1 = lj.minimum_image(d1);
        assert_relative_eq!(result1.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result1.y, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result1.z, 3.0, epsilon = 1e-10);

        // Test wrapping needed
        let d2 = Vector3::new(6.0, -6.0, 8.0);
        let result2 = lj.minimum_image(d2);
        assert_relative_eq!(result2.x, -4.0, epsilon = 1e-10); // 6 - 10 = -4
        assert_relative_eq!(result2.y, 4.0, epsilon = 1e-10); // -6 + 10 = 4
        assert_relative_eq!(result2.z, -2.0, epsilon = 1e-10); // 8 - 10 = -2

        // Test edge case at exactly half box length
        let d3 = Vector3::new(5.0, -5.0, 0.0);
        let result3 = lj.minimum_image(d3);
        assert_relative_eq!(result3.x, -5.0, epsilon = 1e-10);
        assert_relative_eq!(result3.y, 5.0, epsilon = 1e-10);
        assert_relative_eq!(result3.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lj_potential_calculation() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));

        // Test at sigma (r = sigma, so r^2 = sigma^2 = 1.0)
        let r2_at_sigma = 1.0;
        let potential_at_sigma = lj.lj_potential(r2_at_sigma);
        assert_relative_eq!(potential_at_sigma, 0.0, epsilon = 1e-10);

        // Test at minimum (r = 2^(1/6) * sigma ≈ 1.122, r^2 ≈ 1.26)
        let r2_at_min = 2_f64.powf(1.0 / 3.0); // This is 2^(2/6) = 2^(1/3)
        let potential_at_min = lj.lj_potential(r2_at_min);
        assert_relative_eq!(potential_at_min, -1.0, epsilon = 1e-10);

        // Test at very large distance (should approach 0 but allow for numerical precision)
        let r2_large = 100.0;
        let potential_large = lj.lj_potential(r2_large);
        assert!(potential_large.abs() < 1e-3); // Relaxed tolerance for numerical precision

        // Test at very small distance (should be very positive)
        let r2_small = 0.1;
        let potential_small = lj.lj_potential(r2_small);
        assert!(potential_small > 1000.0);
    }

    #[test]
    fn test_compute_potential_energy_two_atoms() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Two atoms separated by sigma (should give 0 potential)
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0), // distance = sigma = 1.0
        ];

        let potential = lj.compute_potential_energy(&positions);
        assert_relative_eq!(potential, 0.0, epsilon = 1e-10);

        // Two atoms at minimum distance (r = 2^(1/6) * sigma)
        let r_min = 2_f64.powf(1.0 / 6.0);
        let positions_min = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(r_min, 0.0, 0.0)];

        let potential_min = lj.compute_potential_energy(&positions_min);
        assert_relative_eq!(potential_min, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_potential_energy_beyond_cutoff() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Two atoms beyond cutoff distance (2.5 * sigma = 2.5)
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0), // distance = 3.0 > 2.5
        ];

        let potential = lj.compute_potential_energy(&positions);
        assert_relative_eq!(potential, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_forces_two_atoms() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Two atoms separated by sigma
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];

        let forces = lj.compute_forces(&positions);

        // Forces should be equal and opposite
        assert_relative_eq!(forces[0].x, -forces[1].x, epsilon = 1e-10);
        assert_relative_eq!(forces[0].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[0].z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].z, 0.0, epsilon = 1e-10);

        // At r = sigma, atoms are still attracted (potential = 0 but derivative != 0)
        // The minimum is at r = 2^(1/6) * sigma ≈ 1.122, so at r = sigma we're still attractive
        // Force on atom 0 should be negative (toward atom 1), force on atom 1 should be positive (away from atom 0)
        assert!(forces[0].x < 0.0); // Attractive force on atom 0 toward atom 1
        assert!(forces[1].x > 0.0); // Force on atom 1 away from atom 0 (reaction)
        assert_relative_eq!(forces[0].x, -forces[1].x, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_forces_beyond_cutoff() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Two atoms beyond cutoff distance
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0)];

        let forces = lj.compute_forces(&positions);

        // Forces should be zero beyond cutoff
        assert_relative_eq!(forces[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[0].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[0].z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_force_energy_consistency() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Test that force is negative gradient of potential energy
        let dx = 1e-6;
        let positions1 = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.5, 0.0, 0.0)];
        let positions2 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.5 + dx, 0.0, 0.0),
        ];

        let e1 = lj.compute_potential_energy(&positions1);
        let e2 = lj.compute_potential_energy(&positions2);
        let numerical_force = -(e2 - e1) / dx;

        let forces = lj.compute_forces(&positions1);
        let analytical_force = forces[1].x;

        assert_relative_eq!(numerical_force, analytical_force, epsilon = 1e-4);
    }

    #[test]
    fn test_compute_forces_three_atoms() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Three atoms in a line
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.5, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];

        let forces = lj.compute_forces(&positions);

        // Newton's third law: total force should be zero
        let total_force = forces[0] + forces[1] + forces[2];
        assert_relative_eq!(total_force.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(total_force.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(total_force.z, 0.0, epsilon = 1e-10);

        // Outer atoms should have non-zero forces
        assert!(forces[0].x != 0.0);
        assert!(forces[2].x != 0.0);

        // Middle atom experiences equal and opposite forces from neighbors, so net force is zero
        assert_relative_eq!(forces[1].x, 0.0, epsilon = 1e-10);

        // Forces should be in y and z = 0 (atoms arranged along x-axis)
        assert_relative_eq!(forces[0].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[0].z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[1].z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[2].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forces[2].z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_lattice_orthogonal() {
        // Test that from_lattice gives same results as new() for orthogonal boxes
        let box_lengths = Vector3::new(10.0, 12.0, 15.0);
        let lattice = Matrix3::from_diagonal(&box_lengths);

        let lj1 = LennardJones::new(1.0, 1.0, box_lengths);
        let lj2 = LennardJones::from_lattice(1.0, 1.0, lattice);

        // Test that minimum image convention gives same results
        let d = Vector3::new(7.0, -8.0, 9.0);
        let d1 = lj1.minimum_image(d);
        let d2 = lj2.minimum_image(d);

        assert_relative_eq!(d1.x, d2.x, epsilon = 1e-10);
        assert_relative_eq!(d1.y, d2.y, epsilon = 1e-10);
        assert_relative_eq!(d1.z, d2.z, epsilon = 1e-10);
    }

    #[test]
    fn test_triclinic_lattice() {
        // Create a triclinic box with 60-degree angle between a and b
        let a = Vector3::new(10.0, 0.0, 0.0);
        let b = Vector3::new(5.0, 8.66025404, 0.0); // 60 degrees from a
        let c = Vector3::new(0.0, 0.0, 10.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);

        let lj = LennardJones::from_lattice(1.0, 1.0, lattice);

        // Test that lattice is stored correctly
        assert_relative_eq!(lj.lattice.column(0)[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(lj.lattice.column(1)[0], 5.0, epsilon = 1e-6);
        assert_relative_eq!(lj.lattice.column(1)[1], 8.66025404, epsilon = 1e-6);

        // Test minimum image convention
        // Point at (11, 0, 0) should wrap to (1, 0, 0) through a-vector
        let d = Vector3::new(11.0, 0.0, 0.0);
        let wrapped = lj.minimum_image(d);
        assert_relative_eq!(wrapped.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(wrapped.y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triclinic_pbc_wrapping() {
        // Create a triclinic box
        let a = Vector3::new(8.0, 0.0, 0.0);
        let b = Vector3::new(4.0, 6.928, 0.0); // 60 degrees
        let c = Vector3::new(0.0, 0.0, 8.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);

        let lj = LennardJones::from_lattice(1.0, 1.0, lattice);

        // Two atoms that are close through PBC in triclinic box
        let positions = vec![
            Vector3::new(0.5, 0.5, 0.0),
            Vector3::new(7.8, 0.5, 0.0), // Near opposite edge
        ];

        // Distance should wrap through PBC
        let rij = lj.minimum_image(positions[0] - positions[1]);
        let dist = rij.norm();

        // Should be much closer than the direct distance
        let direct_dist = (positions[0] - positions[1]).norm();
        assert!(dist < direct_dist);
        assert!(dist < 4.0); // Should be close to 0.7 + wrap
    }

    #[test]
    fn test_set_lattice() {
        let mut lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));

        // Create a new triclinic lattice
        let a = Vector3::new(12.0, 0.0, 0.0);
        let b = Vector3::new(6.0, 10.392, 0.0);
        let c = Vector3::new(0.0, 0.0, 12.0);
        let new_lattice = Matrix3::from_columns(&[a, b, c]);

        lj.set_lattice(new_lattice);

        // Verify the lattice was updated
        assert_relative_eq!(lj.lattice.column(0)[0], 12.0, epsilon = 1e-10);
        assert_relative_eq!(lj.lattice.column(1)[0], 6.0, epsilon = 1e-10);

        // Verify inverse was recomputed (test by checking minimum image)
        let d = Vector3::new(13.0, 0.0, 0.0);
        let wrapped = lj.minimum_image(d);
        assert_relative_eq!(wrapped.x, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triclinic_forces_conservation() {
        // Test that forces still conserve momentum in triclinic box
        let a = Vector3::new(15.0, 0.0, 0.0);
        let b = Vector3::new(7.5, 12.99, 0.0);
        let c = Vector3::new(0.0, 0.0, 15.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);

        let lj = LennardJones::from_lattice(1.0, 1.0, lattice);

        // Three atoms in triclinic box
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(3.0, 2.0, 1.0),
            Vector3::new(5.0, 4.0, 1.0),
        ];

        let forces = lj.compute_forces(&positions);

        // Total force should be zero (momentum conservation)
        let total_force = forces[0] + forces[1] + forces[2];
        assert_relative_eq!(total_force.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(total_force.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(total_force.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_triclinic_energy_symmetry() {
        // Test that energy is symmetric under lattice translation
        let a = Vector3::new(10.0, 0.0, 0.0);
        let b = Vector3::new(5.0, 8.66, 0.0);
        let c = Vector3::new(0.0, 0.0, 10.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);

        let lj = LennardJones::from_lattice(1.0, 1.0, lattice);

        // Two atoms at minimum distance
        let r_min = 2_f64.powf(1.0 / 6.0);
        let positions1 = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0 + r_min, 1.0, 1.0),
        ];

        // Same configuration translated by lattice vector a
        let positions2 = vec![
            Vector3::new(11.0, 1.0, 1.0), // Wrapped through PBC
            Vector3::new(11.0 + r_min, 1.0, 1.0),
        ];

        let e1 = lj.compute_potential_energy(&positions1);
        let e2 = lj.compute_potential_energy(&positions2);

        // Energies should be identical due to PBC
        assert_relative_eq!(e1, e2, epsilon = 1e-8);
        assert_relative_eq!(e1, -1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_volume_calculation() {
        // Orthogonal box
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 12.0, 15.0));
        assert_relative_eq!(lj.volume(), 1800.0, epsilon = 1e-10);

        // Triclinic box
        let a = Vector3::new(10.0, 0.0, 0.0);
        let b = Vector3::new(5.0, 8.66025404, 0.0);
        let c = Vector3::new(0.0, 0.0, 10.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);
        let lj_tri = LennardJones::from_lattice(1.0, 1.0, lattice);

        let expected_volume = 866.025404;
        assert_relative_eq!(lj_tri.volume(), expected_volume, epsilon = 0.01);
    }

    #[test]
    fn test_virial_calculation() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Two atoms at minimum energy distance
        let r_min = 2_f64.powf(1.0 / 6.0);
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(r_min, 0.0, 0.0)];

        let virial = lj.compute_virial(&positions);

        // At minimum, the force is zero, so virial should be ~0
        assert_relative_eq!(virial, 0.0, epsilon = 1e-10);

        // Two atoms closer than minimum (repulsive)
        let positions_close = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.9, 0.0, 0.0)];

        let virial_close = lj.compute_virial(&positions_close);
        // Repulsive force, virial should be positive
        assert!(virial_close > 0.0);
    }

    #[test]
    fn test_pressure_calculation() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));

        // Single atom (ideal gas limit)
        let positions = vec![Vector3::new(5.0, 5.0, 5.0)];
        let kinetic_energy = 1.5; // k_B * T for 3D

        let pressure = lj.compute_pressure(&positions, kinetic_energy);

        // P = 2K/(3V) for non-interacting
        let expected_pressure = 2.0 * kinetic_energy / (3.0 * 1000.0);
        assert_relative_eq!(pressure, expected_pressure, epsilon = 1e-10);
    }

    #[test]
    fn test_pressure_tensor_calculation() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        // Two atoms along x-axis
        let positions = vec![
            Vector3::new(9.0, 10.0, 10.0),
            Vector3::new(11.0, 10.0, 10.0),
        ];

        let tensor = lj.compute_pressure_tensor(&positions);

        // Force is along x-direction, so P_xx should be non-zero
        // P_yy and P_zz should be zero (no y or z components)
        assert!(tensor[(0, 0)].abs() > 1e-10);
        assert_relative_eq!(tensor[(1, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tensor[(2, 2)], 0.0, epsilon = 1e-10);

        // Off-diagonal terms should be zero (symmetric system)
        assert_relative_eq!(tensor[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tensor[(0, 2)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tensor[(1, 2)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pressure_from_tensor() {
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));
        let mass = 1.0;

        // Create a simple system with known velocities
        let positions = vec![Vector3::new(2.0, 2.0, 2.0), Vector3::new(8.0, 8.0, 8.0)];
        let velocities = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)];

        let pressure = lj.compute_pressure_from_tensor(&positions, &velocities, mass);

        // Pressure should be positive
        assert!(pressure > 0.0);

        // Compare with direct calculation
        let ke = LennardJones::kinetic_energy(&velocities, mass);
        let pressure_direct = lj.compute_pressure(&positions, ke);

        // Should give same result (within numerical precision)
        assert_relative_eq!(pressure, pressure_direct, epsilon = 1e-10);
    }

    #[test]
    fn test_kinetic_energy_calculation() {
        let mass = 2.0;
        let velocities = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
        ];

        let ke = LennardJones::kinetic_energy(&velocities, mass);

        // KE = 0.5 * m * (v1² + v2² + v3²)
        // = 0.5 * 2.0 * (1.0 + 4.0 + 9.0) = 14.0
        assert_relative_eq!(ke, 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_temperature_from_kinetic_energy() {
        let n_particles = 100;
        let kinetic_energy = 150.0;

        let temp = LennardJones::temperature_from_kinetic_energy(kinetic_energy, n_particles);

        // T = 2K/(3N) in units where k_B = 1
        let expected_temp = 2.0 * 150.0 / (3.0 * 100.0);
        assert_relative_eq!(temp, expected_temp, epsilon = 1e-10);
    }

    #[test]
    fn test_ideal_gas_pressure() {
        // Test that ideal gas law is satisfied for non-interacting particles
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));

        // Place particles far apart (no interactions)
        let positions = vec![Vector3::new(1.0, 1.0, 1.0), Vector3::new(9.0, 9.0, 9.0)];

        let n = 2;
        let temp = 1.5; // Reduced temperature
        let ke = 1.5 * n as f64 * temp; // K = (3/2)*N*k*T

        let pressure = lj.compute_pressure(&positions, ke);

        // Ideal gas: P = N*k*T/V
        let volume = lj.volume();
        let expected_pressure = n as f64 * temp / volume;

        assert_relative_eq!(pressure, expected_pressure, epsilon = 1e-10);
    }

    #[test]
    fn test_virial_theorem() {
        // Test virial theorem for LJ system
        let lj = LennardJones::new(1.0, 1.0, Vector3::new(20.0, 20.0, 20.0));

        let r_min = 2_f64.powf(1.0 / 6.0);

        // Test in repulsive regime (r < r_min where energy > 0)
        // At r = sigma, U = 0, so we need r < sigma for positive energy
        let r = 0.9; // Less than sigma = 1.0
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(r, 0.0, 0.0)];

        let virial = lj.compute_virial(&positions);
        let energy = lj.compute_potential_energy(&positions);

        // In repulsive regime (r < sigma), both virial and energy should be positive
        assert!(virial > 0.0);
        assert!(energy > 0.0);

        // Test in attractive regime (r > r_min)
        let r_attract = 1.5 * r_min;
        let positions_attract = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(r_attract, 0.0, 0.0),
        ];

        let virial_attract = lj.compute_virial(&positions_attract);
        let energy_attract = lj.compute_potential_energy(&positions_attract);

        // In attractive regime (r > r_min), energy is negative
        assert!(energy_attract < 0.0);
        // Virial can be negative in attractive regime
        assert!(virial_attract < 0.0);
    }

    #[test]
    fn test_pressure_triclinic_box() {
        // Test pressure calculation works with triclinic boxes
        let a = Vector3::new(8.0, 0.0, 0.0);
        let b = Vector3::new(4.0, 6.928, 0.0);
        let c = Vector3::new(0.0, 0.0, 8.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);

        let lj = LennardJones::from_lattice(1.0, 1.0, lattice);

        let positions = vec![Vector3::new(2.0, 2.0, 2.0), Vector3::new(6.0, 4.0, 6.0)];

        let kinetic_energy = 3.0;
        let pressure = lj.compute_pressure(&positions, kinetic_energy);

        // Should give a reasonable pressure value
        assert!(pressure.is_finite());
        assert!(pressure > 0.0 || pressure < 0.0); // Non-zero for this configuration
    }
}
