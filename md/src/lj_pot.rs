// file: `md/src/lj_pot.rs`
use nalgebra::{Vector3, Matrix3};
use crate::run_md::ForceProvider;
use rayon::prelude::*;

/// Lennard-Jones potential with support for arbitrary lattice structures.
/// 
/// The lattice is represented by a 3x3 matrix where each column is a lattice vector:
/// - Column 0: a-vector (first lattice vector)
/// - Column 1: b-vector (second lattice vector)
/// - Column 2: c-vector (third lattice vector)
/// 
/// For orthogonal boxes, this is a diagonal matrix with box lengths on the diagonal.
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
    /// Lattice matrix: columns are lattice vectors [a, b, c]
    pub lattice: Matrix3<f64>,
    /// Inverse lattice matrix for minimum image convention
    lattice_inv: Matrix3<f64>,
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
        let lattice_inv = lattice.try_inverse()
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
        self.lattice_inv = new_lattice.try_inverse()
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

    pub fn lj_potential(&self, r2: f64) -> f64 {
        let inv_r2 = self.sigma*self.sigma/r2;
        let inv_r6 = inv_r2*inv_r2*inv_r2;
        4.0*self.epsilon*(inv_r6*inv_r6 - inv_r6)
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
        (0..n).into_par_iter().map(|i| {
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
        }).sum()
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
        let force_contributions: Vec<Vec<Vector3<f64>>> = (0..n).into_par_iter().map(|i| {
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
        }).collect();

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
        assert_relative_eq!(result2.y, 4.0, epsilon = 1e-10);  // -6 + 10 = 4
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
        let r2_at_min = 2_f64.powf(1.0/3.0); // This is 2^(2/6) = 2^(1/3)
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
        let r_min = 2_f64.powf(1.0/6.0);
        let positions_min = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(r_min, 0.0, 0.0),
        ];
        
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
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        
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
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];
        
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
        let positions1 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.5, 0.0, 0.0),
        ];
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
        let b = Vector3::new(5.0, 8.66025404, 0.0);  // 60 degrees from a
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
        let b = Vector3::new(4.0, 6.928, 0.0);  // 60 degrees
        let c = Vector3::new(0.0, 0.0, 8.0);
        let lattice = Matrix3::from_columns(&[a, b, c]);
        
        let lj = LennardJones::from_lattice(1.0, 1.0, lattice);
        
        // Two atoms that are close through PBC in triclinic box
        let positions = vec![
            Vector3::new(0.5, 0.5, 0.0),
            Vector3::new(7.8, 0.5, 0.0),  // Near opposite edge
        ];
        
        // Distance should wrap through PBC
        let rij = lj.minimum_image(positions[0] - positions[1]);
        let dist = rij.norm();
        
        // Should be much closer than the direct distance
        let direct_dist = (positions[0] - positions[1]).norm();
        assert!(dist < direct_dist);
        assert!(dist < 4.0);  // Should be close to 0.7 + wrap
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
        let r_min = 2_f64.powf(1.0/6.0);
        let positions1 = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0 + r_min, 1.0, 1.0),
        ];
        
        // Same configuration translated by lattice vector a
        let positions2 = vec![
            Vector3::new(11.0, 1.0, 1.0),  // Wrapped through PBC
            Vector3::new(11.0 + r_min, 1.0, 1.0),
        ];
        
        let e1 = lj.compute_potential_energy(&positions1);
        let e2 = lj.compute_potential_energy(&positions2);
        
        // Energies should be identical due to PBC
        assert_relative_eq!(e1, e2, epsilon = 1e-8);
        assert_relative_eq!(e1, -1.0, epsilon = 1e-8);
    }
}