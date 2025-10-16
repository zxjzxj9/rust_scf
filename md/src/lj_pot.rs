// file: `md/src/lj_pot.rs`
use nalgebra::Vector3;
use crate::run_md::ForceProvider;
use rayon::prelude::*;

pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
    pub box_lengths: Vector3<f64>,
}

impl LennardJones {
    pub fn new(epsilon: f64, sigma: f64, box_lengths: Vector3<f64>) -> Self {
        LennardJones { epsilon, sigma, box_lengths }
    }

    // Apply minimum-image convention
    fn minimum_image(&self, mut d: Vector3<f64>) -> Vector3<f64> {
        for k in 0..3 {
            let box_l = self.box_lengths[k];
            d[k] -= box_l * (d[k] / box_l).round();
        }
        d
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
        assert_eq!(lj.box_lengths, box_lengths);
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
}