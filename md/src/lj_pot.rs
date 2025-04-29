// file: `md/src/lj_pot.rs`
use nalgebra::Vector3;
use crate::run_md::ForceProvider;

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
            let L = self.box_lengths[k];
            d[k] -= L * (d[k] / L).round();
        }
        d
    }
}

impl ForceProvider for LennardJones {
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = positions.len();
        let mut forces = vec![Vector3::zeros(); n];
        let sigma2 = self.sigma * self.sigma;

        for i in 0..n {
            for j in (i + 1)..n {
                let rij = self.minimum_image(positions[i] - positions[j]);
                let r2 = rij.dot(&rij);
                if r2 == 0.0 { continue; }

                let inv_r2 = sigma2 / r2;
                let inv_r6 = inv_r2 * inv_r2 * inv_r2;
                let f_mag = 48.0 * self.epsilon * inv_r6 * (inv_r6 - 0.5) / r2;
                let fij = rij * f_mag;

                forces[i] += fij;
                forces[j] -= fij;
            }
        }

        forces
    }
}