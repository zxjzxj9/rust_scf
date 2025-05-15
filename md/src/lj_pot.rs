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

    fn lj_potential(&self, r2: f64) -> f64 {
        let inv_r2 = self.sigma*self.sigma/r2;
        let inv_r6 = inv_r2*inv_r2*inv_r2;
        4.0*self.epsilon*(inv_r6*inv_r6 - inv_r6)
    }
}

impl ForceProvider for LennardJones {
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = positions.len();
        let mut forces = vec![Vector3::zeros(); n];
        let sigma2 = self.sigma * self.sigma;

        // cutoff at 2.5 sigma
        let r_cut = 2.5 * self.sigma;
        let r_cut2 = r_cut * r_cut;

        // prevent singularity for r^2 < min_r2
        let min_r2 = 0.01 * sigma2;

        for i in 0..n {
            for j in (i + 1)..n {
                let rij = self.minimum_image(positions[i] - positions[j]);
                let mut r2 = rij.norm_squared();

                // // skip outside cutoff, clamp below min_r2
                // if r2 > r_cut2 {
                //     continue;
                // }
                // if r2 < min_r2 {
                //     r2 = min_r2;
                // }

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