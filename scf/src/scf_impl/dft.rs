//! Minimal grid-based DFT helpers (currently: LDA exchange-only).
//!
//! This is intentionally simple:
//! - Atom-centered grid: Gauss–Legendre radial × 6-point octahedral angular grid
//! - Becke partitioning to avoid double-counting overlapping atom grids
//! - LDA exchange only (Slater exchange), no correlation

extern crate nalgebra as na;

use na::{DMatrix, Vector3};
use rayon::prelude::*;

/// Exchange-only LDA (Slater exchange), unpolarized.
///
/// Energy density:  e_x(ρ) = c_x ρ^(4/3),  c_x = -(3/4) (3/π)^(1/3)
/// Potential:       v_x(ρ) = d/dρ [ρ ε_x(ρ)] = -(3/π)^(1/3) ρ^(1/3)
pub(crate) fn lda_x_energy_density(rho: f64) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }
    let cx = -0.75 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0);
    cx * rho.powf(4.0 / 3.0)
}

pub(crate) fn lda_x_potential(rho: f64) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }
    -(3.0 / std::f64::consts::PI).powf(1.0 / 3.0) * rho.powf(1.0 / 3.0)
}

#[derive(Clone, Debug)]
pub struct GridPoint {
    pub r: Vector3<f64>,
    pub w: f64,
}

#[derive(Clone, Debug)]
pub struct DftGridParams {
    pub radial_points: usize,
    pub r_max: f64,
}

impl Default for DftGridParams {
    fn default() -> Self {
        Self {
            radial_points: 24,
            r_max: 12.0, // bohr
        }
    }
}

/// Build a simple global atom-centered grid with Becke partition weights.
pub fn build_becke_atom_grid(coords: &[Vector3<f64>], params: &DftGridParams) -> Vec<GridPoint> {
    let (r_nodes, r_weights) = gauss_legendre(params.radial_points, 0.0, params.r_max);
    let ang = octahedral_6();

    let mut points = Vec::with_capacity(coords.len() * r_nodes.len() * ang.len());
    for (a, ra) in coords.iter().enumerate() {
        for (ri, &r) in r_nodes.iter().enumerate() {
            let wr = r_weights[ri] * r * r; // Jacobian r^2
            for &(dir, wang) in &ang {
                let p = *ra + dir * r;
                let w_base = wr * wang;
                let w_becke = becke_weight_for_atom(a, &p, coords);
                let w = w_base * w_becke;
                if w.is_finite() && w > 0.0 {
                    points.push(GridPoint { r: p, w });
                }
            }
        }
    }
    points
}

/// Compute (E_xc, V_xc) for a given density matrix on a fixed grid.
///
/// - `basis_values(point)` must return all AO values φ_i(r) for that point.
pub fn lda_xc_on_grid<F>(
    density: &DMatrix<f64>,
    num_basis: usize,
    grid: &[GridPoint],
    basis_values: F,
) -> (f64, DMatrix<f64>)
where
    F: Fn(&Vector3<f64>) -> Vec<f64> + Sync,
{
    let (e_x, v_x) = grid
        .par_iter()
        .fold(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(mut e_acc, mut v_acc), gp| {
                let phi = basis_values(&gp.r);
                if phi.len() != num_basis {
                    return (e_acc, v_acc);
                }

                // rho = phi^T * P * phi
                let mut tmp = vec![0.0_f64; num_basis];
                for i in 0..num_basis {
                    let mut s = 0.0;
                    for j in 0..num_basis {
                        s += density[(i, j)] * phi[j];
                    }
                    tmp[i] = s;
                }
                let mut rho = 0.0;
                for i in 0..num_basis {
                    rho += phi[i] * tmp[i];
                }

                if !rho.is_finite() || rho <= 1e-14 {
                    return (e_acc, v_acc);
                }

                let ex = lda_x_energy_density(rho);
                let vx = lda_x_potential(rho);

                e_acc += gp.w * ex;

                // V_xc_ij += w * v_x(ρ) * φ_i φ_j
                let scale = gp.w * vx;
                for i in 0..num_basis {
                    let pi = phi[i];
                    for j in 0..num_basis {
                        v_acc[(i, j)] += scale * pi * phi[j];
                    }
                }

                (e_acc, v_acc)
            },
        )
        .reduce(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(e1, v1), (e2, v2)| (e1 + e2, v1 + v2),
        );

    (e_x, v_x)
}

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------

fn octahedral_6() -> Vec<(Vector3<f64>, f64)> {
    // 6 points on axes; weights sum to 4π
    let w = 4.0 * std::f64::consts::PI / 6.0;
    vec![
        (Vector3::new(1.0, 0.0, 0.0), w),
        (Vector3::new(-1.0, 0.0, 0.0), w),
        (Vector3::new(0.0, 1.0, 0.0), w),
        (Vector3::new(0.0, -1.0, 0.0), w),
        (Vector3::new(0.0, 0.0, 1.0), w),
        (Vector3::new(0.0, 0.0, -1.0), w),
    ]
}

/// Classic Becke partition weight for atom `a` at point `r`.
fn becke_weight_for_atom(a: usize, r: &Vector3<f64>, coords: &[Vector3<f64>]) -> f64 {
    let na = coords.len();
    if na == 1 {
        return 1.0;
    }

    // w_a = Π_{b≠a} p_ab, normalized across atoms.
    let mut raw = vec![1.0_f64; na];
    for i in 0..na {
        for j in 0..na {
            if i == j {
                continue;
            }
            raw[i] *= p_ij(i, j, r, coords);
        }
    }
    let denom: f64 = raw.iter().sum();
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    raw[a] / denom
}

fn p_ij(i: usize, j: usize, r: &Vector3<f64>, coords: &[Vector3<f64>]) -> f64 {
    let ri = (r - coords[i]).norm();
    let rj = (r - coords[j]).norm();
    let rij = (coords[i] - coords[j]).norm();
    if rij < 1e-12 {
        return 0.5;
    }
    let mut mu = (ri - rj) / rij;
    // Apply Becke's smooth step function f(mu) three times.
    for _ in 0..3 {
        mu = (3.0 * mu - mu * mu * mu) / 2.0;
    }
    0.5 * (1.0 - mu)
}

/// Gauss–Legendre nodes and weights on [a, b].
///
/// Simple Newton solver for roots of P_n(x); adapted from standard numerical recipes style.
fn gauss_legendre(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2);
    let m = (n + 1) / 2;
    let mut x = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];

    let eps = 1e-14;
    for i in 0..m {
        // Initial guess
        let i1 = i as f64 + 1.0;
        let nn = n as f64;
        let mut z = (std::f64::consts::PI * (i1 - 0.25) / (nn + 0.5)).cos();
        loop {
            let (p1, p2) = legendre_pn(n, z);
            let pp = (nn * (z * p1 - p2)) / (z * z - 1.0); // P'_n(z)
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < eps {
                // Map from [-1,1] to [a,b]
                let xm = 0.5 * (b + a);
                let xl = 0.5 * (b - a);
                x[i] = xm - xl * z;
                x[n - 1 - i] = xm + xl * z;
                let wi = 2.0 * xl / ((1.0 - z * z) * pp * pp);
                w[i] = wi;
                w[n - 1 - i] = wi;
                break;
            }
        }
    }
    (x, w)
}

/// Returns (P_n(z), P_{n-1}(z)).
fn legendre_pn(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = 1.0;
    let mut p2 = 0.0;
    for j in 1..=n {
        let p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / (j as f64);
    }
    (p1, p2)
}


