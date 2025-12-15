//! Minimal grid-based DFT helpers (currently: LDA exchange-only).
//!
//! This is intentionally simple:
//! - Atom-centered grid: Gauss–Legendre radial × 6-point octahedral angular grid
//! - Becke partitioning to avoid double-counting overlapping atom grids
//! - LDA exchange only (Slater exchange), no correlation

extern crate nalgebra as na;

use na::{DMatrix, Vector3};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XcFunctional {
    /// Local-density approximation exchange only (Slater exchange)
    LdaX,
    /// PBE GGA exchange only (no correlation)
    PbeX,
}

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

// ---------------------------------------------------------------------------
// PBE exchange-only (GGA) helpers
// ---------------------------------------------------------------------------

#[inline]
fn pbe_kappa() -> f64 {
    0.804
}

#[inline]
fn pbe_mu() -> f64 {
    0.219_514_972_764_517_1
}

#[inline]
fn c_x() -> f64 {
    -0.75 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0)
}

/// PBE exchange enhancement factor F_x(s)
#[inline]
fn pbe_fx(s: f64) -> f64 {
    let kappa = pbe_kappa();
    let mu = pbe_mu();
    let t = 1.0 + (mu / kappa) * s * s;
    1.0 + kappa - kappa / t
}

/// dF_x/ds
#[inline]
fn pbe_dfx_ds(s: f64) -> f64 {
    let kappa = pbe_kappa();
    let mu = pbe_mu();
    let t = 1.0 + (mu / kappa) * s * s;
    // d/ds [ 1 + kappa - kappa / t ] = kappa * (1/t^2) * d t/ds
    // dt/ds = 2 (mu/kappa) s
    2.0 * mu * s / (t * t)
}

/// Compute PBE exchange energy density e_x (per volume), and partial derivatives w.r.t.
/// rho and grad_rho (vector), using the reduced gradient
///
/// s = |∇ρ| / ( 2 (3π^2)^(1/3) ρ^(4/3) )
fn pbe_x_energy_density_and_partials(rho: f64, grad_rho: Vector3<f64>) -> (f64, f64, Vector3<f64>) {
    if rho <= 0.0 {
        return (0.0, 0.0, Vector3::zeros());
    }

    let g = grad_rho.norm();

    // LDA exchange energy density (per volume)
    let e_lda = c_x() * rho.powf(4.0 / 3.0);
    let de_lda_drho = (4.0 / 3.0) * c_x() * rho.powf(1.0 / 3.0);

    // s denominator: 2 (3π^2)^(1/3) ρ^(4/3)
    let c = 2.0 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(1.0 / 3.0);
    let denom = c * rho.powf(4.0 / 3.0);
    let s = if denom > 0.0 { g / denom } else { 0.0 };

    let fx = pbe_fx(s);
    let dfx_ds = pbe_dfx_ds(s);

    let e = e_lda * fx;

    // ∂s/∂ρ = -(4/3) * s / ρ  (for fixed |∇ρ|)
    let ds_drho = if rho > 0.0 { -(4.0 / 3.0) * s / rho } else { 0.0 };

    // ∂s/∂(∇ρ) = (1/denom) * (∇ρ / |∇ρ|)
    let ds_dgrad = if g > 1e-14 && denom > 0.0 {
        (1.0 / denom) * (grad_rho / g)
    } else {
        Vector3::zeros()
    };

    // ∂e/∂ρ and ∂e/∂(∇ρ)
    let de_drho = de_lda_drho * fx + e_lda * dfx_ds * ds_drho;
    let de_dgrad = e_lda * dfx_ds * ds_dgrad;

    (e, de_drho, de_dgrad)
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

/// Compute (E_xc, V_xc) for either LDA-x or PBE-x on a fixed grid, using finite-difference
/// gradients of the density (so we don't need analytic AO gradients yet).
///
/// This yields a *discretisation-consistent* approximate functional derivative with respect to
/// the AO density matrix elements.
pub fn xc_on_grid_fdiff<F>(
    functional: XcFunctional,
    density: &DMatrix<f64>,
    num_basis: usize,
    grid: &[GridPoint],
    fd_delta: f64,
    basis_values: F,
) -> (f64, DMatrix<f64>)
where
    F: Fn(&Vector3<f64>) -> Vec<f64> + Sync,
{
    match functional {
        XcFunctional::LdaX => lda_xc_on_grid(density, num_basis, grid, basis_values),
        XcFunctional::PbeX => pbe_xc_on_grid_fdiff(density, num_basis, grid, fd_delta, basis_values),
    }
}

fn pbe_xc_on_grid_fdiff<F>(
    density: &DMatrix<f64>,
    num_basis: usize,
    grid: &[GridPoint],
    fd_delta: f64,
    basis_values: F,
) -> (f64, DMatrix<f64>)
where
    F: Fn(&Vector3<f64>) -> Vec<f64> + Sync,
{
    let delta = fd_delta.max(1e-6);

    let (e_x, v_x) = grid
        .par_iter()
        .fold(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(mut e_acc, mut v_acc), gp| {
                // AO values at r and shifted points for finite-difference rho gradients
                let r0 = gp.r;
                let rxp = r0 + Vector3::new(delta, 0.0, 0.0);
                let rxm = r0 - Vector3::new(delta, 0.0, 0.0);
                let ryp = r0 + Vector3::new(0.0, delta, 0.0);
                let rym = r0 - Vector3::new(0.0, delta, 0.0);
                let rzp = r0 + Vector3::new(0.0, 0.0, delta);
                let rzm = r0 - Vector3::new(0.0, 0.0, delta);

                let phi0 = basis_values(&r0);
                if phi0.len() != num_basis {
                    return (e_acc, v_acc);
                }

                let phixp = basis_values(&rxp);
                let phixm = basis_values(&rxm);
                let phiyp = basis_values(&ryp);
                let phiym = basis_values(&rym);
                let phizp = basis_values(&rzp);
                let phizm = basis_values(&rzm);
                if phixp.len() != num_basis
                    || phixm.len() != num_basis
                    || phiyp.len() != num_basis
                    || phiym.len() != num_basis
                    || phizp.len() != num_basis
                    || phizm.len() != num_basis
                {
                    return (e_acc, v_acc);
                }

                let rho0 = rho_from_phi(density, &phi0);
                if !rho0.is_finite() || rho0 <= 1e-14 {
                    return (e_acc, v_acc);
                }

                let rhoxp = rho_from_phi(density, &phixp);
                let rhoxm = rho_from_phi(density, &phixm);
                let rhoyp = rho_from_phi(density, &phiyp);
                let rhoym = rho_from_phi(density, &phiym);
                let rhozp = rho_from_phi(density, &phizp);
                let rhozm = rho_from_phi(density, &phizm);
                if !rhoxp.is_finite()
                    || !rhoxm.is_finite()
                    || !rhoyp.is_finite()
                    || !rhoym.is_finite()
                    || !rhozp.is_finite()
                    || !rhozm.is_finite()
                {
                    return (e_acc, v_acc);
                }

                let grad_rho = Vector3::new(
                    (rhoxp - rhoxm) / (2.0 * delta),
                    (rhoyp - rhoym) / (2.0 * delta),
                    (rhozp - rhozm) / (2.0 * delta),
                );

                let (e, de_drho, de_dgrad) = pbe_x_energy_density_and_partials(rho0, grad_rho);
                e_acc += gp.w * e;

                // dE/dP_ij = Σ_p w_p [ (∂e/∂ρ)_p * φ_i(r_p) φ_j(r_p)
                //                   + Σ_a (∂e/∂(∂_a ρ))_p * ∂(∂_a ρ)/∂P_ij ]
                //
                // with ∂(∂x ρ)/∂P_ij = (φ_i(r+dx)φ_j(r+dx) - φ_i(r-dx)φ_j(r-dx)) / (2δ)
                // and similarly for y,z.
                let w0 = gp.w * de_drho;
                for i in 0..num_basis {
                    let pi = phi0[i];
                    for j in 0..num_basis {
                        v_acc[(i, j)] += w0 * pi * phi0[j];
                    }
                }

                let wx = gp.w * de_dgrad.x / (2.0 * delta);
                let wy = gp.w * de_dgrad.y / (2.0 * delta);
                let wz = gp.w * de_dgrad.z / (2.0 * delta);

                for i in 0..num_basis {
                    for j in 0..num_basis {
                        v_acc[(i, j)] += wx * (phixp[i] * phixp[j] - phixm[i] * phixm[j]);
                        v_acc[(i, j)] += wy * (phiyp[i] * phiyp[j] - phiym[i] * phiym[j]);
                        v_acc[(i, j)] += wz * (phizp[i] * phizp[j] - phizm[i] * phizm[j]);
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

fn rho_from_phi(density: &DMatrix<f64>, phi: &[f64]) -> f64 {
    let n = phi.len();
    let mut tmp = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += density[(i, j)] * phi[j];
        }
        tmp[i] = s;
    }
    let mut rho = 0.0;
    for i in 0..n {
        rho += phi[i] * tmp[i];
    }
    rho
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


