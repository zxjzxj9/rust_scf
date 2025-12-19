//! Minimal grid-based DFT helpers (LDA / PBE / TPSS on a Becke atom grid).
//!
//! This is intentionally simple:
//! - Atom-centered grid: Gauss–Legendre radial × 6-point octahedral angular grid
//! - Becke partitioning to avoid double-counting overlapping atom grids
//! - LDA exchange (Slater), PBE GGA (x / xc), TPSS meta-GGA (xc, unpolarized)

extern crate nalgebra as na;

use na::{DMatrix, Vector3};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XcFunctional {
    /// Local-density approximation exchange only (Slater exchange)
    LdaX,
    /// PBE GGA exchange only (no correlation)
    PbeX,
    /// Full PBE GGA exchange + correlation (unpolarized)
    PbeXc,
    /// Full TPSS meta-GGA exchange + correlation (unpolarized)
    TpssXc,
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

// ---------------------------------------------------------------------------
// PW92 LDA correlation (unpolarized) + PBE correlation gradient correction
// ---------------------------------------------------------------------------

#[inline]
fn pbe_beta() -> f64 {
    0.066_724_550_603_149_22
}

#[inline]
fn pbe_gamma() -> f64 {
    0.031_090_690_869_654_895
}

/// Wigner-Seitz radius rs(ρ) = (3/(4πρ))^(1/3)
#[inline]
fn rs_from_rho(rho: f64) -> f64 {
    (3.0 / (4.0 * std::f64::consts::PI * rho)).powf(1.0 / 3.0)
}

/// PW92 LDA correlation energy per particle ε_c(rs) for unpolarized system, and derivative dε/drs.
///
/// Reference: Perdew & Wang, Phys. Rev. B 45, 13244 (1992) (parametrization).
fn pw92_eps_c_unpol_and_deriv(rs: f64) -> (f64, f64) {
    // Constants for unpolarized case
    let a = 0.031_090_7;
    let a1 = 0.213_70;
    let b1 = 7.595_7;
    let b2 = 3.587_6;
    let b3 = 1.638_2;
    let b4 = 0.492_94;

    let s = rs.sqrt();
    let q = 2.0 * a * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs);
    let x = 1.0 + 1.0 / q;
    let eps = -2.0 * a * (1.0 + a1 * rs) * x.ln();

    // derivative
    // eps = -2a(1+a1 rs) ln(1+1/q)
    // d/d rs: -2a[ a1 ln(x) + (1+a1 rs) * (1/x) * d x/d rs ]
    // x = 1 + 1/q => dx/drs = -(1/q^2) dq/drs
    // dq/drs = 2a[ b1*(1/(2sqrt(rs))) + b2 + b3*(3/2)s + b4*2rs ]
    let dq_drs = 2.0
        * a
        * (b1 * (0.5 / s)
            + b2
            + b3 * (1.5 * s)
            + b4 * (2.0 * rs));
    let dx_drs = -(dq_drs) / (q * q);
    let deps_drs = -2.0
        * a
        * (a1 * x.ln() + (1.0 + a1 * rs) * (1.0 / x) * dx_drs);

    (eps, deps_drs)
}

/// Compute PBE correlation energy density (per volume) and partial derivatives w.r.t rho and grad_rho.
///
/// Uses unpolarized formulation: e_c = ρ(ε_c^LDA(rs) + H(rs, t)).
fn pbe_c_energy_density_and_partials(rho: f64, grad_rho: Vector3<f64>) -> (f64, f64, Vector3<f64>) {
    if rho <= 0.0 {
        return (0.0, 0.0, Vector3::zeros());
    }
    let g = grad_rho.norm();

    let rs = rs_from_rho(rho);
    let (eps_lda, deps_drs) = pw92_eps_c_unpol_and_deriv(rs);

    // drs/drho = -(1/3) rs / rho
    let drs_drho = -(1.0 / 3.0) * rs / rho;
    let deps_drho = deps_drs * drs_drho;

    // Correlation reduced gradient t = |∇ρ| / (2 k_s ρ)
    // k_s = sqrt(4 k_F / π), k_F = (3π^2 ρ)^(1/3)
    let kf = (3.0 * std::f64::consts::PI * std::f64::consts::PI * rho).powf(1.0 / 3.0);
    let ks = (4.0 * kf / std::f64::consts::PI).sqrt();
    let denom = 2.0 * rho * ks;
    let t = if denom > 0.0 { g / denom } else { 0.0 };

    let gamma = pbe_gamma();
    let beta = pbe_beta();

    // A = (β/γ) / (exp(-ε_c^LDA/γ) - 1)
    let exp_arg = (-eps_lda / gamma).exp();
    let b = exp_arg - 1.0;
    let a_corr = if b.abs() > 1e-14 { (beta / gamma) / b } else { 0.0 };

    // H = γ ln(1 + Q), Q = (β/γ) t^2 * (1 + u) / (1 + u + u^2), u = A t^2
    let u = a_corr * t * t;
    let den_u = 1.0 + u + u * u;
    let f = if den_u > 0.0 { (1.0 + u) / den_u } else { 0.0 };
    let q = (beta / gamma) * t * t * f;
    let h = gamma * (1.0 + q).ln();

    // Derivatives: dH/dt and dH/dA (via u)
    let dH_dQ = if (1.0 + q) > 0.0 { gamma / (1.0 + q) } else { 0.0 };

    // f'(u) = -(u(2+u)) / (1+u+u^2)^2
    let fprime = if den_u > 0.0 {
        -(u * (2.0 + u)) / (den_u * den_u)
    } else {
        0.0
    };

    // dQ/dt = (β/γ)[2 t f + t^2 f'(u) * d u/dt], du/dt = 2 A t
    let dQ_dt = (beta / gamma) * (2.0 * t * f + t * t * fprime * (2.0 * a_corr * t));
    let dH_dt = dH_dQ * dQ_dt;

    // dQ/dA = (β/γ) t^2 * f'(u) * du/dA, du/dA = t^2
    let dQ_dA = (beta / gamma) * (t * t) * fprime * (t * t);
    let dH_dA = dH_dQ * dQ_dA;

    // dA/dε = (β/γ) * exp(-ε/γ) / (γ (exp(-ε/γ)-1)^2)
    let dA_deps = if b.abs() > 1e-14 {
        (beta / gamma) * exp_arg / (gamma * b * b)
    } else {
        0.0
    };
    let dA_drho = dA_deps * deps_drho;

    // dt/drho = -(7/6) t / rho  (derived for ks ∝ ρ^(1/6))
    let dt_drho = if rho > 0.0 { -(7.0 / 6.0) * t / rho } else { 0.0 };
    // dt/d(∇ρ) = (1/denom) * ∇ρ/|∇ρ|
    let dt_dgrad = if g > 1e-14 && denom > 0.0 {
        (1.0 / denom) * (grad_rho / g)
    } else {
        Vector3::zeros()
    };

    // Total derivatives of H
    let dH_drho = dH_dA * dA_drho + dH_dt * dt_drho;
    let dH_dgrad = dH_dt * dt_dgrad;

    // e_c = ρ (ε_lda + H)
    let e = rho * (eps_lda + h);
    let de_drho = eps_lda + h + rho * (deps_drho + dH_drho);
    let de_dgrad = rho * dH_dgrad;

    (e, de_drho, de_dgrad)
}

fn pbe_xc_energy_density_and_partials(rho: f64, grad_rho: Vector3<f64>) -> (f64, f64, Vector3<f64>) {
    let (ex, dex_drho, dex_dgrad) = pbe_x_energy_density_and_partials(rho, grad_rho);
    let (ec, dec_drho, dec_dgrad) = pbe_c_energy_density_and_partials(rho, grad_rho);
    (ex + ec, dex_drho + dec_drho, dex_dgrad + dec_dgrad)
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

/// Compute (E_xc, V_xc) for either LDA-x, PBE-x, or PBE-xc on a fixed grid using AO
/// values and AO analytic gradients (∇φ).
///
/// This avoids finite-difference gradients of the density and is substantially faster for GGA.
///
/// - `basis_values_and_grads(r)` must return `(phi, grad_phi)` where:
///   - `phi[i] = φ_i(r)`
///   - `grad_phi[i] = ∇φ_i(r)`
pub fn xc_on_grid_ao_grad<F>(
    functional: XcFunctional,
    density: &DMatrix<f64>,
    num_basis: usize,
    grid: &[GridPoint],
    basis_values_and_grads: F,
) -> (f64, DMatrix<f64>)
where
    F: Fn(&Vector3<f64>) -> (Vec<f64>, Vec<Vector3<f64>>) + Sync,
{
    match functional {
        XcFunctional::LdaX => {
            // LDA does not depend on ∇ρ; fall back to the simpler (and faster) implementation.
            lda_xc_on_grid(density, num_basis, grid, |r| basis_values_and_grads(r).0)
        }
        XcFunctional::PbeX => pbe_xc_on_grid_ao_grad_impl(density, num_basis, grid, basis_values_and_grads, false),
        XcFunctional::PbeXc => pbe_xc_on_grid_ao_grad_impl(density, num_basis, grid, basis_values_and_grads, true),
        XcFunctional::TpssXc => tpss_xc_on_grid_ao_grad_impl(density, num_basis, grid, basis_values_and_grads),
    }
}

fn pbe_xc_on_grid_ao_grad_impl<F>(
    density: &DMatrix<f64>,
    num_basis: usize,
    grid: &[GridPoint],
    basis_values_and_grads: F,
    include_correlation: bool,
) -> (f64, DMatrix<f64>)
where
    F: Fn(&Vector3<f64>) -> (Vec<f64>, Vec<Vector3<f64>>) + Sync,
{
    let (e_xc, v_xc) = grid
        .par_iter()
        .fold(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(mut e_acc, mut v_acc), gp| {
                let (phi, grad_phi) = basis_values_and_grads(&gp.r);
                if phi.len() != num_basis || grad_phi.len() != num_basis {
                    return (e_acc, v_acc);
                }

                // tmp = P * phi, tmp_t = P^T * phi
                let mut tmp = vec![0.0_f64; num_basis];
                let mut tmp_t = vec![0.0_f64; num_basis];
                for i in 0..num_basis {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    for j in 0..num_basis {
                        s1 += density[(i, j)] * phi[j];
                        s2 += density[(j, i)] * phi[j];
                    }
                    tmp[i] = s1;
                    tmp_t[i] = s2;
                }

                let mut rho = 0.0;
                for i in 0..num_basis {
                    rho += phi[i] * tmp[i];
                }

                if !rho.is_finite() || rho <= 1e-14 {
                    return (e_acc, v_acc);
                }

                // ∇ρ = (∇φ)^T P φ + φ^T P ∇φ = Σ_i (tmp_i + tmp_t_i) ∇φ_i
                let mut grad_rho = Vector3::zeros();
                for i in 0..num_basis {
                    grad_rho += (tmp[i] + tmp_t[i]) * grad_phi[i];
                }

                let (e, de_drho, de_dgrad) = if include_correlation {
                    pbe_xc_energy_density_and_partials(rho, grad_rho)
                } else {
                    pbe_x_energy_density_and_partials(rho, grad_rho)
                };

                e_acc += gp.w * e;

                // Vxc_ij = w * [ (∂e/∂ρ) φ_i φ_j + (∂e/∂∇ρ) · (φ_i ∇φ_j + φ_j ∇φ_i) ]
                let w0 = gp.w * de_drho;
                let wg = gp.w * de_dgrad;
                for i in 0..num_basis {
                    for j in 0..num_basis {
                        let phi_i = phi[i];
                        let phi_j = phi[j];
                        let gterm = wg.dot(&(phi_i * grad_phi[j] + phi_j * grad_phi[i]));
                        v_acc[(i, j)] += w0 * phi_i * phi_j + gterm;
                    }
                }

                (e_acc, v_acc)
            },
        )
        .reduce(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(e1, v1), (e2, v2)| (e1 + e2, v1 + v2),
        );

    (e_xc, v_xc)
}

// ---------------------------------------------------------------------------
// TPSS meta-GGA (unpolarized) helpers
// ---------------------------------------------------------------------------

#[inline]
fn tpss_c() -> f64 {
    1.590_96
}

#[inline]
fn tpss_e() -> f64 {
    1.537
}

#[inline]
fn tpss_b() -> f64 {
    0.40
}

#[inline]
fn tpss_d() -> f64 {
    2.8 // hartree^-1
}

#[inline]
fn tpss_kappa() -> f64 {
    0.804
}

/// Uniform-gas exchange energy per particle: ε_x^unif(n) = -(3/(4π)) (3π^2 n)^(1/3)
#[inline]
fn eps_x_unif(n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    -3.0 / (4.0 * std::f64::consts::PI)
        * (3.0 * std::f64::consts::PI * std::f64::consts::PI * n).powf(1.0 / 3.0)
}

/// Uniform-gas kinetic energy density τ_unif(n) = (3/10) (3π^2)^(2/3) n^(5/3)
#[inline]
fn tau_unif(n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    (3.0 / 10.0) * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0) * n.powf(5.0 / 3.0)
}

/// PW92 LDA correlation ε_c(rs) per particle for unpolarized or fully polarized cases.
/// (Perdew & Wang, Phys. Rev. B 45, 13244 (1992))
fn pw92_eps_c(rs: f64, fully_polarized: bool) -> f64 {
    if rs <= 0.0 || !rs.is_finite() {
        return 0.0;
    }
    // Constants for the PW92 parametrization.
    // Unpolarized (ζ=0) and fully polarized (ζ=1) parameter sets.
    let (a, a1, b1, b2, b3, b4) = if fully_polarized {
        // ζ = 1
        (0.015_545_35, 0.205_48, 14.118_9, 6.197_7, 3.366_2, 0.625_17)
    } else {
        // ζ = 0
        (0.031_090_7, 0.213_70, 7.595_7, 3.587_6, 1.638_2, 0.492_94)
    };

    let s = rs.sqrt();
    let q = 2.0 * a * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs);
    let x = 1.0 + 1.0 / q;
    -2.0 * a * (1.0 + a1 * rs) * x.ln()
}

#[inline]
fn rs_from_n(n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    (3.0 / (4.0 * std::f64::consts::PI * n)).powf(1.0 / 3.0)
}

#[inline]
fn phi_from_zeta(zeta: f64) -> f64 {
    // φ(ζ) = [ (1+ζ)^(2/3) + (1-ζ)^(2/3) ] / 2
    let zp = (1.0 + zeta).max(0.0);
    let zm = (1.0 - zeta).max(0.0);
    0.5 * (zp.powf(2.0 / 3.0) + zm.powf(2.0 / 3.0))
}

/// PBE correlation ε_c per particle for a (possibly spin-polarized) density, using the standard PBE form:
/// ε_c = ε_c^LDA(rs, ζ) + H(rs, ζ, t)
/// with H = γ φ^3 ln(1 + (β/γ) t^2 (1 + A t^2)/(1 + A t^2 + (A t^2)^2)),
/// A = (β/γ) / (exp(-ε_c^LDA/(γ φ^3)) - 1),
/// t = |∇n| / (2 φ k_s n), k_s = sqrt(4 k_F/π), k_F = (3π^2 n)^(1/3).
fn pbe_eps_c(n: f64, grad_n: Vector3<f64>, zeta: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let g = grad_n.norm();
    let rs = rs_from_n(n);
    let fully_polarized = zeta >= 1.0 - 1e-14;
    let eps_lda = pw92_eps_c(rs, fully_polarized);

    let phi = phi_from_zeta(zeta);
    let phi3 = phi * phi * phi;

    let kf = (3.0 * std::f64::consts::PI * std::f64::consts::PI * n).powf(1.0 / 3.0);
    let ks = (4.0 * kf / std::f64::consts::PI).sqrt();
    let denom = 2.0 * phi * ks * n;
    let t = if denom > 0.0 { g / denom } else { 0.0 };

    let gamma = pbe_gamma();
    let beta = pbe_beta();

    let exp_arg = (-eps_lda / (gamma * phi3)).exp();
    let b = exp_arg - 1.0;
    let a_corr = if b.abs() > 1e-14 { (beta / gamma) / b } else { 0.0 };

    let u = a_corr * t * t;
    let den_u = 1.0 + u + u * u;
    let f = if den_u > 0.0 { (1.0 + u) / den_u } else { 0.0 };
    let q = (beta / gamma) * t * t * f;
    let h = gamma * phi3 * (1.0 + q).ln();

    eps_lda + h
}

/// TPSS exchange enhancement factor F_x(p,z) for unpolarized case.
fn tpss_fx(p: f64, z: f64, qtilde_b: f64) -> f64 {
    let kappa = tpss_kappa();
    let c = tpss_c();
    let e = tpss_e();
    let mu = pbe_mu();

    let z2 = z * z;
    let one_plus_z2 = 1.0 + z2;

    // Eq. (10) from Tao et al. PRL 91, 146401 (2003).
    // x = { [10/81 + c z^2/(1+z^2)^2] p + 146/2025 q~_b^2
    //       - 73/405 q~_b sqrt( 1/2 (3/5 z)^2 + 1/2 p^2 )
    //       + (1/κ)(10/81)^2 p^2
    //       + 2√e (10/81) (3/5 z)^2
    //       + e μ p^3 } / (1 + √e p)^2
    let term1 = (10.0 / 81.0 + c * (z2 / (one_plus_z2 * one_plus_z2))) * p;
    let term2 = (146.0 / 2025.0) * qtilde_b * qtilde_b;
    let sqrt_arg = 0.5 * (3.0 / 5.0 * z).powi(2) + 0.5 * p * p;
    let term3 = -(73.0 / 405.0) * qtilde_b * sqrt_arg.max(0.0).sqrt();
    let a = 10.0_f64 / 81.0;
    let term4 = (1.0 / kappa) * (a * a) * p * p;
    let term5 = 2.0 * e.sqrt() * (10.0 / 81.0) * (3.0 / 5.0 * z).powi(2);
    let term6 = e * mu * p * p * p;

    let denom = (1.0 + e.sqrt() * p).powi(2);
    let x = (term1 + term2 + term3 + term4 + term5 + term6) / denom;

    // Eq. (5): F_x = 1 + κ - κ/(1 + x/κ)
    let t = 1.0 + x / kappa;
    1.0 + kappa - kappa / t
}

/// TPSS exchange + correlation energy density per volume e_xc(n, |∇n|^2, τ) for unpolarized case.
fn tpss_xc_energy_density(n: f64, sigma: f64, tau: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let sigma = sigma.max(0.0);
    let tau = tau.max(0.0);

    let tau_w = sigma / (8.0 * n);
    let z = if tau > 1e-20 { (tau_w / tau).clamp(0.0, 1.0) } else { 1.0 };
    let t_unif = tau_unif(n);
    let alpha = if t_unif > 1e-30 { (tau - tau_w) / t_unif } else { 0.0 };

    // p = s^2 = |∇n|^2 / [4 (3π^2)^(2/3) n^(8/3)]
    let denom_p = 4.0 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0) * n.powf(8.0 / 3.0);
    let p = if denom_p > 0.0 { sigma / denom_p } else { 0.0 };

    // q~_b (Eq. 7)
    let b = tpss_b();
    let qtilde_b = {
        let num = alpha - 1.0;
        let den = (1.0 + b * alpha * (alpha - 1.0)).max(1e-14).sqrt();
        (9.0 / 20.0) * (num / den) + (2.0 / 3.0) * p
    };

    // Exchange: e_x = n ε_x^unif(n) F_x(p,z)
    let fx = tpss_fx(p, z, qtilde_b);
    let ex = n * eps_x_unif(n) * fx;

    // Correlation (TPSS revPKZB construction, Eqs. 11-14) for unpolarized density:
    // ζ = 0 => C(ζ, ξ) = C(0,0) = 0.53 (since ∇ζ = 0 => ξ = 0)
    let c_self = 0.53;
    let d = tpss_d();

    // ε_c^PBE(n↑=n/2,n↓=n/2)   and   ε_c^PBE(n↑=n/2,n↓=0) for the "tilde" term.
    // For the spin-channel densities, we approximate |∇nσ| = |∇n|/2 with the same direction.
    let grad_n = if sigma > 0.0 {
        // direction doesn't matter for |∇n| in PBE correlation formula, but we need a Vector3.
        // Put everything on x-axis for determinism.
        Vector3::new(sigma.sqrt(), 0.0, 0.0)
    } else {
        Vector3::zeros()
    };
    let eps_pbe_unpol = pbe_eps_c(n, grad_n, 0.0);
    let eps_pbe_spin_channel = pbe_eps_c(0.5 * n, 0.5 * grad_n, 1.0);
    let eps_tilde = eps_pbe_spin_channel.max(eps_pbe_unpol);

    // Eq. (12) unpolarized simplification (Σσ nσ/n = 1):
    let z2 = z * z;
    let eps_rev = eps_pbe_unpol * (1.0 + c_self * z2) - (1.0 + c_self) * z2 * eps_tilde;

    // Eq. (11): e_c = n ε_rev [1 + d ε_rev z^3]
    let ec = n * eps_rev * (1.0 + d * eps_rev * z.powi(3));

    ex + ec
}

/// Numerical partial derivatives for TPSS meta-GGA:
/// returns (e, ∂e/∂n, ∂e/∂∇n (vector), ∂e/∂τ)
fn tpss_xc_energy_density_and_partials_numeric(
    n: f64,
    grad_n: Vector3<f64>,
    tau: f64,
) -> (f64, f64, Vector3<f64>, f64) {
    if n <= 0.0 {
        return (0.0, 0.0, Vector3::zeros(), 0.0);
    }
    let sigma = grad_n.dot(&grad_n).max(0.0);
    let tau = tau.max(0.0);

    let e0 = tpss_xc_energy_density(n, sigma, tau);

    // Steps (relative with floors), keep positivity where needed.
    let dn = (1e-6 * n).max(1e-8);
    let ds = (1e-6 * sigma).max(1e-10);
    let dt = (1e-6 * tau).max(1e-10);

    // ∂e/∂n
    let (n1, n2) = if n > dn { (n - dn, n + dn) } else { (n, n + dn) };
    let e_n1 = tpss_xc_energy_density(n1, sigma, tau);
    let e_n2 = tpss_xc_energy_density(n2, sigma, tau);
    let de_dn = if n1 != n2 { (e_n2 - e_n1) / (n2 - n1) } else { 0.0 };

    // ∂e/∂σ
    let (s1, s2) = if sigma > ds { (sigma - ds, sigma + ds) } else { (sigma, sigma + ds) };
    let e_s1 = tpss_xc_energy_density(n, s1, tau);
    let e_s2 = tpss_xc_energy_density(n, s2, tau);
    let de_dsigma = if s1 != s2 { (e_s2 - e_s1) / (s2 - s1) } else { 0.0 };

    // ∂e/∂τ
    let (t1, t2) = if tau > dt { (tau - dt, tau + dt) } else { (tau, tau + dt) };
    let e_t1 = tpss_xc_energy_density(n, sigma, t1);
    let e_t2 = tpss_xc_energy_density(n, sigma, t2);
    let de_dtau = if t1 != t2 { (e_t2 - e_t1) / (t2 - t1) } else { 0.0 };

    // ∂e/∂∇n = 2 (∂e/∂σ) ∇n
    let de_dgrad = 2.0 * de_dsigma * grad_n;

    (e0, de_dn, de_dgrad, de_dtau)
}

fn tpss_xc_on_grid_ao_grad_impl<F>(
    density: &DMatrix<f64>,
    num_basis: usize,
    grid: &[GridPoint],
    basis_values_and_grads: F,
) -> (f64, DMatrix<f64>)
where
    F: Fn(&Vector3<f64>) -> (Vec<f64>, Vec<Vector3<f64>>) + Sync,
{
    let (e_xc, v_xc) = grid
        .par_iter()
        .fold(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(mut e_acc, mut v_acc), gp| {
                let (phi, grad_phi) = basis_values_and_grads(&gp.r);
                if phi.len() != num_basis || grad_phi.len() != num_basis {
                    return (e_acc, v_acc);
                }

                // tmp = P * phi, tmp_t = P^T * phi
                let mut tmp = vec![0.0_f64; num_basis];
                let mut tmp_t = vec![0.0_f64; num_basis];
                for i in 0..num_basis {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    for j in 0..num_basis {
                        s1 += density[(i, j)] * phi[j];
                        s2 += density[(j, i)] * phi[j];
                    }
                    tmp[i] = s1;
                    tmp_t[i] = s2;
                }

                let mut rho = 0.0;
                for i in 0..num_basis {
                    rho += phi[i] * tmp[i];
                }
                if !rho.is_finite() || rho <= 1e-14 {
                    return (e_acc, v_acc);
                }

                // ∇ρ = Σ_i (tmp_i + tmp_t_i) ∇φ_i
                let mut grad_rho = Vector3::zeros();
                for i in 0..num_basis {
                    grad_rho += (tmp[i] + tmp_t[i]) * grad_phi[i];
                }

                // τ = 1/2 Σ_ij P_ij ∇φ_i · ∇φ_j
                let mut gx = vec![0.0_f64; num_basis];
                let mut gy = vec![0.0_f64; num_basis];
                let mut gz = vec![0.0_f64; num_basis];
                for i in 0..num_basis {
                    gx[i] = grad_phi[i].x;
                    gy[i] = grad_phi[i].y;
                    gz[i] = grad_phi[i].z;
                }
                let mut pgx = vec![0.0_f64; num_basis];
                let mut pgy = vec![0.0_f64; num_basis];
                let mut pgz = vec![0.0_f64; num_basis];
                let mut pgx_t = vec![0.0_f64; num_basis];
                let mut pgy_t = vec![0.0_f64; num_basis];
                let mut pgz_t = vec![0.0_f64; num_basis];
                for i in 0..num_basis {
                    let mut sx1 = 0.0;
                    let mut sy1 = 0.0;
                    let mut sz1 = 0.0;
                    let mut sx2 = 0.0;
                    let mut sy2 = 0.0;
                    let mut sz2 = 0.0;
                    for j in 0..num_basis {
                        let pij = density[(i, j)];
                        let pji = density[(j, i)];
                        sx1 += pij * gx[j];
                        sy1 += pij * gy[j];
                        sz1 += pij * gz[j];
                        sx2 += pji * gx[j];
                        sy2 += pji * gy[j];
                        sz2 += pji * gz[j];
                    }
                    pgx[i] = sx1;
                    pgy[i] = sy1;
                    pgz[i] = sz1;
                    pgx_t[i] = sx2;
                    pgy_t[i] = sy2;
                    pgz_t[i] = sz2;
                }
                let dot_sym = |a: &Vec<f64>, pa: &Vec<f64>, pat: &Vec<f64>| -> f64 {
                    let mut s = 0.0;
                    for i in 0..num_basis {
                        s += 0.5 * (pa[i] + pat[i]) * a[i];
                    }
                    s
                };
                let tau = 0.5 * (dot_sym(&gx, &pgx, &pgx_t) + dot_sym(&gy, &pgy, &pgy_t) + dot_sym(&gz, &pgz, &pgz_t));

                let (e, de_drho, de_dgrad, de_dtau) = tpss_xc_energy_density_and_partials_numeric(rho, grad_rho, tau);

                e_acc += gp.w * e;

                // Vxc_ij = w * [ (∂e/∂ρ) φ_i φ_j
                //               + (∂e/∂∇ρ)·(φ_i ∇φ_j + φ_j ∇φ_i)
                //               + (∂e/∂τ) * (1/2) (∇φ_i · ∇φ_j) ]
                let w0 = gp.w * de_drho;
                let wg = gp.w * de_dgrad;
                let wt = gp.w * de_dtau;
                for i in 0..num_basis {
                    for j in 0..num_basis {
                        let phi_i = phi[i];
                        let phi_j = phi[j];
                        let gterm = wg.dot(&(phi_i * grad_phi[j] + phi_j * grad_phi[i]));
                        let tterm = 0.5 * wt * grad_phi[i].dot(&grad_phi[j]);
                        v_acc[(i, j)] += w0 * phi_i * phi_j + gterm + tterm;
                    }
                }

                (e_acc, v_acc)
            },
        )
        .reduce(
            || (0.0_f64, DMatrix::<f64>::zeros(num_basis, num_basis)),
            |(e1, v1), (e2, v2)| (e1 + e2, v1 + v2),
        );

    (e_xc, v_xc)
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


