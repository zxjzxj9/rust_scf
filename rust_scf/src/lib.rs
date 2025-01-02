extern crate ndarray;

use ndarray::prelude::*;
use ndarray_linalg::*;
use std::collections::VecDeque;

/// L-BFGS algorithm implementation.
///
/// # Arguments
///
/// * `f` - The objective function to minimize.
/// * `grad` - The gradient of the objective function.
/// * `x0` - The initial point.
/// * `m` - The number of updates to store for the limited-memory approximation.
/// * `max_iter` - The maximum number of iterations.
/// * `tolerance` - The tolerance for convergence.
/// * `verbose` - Whether to print iteration information.
///
/// # Returns
///
/// The optimal point found by the algorithm, or an error if one occurred.
pub fn lbfgs<F, G>(
    f: F,
    grad: G,
    x0: Array1<f64>,
    m: usize,
    max_iter: usize,
    tolerance: f64,
    verbose: bool,
) -> Result<Array1<f64>, &'static str>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n = x0.len();
    let mut x = x0.clone();
    let mut g = grad(&x);

    let mut s_history = VecDeque::with_capacity(m);
    let mut y_history = VecDeque::with_capacity(m);
    let mut rho_history = VecDeque::with_capacity(m);

    for iter in 0..max_iter {
        if verbose {
            println!("Iteration: {}", iter);
            println!("x: {:?}", x);
            println!("g: {:?}", g);
            println!("f(x): {}", f(&x));
        }

        // Check for convergence
        if  g.norm_l2() < tolerance {
            if verbose {
                println!("Converged after {} iterations.", iter);
            }
            return Ok(x);
        }

        // Compute the search direction
        let q = g.clone();
        let p = compute_search_direction(q, &s_history, &y_history, &rho_history);

        // Perform line search
        let alpha = line_search(&f, &grad, &x, &p, &g, tolerance)?;

        // Update the current point and gradient
        let s = alpha * &p;
        let x_next = &x + &s;
        let g_next = grad(&x_next);
        let y = &g_next - &g;

        // Update history
        if s_history.len() == m {
            s_history.pop_front();
            y_history.pop_front();
            rho_history.pop_front();
        }

        let rho = 1.0 / (y.dot(&s));
        if !rho.is_finite() {
            return Err("Invalid rho value encountered during optimization. This typically indicates an issue with the gradient or the objective function");
        }

        s_history.push_back(s);
        y_history.push_back(y);
        rho_history.push_back(rho);

        x = x_next;
        g = g_next;
    }

    if verbose {
        println!("Maximum iterations reached.");
    }
    Ok(x)
}

/// Computes the L-BFGS search direction.
fn compute_search_direction(
    mut q: Array1<f64>,
    s_history: &VecDeque<Array1<f64>>,
    y_history: &VecDeque<Array1<f64>>,
    rho_history: &VecDeque<f64>,
) -> Array1<f64> {
    let m = s_history.len();
    let mut alphas = vec![0.0; m];

    for i in (0..m).rev() {
        alphas[i] = rho_history[i] * s_history[i].dot(&q);
        q = &q - alphas[i] * &y_history[i];
    }

    let gamma = if m > 0 {
        s_history.back().unwrap().dot(y_history.back().unwrap())
            / y_history.back().unwrap().dot(y_history.back().unwrap())
    } else {
        1.0
    };

    let mut r = gamma * q;

    for i in 0..m {
        let beta = rho_history[i] * y_history[i].dot(&r);
        r = &r + s_history[i].clone()* (alphas[i] - beta);
    }

    -r
}

/// Performs a simple backtracking line search.
fn line_search<F, G>(
    f: F,
    grad: G,
    x: &Array1<f64>,
    p: &Array1<f64>,
    g: &Array1<f64>,
    tolerance: f64
) -> Result<f64, &'static str>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let c2 = 0.9;
    let fx = f(x);
    let initial_slope = g.dot(p);

    // Check if the initial slope is positive. This would indicate that the search direction is not a descent direction.
    if initial_slope > 0.0 {
        return Err("Initial slope is positive, indicating that the search direction is not a descent direction. This can be caused by an incorrect gradient or a non-convex objective function.");
    }


    loop {
        let x_new = x + alpha * p;
        let fx_new = f(&x_new);

        // Check Wolfe conditions
        if fx_new <= fx + c1 * alpha * initial_slope {
            let g_new = grad(&x_new);
            if g_new.dot(p) >= c2 * initial_slope {
                return Ok(alpha);
            }
        }

        // Reduce alpha
        alpha *= 0.5;

        // Check if alpha is too small
        if alpha < tolerance {
            return Err("Line search failed to find a suitable step size.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgs_quadratic() {
        // Objective function: f(x) = (x1 - 2)^2 + (x2 - 3)^2
        let f = |x: &Array1<f64>| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        // Gradient of the objective function
        let grad = |x: &Array1<f64>| array![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)];

        // Initial point
        let x0 = array![0.0, 0.0];

        // Run L-BFGS
        let result = lbfgs(f, grad, x0, 5, 100, 1e-6, false).unwrap();

        // Check if the result is close to the optimum (2, 3)
        assert!((result[0] - 2.0).abs() < 1e-4);
        assert!((result[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        // Rosenbrock function: f(x) = (1 - x1)^2 + 100(x2 - x1^2)^2
        let f = |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);

        // Gradient of the Rosenbrock function
        let grad = |x: &Array1<f64>| {
            array![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
                200.0 * (x[1] - x[0].powi(2)),
            ]
        };

        // Initial point
        let x0 = array![0.0, 0.0];

        // Run L-BFGS
        let result = lbfgs(f, grad, x0, 5, 100, 1e-6, false).unwrap();

        // Check if the result is close to the optimum (1, 1)
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_lbfgs_invalid_rho() {
        // Objective function: f(x) = x1^4 + x2^4
        // This function can cause numerical instability for certain initial points
        // when using a backtracking line search with insufficient precision.
        let f = |x: &Array1<f64>| x[0].powi(4) + x[1].powi(4);

        // Gradient of the objective function
        let grad = |x: &Array1<f64>| array![4.0 * x[0].powi(3), 4.0 * x[1].powi(3)];

        // Initial point that might cause issues
        let x0 = array![1.0, 1.0];

        // Run L-BFGS
        let result = lbfgs(f, grad, x0, 5, 100, 1e-6, true);

        // Check for an error due to an invalid rho value
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Invalid rho value encountered during optimization. This typically indicates an issue with the gradient or the objective function");
    }

    #[test]
    fn test_lbfgs_non_descent_direction() {
        // Define a function and a gradient that will result in a non-descent direction.
        // We intentionally define an incorrect gradient that points away from the minimum.

        // Objective function: f(x) = x1^2 + x2^2
        let f = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);

        // Incorrect gradient (should be [2*x1, 2*x2], but we use [-2*x1, -2*x2])
        let grad = |x: &Array1<f64>| array![-2.0 * x[0], -2.0 * x[1]];

        // Initial point
        let x0 = array![1.0, 1.0];

        // Run L-BFGS
        let result = lbfgs(f, grad, x0, 5, 100, 1e-6, true);

        // Check for an error due to a non-descent direction
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Initial slope is positive, indicating that the search direction is not a descent direction. This can be caused by an incorrect gradient or a non-convex objective function");
    }
}